import ast
import re
import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = EXPERIMENT_ROOT / "run_all_variants.sh"

MAX_PYTHON_FILE_LINES = 1000
MAX_FUNCTION_LINES = 300
MAX_LAUNCHER_LINES = 100

EXCLUDED_SOURCE_DIRECTORIES = {
    ".aidd-local",
    "__pycache__",
    "evaluation",
    "graphify-out",
    "results",
    "tests",
}
EXCLUDED_SOURCE_FILES = {"validate_experiment_release.py"}

DYNAMIC_IMPORT_CALLS = {
    "__import__",
    "importlib.import_module",
    "importlib.util.module_from_spec",
    "importlib.util.spec_from_file_location",
}
PYTHON_HEREDOC = re.compile(
    r"<<-?\s*(?P<quote>['\"]?)(?:PY|PYTHON)(?P=quote)\s*$"
)


def _active_python_files():
    """Return generation/control sources, never evaluators or test artifacts."""
    active_files = []
    for path in EXPERIMENT_ROOT.rglob("*.py"):
        relative_path = path.relative_to(EXPERIMENT_ROOT)
        if relative_path.name in EXCLUDED_SOURCE_FILES:
            continue
        if any(
            part in EXCLUDED_SOURCE_DIRECTORIES
            for part in relative_path.parts[:-1]
        ):
            continue
        active_files.append(path)
    return sorted(active_files)


def _relative(path):
    return path.relative_to(EXPERIMENT_ROOT).as_posix()


def _parsed(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _failure_message(contract, violations):
    rendered = "\n".join(f"  - {violation}" for violation in violations)
    return f"{contract}\n{rendered}"


class GenerationArchitectureContractTests(unittest.TestCase):
    def test_active_sources_use_explicit_imports(self):
        violations = []
        for path in _active_python_files():
            for node in ast.walk(_parsed(path)):
                if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "*" for alias in node.names
                ):
                    module = node.module or "."
                    violations.append(
                        f"{_relative(path)}:{node.lineno}: "
                        f"wildcard import from {module}"
                    )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "Active generation/control code must use explicit imports.",
                violations,
            ),
        )

    def test_active_sources_do_not_load_modules_dynamically(self):
        violations = []
        for path in _active_python_files():
            for node in ast.walk(_parsed(path)):
                if isinstance(node, ast.Call):
                    called_name = _dotted_name(node.func)
                    if (
                        called_name in DYNAMIC_IMPORT_CALLS
                        or (
                            isinstance(called_name, str)
                            and called_name.endswith(".exec_module")
                        )
                    ):
                        violations.append(
                            f"{_relative(path)}:{node.lineno}: "
                            f"dynamic module loading via {called_name}"
                        )
                if (
                    isinstance(node, ast.Attribute)
                    and _dotted_name(node) == "sys.modules"
                ):
                    violations.append(
                        f"{_relative(path)}:{node.lineno}: "
                        "runtime mutation/access through sys.modules"
                    )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "Active sources must be an importable package without dynamic "
                "module loading or sys.modules manipulation.",
                violations,
            ),
        )

    def test_active_python_modules_are_at_most_one_thousand_lines(self):
        violations = []
        for path in _active_python_files():
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_PYTHON_FILE_LINES:
                violations.append(
                    f"{_relative(path)}: {line_count} lines "
                    f"(maximum {MAX_PYTHON_FILE_LINES})"
                )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "Active Python modules must have one focused responsibility.",
                violations,
            ),
        )

    def test_active_functions_are_at_most_three_hundred_lines(self):
        violations = []
        for path in _active_python_files():
            for node in ast.walk(_parsed(path)):
                if not isinstance(
                    node,
                    (ast.AsyncFunctionDef, ast.FunctionDef),
                ):
                    continue
                line_count = node.end_lineno - node.lineno + 1
                if line_count > MAX_FUNCTION_LINES:
                    violations.append(
                        f"{_relative(path)}:{node.lineno}: {node.name} spans "
                        f"{line_count} lines (maximum {MAX_FUNCTION_LINES})"
                    )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "Active functions must delegate instead of becoming "
                "orchestrator monoliths.",
                violations,
            ),
        )

    def test_campaign_launcher_is_a_thin_wrapper(self):
        launcher_lines = LAUNCHER_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        violations = []
        if len(launcher_lines) > MAX_LAUNCHER_LINES:
            violations.append(
                f"{_relative(LAUNCHER_PATH)}: {len(launcher_lines)} lines "
                f"(maximum {MAX_LAUNCHER_LINES})"
            )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "The shell launcher must only delegate to the Python campaign "
                "CLI.",
                violations,
            ),
        )

    def test_campaign_launcher_contains_no_embedded_python(self):
        violations = []
        for line_number, line in enumerate(
            LAUNCHER_PATH.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if PYTHON_HEREDOC.search(line):
                violations.append(
                    f"{_relative(LAUNCHER_PATH)}:{line_number}: "
                    f"Python heredoc delimiter {line.strip()!r}"
                )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "Python validation/orchestration belongs in importable Python "
                "modules, not shell heredocs.",
                violations,
            ),
        )

    def test_domain_packages_use_only_the_standard_library_and_domain_code(self):
        violations = []
        domain_directories = sorted(
            path
            for path in EXPERIMENT_ROOT.rglob("domain")
            if path.is_dir()
            and not any(
                part in EXCLUDED_SOURCE_DIRECTORIES
                for part in path.relative_to(EXPERIMENT_ROOT).parts
            )
        )

        for domain_directory in domain_directories:
            relative_domain = domain_directory.relative_to(EXPERIMENT_ROOT)
            domain_prefix = ".".join(relative_domain.parts)
            allowed_domain_prefixes = {
                domain_prefix,
                f"few_shot_experiments.{domain_prefix}",
            }
            for path in sorted(domain_directory.rglob("*.py")):
                for node in ast.walk(_parsed(path)):
                    imported_modules = []
                    if isinstance(node, ast.Import):
                        imported_modules.extend(
                            alias.name for alias in node.names
                        )
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            continue
                        if node.module:
                            imported_modules.append(node.module)
                    else:
                        continue

                    for module in imported_modules:
                        top_level = module.split(".", 1)[0]
                        is_domain_import = any(
                            module == prefix
                            or module.startswith(f"{prefix}.")
                            for prefix in allowed_domain_prefixes
                        )
                        if (
                            top_level not in sys.stdlib_module_names
                            and not is_domain_import
                        ):
                            violations.append(
                                f"{_relative(path)}:{node.lineno}: "
                                f"domain imports non-domain dependency {module}"
                            )

        self.assertEqual(
            violations,
            [],
            _failure_message(
                "Domain packages must depend only on the standard library and "
                "their own domain modules.",
                violations,
            ),
        )


if __name__ == "__main__":
    unittest.main()
