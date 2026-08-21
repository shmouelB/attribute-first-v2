import importlib.util
import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
UTILS_SPEC = importlib.util.spec_from_file_location(
    "_core_utils_text_normalization",
    EXPERIMENT_ROOT / "utils.py",
)
utils = importlib.util.module_from_spec(UTILS_SPEC)
sys.modules[UTILS_SPEC.name] = utils
UTILS_SPEC.loader.exec_module(utils)


class TextNormalizationTests(unittest.TestCase):
    def test_remove_spaces_and_punctuation_keeps_only_alphanumeric_text(self):
        self.assertEqual(
            utils.remove_spaces_and_punctuation(" A—b!\t é_2.\n"),
            "Abé2",
        )

    def test_find_substring_maps_through_punctuation_and_whitespace(self):
        source = "Prefix: Alpha,\n beta! Suffix."
        start, end = utils.find_substring(source, "Alpha beta")
        self.assertEqual(source[start:end], "Alpha,\n beta")


if __name__ == "__main__":
    unittest.main()
