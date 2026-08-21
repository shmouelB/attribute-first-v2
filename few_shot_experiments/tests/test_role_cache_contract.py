import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from google.generativeai import caching as genai_caching


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_INSTRUCTION = "Fuse evidence faithfully and preserve attribution."
SHARED_DEMO_TURNS = [
    {"role": "user", "parts": ["Shared demonstration input."]},
    {"role": "model", "parts": ["Shared demonstration answer."]},
]
INSTANCE_TAILS = {
    "example-1": {"role": "user", "parts": ["Target documents for example one."]},
    "example-2": {"role": "user", "parts": ["Target documents for example two."]},
}


def _load_utils():
    spec = importlib.util.spec_from_file_location(
        "role_cache_contract_utils",
        EXPERIMENT_ROOT / "utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


class RoleCacheContractTests(unittest.TestCase):
    def setUp(self):
        utils.reset_token_usage()

    def tearDown(self):
        utils.reset_token_usage()

    @staticmethod
    def _prompts_and_roles():
        prompts = {
            "example-1": "Full parser prompt for example one.",
            "example-2": "Full parser prompt for example two.",
        }
        role_messages = {
            instance_id: {
                "system": SYSTEM_INSTRUCTION,
                "contents": copy.deepcopy(SHARED_DEMO_TURNS)
                + [copy.deepcopy(target_tail)],
            }
            for instance_id, target_tail in INSTANCE_TAILS.items()
        }
        return prompts, role_messages

    @staticmethod
    def _fake_cache():
        return SimpleNamespace(
            model="models/gemini-test",
            name="cachedContents/role-cache-contract",
            usage_metadata=SimpleNamespace(total_token_count=42),
            delete=mock.Mock(name="delete_role_cache"),
        )

    def _run_prompt_model(self, *, model_side_effect=None):
        prompts, role_messages = self._prompts_and_roles()
        fake_cache = self._fake_cache()
        model_return = {
            "final_output": "Generated summary.",
            "full_model_response": "Generated summary.",
        }

        with mock.patch.dict(
            os.environ,
            {"AF_USE_ROLES": "true", "AF_CONTEXT_CACHE": "true"},
            clear=False,
        ), mock.patch.object(
            genai_caching.CachedContent,
            "create",
            return_value=fake_cache,
        ) as create_cache, mock.patch.object(
            utils,
            "model_call_wrapper",
            side_effect=model_side_effect,
            return_value=model_return,
        ) as model_boundary:
            results = utils.prompt_model(
                prompts=prompts,
                model_name="models/gemini-test",
                parse_response_fn=lambda response, prompt: {
                    "final_output": response,
                    "prompt": prompt,
                },
                role_messages=role_messages,
                verbose=False,
            )

        return results, fake_cache, create_cache, model_boundary

    def test_role_cache_contains_system_instruction_and_common_demo_turns(self):
        _, _, create_cache, _ = self._run_prompt_model()

        create_cache.assert_called_once()
        create_kwargs = create_cache.call_args.kwargs
        self.assertEqual(create_kwargs["model"], "models/gemini-test")
        self.assertEqual(
            create_kwargs["system_instruction"],
            SYSTEM_INSTRUCTION,
        )
        self.assertEqual(create_kwargs["contents"], SHARED_DEMO_TURNS)

    def test_role_cache_keeps_roles_and_sends_only_each_instance_user_tail(self):
        _, _, _, model_boundary = self._run_prompt_model()

        self.assertEqual(
            [call.kwargs["contents"] for call in model_boundary.call_args_list],
            [
                [INSTANCE_TAILS["example-1"]],
                [INSTANCE_TAILS["example-2"]],
            ],
        )
        self.assertTrue(
            all(
                call.kwargs.get("system_instruction") is None
                for call in model_boundary.call_args_list
            ),
            "the system instruction belongs in the cache and must not be resent",
        )

    def test_role_cache_uses_the_model_bound_to_cached_content(self):
        _, fake_cache, _, model_boundary = self._run_prompt_model()

        bound_models = [
            call.kwargs["model_override"]
            for call in model_boundary.call_args_list
        ]
        self.assertTrue(
            all(model is not None for model in bound_models),
            "role transport must stay enabled through the cache-bound model",
        )
        self.assertEqual(
            [model._cached_content for model in bound_models],
            [fake_cache.name, fake_cache.name],
        )

    def test_role_cache_is_deleted_in_finally_when_model_boundary_raises(self):
        prompts, role_messages = self._prompts_and_roles()
        fake_cache = self._fake_cache()

        with mock.patch.dict(
            os.environ,
            {"AF_USE_ROLES": "true", "AF_CONTEXT_CACHE": "true"},
            clear=False,
        ), mock.patch.object(
            genai_caching.CachedContent,
            "create",
            return_value=fake_cache,
        ), mock.patch.object(
            utils,
            "model_call_wrapper",
            side_effect=RuntimeError("model boundary failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model boundary failed"):
                utils.prompt_model(
                    prompts=prompts,
                    model_name="models/gemini-test",
                    parse_response_fn=lambda response, prompt: {
                        "final_output": response,
                        "prompt": prompt,
                    },
                    role_messages=role_messages,
                    verbose=False,
                )

        fake_cache.delete.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
