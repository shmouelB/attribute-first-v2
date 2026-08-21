from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from attribute_first.runtime.retry_policy import (  # noqa: E402
    RetryDelayPolicy,
)
from attribute_first.stages.planned import StageExecutor  # noqa: E402


class _CodeOnlyRateLimitError(RuntimeError):
    def code(self):
        return 429

    def __str__(self):
        return "provider capacity unavailable"


class _ResponseRateLimitError(RuntimeError):
    response = type("Response", (), {"status_code": 429})()

    def __str__(self):
        return "request rejected"


class RetryDelayPolicyTests(unittest.TestCase):
    def test_structured_status_detects_quota_without_literal_429_text(self):
        policy = RetryDelayPolicy()

        self.assertEqual(
            policy.delay_seconds(_CodeOnlyRateLimitError()),
            60,
        )
        self.assertEqual(
            policy.delay_seconds(_ResponseRateLimitError()),
            60,
        )
        self.assertEqual(
            policy.delay_seconds(RuntimeError("ResourceExhausted")),
            60,
        )

    def test_non_quota_failure_uses_short_retry(self):
        self.assertEqual(
            RetryDelayPolicy().delay_seconds(
                RuntimeError("invalid structured response")
            ),
            1,
        )

    def test_derived_stage_uses_the_shared_policy(self):
        self.assertEqual(
            StageExecutor._retry_seconds(_CodeOnlyRateLimitError()),
            60,
        )


if __name__ == "__main__":
    unittest.main()
