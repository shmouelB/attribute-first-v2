"""Transactional conversation-state contracts."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


from attribute_first.runtime.conversation import Conversation  # noqa: E402


class ConversationTests(unittest.TestCase):
    def test_transaction_rollback_restores_the_exact_history(self):
        session = SimpleNamespace(
            history=[
                {"role": "user", "parts": ["first task"]},
                {"role": "model", "parts": ["first answer"]},
            ]
        )
        conversation = Conversation(session)
        transaction = conversation.begin()

        session.history.extend(
            [
                {"role": "user", "parts": ["invalid retry"]},
                {"role": "model", "parts": ["invalid answer"]},
            ]
        )
        transaction.rollback()

        self.assertEqual(
            conversation.history,
            [
                {"role": "user", "parts": ["first task"]},
                {"role": "model", "parts": ["first answer"]},
            ],
        )

    def test_commit_keeps_turn_and_reset_replaces_provider_history(self):
        session = SimpleNamespace(history=[])
        conversation = Conversation(session)
        transaction = conversation.begin()
        session.history.extend(
            [
                {"role": "user", "parts": ["new task only"]},
                {"role": "model", "parts": ["new answer"]},
            ]
        )

        transaction.commit()
        self.assertEqual(len(conversation.history), 2)

        conversation.reset(
            [{"role": "user", "parts": ["replacement context"]}]
        )
        self.assertEqual(
            session.history,
            [{"role": "user", "parts": ["replacement context"]}],
        )

    def test_appending_stage_history_does_not_alias_the_input(self):
        session = SimpleNamespace(history=[])
        conversation = Conversation(session)
        stage_history = [
            {"role": "user", "parts": ["demonstration task"]}
        ]

        conversation.append_history(stage_history)
        stage_history[0]["parts"][0] = "mutated caller value"

        self.assertEqual(
            conversation.history[0]["parts"][0],
            "demonstration task",
        )


if __name__ == "__main__":
    unittest.main()
