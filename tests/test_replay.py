"""Deadline regression checks; these do not load models or run stage commands."""
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import deadline_from_environment, replay_environment


class ReplayDeadlineTest(unittest.TestCase):
    def test_historical_deadline_is_preserved_without_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(deadline_from_environment("SOUP_REPLAY_DEADLINE_EPOCH", 123), 123)

    def test_invalid_budgets_and_epoch_are_rejected(self):
        for minutes in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                replay_environment(minutes)
        with self.assertRaises(ValueError):
            replay_environment(10, 11)
        for value in ("invalid", "nan", "inf", "0"):
            with patch.dict(os.environ, {"TEST_DEADLINE": value}):
                with self.assertRaises(ValueError):
                    deadline_from_environment("TEST_DEADLINE", 123)

    def test_required_optional_and_training_commands_get_fresh_budget(self):
        environment, end = replay_environment(10, 5, now=2_000_000_000)
        self.assertEqual(end, 2_000_000_600)
        with patch.dict(os.environ, environment):
            import patch_common
            import train_stage
            importlib.reload(patch_common)
            importlib.reload(train_stage)
            with patch("time.time", return_value=2_000_000_100):
                with patch.object(patch_common, "command", return_value=0) as command:
                    self.assertEqual(patch_common.step(["unused"], "required"), 0)
                    self.assertEqual(command.call_args.kwargs["timeout"], 500)
                    self.assertEqual(patch_common.step(["unused"], "optional", optional=True), 0)
                    self.assertEqual(command.call_args.kwargs["timeout"], 200)
                with patch.object(train_stage, "command", return_value=0) as command:
                    self.assertEqual(train_stage.run(["unused"], "training"), 0)
                    self.assertEqual(command.call_args.kwargs["timeout"], 500)
            with patch("time.time", return_value=2_000_000_601):
                with patch.object(patch_common, "command") as command:
                    self.assertIsNone(patch_common.step(["unused"], "expired"))
                    command.assert_not_called()
                with self.assertRaises(TimeoutError):
                    train_stage.run(["unused"], "expired")

    def test_launcher_passes_deadline_to_a_real_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "child.py"
            script.write_text(
                "import json, os, sys\n"
                "print(json.dumps({'epoch': float(os.environ['SOUP_REPLAY_DEADLINE_EPOCH']), "
                "'optional': float(os.environ['SOUP_REPLAY_OPTIONAL_EPOCH']), 'argument': sys.argv[1]}))\n"
            )
            started = time.time()
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / "replay.py"),
                 "--minutes", "1", "--optional-minutes", "0.5", str(script), "@deadline"],
                text=True, capture_output=True, check=True,
            )
            record = json.loads(result.stdout.splitlines()[-1])
            from datetime import datetime
            self.assertGreaterEqual(record["epoch"], started + 60)
            self.assertLessEqual(record["epoch"], time.time() + 60)
            self.assertEqual(record["epoch"] - record["optional"], 30)
            self.assertAlmostEqual(datetime.fromisoformat(record["argument"]).timestamp(), record["epoch"], places=5)


if __name__ == "__main__":
    unittest.main()
