import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.tictactoe import game, settings


class LoadSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data):
        path = self.dir / "env.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_default_path_is_project_root(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(settings.DEFAULT_ENV_PATH, root / "env.json")
        self.assertTrue((settings.DEFAULT_ENV_PATH.parent / "env.example.json").exists())

    def test_defaults_when_no_file(self):
        cfg = settings.load_settings(self.dir / "missing.json")
        self.assertEqual(cfg, {"FIRST_PLAYER": "human", "DIFFICULTY": "strong"})

    def test_defaults_when_key_missing(self):
        self.assertEqual(settings.load_settings(self.write({}))["FIRST_PLAYER"], "human")

    def test_reads_ai_first(self):
        self.assertEqual(settings.load_settings(self.write({"FIRST_PLAYER": "ai"}))["FIRST_PLAYER"], "ai")

    def test_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            settings.load_settings(self.write({"FIRST_PLAYER": "random"}))

    def test_defaults_difficulty_when_key_missing(self):
        self.assertEqual(settings.load_settings(self.write({}))["DIFFICULTY"], "strong")

    def test_reads_each_difficulty(self):
        for level in ("strong", "medium", "weak"):
            with self.subTest(level=level):
                cfg = settings.load_settings(self.write({"DIFFICULTY": level}))
                self.assertEqual(cfg["DIFFICULTY"], level)

    def test_rejects_unknown_difficulty(self):
        with self.assertRaises(ValueError):
            settings.load_settings(self.write({"DIFFICULTY": "hard"}))

    def test_ai_first_moves_before_human(self):
        # Order: AI 0, human 4, AI 1, human 5, AI 2 -> AI completes the top row.
        scores = {"X": 0, "O": 0, "D": 0}
        inputs = iter(["4", "5"])
        with patch("builtins.input", lambda _: next(inputs)), \
             redirect_stdout(io.StringIO()), \
             patch.object(game, "best_ai_move", side_effect=[0, 1, 2]):
            game.play_round(scores, first_player="ai")
        self.assertEqual(scores, {"X": 0, "O": 1, "D": 0})

    def test_human_first_moves_before_ai(self):
        # Order: human 1, AI 3, human 2, AI 4, human 3 -> human completes the top row.
        scores = {"X": 0, "O": 0, "D": 0}
        inputs = iter(["1", "2", "3"])
        with patch("builtins.input", lambda _: next(inputs)), \
             redirect_stdout(io.StringIO()), \
             patch.object(game, "best_ai_move", side_effect=[3, 4]):
            game.play_round(scores, first_player="human")
        self.assertEqual(scores, {"X": 1, "O": 0, "D": 0})


if __name__ == "__main__":
    unittest.main()
