import unittest

from src.tictactoe import game
from src.tictactoe.model import DEFAULT_MODEL, MODEL_NAMES, load_model


class LoadModelTests(unittest.TestCase):
    def test_default_is_minimax(self):
        self.assertEqual(DEFAULT_MODEL, "minimax")
        board = ["X", "X", None, "O", "O", None, "X", None, None]
        self.assertEqual(load_model()(board), game.best_ai_move(board))

    def test_lists_all_models(self):
        self.assertEqual(set(MODEL_NAMES), {"minimax", "mlmodel"})

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            load_model("gpt")

    def test_minimax_returns_legal_move(self):
        move = load_model("minimax")([None] * 9)
        self.assertIn(move, range(9))

if __name__ == "__main__":
    unittest.main()
