import random
import unittest

from src.tictactoe import game
from src.tictactoe.model import (
    DEFAULT_MODEL,
    DIFFICULTY_LEVELS,
    MODEL_NAMES,
    RANDOM_MOVE_RATE,
    load_model,
    with_difficulty,
)


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


class DifficultyTests(unittest.TestCase):
    BEST = 4  # the stub model always wants the centre

    def best(self, board):
        return self.BEST

    def moves(self, difficulty, n=2000):
        move = with_difficulty(self.best, difficulty, rng=random.Random(0))
        return [move([None] * 9) for _ in range(n)]

    def test_levels(self):
        self.assertEqual(DIFFICULTY_LEVELS, ("strong", "medium", "weak"))
        self.assertEqual(set(RANDOM_MOVE_RATE), set(DIFFICULTY_LEVELS))

    def test_strong_returns_model_unchanged(self):
        best = self.best
        self.assertIs(with_difficulty(best, "strong"), best)

    def test_random_move_rate_matches_level(self):
        for level in DIFFICULTY_LEVELS:
            with self.subTest(level=level):
                moves = self.moves(level)
                rate = sum(m != self.BEST for m in moves) / len(moves)
                # A random pick lands on the best cell 1 time in 9.
                expected = RANDOM_MOVE_RATE[level] * 8 / 9
                self.assertAlmostEqual(rate, expected, delta=0.04)

    def test_weaker_levels_make_more_mistakes(self):
        rates = [RANDOM_MOVE_RATE[level] for level in DIFFICULTY_LEVELS]
        self.assertEqual(rates, sorted(rates))
        self.assertEqual(rates[0], 0.0)

    def test_random_moves_are_legal(self):
        board = ["X", "O", "X", None, "O", None, "X", None, None]
        move = with_difficulty(lambda b: 3, "weak", rng=random.Random(1))
        for _ in range(200):
            self.assertIn(move(board), (3, 5, 7, 8))

    def test_unknown_difficulty_raises(self):
        with self.assertRaises(ValueError):
            with_difficulty(self.best, "hard")


if __name__ == "__main__":
    unittest.main()
