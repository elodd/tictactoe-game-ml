import re
import unittest

from src.tictactoe import game
from src.tictactoe.model import build_mlmodel as build
from src.tictactoe.model.mlmodel import MLP

TINY = [
    (["X", "X", None, "O", "O", None, "X", None, None], [5]),
    (["X", None, None, None, None, None, None, None, None], [1, 2, 3, 4, 5, 6, 7, 8]),
    (["X", "O", None, "X", None, None, None, None, None], [2, 4, 5, 6, 7, 8]),
]


class MinimaxMoves(unittest.TestCase):
    def test_optimal_moves_is_the_immediate_win(self):
        board = ["X", "X", None, "O", "O", None, "X", None, None]
        self.assertEqual(build.optimal_moves(board), [5])

    def test_optimal_moves_restores_the_board(self):
        board = ["X", None, None, "O", None, None, None, None, None]
        before = list(board)
        build.optimal_moves(board)
        self.assertEqual(board, before)


class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build.build_dataset()

    def test_positions_have_o_to_move(self):
        for board, _ in self.data[:300]:
            self.assertEqual(board.count("X"), board.count("O") + 1)

    def test_labels_are_legal_cells(self):
        for board, best in self.data[:300]:
            for cell in best:
                self.assertIsNone(board[cell])


class Training(unittest.TestCase):
    def test_loss_goes_down(self):
        lines = []
        build.train(list(TINY), hidden=8, epochs=5, lr=0.1, log=lines.append)
        losses = [float(re.search(r"loss ([\d.]+)", line).group(1)) for line in lines]
        self.assertLess(losses[-1], losses[0])

    def test_same_seed_same_weights(self):
        a = build.train(list(TINY), hidden=4, epochs=2, seed=7, log=lambda _: None)
        b = build.train(list(TINY), hidden=4, epochs=2, seed=7, log=lambda _: None)
        self.assertEqual(a.w1, b.w1)
        self.assertEqual(a.w2, b.w2)

    def test_returns_an_mlp_of_requested_size(self):
        m = build.train(list(TINY), hidden=5, epochs=1, log=lambda _: None)
        self.assertIsInstance(m, MLP)
        self.assertEqual(m.hidden, 5)


class Checks(unittest.TestCase):
    def test_agreement_is_one_for_minimax(self):
        class Minimax:
            def move(self, board):
                return game.best_ai_move(board)

        data = [(board, build.optimal_moves(list(board))) for board, _ in TINY]
        self.assertEqual(build.agreement(Minimax(), data), 1.0)

    def test_agreement_counts_partial_matches(self):
        class AlwaysFour:
            def move(self, board):
                return 4

        data = [(["X"] + [None] * 8, [4]), (["X"] + [None] * 8, [1])]
        self.assertEqual(build.agreement(AlwaysFour(), data), 0.5)

    def test_exhaustive_result_is_draw_for_minimax(self):
        class Minimax:
            def move(self, board):
                return game.best_ai_move(board)

        self.assertEqual(build.exhaustive_result(Minimax()), "draw")

    def test_random_play_has_no_losses_for_minimax(self):
        class Minimax:
            def move(self, board):
                return game.best_ai_move(board)

        self.assertEqual(build.random_play_losses(Minimax(), games=50), 0)

    def test_random_play_detects_a_bad_model(self):
        class Lowest:
            def move(self, board):
                return next(i for i, v in enumerate(board) if v is None)

        self.assertGreater(build.random_play_losses(Lowest(), games=200), 0)


if __name__ == "__main__":
    unittest.main()
