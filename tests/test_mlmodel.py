import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tictactoe import game
from src.tictactoe.model import load_model, mlmodel
from src.tictactoe.model.mlmodel import MLP, encode, load_weights


def tiny_mlp(hidden=3):
    w1 = [[0.1 * (j + k) for k in range(18)] for j in range(hidden)]
    b1 = [0.0] * hidden
    w2 = [[0.2 * (i - j) for j in range(hidden)] for i in range(9)]
    b2 = [0.01 * i for i in range(9)]
    return MLP(w1, b1, w2, b2)


class EncodeTests(unittest.TestCase):
    def test_empty_board_is_all_zero(self):
        self.assertEqual(encode([None] * 9), [0.0] * 18)

    def test_x_and_o_flags(self):
        x = encode(["X", None, "O"] + [None] * 6)
        self.assertEqual(x[0], 1.0)   # X at cell 0
        self.assertEqual(x[5], 1.0)   # O at cell 2
        self.assertEqual(sum(x), 2.0)


class MLPTests(unittest.TestCase):
    def test_move_is_legal(self):
        board = ["X", "O", None, "X", None, "O", None, None, None]
        self.assertIn(tiny_mlp().move(board), [2, 4, 6, 7, 8])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            m = tiny_mlp()
            m.save(path)
            loaded = MLP.load(path)
        self.assertEqual(loaded.hidden, 3)
        for a, b in zip(loaded.w2, m.w2):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=5)


class BinaryFormatTests(unittest.TestCase):
    def test_round_trip_is_exact(self):
        m = tiny_mlp()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.bin"
            m.save(path)
            loaded = MLP.load(path)
        self.assertEqual((loaded.w1, loaded.b1, loaded.w2, loaded.b2), (m.w1, m.b1, m.w2, m.b2))

    def test_rejects_wrong_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bin"
            tiny_mlp().save(path)
            data = bytearray(path.read_bytes())
            data[:4] = b"NOPE"
            path.write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                MLP.load(path)

    def test_rejects_truncated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.bin"
            tiny_mlp().save(path)
            path.write_bytes(path.read_bytes()[:-8])
            with self.assertRaises(ValueError):
                MLP.load(path)


class LoadWeightsTests(unittest.TestCase):
    def test_missing_file_names_the_build_command(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(mlmodel, "OUTPUT_DIR", Path(tmp)), \
             self.assertRaises(FileNotFoundError) as ctx:
            load_weights()
        self.assertIn("build_mlmodel", str(ctx.exception))

    def test_load_model_mlmodel_fails_cleanly_when_unbuilt(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(mlmodel, "OUTPUT_DIR", Path(tmp)), \
             self.assertRaises(FileNotFoundError):
            load_model("mlmodel")


class BundledWeightsTests(unittest.TestCase):
    """Runs only when mlmodel/mlmodel.bin has been built."""

    @classmethod
    def setUpClass(cls):
        if not mlmodel.weights_path().exists():
            raise unittest.SkipTest("mlmodel.bin not built; run python3 -m src.tictactoe.model.build_mlmodel")
        cls.move = load_model("mlmodel")

    def test_no_losses_in_random_games(self):
        rng = random.Random(42)
        for _ in range(200):
            board = [None] * 9
            human_turn = True
            while not (game.winner(board) or game.is_full(board)):
                if human_turn:
                    board[rng.choice([i for i in range(9) if board[i] is None])] = "X"
                else:
                    board[self.move(board)] = "O"
                human_turn = not human_turn
            self.assertNotEqual(game.winner(board), "X")

    def test_never_plays_an_occupied_cell(self):
        board = ["X", None, None, None, "O", None, None, None, None]
        self.assertIsNone(board[self.move(board)])

    def test_draws_every_human_first_game(self):
        def tree(board, human_turn):
            w = game.winner(board)
            if w == "O":
                return "win"
            if w == "X":
                return "loss"
            if game.is_full(board):
                return "draw"
            if human_turn:
                results = []
                for i in range(9):
                    if board[i] is None:
                        board[i] = "X"
                        results.append(tree(board, False))
                        board[i] = None
                if "loss" in results:
                    return "loss"
                return "draw" if "draw" in results else "win"
            i = self.move(board)
            board[i] = "O"
            result = tree(board, True)
            board[i] = None
            return result

        self.assertEqual(tree([None] * 9, True), "draw")
