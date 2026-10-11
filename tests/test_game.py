import functools
import io
import random
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.tictactoe import game


def board_from(text):
    """'XO.' style rows, '.' = empty -> list of None/'X'/'O'."""
    return [None if ch == "." else ch for ch in text.replace("\n", "").replace(" ", "")]


class WinnerTests(unittest.TestCase):
    def test_no_winner_on_empty_board(self):
        self.assertIsNone(game.winner([None] * 9))

    def test_row_column_and_diagonal_wins(self):
        self.assertEqual(game.winner(board_from("XXX......")), "X")
        self.assertEqual(game.winner(board_from("O..O..O..")), "O")
        self.assertEqual(game.winner(board_from("X...X...X")), "X")
        self.assertEqual(game.winner(board_from("..O.O.O..")), "O")

    def test_mixed_line_is_not_a_win(self):
        self.assertIsNone(game.winner(board_from("XOX......")))

    def test_is_full(self):
        self.assertTrue(game.is_full(board_from("XOXXOOOXX")))
        self.assertFalse(game.is_full(board_from("XOXXOOOX.")))


class AITests(unittest.TestCase):
    def test_ai_takes_immediate_win(self):
        board = board_from("OO.X.X...")
        self.assertEqual(game.best_ai_move(board), 2)

    def test_ai_blocks_human_win(self):
        board = board_from("XX.O.....")
        self.assertEqual(game.best_ai_move(board), 2)

    def test_ai_never_loses_against_any_human_strategy(self):
        """Exhaustively explore every game where the human moves first."""

        def outcome(board, human_turn):
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
                        results.append(outcome(board, False))
                        board[i] = None
                if "loss" in results:
                    return "loss"
                return "draw" if "draw" in results else "win"
            move = game.best_ai_move(board)
            board[move] = "O"
            result = outcome(board, True)
            board[move] = None
            return result

        self.assertNotEqual(outcome([None] * 9, True), "loss")


def _outcome(board, human_turn):
    """Game-tree result from O's point of view: 'win', 'draw' or 'loss'."""
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
                results.append(_outcome(board, False))
                board[i] = None
        if "loss" in results:
            return "loss"
        return "draw" if "draw" in results else "win"
    move = game.best_ai_move(board)
    board[move] = "O"
    result = _outcome(board, True)
    board[move] = None
    return result


def _ai_turn_positions(board, human_turn=True, seen=None):
    """Yield every board where the AI is to move, assuming the AI always plays
    its best move. The human explores every legal move."""
    seen = set() if seen is None else seen
    key = (tuple(board), human_turn)
    if key in seen or game.winner(board) or game.is_full(board):
        return
    seen.add(key)
    if human_turn:
        for i in range(9):
            if board[i] is None:
                board[i] = "X"
                yield from _ai_turn_positions(board, False, seen)
                board[i] = None
    else:
        yield list(board)
        move = game.best_ai_move(board)
        board[move] = "O"
        yield from _ai_turn_positions(board, True, seen)
        board[move] = None


class AIMoveQualityTests(unittest.TestCase):
    def test_wins_immediately_instead_of_blocking(self):
        # O can complete row 3-4-5 right now; X also threatens row 0-1-2.
        board = board_from("XX.OO....")
        self.assertEqual(game.best_ai_move(board), 5)

    def test_blocks_diagonal_threat(self):
        # X threatens 0-4-8 and O has no winning move of its own.
        board = board_from("XO..X....")
        self.assertEqual(game.best_ai_move(board), 8)

    def test_blocks_column_threat(self):
        # X threatens column 2-5-8; O cells 0 and 7 share no line.
        board = board_from("O.X..X.O.")
        self.assertEqual(game.best_ai_move(board), 8)

    def test_full_board_has_no_move(self):
        self.assertIsNone(game.best_ai_move(board_from("XOXXOOOXX")))

    def test_move_is_always_an_empty_cell(self):
        for board in _ai_turn_positions([None] * 9):
            move = game.best_ai_move(board)
            self.assertIsNotNone(move)
            self.assertIsNone(board[move], f"AI chose occupied cell {move} on {board}")

    def test_does_not_modify_the_board(self):
        board = board_from("XO..X....")
        before = list(board)
        game.best_ai_move(board)
        self.assertEqual(board, before)

    def test_ai_never_loses_after_any_opening_move(self):
        # Human opens with X on any square; the AI must not lose from there.
        for opening in range(9):
            board = [None] * 9
            board[opening] = "X"
            self.assertNotEqual(_outcome(board, False), "loss", f"lost after X at {opening}")

    def test_every_ai_turn_position_is_not_lost(self):
        for board in _ai_turn_positions([None] * 9):
            self.assertNotEqual(_outcome(board, False), "loss", f"lost from {board}")

    def test_ai_beats_random_human_players(self):
        # Random games revisit the same boards, so cache the AI's answers.
        cached_ai_move = functools.lru_cache(maxsize=None)(
            lambda key: game.best_ai_move(list(key))
        )
        rng = random.Random(1234)
        for _ in range(200):
            board = [None] * 9
            human_turn = True
            while not (game.winner(board) or game.is_full(board)):
                if human_turn:
                    move = rng.choice([i for i in range(9) if board[i] is None])
                    board[move] = "X"
                else:
                    board[cached_ai_move(tuple(board))] = "O"
                human_turn = not human_turn
            self.assertNotEqual(game.winner(board), "X", f"AI lost to random play: {board}")


class HumanInputTests(unittest.TestCase):
    def test_rejects_taken_cell_then_accepts_valid(self):
        board = board_from("X........")
        inputs = iter(["1", "abc", "5"])
        with patch("builtins.input", lambda _: next(inputs)), redirect_stdout(io.StringIO()):
            self.assertEqual(game.human_move(board), 4)

    def test_q_quits(self):
        with patch("builtins.input", lambda _: "q"), redirect_stdout(io.StringIO()), \
             self.assertRaises(KeyboardInterrupt):
            game.human_move([None] * 9)


class RoundTests(unittest.TestCase):
    def test_human_win_updates_score(self):
        # Human plays 1, 2, 3 on top row; AI plays 4, 5 along the way.
        scores = {"X": 0, "O": 0, "D": 0}
        inputs = iter(["1", "2", "3"])
        with patch("builtins.input", lambda _: next(inputs)), redirect_stdout(io.StringIO()), \
             patch.object(game, "best_ai_move", side_effect=[3, 4]):
            game.play_round(scores)
        self.assertEqual(scores, {"X": 1, "O": 0, "D": 0})

    def test_ai_win_updates_score(self):
        scores = {"X": 0, "O": 0, "D": 0}
        # Human plays cells 9, 5, 8; AI completes the top row (cells 1-3) first.
        inputs = iter(["9", "5", "8"])
        with patch("builtins.input", lambda _: next(inputs)), redirect_stdout(io.StringIO()), \
             patch.object(game, "best_ai_move", side_effect=[0, 1, 2]):
            game.play_round(scores)
        self.assertEqual(scores, {"X": 0, "O": 1, "D": 0})


if __name__ == "__main__":
    unittest.main()
