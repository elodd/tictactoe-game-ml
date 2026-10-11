import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.tictactoe import game


def capture(fn, *args, **kwargs):
    out = io.StringIO()
    with redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class FreeCellsTests(unittest.TestCase):
    def test_lists_empty_cells_as_numbers(self):
        self.assertEqual(game.free_cells(["X", None, "O"] + [None] * 6), ["2", "4", "5", "6", "7", "8", "9"])

    def test_full_board_has_no_free_cells(self):
        self.assertEqual(game.free_cells(["X", "O"] * 4 + ["X"]), [])


class ColorTests(unittest.TestCase):
    def test_paint_is_plain_when_colour_is_off(self):
        with patch.object(game, "use_color", return_value=False):
            self.assertEqual(game.paint("X", "1;34"), "X")

    def test_paint_adds_ansi_codes_when_colour_is_on(self):
        with patch.object(game, "use_color", return_value=True):
            self.assertEqual(game.paint("X", "1;34"), "\033[1;34mX\033[0m")

    def test_no_color_variable_turns_colour_off(self):
        with patch.object(game, "interactive", return_value=True), \
             patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertFalse(game.use_color())

    def test_colour_is_off_when_not_a_terminal(self):
        with patch.object(game, "interactive", return_value=False):
            self.assertFalse(game.use_color())


class ClearScreenTests(unittest.TestCase):
    def test_clears_only_in_a_terminal(self):
        _, text = capture(game.clear_screen)
        self.assertEqual(text, "")
        with patch.object(game, "interactive", return_value=True):
            _, text = capture(game.clear_screen)
        self.assertIn("\033[2J", text)


class RenderTests(unittest.TestCase):
    def test_plain_board_text(self):
        board = ["X", None, None, None, "O", None, None, None, None]
        with patch.object(game, "use_color", return_value=False):
            _, text = capture(game.render, board)
        self.assertIn(" X | 2 | 3 ", text)
        self.assertIn(" 4 | O | 6 ", text)

    def test_last_move_is_highlighted_in_colour(self):
        board = ["X", None, None, None, None, None, None, None, None]
        with patch.object(game, "use_color", return_value=True):
            _, text = capture(game.render, board, 0)
        self.assertIn("\033[1;34;7mX\033[0m", text)

    def test_other_marks_are_not_highlighted(self):
        board = ["X", None, None, None, "O", None, None, None, None]
        with patch.object(game, "use_color", return_value=True):
            _, text = capture(game.render, board, 0)
        self.assertIn("\033[1;31mO\033[0m", text)
        self.assertNotIn("\033[1;31;7mO", text)


class HumanMoveMessageTests(unittest.TestCase):
    def ask(self, inputs, board):
        feed = iter(inputs)
        out = io.StringIO()
        with patch("builtins.input", lambda _: next(feed)), redirect_stdout(out):
            move = game.human_move(board)
        return move, out.getvalue()

    def test_taken_cell_message_names_the_cell(self):
        board = ["X", None, None, None, None, None, None, None, None]
        move, text = self.ask(["1", "2"], board)
        self.assertEqual(move, 1)
        self.assertIn("Cell 1 is not available", text)

    def test_non_number_message_lists_free_cells(self):
        board = [None] * 9
        move, text = self.ask(["abc", "5"], board)
        self.assertEqual(move, 4)
        self.assertIn("Enter a number from", text)

    def test_prompt_lists_free_cells(self):
        board = ["X"] + [None] * 8
        prompts = []
        feed = iter(["2"])
        with patch("builtins.input", lambda p: (prompts.append(p), next(feed))[1]), redirect_stdout(io.StringIO()):
            game.human_move(board)
        self.assertIn("2, 3, 4, 5, 6, 7, 8, 9", prompts[0])


if __name__ == "__main__":
    unittest.main()
