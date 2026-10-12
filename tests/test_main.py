import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.tictactoe import game


def run_main(inputs, first="human", ai_moves=None, difficulty="strong"):
    """Run main() with scripted input and AI moves; return captured stdout."""
    feed = iter(inputs)
    out = io.StringIO()
    settings = {"FIRST_PLAYER": first, "DIFFICULTY": difficulty}
    scripted = patch.object(game, "best_ai_move", side_effect=ai_moves or [])
    with patch("builtins.input", lambda _: next(feed)), \
         patch("src.tictactoe.settings.load_settings", return_value=settings), \
         patch("src.tictactoe.model.load_model", side_effect=lambda name, **kw: lambda b: game.best_ai_move(b)), \
         scripted, \
         redirect_stdout(out):
        game.main()
    return out.getvalue()


class RenderTests(unittest.TestCase):
    def capture(self, board):
        out = io.StringIO()
        with redirect_stdout(out):
            game.render(board)
        return out.getvalue()

    def test_empty_board_shows_numbers(self):
        text = self.capture([None] * 9)
        self.assertIn(" 1 | 2 | 3 ", text)
        self.assertIn(" 7 | 8 | 9 ", text)

    def test_marks_replace_numbers(self):
        text = self.capture(["X", None, "O", None, "X", None, None, None, "O"])
        self.assertIn(" X | 2 | O ", text)
        self.assertIn(" 4 | X | 6 ", text)
        self.assertIn(" 7 | 8 | O ", text)


class HumanMoveTests(unittest.TestCase):
    def ask(self, inputs, board=None):
        feed = iter(inputs)
        with patch("builtins.input", lambda _: next(feed)), redirect_stdout(io.StringIO()):
            return game.human_move(board or [None] * 9)

    def test_accepts_first_and_last_cells(self):
        self.assertEqual(self.ask(["1"]), 0)
        self.assertEqual(self.ask(["9"]), 8)

    def test_strips_whitespace(self):
        self.assertEqual(self.ask(["  5  "]), 4)

    def test_rejects_zero_and_out_of_range(self):
        self.assertEqual(self.ask(["0", "10", "5"]), 4)

    def test_rejects_non_numeric(self):
        self.assertEqual(self.ask(["x", "", "2"]), 1)

    def test_quit_is_case_insensitive(self):
        with self.assertRaises(KeyboardInterrupt):
            self.ask(["Q"])


class MainLoopTests(unittest.TestCase):
    def test_human_win_then_quit_after_play_again(self):
        # Human completes the top row; decline another game.
        out = run_main(["1", "2", "3", "n"], first="human", ai_moves=[3, 4])
        self.assertIn("You win!", out)
        self.assertIn("Score  You: 1  AI: 0  Draws: 0", out)
        self.assertIn("Thanks for playing.", out)

    def test_ai_first_announces_ai_opening(self):
        out = run_main(["4", "5", "n"], first="ai", ai_moves=[0, 1, 2])
        self.assertIn("AI (O) goes first.", out)
        self.assertIn("AI wins.", out)
        self.assertIn("Score  You: 0  AI: 1  Draws: 0", out)

    def test_draw_is_counted(self):
        # Final board: X O X / O O X / X X O -> draw.
        out = run_main(
            ["1", "3", "6", "7", "8", "n"],
            first="human",
            ai_moves=[1, 3, 4, 8],
        )
        self.assertIn("It's a draw.", out)
        self.assertIn("Draws: 1", out)

    def test_play_again_starts_new_round_and_scores_accumulate(self):
        out = run_main(
            ["1", "2", "3", "y", "1", "2", "3", "n"],
            first="human",
            ai_moves=[3, 4, 3, 4],
        )
        self.assertEqual(out.count("New game."), 2)
        self.assertIn("Score  You: 2  AI: 0  Draws: 0", out)

    def test_title_shows_difficulty(self):
        out = run_main(["q"], difficulty="weak")
        self.assertIn("Tic-Tac-Toe vs AI (weak)", out)

    def test_q_quits_mid_game_cleanly(self):
        out = run_main(["q"], first="human")
        self.assertIn("Thanks for playing.", out)
        self.assertNotIn("You win!", out)

    def test_eof_exits_cleanly(self):
        out = io.StringIO()
        with patch("builtins.input", side_effect=EOFError), \
             patch("src.tictactoe.settings.load_settings",
                   return_value={"FIRST_PLAYER": "human", "DIFFICULTY": "strong"}), \
             patch("src.tictactoe.model.load_model", return_value=game.best_ai_move), \
             redirect_stdout(out):
            game.main()
        self.assertIn("Thanks for playing.", out.getvalue())


if __name__ == "__main__":
    unittest.main()
