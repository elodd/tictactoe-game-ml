"""Tic-tac-toe in the terminal. You are X, the AI is O."""

import os
import sys

HUMAN, AI = "X", "O"
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board):
    return all(board)


def minimax(board, ai_turn, depth):
    w = winner(board)
    if w == AI:
        return 10 - depth
    if w == HUMAN:
        return depth - 10
    if is_full(board):
        return 0

    scores = []
    for i in range(9):
        if board[i] is None:
            board[i] = AI if ai_turn else HUMAN
            scores.append(minimax(board, not ai_turn, depth + 1))
            board[i] = None
    return max(scores) if ai_turn else min(scores)


def best_ai_move(board):
    best_score, best_move = -float("inf"), None
    for i in range(9):
        if board[i] is None:
            board[i] = AI
            score = minimax(board, False, 0)
            board[i] = None
            if score > best_score:
                best_score, best_move = score, i
    return best_move


def interactive():
    """True when writing to a real terminal (not a pipe or test capture)."""
    return sys.stdout.isatty()


def use_color():
    return interactive() and "NO_COLOR" not in os.environ


def paint(text, code):
    if not use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def clear_screen():
    if interactive():
        print("\033[2J\033[H", end="")


def free_cells(board):
    return [str(i + 1) for i, v in enumerate(board) if v is None]


def render(board, last_move=None):
    cells = []
    for i, v in enumerate(board):
        if v is None:
            cells.append(paint(str(i + 1), "90"))
            continue
        color = "34" if v == HUMAN else "31"
        code = f"1;{color}" + (";7" if i == last_move else "")
        cells.append(paint(v, code))
    print()
    for r in range(3):
        row = cells[r * 3:r * 3 + 3]
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if r < 2:
            print("---+---+---")
    print()


def human_move(board):
    while True:
        free = free_cells(board)
        raw = input(f"Your move ({', '.join(free)}, q to quit): ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw in free:
            return int(raw) - 1
        if raw.isdigit():
            print(f"Cell {raw} is not available. Choose one of: {', '.join(free)}.")
        else:
            print(f"Enter a number from: {', '.join(free)}, or q to quit.")


def play_round(scores, first_player="human", ai_move=None):
    if ai_move is None:
        ai_move = best_ai_move
    board = [None] * 9
    human_turn = first_player == "human"
    who = "You are X and go first." if human_turn else "AI (O) goes first."
    clear_screen()
    print(f"New game. {who}")
    render(board)

    last_move = None
    while True:
        if human_turn:
            last_move = human_move(board)
            board[last_move] = HUMAN
        else:
            print("AI is thinking...")
            last_move = ai_move(board)
            board[last_move] = AI
        if winner(board) or is_full(board):
            break
        human_turn = not human_turn
        clear_screen()
        print(who)
        render(board, last_move)

    render(board, last_move)
    if winner(board) == HUMAN:
        scores["X"] += 1
        print(paint("You win!", "1;32"))
    elif winner(board) == AI:
        scores["O"] += 1
        print(paint("AI wins.", "1;31"))
    else:
        scores["D"] += 1
        print("It's a draw.")


def main():
    from .settings import load_settings

    cfg = load_settings()
    first = cfg["FIRST_PLAYER"]
    from .model import load_model

    try:
        ai_move = load_model("mlmodel")
    except FileNotFoundError as err:
        print(err)
        return
    scores = {"X": 0, "O": 0, "D": 0}
    print("Tic-Tac-Toe vs AI")
    try:
        while True:
            play_round(scores, first, ai_move)
            print(f"Score  You: {scores['X']}  AI: {scores['O']}  Draws: {scores['D']}")
            again = input("Play again? (y/n): ").strip().lower()
            if again != "y":
                break
    except (KeyboardInterrupt, EOFError):
        print()
    print("Thanks for playing.")


if __name__ == "__main__":
    main()
