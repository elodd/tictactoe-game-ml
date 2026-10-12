"""Registry of AI move models.

Each model is a function board -> move, where board is a list of 9 items
(None, "X", or "O") and the return value is an empty cell index for O.
"""

import random

from .. import game

MODEL_NAMES = ("minimax", "mlmodel")
DEFAULT_MODEL = "minimax"
DIFFICULTY_LEVELS = ("strong", "medium", "weak")
# Chance that the AI ignores the model and plays a random empty cell.
RANDOM_MOVE_RATE = {"strong": 0.0, "medium": 0.3, "weak": 0.7}


def load_model(name=DEFAULT_MODEL, seed=0):
    """Return a move function for the named model.

    minimax: exhaustive search, never loses. Ready immediately.
           Plays the lowest empty cell.
    mlmodel: MLP trained to imitate minimax. Weights come from
             python3 -m src.tictactoe.model.build_mlmodel and must be built first.
    """
    if name == "minimax":
        # Look up at call time so tests can patch game.best_ai_move.
        return lambda board: game.best_ai_move(board)
    if name == "mlmodel":
        from .mlmodel import load_weights

        return load_weights().move
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")


def with_difficulty(move, difficulty, rng=None):
    """Wrap a move function so it sometimes plays a random empty cell.

    strong: the model's move every time (returns move unchanged).
    medium: a random move 30% of the time.
    weak:   a random move 70% of the time.
    """
    if difficulty not in RANDOM_MOVE_RATE:
        raise ValueError(f"Unknown difficulty {difficulty!r}; expected one of {DIFFICULTY_LEVELS}")
    rate = RANDOM_MOVE_RATE[difficulty]
    if rate == 0.0:
        return move
    rng = rng or random.Random()

    def play(board):
        if rng.random() < rate:
            return rng.choice([i for i, v in enumerate(board) if v is None])
        return move(board)

    return play
