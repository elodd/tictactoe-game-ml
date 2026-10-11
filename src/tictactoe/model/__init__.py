"""Registry of AI move models.

Each model is a function board -> move, where board is a list of 9 items
(None, "X", or "O") and the return value is an empty cell index for O.
"""

from .. import game

MODEL_NAMES = ("minimax", "mlmodel")
DEFAULT_MODEL = "minimax"


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
