"""Build the ML move model: train an MLP to imitate minimax, verify it, save it.

Run from the project root:

    python3 -m src.tictactoe.model.build_mlmodel

Takes about 11 minutes. The weights are written to mlmodel/mlmodel.bin only
if the saved file passes both checks:
  1. Exhaustive: O draws every human-first game tree (no losses).
  2. Random play: O loses no games in 2,000 random games.
"""

import math
import random
import sys
import time

from src.tictactoe import game
from src.tictactoe.model.mlmodel import (
    MLP,
    N_CELLS,
    N_IN,
    OUTPUT_DIR,
    WEIGHTS_NAME,
    encode,
    legal_moves,
    softmax_over,
)

HIDDEN = 128
EPOCHS = 600
LR = 0.05
SEED = 0
RANDOM_GAMES = 2000


def optimal_moves(board):
    scores = {}
    for i in legal_moves(board):
        board[i] = "O"
        scores[i] = game.minimax(board, False, 0)
        board[i] = None
    best = max(scores.values())
    return [i for i, s in scores.items() if s == best]


def build_dataset():
    seen = {}

    def visit(board, o_turn):
        key = (tuple(board), o_turn)
        if key in seen or game.winner(board) or game.is_full(board):
            return
        if o_turn:
            seen[key] = optimal_moves(board)
        for i in legal_moves(board):
            board[i] = "O" if o_turn else "X"
            visit(board, not o_turn)
            board[i] = None

    visit([None] * N_CELLS, False)
    return [(list(k[0]), v) for k, v in seen.items()]


def train(data, hidden=HIDDEN, epochs=EPOCHS, lr=LR, seed=SEED, log=print):
    """Per-sample SGD on cross-entropy against uniform-over-optimal targets."""
    rng = random.Random(seed)
    scale1 = 1.0 / math.sqrt(N_IN)
    scale2 = 1.0 / math.sqrt(hidden)
    w1 = [[rng.uniform(-scale1, scale1) for _ in range(N_IN)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    w2 = [[rng.uniform(-scale2, scale2) for _ in range(hidden)] for _ in range(N_CELLS)]
    b2 = [0.0] * N_CELLS
    model = MLP(w1, b1, w2, b2)

    for epoch in range(epochs):
        rng.shuffle(data)
        total_loss = 0.0
        for board, best in data:
            x = encode(board)
            h = model.hidden_layer(x)
            logits = model.output_from_hidden(h)
            legal = legal_moves(board)
            p = softmax_over(logits, legal)
            target = {i: 1.0 / len(best) for i in best}
            total_loss += -sum(target[i] * math.log(max(p[i], 1e-12)) for i in best)

            dlog = {i: p[i] - target.get(i, 0.0) for i in legal}

            dh = [0.0] * hidden
            for i, g in dlog.items():
                row = w2[i]
                for j in range(hidden):
                    dh[j] += g * row[j]
                for j in range(hidden):
                    row[j] -= lr * g * h[j]
                b2[i] -= lr * g

            for j in range(hidden):
                g = dh[j] * (1.0 - h[j] * h[j])
                row = w1[j]
                for k in range(N_IN):
                    if x[k]:
                        row[k] -= lr * g
                b1[j] -= lr * g
        log(f"epoch {epoch + 1:3d}  loss {total_loss / len(data):.4f}")
    return model


def agreement(model, data):
    ok = sum(1 for board, best in data if model.move(board) in best)
    return ok / len(data)


def exhaustive_result(model):
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
        i = model.move(board)
        board[i] = "O"
        result = tree(board, True)
        board[i] = None
        return result

    return tree([None] * 9, True)


def random_play_losses(model, games=RANDOM_GAMES, seed=42):
    rng = random.Random(seed)
    losses = 0
    for _ in range(games):
        board = [None] * 9
        human_turn = True
        while not (game.winner(board) or game.is_full(board)):
            if human_turn:
                board[rng.choice([i for i in range(9) if board[i] is None])] = "X"
            else:
                board[model.move(board)] = "O"
            human_turn = not human_turn
        if game.winner(board) == "X":
            losses += 1
    return losses


def main():
    t0 = time.time()
    data = build_dataset()
    print(f"dataset: {len(data)} positions")
    model = train(data)
    print(f"agreement with minimax: {agreement(model, data):.4%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_DIR / (WEIGHTS_NAME + ".tmp")
    model.save(tmp)
    saved = MLP.load(tmp)  # check the file that will be used, not the in-memory model

    exh = exhaustive_result(saved)
    losses = random_play_losses(saved)
    print(f"exhaustive (human first): {exh}")
    print(f"losses in {RANDOM_GAMES} random games: {losses}")
    if exh != "draw" or losses != 0:
        tmp.unlink()
        print("checks failed; mlmodel/mlmodel.bin not written")
        return 1

    tmp.replace(OUTPUT_DIR / WEIGHTS_NAME)
    print(f"saved {OUTPUT_DIR / WEIGHTS_NAME} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
