"""ML move model: a small MLP that imitates minimax.

Weights are produced by python3 -m src.tictactoe.model.build_mlmodel and loaded from
mlmodel/mlmodel.bin. This module only does the forward pass.
"""

import math
import struct
from pathlib import Path

N_CELLS = 9
N_IN = 2 * N_CELLS  # one "X here" and one "O here" flag per cell
OUTPUT_DIR = (Path(__file__).resolve().parent / "../../../mlmodel/").resolve()
WEIGHTS_NAME = "mlmodel.bin"
MAGIC = b"TTTM"
FORMAT_VERSION = 1
HEADER = struct.Struct("<4sII")  # magic, format version, hidden size
BUILD_HINT = "Run: python3 -m src.tictactoe.model.build_mlmodel"


def encode(board):
    x = [0.0] * N_IN
    for i, v in enumerate(board):
        if v == "X":
            x[2 * i] = 1.0
        elif v == "O":
            x[2 * i + 1] = 1.0
    return x


def legal_moves(board):
    return [i for i, v in enumerate(board) if v is None]


def softmax_over(logits, legal):
    m = max(logits[i] for i in legal)
    exps = {i: math.exp(logits[i] - m) for i in legal}
    total = sum(exps.values())
    return {i: e / total for i, e in exps.items()}


class MLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    @property
    def hidden(self):
        return len(self.b1)

    def hidden_layer(self, x):
        return [
            math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
            for row, b in zip(self.w1, self.b1)
        ]

    def output_from_hidden(self, h):
        return [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.w2, self.b2)]

    def logits(self, x):
        return self.output_from_hidden(self.hidden_layer(x))

    def move(self, board):
        """Legal cell with the highest logit."""
        logits = self.logits(encode(board))
        return max(legal_moves(board), key=lambda i: logits[i])

    def save(self, path):
        """Binary format: header (magic, version, hidden size), then float64 values
        in the order w1, b1, w2, b2, little-endian."""
        values = []
        for row in self.w1:
            values.extend(row)
        values.extend(self.b1)
        for row in self.w2:
            values.extend(row)
        values.extend(self.b2)
        with open(path, "wb") as f:
            f.write(HEADER.pack(MAGIC, FORMAT_VERSION, self.hidden))
            f.write(struct.pack(f"<{len(values)}d", *values))

    @classmethod
    def load(cls, path):
        data = Path(path).read_bytes()
        if len(data) < HEADER.size:
            raise ValueError(f"{path} is too short to be a model file")
        magic, version, hidden = HEADER.unpack_from(data)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ValueError(f"{path} is not a supported model file (magic {magic!r}, version {version})")
        count = hidden * N_IN + hidden + N_CELLS * hidden + N_CELLS
        if len(data) != HEADER.size + 8 * count:
            raise ValueError(f"{path} has the wrong size for {hidden} hidden units")
        v = list(struct.unpack_from(f"<{count}d", data, HEADER.size))
        w1 = [v[j * N_IN:(j + 1) * N_IN] for j in range(hidden)]
        pos = hidden * N_IN
        b1 = v[pos:pos + hidden]
        pos += hidden
        w2 = [v[pos + i * hidden:pos + (i + 1) * hidden] for i in range(N_CELLS)]
        pos += N_CELLS * hidden
        b2 = v[pos:pos + N_CELLS]
        return cls(w1, b1, w2, b2)


def weights_path():
    return OUTPUT_DIR / WEIGHTS_NAME


def load_weights(path=None):
    """Load the trained model, or fail with the command that builds it."""
    path = weights_path() if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {BUILD_HINT}")
    return MLP.load(path)
