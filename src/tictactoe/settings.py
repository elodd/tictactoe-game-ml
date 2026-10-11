"""Settings loaded from env.json (copy env.example.json and edit it), if it exists.

Missing file or missing key falls back to the defaults below.
"""

import json
from pathlib import Path

FIRST_PLAYER_OPTIONS = ("human", "ai")
DEFAULT_FIRST_PLAYER = "human"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / "env.json"


def load_settings(path=None):
    path = Path(path) if path else DEFAULT_ENV_PATH
    values = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            values = json.load(f)

    first = values.get("FIRST_PLAYER", DEFAULT_FIRST_PLAYER)
    if first not in FIRST_PLAYER_OPTIONS:
        raise ValueError(f"FIRST_PLAYER must be one of {FIRST_PLAYER_OPTIONS}, got {first!r}")
    return {"FIRST_PLAYER": first}
