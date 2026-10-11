# Tic-Tac-Toe Game ML

[![Tests](https://github.com/elodd/tictactoe-game-ml/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/elodd/tictactoe-game-ml/actions/workflows/test.yml)

A terminal tic-tac-toe game. You play X, and the AI plays O. The AI is an ML model (`mlmodel`) trained to imitate minimax.

## Features

- The AI uses the `mlmodel` model. Its weights are built locally (see [Build the ML model](#build-the-ml-model)).
- Choose who opens each game with `FIRST_PLAYER` in `env.json`.
- The score for your wins, the AI's wins, and draws carries across games.
- Standard library only. No third-party packages are needed to play.

## Requirements

- Python 3.9 or newer

## Run

From the project root:

```bash
python3 -m src.tictactoe
```

Or, equivalently: `python3 play.py`

Enter a cell number from 1 to 9 to move, or `q` to quit. After each game, answer `y` to play again or `n` to exit.

In a terminal, the screen is redrawn before each move, X and O are coloured, and the last move is highlighted. The prompt lists the free cells. Set `NO_COLOR=1` to turn colours off. Piped or redirected output is plain text.

```
 1 | 2 | 3
---+---+---
 4 | 5 | 6
---+---+---
 7 | 8 | 9
```

## Settings

Create your own settings file from the template:

```bash
cp env.example.json env.json
```

| Key            | Values          | Default | Meaning               |
|----------------|-----------------|---------|-----------------------|
| `FIRST_PLAYER` | `human` or `ai` | `human` | Who opens each game.  |

If `env.json` is missing, or a key is missing from it, the default is used. An invalid value stops the program with an error message. `env.json` is git-ignored, and `env.example.json` is committed.

## Build the ML model

The weights are not in git. Build them once from the project root:

```bash
python3 -m src.tictactoe.model.build_mlmodel
```

The build takes about 10 minutes. It trains a small neural network to imitate minimax, then checks the saved file. The weights are written to `mlmodel/mlmodel.bin` only if O draws every human-first game tree and loses none of 2,000 random games. Until the weights exist, the game stops and prints this command.

## How the AI works

The AI is a neural network with one hidden layer of 128 tanh units. It takes an X flag and an O flag for each of the nine cells, outputs a score for each cell, and plays the highest-scoring empty cell.

Minimax is the teacher. `best_ai_move` searches the whole game tree for each empty cell. A win scores higher the sooner it happens, and a loss scores lower the later it happens. The network learns to choose the moves minimax would choose.

## Test

From the project root:

```bash
python3 -m unittest discover -s tests -t . -v
```

The suite has 80 tests in seven files:

| File                     | Covers                                                                                |
|--------------------------|---------------------------------------------------------------------------------------|
| `tests/test_game.py`     | Win detection, minimax wins and blocks, move validity, exhaustive safety, random play |
| `tests/test_main.py`     | Board rendering, input validation, the main loop, scores, draws, quitting             |
| `tests/test_ui.py`       | Free cells, colour and `NO_COLOR`, screen clearing, highlighting, input messages      |
| `tests/test_settings.py` | Loading `env.json`, defaults, invalid values, turn order                              |
| `tests/test_model.py`    | Model registry: default, names, unknown model, minimax produces moves                 |
| `tests/test_mlmodel.py`  | ML model: encoding, binary save and load, missing weights, bundled weights            |
| `tests/test_build.py`    | Build functions: minimax moves, dataset, training, and the checks                     |

The slowest tests take about 10 seconds in total. The bundled-weights tests are skipped until `mlmodel/mlmodel.bin` is built.

## Project structure

```
tictactoe-game-ml/
├── play.py                         # Starts the game: python3 play.py
├── src/
│   └── tictactoe/                  # The game
│       ├── __init__.py             # Public API
│       ├── __main__.py             # Enables `python3 -m src.tictactoe`
│       ├── settings.py             # Loads env.json (FIRST_PLAYER)
│       ├── game.py                 # Board logic, minimax, terminal UI and game loop
│       └── model/                  # ML model package and its build script
│           ├── __init__.py         # Registry: load_model("minimax" | "mlmodel")
│           ├── mlmodel.py          # MLP: forward pass, binary save/load, load_weights()
│           └── build_mlmodel.py    # Trains mlmodel and writes mlmodel/mlmodel.bin
├── mlmodel/                        # Output folder for generated weights
│   ├── .gitkeep                    # Keeps the folder in git
│   └── mlmodel.bin                 # Created by the build (git-ignored)
├── tests/
│   ├── __init__.py
│   ├── test_build.py               # ML build function tests
│   ├── test_game.py                # Board logic and minimax tests
│   ├── test_main.py                # Main loop tests
│   ├── test_model.py               # Model registry tests
│   ├── test_mlmodel.py             # ML model tests
│   ├── test_settings.py            # Settings tests
│   └── test_ui.py                  # Terminal UI tests
├── env.example.json                # Settings template (copy to env.json)
├── pyproject.toml                  # Package metadata and `tictactoe` console script
├── .gitignore
└── README.md
```

## License

MIT
