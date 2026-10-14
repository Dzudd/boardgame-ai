"""Command-line entry point for diagnostics and human play."""

import argparse
import platform
from collections.abc import Sequence
from importlib.metadata import version


def main(argv: Sequence[str] | None = None) -> None:
    """Parse diagnostics or one human-versus-agent game."""
    parser = argparse.ArgumentParser(prog="boardgame-ai")
    parser.add_argument("--version", action="version", version=version("boardgame-ai"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Report Python, NumPy, and PyTorch availability")
    play = commands.add_parser("play", help="Play a supported game against an agent")
    play.add_argument("--game", choices=("tic-tac-toe",), default="tic-tac-toe")
    play.add_argument(
        "--opponent", choices=("random", "heuristic"), default="heuristic"
    )
    play.add_argument(
        "--starts",
        choices=("human", "ai"),
        default="human",
        help="Who plays first; the starter is X (default: human)",
    )
    play.add_argument("--engine-seed", type=int, help="Engine setup seed")
    play.add_argument(
        "--agent-seed", type=int, help="Opponent RNG seed; heuristic is deterministic"
    )
    play.add_argument("--save-replay", metavar="PATH", help="Save the public game log")
    replay = commands.add_parser("replay", help="Validate and inspect a saved game log")
    replay.add_argument("path")
    args = parser.parse_args(argv)

    if args.command == "info":
        # Keep expensive ML imports out of package imports and CLI help.
        import numpy as np
        import torch

        print(f"Python: {platform.python_version()}")
        print(f"NumPy: {np.__version__}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    elif args.command == "play":
        from boardgame_ai.cli_play import play_tic_tac_toe

        play_tic_tac_toe(
            starts=args.starts,
            opponent=args.opponent,
            engine_seed=args.engine_seed,
            agent_seed=args.agent_seed,
            replay_path=args.save_replay,
        )
    elif args.command == "replay":
        from boardgame_ai.cli_play import inspect_replay

        try:
            inspect_replay(args.path)
        except (OSError, ValueError):
            parser.error("Cannot read or validate replay")
