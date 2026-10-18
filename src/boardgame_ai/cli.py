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
    play.add_argument(
        "--game",
        choices=("tic-tac-toe", "guildworks", "connect-four"),
        default="tic-tac-toe",
    )
    play.add_argument(
        "--opponent",
        choices=("random", "heuristic"),
        help="Default: heuristic for Tic-Tac-Toe, random for other games",
    )
    play.add_argument(
        "--starts",
        choices=("human", "ai"),
        default="human",
        help="Who plays first; starter is seat 0 (default: human)",
    )
    play.add_argument("--engine-seed", type=int, help="Engine setup seed")
    play.add_argument(
        "--agent-seed", type=int, help="Opponent RNG seed; heuristic is deterministic"
    )
    play.add_argument(
        "--save-replay", metavar="PATH", help="Save a public Tic-Tac-Toe game log"
    )
    play.add_argument(
        "--checkpoint", metavar="PATH", help="Connect Four portable trainer snapshot"
    )
    play.add_argument(
        "--policy-mode",
        choices=("sampled", "greedy"),
        help="Checkpoint action selection (default: sampled)",
    )
    replay = commands.add_parser("replay", help="Validate and inspect a saved game log")
    replay.add_argument("path")
    train = commands.add_parser("train", help="Run Tic-Tac-Toe PPO self-play training")
    train.add_argument(
        "--device", default="cuda", help="cuda (default), cuda:N, or cpu"
    )
    train.add_argument(
        "--output", default="runs/tic-tac-toe", help="Training output folder"
    )
    train.add_argument(
        "--updates",
        type=int,
        default=1000,
        help="Total update budget, including resumed updates",
    )
    train.add_argument("--games-per-update", type=int, default=64)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--resume", action="store_true", help="Resume latest.pt in output folder"
    )
    train.add_argument(
        "--check-device",
        action="store_true",
        help="Check device and exit without creating files or training",
    )
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--clip-ratio", type=float, default=0.2)
    train.add_argument("--value-coefficient", type=float, default=0.5)
    train.add_argument("--entropy-coefficient", type=float, default=0.01)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--no-normalize-advantages", action="store_true")
    train.add_argument(
        "--monitor-interval",
        type=int,
        default=0,
        help="Enable isolated development monitoring every N updates; 0 disables",
    )
    train.add_argument("--monitor-retain", type=int, default=3)
    train.add_argument("--monitor-games-per-seat", type=int, default=128)
    train.add_argument("--monitor-self-play-games", type=int, default=64)
    train.add_argument("--monitor-seed", type=int, default=20261022)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a checkpoint against fixed opponents on CPU"
    )
    evaluate.add_argument("checkpoint", help="Completed training checkpoint")
    evaluate.add_argument("--output", default="runs/tic-tac-toe-evaluation.json")
    evaluate.add_argument("--games-per-seat", type=int, default=1000)
    evaluate.add_argument(
        "--seed", type=int, default=20261015, help="Held-out evaluation root seed"
    )
    evaluate.add_argument("--start-index", type=int, default=0)
    evaluate.add_argument("--mode", choices=("sampled", "greedy"), default="sampled")
    analyze = commands.add_parser(
        "analyze", help="Plot training curves and compare baseline controls"
    )
    analyze.add_argument("checkpoint")
    analyze.add_argument(
        "--training-log", help="Defaults to updates.jsonl beside checkpoint"
    )
    analyze.add_argument("--output", default="runs/phase4-analysis")
    analyze.add_argument("--games-per-seat", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=20261015)
    analyze.add_argument("--start-index", type=int, default=0)
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
        if args.checkpoint is not None:
            if args.game != "connect-four":
                parser.error("Checkpoint play currently supports Connect Four only")
            if args.opponent is not None:
                parser.error("Choose either --checkpoint or --opponent")
        elif args.policy_mode is not None:
            parser.error("--policy-mode requires --checkpoint")
        if args.game == "connect-four":
            if args.opponent not in (None, "random"):
                parser.error("Connect Four supports only the random opponent")
            if args.save_replay is not None:
                parser.error("Connect Four replay saving is not supported")
            from pickle import UnpicklingError

            from boardgame_ai.cli_connect_four import play_connect_four

            try:
                play_connect_four(
                    starts=args.starts,
                    engine_seed=args.engine_seed,
                    agent_seed=args.agent_seed,
                    checkpoint=args.checkpoint,
                    policy_mode=args.policy_mode,
                )
            except (OSError, ValueError, RuntimeError, UnpicklingError, EOFError):
                parser.error(
                    "Cannot load or use Connect Four opponent; --checkpoint requires "
                    "a valid portable trainer snapshot (initial.pt/final.pt), "
                    "and a policy seed must be unsigned 64-bit"
                )
            return
        if args.game == "guildworks":
            if args.opponent not in (None, "random"):
                parser.error("Guildworks supports only the random opponent")
            if args.save_replay is not None:
                parser.error("Guildworks replay saving is not supported")
            from boardgame_ai.cli_guildworks import play_guildworks

            play_guildworks(
                starts=args.starts,
                engine_seed=args.engine_seed,
                agent_seed=args.agent_seed,
            )
            return
        from boardgame_ai.cli_play import play_tic_tac_toe

        play_tic_tac_toe(
            starts=args.starts,
            opponent=args.opponent or "heuristic",
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
    elif args.command == "train":
        from boardgame_ai.cli_train import train_from_args

        try:
            train_from_args(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        except KeyboardInterrupt:
            print("Training interrupted; latest.pt retains the last saved update.")
    elif args.command == "evaluate":
        from pickle import UnpicklingError

        from boardgame_ai.cli_evaluate import evaluate_from_args

        try:
            evaluate_from_args(args)
        except (OSError, ValueError, RuntimeError, UnpicklingError) as error:
            parser.error(str(error))
    elif args.command == "analyze":
        from pickle import UnpicklingError

        from boardgame_ai.cli_analyze import analyze_from_args

        try:
            analyze_from_args(args)
        except (OSError, ValueError, RuntimeError, UnpicklingError) as error:
            parser.error(str(error))
