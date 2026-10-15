"""User-launched training command; no work occurs during imports or help."""

from argparse import Namespace
from typing import Any

import torch

from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.monitoring import MonitoringConfig
from boardgame_ai.training.ppo import PPOConfig
from boardgame_ai.training.run import TrainingConfig, run_self_play_training


def train_from_args(args: Namespace) -> None:
    device = resolve_training_device(args.device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Training device: {device} ({name}); PyTorch {torch.__version__}")
    if args.check_device:
        print("Device check complete; no training or output files created.")
        return
    ppo = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        normalize_advantages=not args.no_normalize_advantages,
    )
    config = TrainingConfig(
        updates=args.updates,
        games_per_update=args.games_per_update,
        seed=args.seed,
        device=str(device),
        ppo=ppo,
    )

    def progress(record: dict[str, Any]) -> None:
        report = record["ppo"]
        print(
            f"Update {record['update']}/{config.updates}: "
            f"games={record['games']} samples={report['num_samples']} "
            f"loss={report['total_loss']:.4f} entropy={report['entropy']:.4f} "
            f"KL={report['approximate_kl']:.5f}",
            flush=True,
        )

    if args.monitor_interval < 0:
        raise ValueError("Monitoring interval must be nonnegative")
    monitoring = (
        MonitoringConfig(
            interval=args.monitor_interval,
            retain=args.monitor_retain,
            games_per_seat=args.monitor_games_per_seat,
            self_play_games=args.monitor_self_play_games,
            seed=args.monitor_seed,
        )
        if args.monitor_interval
        else None
    )
    trainer = run_self_play_training(
        args.output,
        config,
        resume=args.resume,
        on_update=progress,
        monitoring=monitoring,
    )
    print(f"Saved {trainer.update_count} completed updates in {args.output}/latest.pt")
