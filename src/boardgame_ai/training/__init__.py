"""Trusted rollout orchestration and player-specific learning targets."""

from boardgame_ai.training.advantages import generalized_advantage_estimation
from boardgame_ai.training.checkpoint import load_ppo_checkpoint, save_ppo_checkpoint
from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.ppo import (
    PPOBatch,
    PPOConfig,
    PPOLoss,
    PPOTrainer,
    PPOUpdate,
    ppo_loss,
    prepare_ppo_batch,
)
from boardgame_ai.training.rollout import (
    PlayerTrajectory,
    TicTacToeRollout,
    collect_tic_tac_toe_rollout,
)
from boardgame_ai.training.run import TrainingConfig, run_self_play_training

__all__ = [
    "PPOBatch",
    "PPOConfig",
    "PPOLoss",
    "PPOTrainer",
    "PPOUpdate",
    "PlayerTrajectory",
    "TicTacToeRollout",
    "TrainingConfig",
    "collect_tic_tac_toe_rollout",
    "generalized_advantage_estimation",
    "load_ppo_checkpoint",
    "ppo_loss",
    "prepare_ppo_batch",
    "resolve_training_device",
    "run_self_play_training",
    "save_ppo_checkpoint",
]
