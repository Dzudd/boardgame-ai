"""Frozen observation-only neural evaluator."""

from copy import deepcopy

import torch

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.encoders import (
    encode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.models.masking import mask_action_logits


class NeuralEvaluator:
    def __init__(self, model: TicTacToeActorCritic) -> None:
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError("Requires CPU float32 model")
        self.model = deepcopy(model).eval().requires_grad_(False)

    def __call__(
        self, observation: Observation[TicTacToeObservation], legal: tuple[Action, ...]
    ) -> tuple[tuple[float, ...], float]:
        with torch.no_grad():
            logits, value = self.model(encode_tic_tac_toe_observation(observation))
            mask = encode_tic_tac_toe_legal_actions(legal)
            probabilities = mask_action_logits(logits, mask).softmax(-1)
            return tuple(
                float(probabilities[encode_tic_tac_toe_action(a)]) for a in legal
            ), float(value)
