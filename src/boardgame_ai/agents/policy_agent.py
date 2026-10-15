"""Frozen CPU policy adapter using only the existing player-input contract."""

from copy import deepcopy
from typing import Literal

import torch

from boardgame_ai.agents.base import Agent
from boardgame_ai.core import Action, Observation, create_rng
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.models.masking import mask_action_logits

type PolicyMode = Literal["sampled", "greedy"]


class TicTacToePolicyAgent(Agent[TicTacToeObservation]):
    """Own an immutable-in-use model copy and separate per-game sampling RNG.

    The supplied model must be CPU float32. Snapshotting leaves the source's
    parameters, mode, and gradients unchanged. Greedy ties use lowest cell index;
    sampled play uses the masked distribution and a private CPU generator.
    No engine reference, authoritative state, rewards, or search is accepted.
    """

    def __init__(
        self, model: TicTacToeActorCritic, *, mode: PolicyMode = "sampled"
    ) -> None:
        if mode not in ("sampled", "greedy"):
            raise ValueError("Policy mode must be sampled or greedy")
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError("Evaluation policy requires CPU float32 parameters")
        self._model = deepcopy(model).eval().requires_grad_(False)
        self.mode = mode
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        generator = torch.Generator(device="cpu").manual_seed(
            create_rng(seed).getrandbits(64)
        )
        self._generator = generator

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        encoded = encode_tic_tac_toe_observation(observation)
        if observation.player_id != observation.current_player:
            raise ValueError("Policy selection requires the actor's observation")
        mask = encode_tic_tac_toe_legal_actions(legal_actions)
        with torch.no_grad():
            logits, _ = self._model.forward(encoded)
            masked = mask_action_logits(logits, mask)
            if self.mode == "greedy":
                index = int(masked.argmax().item())
            else:
                probabilities = masked.log_softmax(dim=-1).exp()
                index = int(
                    torch.multinomial(
                        probabilities, 1, generator=self._generator
                    ).item()
                )
        return decode_tic_tac_toe_action(index, legal_actions)
