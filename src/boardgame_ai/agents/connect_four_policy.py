"""Frozen Connect Four policy acting only on filtered decision inputs."""

from copy import deepcopy
from typing import Literal

import torch

from boardgame_ai.agents.base import Agent
from boardgame_ai.core import Action, Observation, create_rng
from boardgame_ai.games.connect_four.observation import ConnectFourObservation
from boardgame_ai.models.connect_four import ConnectFourActorCritic
from boardgame_ai.models.encoders import (
    decode_connect_four_action,
    encode_connect_four_legal_actions,
    encode_connect_four_observation,
)
from boardgame_ai.models.masking import mask_action_logits

type ConnectFourPolicyMode = Literal["sampled", "greedy"]


class ConnectFourPolicyAgent(Agent[ConnectFourObservation]):
    """Own a frozen CPU float32 copy and a private per-game CPU generator.

    CPU/CUDA source parameters, gradients and child modes are preserved. Seeds
    are unsigned 64-bit integers used directly, matching Connect Four collection;
    None draws private system entropy. Greedy ties use the lowest column index.
    No engine, authoritative state, rewards or optimizer is retained.
    """

    def __init__(
        self,
        model: ConnectFourActorCritic,
        *,
        mode: ConnectFourPolicyMode = "sampled",
    ) -> None:
        if mode not in ("sampled", "greedy"):
            raise ValueError("Policy mode must be sampled or greedy")
        if not isinstance(model, ConnectFourActorCritic):
            raise ValueError("Expected a Connect Four actor-critic model")
        reference = model.policy_head.weight
        if reference.device.type not in ("cpu", "cuda") or any(
            p.dtype != torch.float32
            or p.device != reference.device
            or not torch.isfinite(p).all().item()
            for p in model.parameters()
        ):
            raise ValueError("Policy requires finite CPU/CUDA float32 parameters")
        self._model = deepcopy(model).cpu().eval().requires_grad_(False)
        self._model.zero_grad(set_to_none=True)
        self.mode = mode
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("Policy seed must be an unsigned 64-bit integer or None")
        actual = create_rng().getrandbits(64) if seed is None else seed
        self._generator = torch.Generator(device="cpu").manual_seed(actual)

    def select_action(
        self,
        observation: Observation[ConnectFourObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        encoded = encode_connect_four_observation(observation)
        if observation.player_id != observation.current_player:
            raise ValueError("Policy selection requires the actor's observation")
        mask = encode_connect_four_legal_actions(legal_actions)
        with torch.no_grad():
            logits, value = self._model(encoded)
            if value.shape != () or not torch.isfinite(value).item():
                raise ValueError("Policy value must be a finite scalar")
            masked = mask_action_logits(logits, mask)
            index = (
                int(masked.argmax().item())
                if self.mode == "greedy"
                else int(
                    torch.multinomial(
                        masked.log_softmax(-1).exp(), 1, generator=self._generator
                    ).item()
                )
            )
        return decode_connect_four_action(index, legal_actions)
