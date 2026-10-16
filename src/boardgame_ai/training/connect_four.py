"""Complete Connect Four latest-policy self-play; no optimizer or evaluation."""

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from time import monotonic

import torch

from boardgame_ai.core import GameEngine, create_rng
from boardgame_ai.games.connect_four.observation import ConnectFourObservation
from boardgame_ai.models.connect_four import ConnectFourActorCritic
from boardgame_ai.models.encoders import (
    decode_connect_four_action,
    encode_connect_four_legal_actions,
    encode_connect_four_observation,
)
from boardgame_ai.models.masking import mask_action_logits


def derive_connect_four_seed(root: int, index: int, stream: str) -> int:
    """Protocol v1 training seed; evaluation has a separate future namespace."""
    if type(root) is not int or type(index) is not int or index < 0:
        raise ValueError("Seed root/index must be integers with nonnegative index")
    if stream not in ("initialization", "shuffle", "engine", "seat:0", "seat:1"):
        raise ValueError("Unsupported Connect Four training seed stream")
    text = f"boardgame-ai/connect-four/ppo/v1:{root}:{index}:{stream}"
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectFourTrajectory:
    """Owned detached CPU tensors for one seat's complete decision sequence.

    Observations (T,6,6,7), bool masks (T,7), int64 actions/step_indices (T,),
    float32 log_probs/values/rewards (T,). Rewards include intervening opponent
    moves through termination. Step indices are zero-based global game steps.
    Value/reward perspective is this seat; terminal bootstrap is always zero.
    These are trusted learning records, not agent inputs or public exports.
    """

    player_id: int
    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    step_indices: torch.Tensor
    bootstrap_value: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectFourRollout:
    """One complete game only; a cutoff raises instead of returning samples."""

    players: tuple[ConnectFourTrajectory, ConnectFourTrajectory]
    num_steps: int
    total_rewards: tuple[float, float]


@dataclass
class _Pending:
    observations: list[torch.Tensor] = field(default_factory=list)
    masks: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    def finish(self, seat: int) -> ConnectFourTrajectory:
        if not self.actions:
            raise RuntimeError("A complete Connect Four game needs both seat sequences")
        result = ConnectFourTrajectory(
            player_id=seat,
            observations=torch.stack(self.observations),
            legal_masks=torch.stack(self.masks),
            actions=torch.tensor(self.actions, dtype=torch.int64, device="cpu"),
            log_probs=torch.tensor(self.log_probs, dtype=torch.float32, device="cpu"),
            values=torch.tensor(self.values, dtype=torch.float32, device="cpu"),
            rewards=torch.tensor(self.rewards, dtype=torch.float32, device="cpu"),
            step_indices=torch.tensor(self.steps, dtype=torch.int64, device="cpu"),
        )
        for scalars in (result.log_probs, result.values, result.rewards):
            if not torch.isfinite(scalars).all().item():
                raise ValueError("Trajectory scalars must fit finite float32 storage")
        return result


def collect_connect_four_self_play(
    game: GameEngine[ConnectFourObservation],
    model: ConnectFourActorCritic,
    *,
    sampling_seeds: tuple[int, int],
    engine_seed: int | None = None,
    max_steps: int = 42,
    deadline: float | None = None,
) -> ConnectFourRollout:
    """Reset and collect with one frozen CPU copy shared by both seats.

    Seed private per-seat CPU sampling generators directly with unsigned 64-bit
    seeds. No global RNG is consumed. Original parameters, gradients and modes
    are preserved, including for a CUDA source model. Only filtered decision
    tensors reach forward. No terminal inference, baseline data or replay.

    A live step cutoff raises RuntimeError. A monotonic-clock deadline raises
    TimeoutError before reset or between decisions; partial samples never return.
    Configuration/model failures reject before resetting the supplied game.
    Later failures leave executed engine steps in place and never retry them.
    """
    if type(max_steps) is not int or not 1 <= max_steps <= 42:
        raise ValueError("max_steps must be an integer from 1 to 42")
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not isfinite(deadline)
    ):
        raise ValueError("Deadline must be a finite monotonic-clock time")
    if game.num_players != 2 or game.spec.game_id != "connect_four":
        raise ValueError("Expected a two-seat Connect Four engine")
    if not isinstance(model, ConnectFourActorCritic):
        raise ValueError("Expected a Connect Four actor-critic model")
    if (
        not isinstance(sampling_seeds, tuple)
        or len(sampling_seeds) != 2
        or any(
            type(seed) is not int or not 0 <= seed < 2**64 for seed in sampling_seeds
        )
    ):
        raise ValueError("Provide two unsigned 64-bit sampling seeds")
    create_rng(engine_seed)
    reference = model.policy_head.weight
    if reference.device.type not in ("cpu", "cuda") or any(
        p.dtype != torch.float32 or p.device != reference.device
        for p in model.parameters()
    ):
        raise ValueError("Collection requires a CPU/CUDA float32 model")
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Collection model parameters must be finite")

    def check_deadline() -> None:
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("Connect Four collection deadline reached")

    check_deadline()
    policy = deepcopy(model).cpu().eval()
    policy.zero_grad(set_to_none=True)
    generators = tuple(
        torch.Generator(device="cpu").manual_seed(seed) for seed in sampling_seeds
    )
    check_deadline()
    game.reset(engine_seed)
    pending = (_Pending(), _Pending())
    totals = [0.0, 0.0]
    steps = 0
    while True:
        check_deadline()
        actor = game.current_player()
        if game.is_terminal() != (actor is None):
            raise RuntimeError("Engine actor and termination disagree")
        if actor is None:
            break
        if steps == max_steps:
            raise RuntimeError("Connect Four collection requires a complete game")
        game.validate_player(actor)
        view, legal = game.decision_for(actor)
        if view.player_id != actor or view.current_player != actor:
            raise RuntimeError("Engine decision view does not identify the actor")
        encoded = encode_connect_four_observation(view)
        mask = encode_connect_four_legal_actions(legal)
        with torch.no_grad():
            logits, value = policy.forward(encoded)
            if (
                logits.shape != (7,)
                or value.shape != ()
                or logits.dtype != torch.float32
                or value.dtype != torch.float32
                or logits.device.type != "cpu"
                or value.device.type != "cpu"
                or not torch.isfinite(value).item()
            ):
                raise ValueError(
                    "Model must produce seven CPU logits and a finite value"
                )
            log_probs = mask_action_logits(logits, mask).log_softmax(-1)
            index = int(
                torch.multinomial(
                    log_probs.exp(), 1, generator=generators[actor]
                ).item()
            )
        action = decode_connect_four_action(index, legal)
        record = pending[actor]
        record.observations.append(encoded)
        record.masks.append(mask)
        record.actions.append(index)
        record.log_probs.append(float(log_probs[index].item()))
        record.values.append(float(value.item()))
        record.rewards.append(0.0)
        record.steps.append(steps)
        check_deadline()
        result = game.step(action)
        if (
            result.next_player != game.current_player()
            or result.terminated != game.is_terminal()
            or len(result.rewards) != 2
        ):
            raise RuntimeError("Engine step result disagrees with engine contract")
        for seat, reward in enumerate(result.rewards):
            totals[seat] += reward
            if pending[seat].rewards:
                pending[seat].rewards[-1] += reward
            if not isfinite(totals[seat]):
                raise ValueError("Accumulated rewards must remain finite")
        steps += 1

    return ConnectFourRollout(
        players=(pending[0].finish(0), pending[1].finish(1)),
        num_steps=steps,
        total_rewards=(totals[0], totals[1]),
    )
