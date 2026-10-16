"""Version-1 visible-history tokens, fixed action mapping and recurrent policy.

No authoritative engine imports. Token streams contain newly visible events then
one snapshot per own decision. Training recomputes full episodes from zero memory.
"""

from copy import deepcopy

import torch
from torch import nn

from boardgame_ai.agents.base import Agent
from boardgame_ai.core import Action, GameEvent, Observation, create_rng
from boardgame_ai.games.guildworks.observation import GuildworksObservation
from boardgame_ai.models.masking import mask_action_logits

SPACES = ("inkwell", "archive", "festival", "lottery")
EVENTS = ("draw", "card", "place", "points", "ink", "craft", "pass", "round")
ACTIONS = (
    *(Action(kind="place", parameters=(("space", s),)) for s in SPACES),
    *(Action(kind="craft", parameters=(("card", c),)) for c in (1, 2, 3)),
    Action(kind="pass"),
)
FEATURES = 52
ACTION_COUNT = 8
HIDDEN_SIZE = 32
ENCODING_ID = "guildworks-visible-sequence-v1"


def encode_actions(legal: tuple[Action, ...]) -> torch.Tensor:
    """Encode supplied choices only; inactive/terminal empty choices stay empty."""
    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    for action in legal:
        if action not in ACTIONS or any(
            type(v) not in (str, int) for _, v in action.parameters
        ):
            raise ValueError("Unsupported Guildworks action")
        index = ACTIONS.index(action)
        if mask[index]:
            raise ValueError("Duplicate Guildworks action")
        mask[index] = True
    return mask


def decode_action(index: int, legal: tuple[Action, ...]) -> Action:
    if type(index) is not int or not 0 <= index < ACTION_COUNT:
        raise ValueError("Invalid action index")
    encode_actions(legal)
    for action in legal:
        if action == ACTIONS[index]:
            return action
    raise ValueError("Action index is not supplied as legal")


def _relative(owner: int | None, seat: int) -> list[float]:
    return [
        float(owner is None),
        float(owner == seat),
        float(owner is not None and owner != seat),
    ]


def encode_snapshot(view: Observation[GuildworksObservation]) -> torch.Tensor:
    """Encode any valid observer, including inactive and terminal seats.

    Fixed scales are representational only, not reward normalization or clipping.
    Event history is encoded separately and preserves discard/acquisition order.
    """
    data = view.data
    if type(view.player_id) is not int or view.player_id not in (0, 1):
        raise ValueError("Invalid observer")
    if view.phase not in ("place", "craft", "finished"):
        raise ValueError("Invalid phase")
    own, other = view.player_id, 1 - view.player_id
    features = [data.round / 4]
    features += [float(view.phase == p) for p in ("place", "craft", "finished")]
    features += _relative(view.current_player, own)
    for owner in data.placements:
        features += _relative(owner, own)
    for values, scale in ((data.workers, 2), (data.ink, 32), (data.scores, 36)):
        features += [values[own] / scale, values[other] / scale]
    features += [data.own_hand.count(c) / 4 for c in (1, 2, 3)]
    features += [data.hand_counts[own] / 12, data.hand_counts[other] / 12]
    features += [data.deck_count / 12]
    features += [data.discard.count(c) / 4 for c in (1, 2, 3)]
    if len(features) != 34:
        raise ValueError("Invalid Guildworks observation shape")
    return torch.tensor([1.0, 0.0, *features, *([0.0] * 16)], dtype=torch.float32)


def encode_event(event: GameEvent, seat: int) -> torch.Tensor:
    if not event.visible_to(seat) or event.kind not in EVENTS:
        raise ValueError("Event is not a supported visible event")
    parameters = dict(event.parameters)
    owner = parameters.get("player", seat)
    value = parameters.get("card") if event.kind == "card" else parameters["value"]
    if type(owner) is not int or owner not in (0, 1):
        raise ValueError("Invalid event owner")
    features = [float(event.kind == k) for k in EVENTS]
    features += _relative(owner, seat)
    features += [value / 4 if type(value) is int else 0.0]
    features += [float(value == space) for space in SPACES]
    return torch.tensor([0.0, 1.0, *([0.0] * 34), *features], dtype=torch.float32)


def decision_tokens(
    view: Observation[GuildworksObservation], *, history_start: int = 0
) -> torch.Tensor:
    if type(history_start) is not int or not 0 <= history_start <= len(
        view.data.history
    ):
        raise ValueError("Invalid history cursor")
    return torch.stack(
        [
            *(
                encode_event(e, view.player_id)
                for e in view.data.history[history_start:]
            ),
            encode_snapshot(view),
        ]
    )


class GuildworksRecurrentPolicy(nn.Module):
    """GRU(52,32), eight policy logits and an observer-relative own-points value."""

    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(FEATURES, HIDDEN_SIZE, batch_first=True)
        self.policy_head = nn.Linear(HIDDEN_SIZE, ACTION_COUNT)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(
        self, tokens: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            tokens.ndim != 2
            or tokens.shape[1] != FEATURES
            or not len(tokens)
            or tokens.dtype != torch.float32
            or tokens.device != self.policy_head.weight.device
            or not torch.isfinite(tokens).all().item()
        ):
            raise ValueError("Expected finite float32 sequence on the model device")
        if hidden is not None and (
            hidden.shape != (1, 1, HIDDEN_SIZE)
            or hidden.dtype != torch.float32
            or hidden.device != tokens.device
            or not torch.isfinite(hidden).all().item()
        ):
            raise ValueError("Invalid recurrent memory")
        output, next_hidden = self.recurrent(tokens.unsqueeze(0), hidden)
        output = output.squeeze(0)
        return (
            self.policy_head(output),
            self.value_head(output).squeeze(-1),
            next_hidden,
        )


class RecurrentGuildworksAgent(Agent[GuildworksObservation]):
    """Frozen CPU policy with private per-seat memory and sampling generator.

    Tokens and last predictions are available only for trusted on-policy collection.
    Neither the model nor memory receives rewards, engines, seeds or private events.
    """

    def __init__(self, model: GuildworksRecurrentPolicy) -> None:
        self.model = deepcopy(model).cpu().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        generator = torch.Generator().manual_seed(create_rng(seed).getrandbits(64))
        self.generator = generator
        self.hidden: torch.Tensor | None = None
        self.history: tuple[GameEvent, ...] = ()
        self.seat: int | None = None
        self.tokens: list[torch.Tensor] = []
        self.last_log_prob = 0.0
        self.last_value = 0.0

    def select_action(
        self,
        observation: Observation[GuildworksObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        if (
            not legal_actions
            or observation.current_player != observation.player_id
            or observation.phase == "finished"
        ):
            raise ValueError("Policy requires an active decision")
        if self.seat is not None and self.seat != observation.player_id:
            raise ValueError("Use a separate recurrent agent per seat")
        if observation.data.history[: len(self.history)] != self.history:
            raise ValueError("Reset recurrent memory before a new episode")
        mask = encode_actions(legal_actions)
        tokens = decision_tokens(observation, history_start=len(self.history))
        with torch.no_grad():
            logits, values, hidden = self.model(tokens, self.hidden)
            log_probs = mask_action_logits(logits[-1], mask).log_softmax(-1)
            if not torch.isfinite(values[-1]).item():
                raise ValueError("Nonfinite value prediction")
            index = int(
                torch.multinomial(log_probs.exp(), 1, generator=self.generator).item()
            )
        self.hidden = hidden.detach()
        self.tokens.extend(t.detach().clone() for t in tokens)
        self.history = observation.data.history
        self.seat = observation.player_id
        self.last_log_prob = float(log_probs[index])
        self.last_value = float(values[-1])
        return decode_action(index, legal_actions)
