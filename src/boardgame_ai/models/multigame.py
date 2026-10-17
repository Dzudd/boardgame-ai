"""Visible per-game streams and permutation-equivariant candidate scoring.

No engine imports. The common GRU consumes game-specific projected tokens;
candidate rows come only from the supplied legal tuple, in its original order.
"""

from dataclasses import dataclass

import torch
from torch import nn

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.breakthrough_observation import BreakthroughObservation
from boardgame_ai.games.connect_four.observation import ConnectFourObservation
from boardgame_ai.games.guildworks.observation import GuildworksObservation
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.models.encoders.connect_four import encode_connect_four_observation
from boardgame_ai.models.encoders.connect_four_actions import (
    encode_connect_four_action,
)
from boardgame_ai.models.encoders.tic_tac_toe import encode_tic_tac_toe_observation
from boardgame_ai.models.encoders.tic_tac_toe_actions import encode_tic_tac_toe_action
from boardgame_ai.models.guildworks import ACTIONS, decision_tokens, encode_actions

DEVELOPMENT_GAMES = ("tic_tac_toe", "guildworks", "connect_four")
TRANSFER_GAME = "breakthrough"
GAMES = (*DEVELOPMENT_GAMES, TRANSFER_GAME)
OBSERVATION_SIZES = dict(zip(GAMES, (54, 52, 252, 150), strict=True))
CANDIDATE_SIZES = dict(zip(GAMES, (9, 8, 7, 50), strict=True))
REWARD_SCALES = dict(zip(GAMES, (1.0, 36.0, 1.0, 1.0), strict=True))
MODEL_ID = "multigame-visible-gru-candidates-v1"
WIDTH = 32
type View = Observation[
    TicTacToeObservation
    | GuildworksObservation
    | ConnectFourObservation
    | BreakthroughObservation
]


def observation_tokens(
    game: str, view: View, *, history_start: int = 0
) -> torch.Tensor:
    """One board snapshot or newly visible Guildworks events plus snapshot."""
    data = view.data
    if game == "guildworks" and isinstance(data, GuildworksObservation):
        return decision_tokens(
            Observation(
                player_id=view.player_id,
                current_player=view.current_player,
                phase=view.phase,
                data=data,
            ),
            history_start=history_start,
        )
    if history_start != 0:
        raise ValueError("Only Guildworks has a visible event cursor")
    if game == "tic_tac_toe" and isinstance(data, TicTacToeObservation):
        return encode_tic_tac_toe_observation(
            Observation(
                player_id=view.player_id,
                current_player=view.current_player,
                phase=view.phase,
                data=data,
            )
        ).reshape(1, -1)
    if game == "connect_four" and isinstance(data, ConnectFourObservation):
        return encode_connect_four_observation(
            Observation(
                player_id=view.player_id,
                current_player=view.current_player,
                phase=view.phase,
                data=data,
            )
        ).reshape(1, -1)
    if game == TRANSFER_GAME and isinstance(data, BreakthroughObservation):
        seat, actor = view.player_id, view.current_player
        if (
            type(seat) is not int
            or seat not in (0, 1)
            or actor is not None
            and (type(actor) is not int or actor not in (0, 1))
            or view.phase != ("finished" if actor is None else "play")
            or len(data.board) != 25
            or any(
                c is not None and (type(c) is not int or c not in (0, 1))
                for c in data.board
            )
        ):
            raise ValueError("Invalid Breakthrough observation")
        # Rotate seat 1's view to put its home rank at the bottom.
        board = data.board if seat == 0 else data.board[::-1]
        return torch.tensor(
            [
                [float(c == seat) for c in board],
                [float(c == 1 - seat) for c in board],
                [float(c is None) for c in board],
                [float(actor == seat)] * 25,
                [float(actor == 1 - seat)] * 25,
                [float(actor is None)] * 25,
            ],
            dtype=torch.float32,
        ).reshape(1, -1)
    raise ValueError("Game/observation mismatch")


def candidate_features(game: str, legal: tuple[Action, ...], seat: int) -> torch.Tensor:
    """Encode supplied candidates; never reconstruct legality from a board."""
    if game not in GAMES or type(seat) is not int or seat not in (0, 1):
        raise ValueError("Invalid game or seat")
    if not isinstance(legal, tuple) or len(set(legal)) != len(legal):
        raise ValueError("Candidates must be a unique supplied tuple")
    result = torch.zeros((len(legal), CANDIDATE_SIZES[game]), dtype=torch.float32)
    for row, action in enumerate(legal):
        if game == "tic_tac_toe":
            result[row, encode_tic_tac_toe_action(action)] = 1
        elif game == "connect_four":
            result[row, encode_connect_four_action(action)] = 1
        elif game == "guildworks":
            encode_actions((action,))
            result[row, ACTIONS.index(action)] = 1
        else:
            if (
                action.kind != "move"
                or len(action.parameters) != 2
                or tuple(k for k, _ in action.parameters) != ("source", "target")
                or any(
                    type(v) is not int or not 0 <= v < 25 for _, v in action.parameters
                )
            ):
                raise ValueError("Invalid Breakthrough candidate")
            source, target = (v for _, v in action.parameters)
            assert type(source) is int and type(target) is int
            if seat == 1:
                source, target = 24 - source, 24 - target
            result[row, source] = result[row, 25 + target] = 1
    return result


@dataclass(frozen=True)
class DecisionInput:
    tokens: torch.Tensor
    candidates: torch.Tensor


class MultiGamePolicy(nn.Module):
    """Same architecture for shared, separate, fresh and pretrained arms.

    Values predict scaled own-seat return. All adapters exist from initialization
    so the held-out adapter has identical fresh weights in both adaptation arms.
    """

    def __init__(self) -> None:
        super().__init__()
        self.observation = nn.ModuleDict(
            {game: nn.Linear(size, WIDTH) for game, size in OBSERVATION_SIZES.items()}
        )
        self.candidate = nn.ModuleDict(
            {game: nn.Linear(size, WIDTH) for game, size in CANDIDATE_SIZES.items()}
        )
        self.value = nn.ModuleDict({game: nn.Linear(WIDTH, 1) for game in GAMES})
        self.recurrent = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.query = nn.ModuleDict({game: nn.Linear(WIDTH, WIDTH) for game in GAMES})

    def forward(
        self,
        game: str,
        decision: DecisionInput,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if game not in GAMES:
            raise ValueError("Unsupported game")
        tokens, candidates = decision.tokens, decision.candidates
        device = next(self.parameters()).device
        for tensor, size in (
            (tokens, OBSERVATION_SIZES[game]),
            (candidates, CANDIDATE_SIZES[game]),
        ):
            if (
                tensor.ndim != 2
                or tensor.shape[1] != size
                or tensor.dtype != torch.float32
                or tensor.device != device
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError("Invalid model input")
        if not len(tokens):
            raise ValueError("Empty visible stream")
        if hidden is not None and (
            hidden.shape != (1, 1, WIDTH)
            or hidden.dtype != torch.float32
            or hidden.device != device
            or not torch.isfinite(hidden).all().item()
        ):
            raise ValueError("Invalid private memory")
        output, memory = self.recurrent(
            self.observation[game](tokens).tanh().unsqueeze(0),
            hidden,
        )
        context = output[0, -1]
        logits = self.candidate[game](candidates).tanh() @ self.query[game](context)
        return logits / WIDTH**0.5, self.value[game](context).squeeze(-1), memory


def initial_multigame_model(seed: int) -> MultiGamePolicy:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Expected unsigned 64-bit seed")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return MultiGamePolicy()
