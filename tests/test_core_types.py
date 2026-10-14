"""Core contract invariants, independent of any particular game."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from boardgame_ai.core import (
    Action,
    GameEvent,
    GameSpec,
    GameState,
    Observation,
    StepResult,
)


def test_state_and_observation_can_have_distinct_payload_types() -> None:
    state = GameState(current_player=0, phase="draw", data=("secret", "public"))
    observation = Observation(
        player_id=1, current_player=0, phase=state.phase, data=state.data[1]
    )
    assert observation.data == "public"
    assert "secret" not in repr(observation)


def test_action_json_serialization_and_frozen_fields() -> None:
    action = Action(kind="place", parameters=(("row", 1), ("column", 2)))
    assert json.loads(json.dumps(asdict(action), allow_nan=False)) == {
        "kind": "place",
        "parameters": [["row", 1], ["column", 2]],
    }
    with pytest.raises(FrozenInstanceError):
        action.kind = "draw"  # type: ignore[misc]


def test_action_rejects_ambiguous_parameters() -> None:
    with pytest.raises(ValueError, match="unique"):
        Action(kind="place", parameters=(("row", 1), ("row", 2)))
    with pytest.raises(ValueError, match="finite"):
        Action(kind="bid", parameters=(("amount", float("nan")),))
    with pytest.raises(ValueError, match="empty"):
        Action(kind="")


def test_same_player_can_act_again_with_general_sum_rewards() -> None:
    result = StepResult(next_player=1, terminated=False, rewards=(1.0, 2.0, 3.0))
    assert result.next_player == 1
    assert sum(result.rewards) == 6.0


def test_terminal_result_has_no_next_player() -> None:
    result = StepResult(next_player=None, terminated=True, rewards=(1.0, -1.0))
    assert result.terminated


@pytest.mark.parametrize(
    ("next_player", "terminated", "rewards"),
    [
        (0, True, (0.0,)),
        (None, False, (0.0,)),
        (-1, False, (0.0,)),
        (2, False, (0.0, 0.0)),
        (None, True, ()),
        (None, True, (float("inf"),)),
    ],
)
def test_invalid_step_result(
    next_player: int | None, terminated: bool, rewards: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        StepResult(next_player=next_player, terminated=terminated, rewards=rewards)


def test_event_filtering_preserves_order_and_private_recipients() -> None:
    public = GameEvent(kind="draw", parameters=(("player", 0),))
    private = GameEvent(kind="card_drawn", recipients=(0,), parameters=(("card", "A"),))
    shared = GameEvent(kind="team_message", recipients=(0, 2))
    internal = GameEvent(kind="internal", recipients=())
    result = StepResult(
        next_player=0,
        terminated=False,
        rewards=(0.0, 0.0, 0.0),
        events=(public, private, shared, internal),
    )
    assert result.events_for(0) == (public, private, shared)
    assert result.events_for(1) == (public,)
    assert result.events_for(2) == (public, shared)
    with pytest.raises(ValueError):
        result.events_for(3)


@pytest.mark.parametrize(("minimum", "maximum"), [(0, 2), (3, 2)])
def test_invalid_player_counts(minimum: int, maximum: int) -> None:
    with pytest.raises(ValueError, match="Player counts"):
        GameSpec(
            game_id="example", name="Example", min_players=minimum, max_players=maximum
        )


def test_game_spec_supports_single_player_games() -> None:
    spec = GameSpec(game_id="solo", name="Solo", min_players=1, max_players=1)
    assert spec.min_players == spec.max_players == 1
