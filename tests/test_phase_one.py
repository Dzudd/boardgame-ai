"""Visibility, randomness, failure atomicity, and reusable engine contracts."""

import random

import pytest

from boardgame_ai.core import Action, Observation, create_rng
from tests.contracts import assert_engine_contract, assert_rejection_preserves_game
from tests.hidden_game import CardView, HiddenGame
from tests.test_interfaces import DummyGame


def test_deterministic_engine_contract() -> None:
    assert_engine_contract(DummyGame, max_steps=4)


def test_hidden_chance_engine_contract() -> None:
    assert_engine_contract(HiddenGame, max_steps=4)


def test_observations_do_not_alias_private_state() -> None:
    game = HiddenGame()
    game.reset(seed=42)
    original = game.observe(0)
    view, _ = game.decision_for(0)
    view.data.own_hand.append(-1)
    assert game.observe(0) == original
    observation = game.observe(0)
    observation.data.own_hand.clear()
    assert game.observe(0) == original


def test_unseen_changes_do_not_affect_player_inputs() -> None:
    game = HiddenGame()
    game.reset(seed=42)
    other_world = game.clone()
    # Deliberate test-only access: change only information unavailable to seat 0.
    other_world._hands[1] = [-999]
    other_world._deck.reverse()
    assert game.decision_for(0) == other_world.decision_for(0)
    assert game.observe(1) != other_world.observe(1)


def test_draw_outcome_is_revealed_only_to_drawer_after_action() -> None:
    game = HiddenGame()
    game.reset(seed=42)
    _, choices = game.decision_for(0)
    assert choices == (Action(kind="draw"),)
    result = game.step(choices[0])
    assert tuple(event.kind for event in result.events_for(0)) == ("drew", "card")
    assert tuple(event.kind for event in result.events_for(1)) == ("drew",)
    assert game.observe(1).data.hand_sizes == (2, 1)


@pytest.mark.parametrize(
    "action",
    [Action(kind="unknown"), Action(kind="draw", parameters=(("card", 7),))],
)
def test_rejected_action_does_not_change_state_or_rng(action: Action) -> None:
    game = HiddenGame()
    game.reset(seed=42)
    assert_rejection_preserves_game(game, action, max_steps=4)


def test_querying_views_and_cloning_do_not_consume_chance() -> None:
    game = HiddenGame()
    game.reset(seed=42)
    untouched = game.clone()
    for _ in range(5):
        game.decision_for(0)
        game.observe(1)
        game.get_rewards()
        game.clone()
    assert game.step(Action(kind="draw")) == untouched.step(Action(kind="draw"))


def test_rngs_are_independent_and_do_not_touch_global_randomness() -> None:
    global_state = random.getstate()
    first = create_rng(42)
    second = create_rng(42)
    assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]
    create_rng(None).random()
    game = HiddenGame()
    game.reset(seed=42)
    game.step(Action(kind="draw"))
    assert random.getstate() == global_state


def test_rejection_message_does_not_include_private_state() -> None:
    game = HiddenGame()
    game.reset(seed=42)
    game._hands[1] = [-987654321]
    with pytest.raises(ValueError) as exc:
        game.step(Action(kind="unknown"))
    assert str(exc.value) == "Action is not legal"


def test_decision_helper_copies_even_an_aliased_payload() -> None:
    class AliasedViewGame(HiddenGame):
        def observe(self, player_id: int) -> Observation[CardView]:
            view = super().observe(player_id)
            view.data.own_hand = self._hands[player_id]
            return view

    game = AliasedViewGame()
    game.reset(seed=42)
    original = game._hands[0].copy()
    view, _ = game.decision_for(0)
    view.data.own_hand.clear()
    assert game._hands[0] == original


def test_invalid_seed_preserves_existing_game() -> None:
    game = HiddenGame()
    game.reset(seed=42)
    before = game.clone()
    with pytest.raises(ValueError, match="Seed"):
        game.reset(seed=True)
    assert game.step(Action(kind="draw")) == before.step(Action(kind="draw"))
