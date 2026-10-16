"""Rules, complete runner games and adversarial information-boundary checks."""

import random
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from boardgame_ai.agents import Agent, RandomAgent
from boardgame_ai.core import Action, GameEvent, IllegalActionError, Observation
from boardgame_ai.games.guildworks import GuildworksEngine, GuildworksObservation
from boardgame_ai.games.guildworks.engine import MAX_STEPS, SPACES
from boardgame_ai.games.guildworks.evaluation import guildworks_outcome
from boardgame_ai.simulation import evaluate_batch, run_match
from tests.contracts import assert_engine_contract, assert_rejection_preserves_game


def place(space: str) -> Action:
    return Action(kind="place", parameters=(("space", space),))


def craft(card: int) -> Action:
    return Action(kind="craft", parameters=(("card", card),))


PASS = Action(kind="pass")


class ScriptedAgent(Agent[GuildworksObservation]):
    """Pass crafting; choose festival/lottery or archive/inkwell by round order."""

    def select_action(
        self,
        observation: Observation[GuildworksObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        assert type(observation.data) is GuildworksObservation
        if observation.phase == "craft":
            return PASS
        priority = ("festival", "archive", "lottery", "inkwell")
        return next(place(s) for s in priority if place(s) in legal_actions)


def fresh() -> GuildworksEngine:
    game = GuildworksEngine()
    game.reset(123)
    return game


def test_shared_contracts() -> None:
    assert_engine_contract(GuildworksEngine, max_steps=MAX_STEPS + 1)


@pytest.mark.parametrize("phase", ["place", "craft"])
@pytest.mark.parametrize(
    "action",
    [
        Action(kind="draw", parameters=(("card", 1),)),
        craft(99),
        Action(kind="craft", parameters=(("card", True),)),
        Action(kind="craft", parameters=(("card", 1.0),)),
        place("secret"),
        Action(kind="pass", parameters=(("extra", 1),)),
    ],
)
def test_rejection(phase: str, action: Action) -> None:
    game = fresh()
    if phase == "craft":
        game.step(place("festival"))
    assert_rejection_preserves_game(game, action, max_steps=33)


def test_rounds_workers_rewards_and_same_actor() -> None:
    game = fresh()
    total = [0.0, 0.0]
    for round_number in range(1, 5):
        starter = (round_number - 1) % 2
        for turn, space in enumerate(SPACES):
            seat = (starter + turn) % 2
            assert game.current_player() == seat
            assert game.observe(seat).data.round == round_number
            result = game.step(place(space))
            assert result.next_player == seat
            assert game.observe(seat).phase == "craft"
            view = game.observe(seat).data
            assert view.placements[turn] == seat
            assert view.workers[seat] == (1 if turn < 2 else 0)
            expected = [0.0, 0.0]
            expected[seat] = float(space == "festival")
            assert result.rewards == tuple(expected)
            total[seat] += result.rewards[seat]
            game.step(PASS)
            assert game.get_rewards() == (0.0, 0.0)
    assert game.is_terminal()
    assert game.observe(0).data.scores == tuple(total) == (2, 2)
    assert guildworks_outcome(game).draw
    assert game.observe(0).data.deck_count == 0


def test_craft_cost_discard_increment_and_no_hidden_card_actions() -> None:
    game = fresh()
    game.step(place("inkwell"))
    card = game.observe(0).data.own_hand[0]
    before = game.observe(0).data
    result = game.step(craft(card))
    after = game.observe(0).data
    assert result.rewards == (float(card + 1), 0.0)
    assert after.ink == (before.ink[0] - card, 2)
    assert after.discard == (card,)
    assert after.hand_counts == (1, 2)
    assert result.events == (
        GameEvent(kind="craft", parameters=(("player", 0), ("value", card))),
    )


def test_empty_deck_unavailable_space_and_forced_pass() -> None:
    game = fresh()
    game._deck.clear()
    game._hands[0].clear()
    result = game.step(place("archive"))
    assert result.events[-1].kind == "draw"
    assert result.events[-1].parameters[-1] == ("value", 0)
    assert game.legal_actions(0) == (PASS,)
    game.step(PASS)
    assert place("archive") not in game.legal_actions(1)
    assert_rejection_preserves_game(game, place("archive"), max_steps=33)


def test_partial_draw_and_private_history() -> None:
    game = fresh()
    game._deck = [3]
    old = game.observe(0)
    result = game.step(place("archive"))
    assert game.observe(0).data.hand_counts == (3, 2)
    assert game.observe(0).data.deck_count == 0
    assert result.events[-1].recipients == (0,)
    assert result.events[-1] not in result.events_for(1)
    assert result.events[-1] in game.observe(0).data.history
    assert result.events[-1] not in game.observe(1).data.history
    assert old.data.hand_counts == (2, 2)
    with pytest.raises(FrozenInstanceError):
        old.data.deck_count = 900  # type: ignore[misc]


@pytest.mark.parametrize("observer", [0, 1])
@pytest.mark.parametrize("phase", ["place", "craft", "finished"])
def test_indistinguishable_worlds(observer: int, phase: str) -> None:
    game = fresh()
    if phase == "craft":
        game.step(place("inkwell"))
    elif phase == "finished":
        while not game.is_terminal():
            actor = game.current_player()
            assert actor is not None
            choices = game.legal_actions(actor)
            game.step(PASS if PASS in choices else choices[0])
    twin = game.clone()
    # Exchange an opponent setup card with an unseen deck card, preserving the
    # published deck composition and the opponent's legitimate private history.
    other = 1 - observer
    old_card = twin._hands[other][0]
    if twin._deck:
        index = next(i for i, c in enumerate(twin._deck) if c != old_card)
        new_card = twin._deck[index]
        twin._deck[index] = old_card
        twin._hands[other][0] = new_card
        event_index = next(
            i for i, e in enumerate(twin._history) if e.recipients == (other,)
        )
        twin._history[event_index] = replace(
            twin._history[event_index], parameters=(("card", new_card),)
        )
    else:
        # After all draws, public composition constrains remaining card values.
        # Private acquisition order is still secret and can differ coherently.
        indices = [i for i, e in enumerate(twin._history) if e.recipients == (other,)]
        events = [twin._history[i] for i in indices]
        for i, event in zip(indices, reversed(events), strict=True):
            twin._history[i] = event
        twin._hands[other].reverse()
    twin._deck.reverse()
    assert twin.decision_for(observer) == game.decision_for(observer)
    assert twin.render(observer) == game.render(observer)
    # Queries cannot change the private generator.
    assert twin._rng.getstate() == game._rng.getstate()


def test_deck_order_only_revealed_on_draw() -> None:
    game = fresh()
    twin = game.clone()
    twin._deck = list(reversed(game._deck))
    # Ensure distinct next cards even if the shuffle's ends happen to match.
    index = next(i for i, c in enumerate(twin._deck) if c != game._deck[-1])
    twin._deck[index], twin._deck[-1] = twin._deck[-1], twin._deck[index]
    assert game.decision_for(0) == twin.decision_for(0)
    game.step(place("archive"))
    twin.step(place("archive"))
    assert game.decision_for(1) == twin.decision_for(1)
    assert game.observe(0).data.history != twin.observe(0).data.history


def test_clone_reset_and_rejected_seed_preserve_everything() -> None:
    game = fresh()
    twin = game.clone()
    before = game.observe(0)
    twin_result = twin.step(place("lottery"))
    assert game.observe(0) == before
    assert game.step(place("lottery")) == twin_result
    with pytest.raises(ValueError):
        game.reset(True)
    assert game.observe(0) == twin.observe(0)
    assert game._rng.getstate() == twin._rng.getstate()
    twin.reset(99)
    assert game.observe(0) != twin.observe(0)


def test_scripted_and_random_runner_games() -> None:
    game = GuildworksEngine()
    result = run_match(
        game,
        (ScriptedAgent(), ScriptedAgent()),
        engine_seed=7,
        agent_seeds=(1, 2),
        max_steps=MAX_STEPS,
    )
    assert result.terminated and result.num_steps == MAX_STEPS
    assert result.total_rewards == (2.0, 2.0)
    assert guildworks_outcome(game).draw
    assert "finished" in game.render(0)
    for seat in (0, 1):
        assert game.legal_actions(seat) == ()
        assert all(e.visible_to(seat) for e in game.observe(seat).data.history)
    agents = (
        RandomAgent[GuildworksObservation](),
        RandomAgent[GuildworksObservation](),
    )
    report = evaluate_batch(
        game,
        agents,
        outcome_for=guildworks_outcome,
        agent_labels=("random", "random"),
        num_matches=100,
        seed=918,
        max_steps=MAX_STEPS,
    )
    assert report.summary.completed == 100
    assert report.summary.errors == report.summary.truncated == 0
    assert report.summary.min_game_length == report.summary.max_game_length == 32
    assert any(r.outcome and r.outcome.winners for r in report.records)
    assert report == evaluate_batch(
        game,
        agents,
        outcome_for=guildworks_outcome,
        agent_labels=("random", "random"),
        num_matches=100,
        seed=918,
        max_steps=MAX_STEPS,
    )
    for record in report.records:
        assert record.result and record.outcome
        assert record.result.total_rewards == record.outcome.scores


def test_global_rng_preservation() -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    game = fresh()
    while not game.is_terminal():
        seat = game.current_player()
        assert seat is not None
        game.observe(1 - seat)
        game.clone()
        actions = game.legal_actions(seat)
        game.step(place("lottery") if place("lottery") in actions else actions[0])
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_outcome_requires_terminal() -> None:
    with pytest.raises(ValueError):
        guildworks_outcome(fresh())


def test_insufficient_ink_hides_only_own_choices() -> None:
    game = fresh()
    game.step(place("festival"))
    game._ink[0] = 0
    assert game.legal_actions(0) == (PASS,)
    with pytest.raises(IllegalActionError):
        game.step(craft(game.observe(0).data.own_hand[0]))


@pytest.mark.parametrize("kind", ["interrupt", "extra_turn", "skip"])
def test_unsupported_mechanics_reject_atomically(kind: str) -> None:
    assert_rejection_preserves_game(fresh(), Action(kind=kind), max_steps=33)


def test_empty_hands_never_skip_crafting_and_round_boundary_can_repeat_actor() -> None:
    game = fresh()
    game._hands = [[], []]
    game._deck.clear()
    actors = []
    for step in range(32):
        actor = game.current_player()
        assert actor is not None
        actors.append(actor)
        view, legal = game.decision_for(actor)
        if step % 2:
            assert view.phase == "craft" and legal == (PASS,)
        else:
            assert view.phase == "place"
        game.step(legal[0])
    expected = [
        ((r - 1) % 2 + turn) % 2
        for r in range(1, 5)
        for turn in range(4)
        for _ in range(2)
    ]
    assert actors == expected
    assert actors[6:10] == [1, 1, 1, 1]  # final turn of round 1, first of round 2
    assert game.is_terminal()
