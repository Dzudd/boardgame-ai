"""Bounded frozen-policy diagnosis; tactical labels never become learning targets."""

import argparse
import json
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.state import Board
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.training.checkpoint import load_ppo_checkpoint
from boardgame_ai.training.ppo import PPOBatch, PPOTrainer, ppo_loss, prepare_ppo_batch
from boardgame_ai.training.rollout import (
    PlayerTrajectory,
    TicTacToeRollout,
    collect_tic_tac_toe_rollout,
)

_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _won(board: tuple[int | None, ...], seat: int) -> bool:
    return any(all(board[i] == seat for i in line) for line in _LINES)


def tactical_moves(
    observation: Observation[TicTacToeObservation], legal: tuple[Action, ...]
) -> tuple[str, tuple[int, ...]]:
    """One-ply public tactics, win precedence, all valid moves, no solver."""
    encode_tic_tac_toe_observation(observation)  # Validate the public schema.
    mask = encode_tic_tac_toe_legal_actions(legal)
    if observation.current_player != observation.player_id or not mask.any():
        raise ValueError("Tactics require a live actor and supplied choices")
    board = observation.data.board
    cells = tuple(i for i in range(9) if mask[i])
    if any(board[i] is not None for i in cells):
        raise ValueError("Supplied placements must be empty")
    seat = observation.player_id

    def wins(mark: int) -> tuple[int, ...]:
        return tuple(
            i
            for i in cells
            if _won(tuple(mark if j == i else v for j, v in enumerate(board)), mark)
        )

    winning = wins(seat)
    if winning:
        return "win", winning
    threats = wins(1 - seat)
    if not threats:
        return "quiet", ()
    # A placement must remove every immediate opponent winning reply.
    blocks = tuple(i for i in cells if all(i == threat for threat in threats))
    return ("block", blocks) if blocks else ("unavoidable", ())


def reachable_decisions() -> list[
    tuple[Observation[TicTacToeObservation], tuple[Action, ...]]
]:
    """Every live X-first board once; row-major DFS, stop at first terminal."""
    seen: set[tuple[int | None, ...]] = set()
    decisions: list[tuple[Observation[TicTacToeObservation], tuple[Action, ...]]] = []

    def visit(board: tuple[int | None, ...], seat: int) -> None:
        if board in seen:
            return
        seen.add(board)
        if _won(board, 0) or _won(board, 1) or None not in board:
            return
        legal = tuple(
            Action(kind="place", parameters=(("cell", i),))
            for i, mark in enumerate(board)
            if mark is None
        )
        decisions.append(
            (
                Observation(
                    player_id=seat,
                    current_player=seat,
                    phase="play",
                    data=TicTacToeObservation(board=cast(Board, board)),
                ),
                legal,
            )
        )
        for i, mark in enumerate(board):
            if mark is None:
                visit(
                    tuple(seat if j == i else v for j, v in enumerate(board)), 1 - seat
                )

    visit((None,) * 9, 0)
    return decisions


def audit_decision(
    model: TicTacToeActorCritic,
    view: Observation[TicTacToeObservation],
    legal: tuple[Action, ...],
) -> dict[str, Any]:
    category, valid = tactical_moves(view, legal)
    mask = encode_tic_tac_toe_legal_actions(legal)
    with torch.no_grad():
        logits, value = model(encode_tic_tac_toe_observation(view))
        probabilities = mask_action_logits(logits, mask).softmax(-1)
    return {
        "seat": view.player_id,
        "board": view.data.board,
        "category": category,
        "valid_moves": valid,
        "probabilities": probabilities.tolist(),
        "valid_probability": float(probabilities[list(valid)].sum()) if valid else None,
        "greedy_valid": int(probabilities.argmax()) in valid if valid else None,
        "uniform_valid_probability": len(valid) / int(mask.sum()) if valid else None,
        "value": float(value),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for seat in (0, 1):
        for category in ("win", "block", "unavoidable", "quiet"):
            group = [r for r in rows if r["seat"] == seat and r["category"] == category]
            n = len(group)
            measurable = category in ("win", "block") and n > 0
            result.append(
                {
                    "seat": seat,
                    "category": category,
                    "samples": n,
                    "distinct_boards": len({tuple(r["board"]) for r in group}),
                    "mean_valid_probability": sum(r["valid_probability"] for r in group)
                    / n
                    if measurable
                    else None,
                    "greedy_success_rate": sum(r["greedy_valid"] for r in group) / n
                    if measurable
                    else None,
                    "sampled_successes": sum(
                        r["action"] in r["valid_moves"] for r in group
                    )
                    if measurable and "action" in group[0]
                    else None,
                    "uniform_valid_probability": sum(
                        r["uniform_valid_probability"] for r in group
                    )
                    / n
                    if measurable
                    else None,
                }
            )
    return result


def trace_rollout(
    rollout: TicTacToeRollout,
    trainer: PPOTrainer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay public actions; independently check interval rewards and scalar GAE."""
    batch = prepare_ppo_batch(rollout.players, trainer.config)
    rows: list[dict[str, Any]] = []
    moves = sorted(
        (int(step), seat, local, int(segment.actions[local]))
        for seat, segment in enumerate(rollout.players)
        for local, step in enumerate(segment.step_indices)
    )
    game = TicTacToeEngine()
    game.reset()
    increments: list[tuple[float, ...]] = []
    for step, seat, local, cell in moves:
        if game.current_player() != seat or step != len(increments):
            raise AssertionError("Trajectory actor/step mismatch")
        view, legal = game.decision_for(seat)
        segment = rollout.players[seat]
        torch.testing.assert_close(
            encode_tic_tac_toe_observation(view),
            segment.observations[local],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            encode_tic_tac_toe_legal_actions(legal), segment.legal_masks[local]
        )
        row = audit_decision(trainer.model, view, legal)
        row.update(
            step=step, action=cell, selected_probability=row["probabilities"][cell]
        )
        rows.append(row)
        increments.append(game.step(decode_tic_tac_toe_action(cell, legal)).rewards)
    if not game.is_terminal() or not rollout.terminated or rollout.truncated:
        raise AssertionError("Expected complete self-play")
    offset = 0
    players = []
    for seat, segment in enumerate(rollout.players):
        rewards, values = segment.rewards.tolist(), segment.values.tolist()
        steps = segment.step_indices.tolist()
        expected_rewards = [
            sum(r[seat] for r in increments[start:end])
            for start, end in zip(steps, steps[1:] + [len(increments)], strict=True)
        ]
        torch.testing.assert_close(segment.rewards, torch.tensor(expected_rewards))
        if segment.bootstrap_value != 0 or sum(rewards) != rollout.total_rewards[seat]:
            raise AssertionError("Terminal reward/bootstrap mismatch")
        advantages = [0.0] * len(rewards)
        deltas = [0.0] * len(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            deltas[t] = rewards[t] + trainer.config.gamma * next_value - values[t]
            running = (
                deltas[t] + trainer.config.gamma * trainer.config.gae_lambda * running
            )
            advantages[t] = running
        targets = [a + v for a, v in zip(advantages, values, strict=True)]
        section = slice(offset, offset + len(rewards))
        torch.testing.assert_close(
            batch.advantages[section], torch.tensor(advantages), atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            batch.returns[section], torch.tensor(targets), atol=2e-6, rtol=2e-5
        )
        # Stored behavior probabilities and values must match the unchanged actor.
        with torch.no_grad():
            logits, predicted = trainer.model(segment.observations)
            logs = mask_action_logits(logits, segment.legal_masks).log_softmax(-1)
        torch.testing.assert_close(predicted, segment.values, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            logs.gather(1, segment.actions[:, None]).squeeze(1),
            segment.log_probs,
            atol=2e-6,
            rtol=2e-5,
        )
        players.append(
            {
                "seat": seat,
                "steps": steps,
                "actions": segment.actions.tolist(),
                "old_log_probs": segment.log_probs.tolist(),
                "values": values,
                "interval_rewards": rewards,
                "bootstrap": segment.bootstrap_value,
                "deltas": deltas,
                "advantages": advantages,
                "value_targets": targets,
            }
        )
        offset += len(rewards)
    return {
        "moves": rows,
        "step_rewards": increments,
        "total_rewards": rollout.total_rewards,
        "players": players,
        "checks_passed": True,
    }, rows


def _normalized(batch: PPOBatch, trainer: PPOTrainer) -> PPOBatch:
    if not trainer.config.normalize_advantages:
        return batch
    adv = batch.advantages
    return replace(
        batch, advantages=(adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    )


def diagnose_update(trainer: PPOTrainer, batch: PPOBatch) -> dict[str, Any]:
    """Compare production PPO against independent loss arithmetic on two copies.

    Copies retain saved Adam moments and shuffle state. Neither is persisted.
    Only unselected latest-policy self-play samples may be supplied here.
    """
    actual, reference = deepcopy(trainer), deepcopy(trainer)
    normalized = _normalized(batch, trainer)
    config = trainer.config
    minibatches = []
    for epoch in range(config.epochs):
        order = torch.randperm(len(batch.actions), generator=reference.generator)
        for start in range(0, len(order), config.minibatch_size):
            indices = order[start : start + config.minibatch_size]
            part = normalized.subset(indices)
            logits, values = reference.model(part.observations)
            logs = mask_action_logits(logits, part.legal_masks).log_softmax(-1)
            selected = logs.gather(1, part.actions[:, None]).squeeze(1)
            log_ratio = selected - part.old_log_probs
            ratios = log_ratio.exp()
            unclipped = ratios * part.advantages
            clipped = (
                ratios.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                * part.advantages
            )
            policy_terms = -torch.minimum(unclipped, clipped)
            value_terms = 0.5 * (values - part.returns).square()
            entropy_terms = -torch.where(
                part.legal_masks, logs.exp() * logs.masked_fill(~part.legal_masks, 0), 0
            ).sum(-1)
            terms = (policy_terms.mean(), value_terms.mean(), entropy_terms.mean())
            total = (
                terms[0]
                + config.value_coefficient * terms[1]
                - config.entropy_coefficient * terms[2]
            )
            production = ppo_loss(logits, values, part, config)
            for expected, computed in zip(
                (*terms, total),
                (
                    production.policy,
                    production.value,
                    production.entropy,
                    production.total,
                ),
                strict=True,
            ):
                torch.testing.assert_close(expected, computed)
            torch.testing.assert_close(
                production.approximate_kl, ((ratios - 1) - log_ratio).mean()
            )
            torch.testing.assert_close(
                production.clip_fraction,
                ((ratios - 1).abs() > config.clip_ratio).float().mean(),
            )
            reference.optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                reference.model.parameters(),
                config.max_grad_norm,
                error_if_nonfinite=True,
            )
            reference.optimizer.step()
            minibatches.append(
                {
                    "epoch": epoch,
                    "indices": indices.tolist(),
                    "policy_loss": float(terms[0].detach()),
                    "value_loss": float(terms[1].detach()),
                    "entropy": float(terms[2].detach()),
                    "total_loss": float(total.detach()),
                    "gradient_norm_before_clip": float(norm),
                    "ratios": ratios.detach().tolist(),
                    "old_log_probs": part.old_log_probs.tolist(),
                    "new_log_probs": selected.detach().tolist(),
                    "predicted_values": values.detach().tolist(),
                    "normalized_advantages": part.advantages.tolist(),
                    "policy_terms": policy_terms.detach().tolist(),
                    "value_terms": value_terms.detach().tolist(),
                }
            )
    update = actual.update(batch)
    if (
        actual.update_count != trainer.update_count + 1
        or actual.optimizer_steps != trainer.optimizer_steps + len(minibatches)
        or update.num_samples != len(batch.actions)
        or update.optimizer_steps != len(minibatches)
    ):
        raise AssertionError("Update counters disagree")
    for field in ("policy_loss", "value_loss", "entropy", "total_loss"):
        expected_mean = sum(m[field] * len(m["indices"]) for m in minibatches) / (
            len(batch.actions) * config.epochs
        )
        if abs(expected_mean - getattr(update, field)) > 1e-6:
            raise AssertionError("Sample-weighted loss reporting mismatch")
    for name, tensor in reference.model.state_dict().items():
        torch.testing.assert_close(
            tensor, actual.model.state_dict()[name], rtol=0, atol=0
        )
    for key, state in reference.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(
                tensor,
                actual.optimizer.state_dict()["state"][key][name],
                rtol=0,
                atol=0,
            )
    if not torch.equal(reference.generator.get_state(), actual.generator.get_state()):
        raise AssertionError("Shuffle mismatch")
    delta = max(
        float((p - q).abs().max().detach())
        for p, q in zip(
            actual.model.parameters(), trainer.model.parameters(), strict=True
        )
    )
    with torch.no_grad():
        after_logits, after_values = actual.model(batch.observations)
        after_logs = mask_action_logits(after_logits, batch.legal_masks).log_softmax(-1)
        after_selected = after_logs.gather(1, batch.actions[:, None]).squeeze(1)
    return {
        "checks_passed": True,
        "source_updated": False,
        "config": asdict(config),
        "raw_advantages": batch.advantages.tolist(),
        "value_targets": batch.returns.tolist(),
        "update": asdict(update),
        "max_parameter_change": delta,
        "post_update_selected_probabilities": after_selected.exp().tolist(),
        "post_update_values": after_values.tolist(),
        "minibatches": minibatches,
    }


def _seed(root: int, index: int, stream: str) -> int:
    return int.from_bytes(
        sha256(f"boardgame-ai/audit/v1:{root}:{index}:{stream}".encode()).digest()[:8],
        "big",
    )


def audit_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    games: int = 1024,
    update_games: int = 64,
    seed: int = 20261016,
) -> dict[str, Any]:
    """Write one new report only; source files are hashed before and after.

    Requires finished training. Missing outcome classes are explicit, never
    synthesized using scripted/teacher games. One update uses the first games,
    without selecting wins/losses or tactical positions for optimization.
    """
    if (
        type(games) is not int
        or type(update_games) is not int
        or not 1 <= update_games <= games
    ):
        raise ValueError("Require 1 <= update_games <= games")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    source, destination = Path(checkpoint).resolve(), Path(output).resolve()
    if destination.exists() or destination.parent == source.parent:
        raise ValueError("Audit output must be new and outside the source directory")
    protected = [
        source,
        source.parent / "training.json",
        source.parent / "updates.jsonl",
    ]
    snapshots = {p: p.read_bytes() for p in protected if p.exists()}
    hashes = {p.name: sha256(data).hexdigest() for p, data in snapshots.items()}
    with tempfile.TemporaryDirectory(prefix="boardgame-ai-audit-") as folder:
        snapshot = Path(folder) / "snapshot.pt"
        snapshot.write_bytes(snapshots[source])
        trainer = load_ppo_checkpoint(snapshot)
    exhaustive = [
        audit_decision(trainer.model, view, legal)
        for view, legal in reachable_decisions()
    ]
    visits: list[dict[str, Any]] = []
    representatives: dict[str, Any] = {}
    segments: list[PlayerTrajectory] = []
    sample_provenance: list[dict[str, int]] = []
    outcomes = [0, 0, 0]
    for index in range(games):
        rollout = collect_tic_tac_toe_rollout(
            TicTacToeEngine(),
            (trainer.model, trainer.model),
            max_steps=9,
            engine_seed=_seed(seed, index, "engine"),
            sampling_seeds=(_seed(seed, index, "seat:0"), _seed(seed, index, "seat:1")),
        )
        trace, rows = trace_rollout(rollout, trainer)
        for row in rows:
            row["game_index"] = index
        visits.extend(rows)
        outcome = (
            0
            if rollout.total_rewards[0] > 0
            else 1
            if rollout.total_rewards[1] > 0
            else 2
        )
        outcomes[outcome] += 1
        representatives.setdefault(
            ("X_win", "O_win", "draw")[outcome], {"game_index": index, **trace}
        )
        if index < update_games:
            segments.extend(rollout.players)
            for seat, segment in enumerate(rollout.players):
                sample_provenance.extend(
                    {
                        "game_index": index,
                        "seat": seat,
                        "local_index": local,
                        "step": int(step),
                        "action": int(segment.actions[local]),
                    }
                    for local, step in enumerate(segment.step_indices)
                )
    update = diagnose_update(
        trainer, prepare_ppo_batch(tuple(segments), trainer.config)
    )
    update["sample_provenance"] = sample_provenance
    for path, data in snapshots.items():
        if path.read_bytes() != data:
            raise AssertionError("Source changed during audit")
    report = {
        "schema_version": 1,
        "protocol": {
            "positions": (
                "All reachable live X-first boards once; row-major DFS; "
                "no symmetry reduction"
            ),
            "self_play": (
                "Sampled latest frozen shared policy; visit weighted; "
                "first game of each observed outcome traced"
            ),
            "tactics": (
                "Immediate win precedence; all winning moves accepted; "
                "blocks prevent every immediate threat; "
                "unavoidable threats excluded"
            ),
            "update": (
                "First update_games games, all decisions, saved PPO/Adam/"
                "shuffle; two disposable copies; never saved"
            ),
            "perspective": "Observer seat; gamma/GAE advance per own decision",
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
        },
        "checkpoint_hashes": hashes,
        "checkpoint_updates": trainer.update_count,
        "checkpoint_optimizer_steps": trainer.optimizer_steps,
        "seed": seed,
        "seed_schedule": (
            "boardgame-ai/audit/v1:{root}:{index}:{stream}; SHA256 first8 big-endian"
        ),
        "games": games,
        "outcomes_X_O_draw": outcomes,
        "update_games": update_games,
        "exhaustive_summary": summarize(exhaustive),
        "exhaustive_positions": exhaustive,
        "self_play_summary": summarize(visits),
        "self_play_decisions": visits,
        "representatives": representatives,
        "missing_representatives": [
            key for key in ("X_win", "O_win", "draw") if key not in representatives
        ],
        "update_diagnostic": update,
        "source_preserved": True,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--update-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    result = audit_checkpoint(
        args.checkpoint,
        args.output,
        games=args.games,
        update_games=args.update_games,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "checkpoint_hashes",
                    "outcomes_X_O_draw",
                    "exhaustive_summary",
                    "self_play_summary",
                    "missing_representatives",
                    "source_preserved",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
