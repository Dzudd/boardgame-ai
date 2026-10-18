"""Isolated fixed-snapshot evaluation; no trajectories or training updates."""

import json
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from boardgame_ai.core import create_rng
from boardgame_ai.games.guildworks.observation import GuildworksObservation
from boardgame_ai.models.multigame import (
    GAMES,
    DecisionInput,
    MultiGamePolicy,
    candidate_features,
    observation_tokens,
)
from boardgame_ai.training.connect_four_run import digest
from boardgame_ai.training.monitoring import (
    _immutable_snapshot,
    isolated_rng,
)
from boardgame_ai.training.multigame import (
    MAX_STEPS,
    deadline_check,
    derive_seed,
    engine_for,
)
from boardgame_ai.training.multigame_io import atomic_json
from boardgame_ai.training.multigame_run import _load, restore_trainer, validate_run


def evaluate_case(
    model: MultiGamePolicy | None,
    game_id: str,
    *,
    seat: int,
    index: int,
    root: int,
    deadline: float,
) -> dict[str, Any]:
    """Random control substitutes only the designated seat's policy stream."""
    game = engine_for(game_id, index % 2)
    case_index = seat * 1000000 + index
    engine_seed = derive_seed(root, case_index, f"evaluation/{game_id}/engine")
    policy_seed = derive_seed(root, case_index, f"evaluation/{game_id}/policy")
    opponent_seed = derive_seed(root, case_index, f"evaluation/{game_id}/opponent")
    generator = torch.Generator().manual_seed(policy_seed)
    opponent = create_rng(opponent_seed)
    control = create_rng(derive_seed(root, case_index, f"evaluation/{game_id}/control"))
    game.reset(engine_seed)
    totals = [0.0, 0.0]
    memory = None
    cursor = 0
    steps = 0
    while not game.is_terminal():
        deadline_check(deadline)
        if steps >= MAX_STEPS[game_id]:
            raise ValueError("Evaluation truncated")
        actor = game.current_player()
        assert actor is not None
        view, legal = game.decision_for(actor)
        if actor != seat:
            action = opponent.choice(legal)
        elif model is None:
            action = control.choice(legal)
        else:
            decision = DecisionInput(
                observation_tokens(game_id, view, history_start=cursor),
                candidate_features(game_id, legal, seat),
            )
            if isinstance(view.data, GuildworksObservation):
                cursor = len(view.data.history)
            with torch.no_grad():
                logits, _, memory = model(game_id, decision, memory)
                selected = int(
                    torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
                )
            action = legal[selected]
        result = game.step(action)
        for player, reward in enumerate(result.rewards):
            totals[player] += reward
        steps += 1
    score = (
        1.0
        if totals[seat] > totals[1 - seat]
        else (0.5 if totals[seat] == totals[1 - seat] else 0.0)
    )
    return {
        "seat": seat,
        "index": index,
        "starter": index % 2,
        "engine_seed": engine_seed,
        "policy_seed": policy_seed,
        "opponent_seed": opponent_seed,
        "score": score,
        "raw_return": totals[seat],
        "decisions": steps,
    }


def evaluate_job(
    root: Path,
    raw: bytes,
    settings: dict[str, Any],
    job: dict[str, Any],
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    directory = root / job["id"]
    _, payload, ledger = validate_run(
        directory, raw, settings, job, torch.device("cpu")
    )
    if not payload["complete"] or "p100" not in payload["curves"]:
        raise ValueError("Evaluation requires the fixed completed boundary")
    destination = directory / "evaluation.json"
    if destination.exists() or destination.is_symlink():
        raise ValueError("Evaluation report already exists; preserve it")
    remaining = settings["evaluation_seconds_per_job"] - ledger["evaluation"]
    if remaining <= 0:
        raise ValueError("Evaluation cumulative cap exhausted")
    if check_only:
        return {"ready": True, "job": job["id"], "remaining_seconds": remaining}
    start = monotonic()
    deadline = start + remaining
    previous_threads = torch.get_num_threads()
    used = ledger["evaluation"]
    ledger["evaluation"] = settings["evaluation_seconds_per_job"]
    atomic_json(directory / "budget.json", ledger)
    torch.set_num_threads(settings["threads"])
    report: dict[str, Any] = {
        "schema": 1,
        "job": job,
        "identity": payload["identity"],
        "curves": payload["curves"],
        "cases": {},
        "complete": False,
        "errors": [],
        "partial": False,
        "training_budget_seconds": ledger["training"],
        "evaluation_root": job["evaluation_root"],
    }
    # Retention uses original adapters, with memory reset for every match.
    game_ids = list(GAMES) if job["arm"].startswith("adapt-") else job["games"]
    try:
        with isolated_rng():
            for label in ("p0", "p25", "p50", "p75", "p100", "control"):
                if label != "control" and label not in payload["curves"]:
                    continue
                model = None
                if label != "control":
                    info = payload["curves"][label]
                    model = restore_trainer(
                        _load(directory / info["file"]), torch.device("cpu")
                    ).model.eval()
                report["cases"][label] = {}
                for game_id in game_ids:
                    cases: list[dict[str, Any]] = []
                    report["cases"][label][game_id] = cases
                    for seat in (0, 1):
                        for index in range(settings["evaluation_games_per_seat"]):
                            cases.append(
                                evaluate_case(
                                    model,
                                    game_id,
                                    seat=seat,
                                    index=index,
                                    root=job["evaluation_root"],
                                    deadline=deadline,
                                )
                            )
            report["complete"] = True
    except (TimeoutError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        report["partial"] = True
        report["errors"].append(type(error).__name__)
    finally:
        try:
            report["elapsed_seconds"] = monotonic() - start
            report["overrun_seconds"] = max(0.0, monotonic() - deadline)
            _immutable_snapshot(
                destination,
                (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(),
            )
        finally:
            try:
                ledger["evaluation"] = used + monotonic() - start
                atomic_json(directory / "budget.json", ledger)
            finally:
                torch.set_num_threads(previous_threads)
    return {
        "complete": report["complete"],
        "report": str(destination),
        "sha256": digest(destination.read_bytes()),
        "budget": ledger,
    }
