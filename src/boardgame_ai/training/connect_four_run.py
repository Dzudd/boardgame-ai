"""Bounded Connect Four runs with immutable provenance and boundary resume.

Single writer only. A budget reservation charges the entire remaining allowance
before work; a hard process kill therefore exhausts it conservatively. Normal
completion/failure settles actual elapsed time, including failed/unsaved work.
"""

import argparse
import io
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from boardgame_ai.games.connect_four import ConnectFourEngine
from boardgame_ai.models.connect_four import (
    initial_connect_four_model,
)
from boardgame_ai.training.connect_four import (
    collect_connect_four_self_play,
    derive_connect_four_seed,
)
from boardgame_ai.training.connect_four_checkpoint import (
    connect_four_payload,
    trainer_from_connect_four_payload,
)
from boardgame_ai.training.connect_four_ppo import (
    ConnectFourPPOTrainer,
    prepare_connect_four_batch,
)
from boardgame_ai.training.metrics_log import recover_metrics_log
from boardgame_ai.training.monitoring import (
    _immutable_snapshot,
    atomic_json,
    isolated_rng,
)
from boardgame_ai.training.ppo import PPOConfig


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def read_settings(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    raw = Path(path).read_bytes()
    settings = json.loads(raw)
    if not isinstance(settings, dict):
        raise ValueError("Protocol must be an object")
    # Reject unsupported architecture/algorithm changes.
    template = json.loads(
        (
            Path(__file__).parent / "protocols/connect-four-functional-v1.json"
        ).read_bytes()
    )
    for section, mutable in (
        (None, {"protocol_id", "training", "evaluation"}),
        (
            "training",
            {"seed", "updates", "games_per_update", "wall_time_cap_seconds", "ppo"},
        ),
        (
            "evaluation",
            {
                "seed",
                "control_seed",
                "games_per_seat",
                "control_games_per_seat",
                "wall_time_cap_seconds",
                "checkpoints",
            },
        ),
    ):
        actual = settings if section is None else settings[section]
        expected = template if section is None else template[section]
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Unsupported protocol fields")
        if any(
            type(actual[k]) is not type(v) or actual[k] != v
            for k, v in expected.items()
            if k not in mutable
        ):
            raise ValueError("Unsupported protocol identity or execution settings")
    if not isinstance(settings["protocol_id"], str) or not settings["protocol_id"]:
        raise ValueError("Protocol needs an identity")
    if settings["protocol_id"] == template["protocol_id"] and settings != template:
        raise ValueError("Changed settings require a new protocol version")
    for section, names in (
        ("training", ("seed", "updates", "games_per_update")),
        (
            "evaluation",
            ("seed", "control_seed", "games_per_seat", "control_games_per_seat"),
        ),
    ):
        for name in names:
            value = settings[section][name]
            if type(value) is not int or value < 1:
                raise ValueError("Protocol counts/seeds must be positive integers")
        cap = settings[section]["wall_time_cap_seconds"]
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("Protocol caps must be positive finite seconds")
    t, e = settings["training"], settings["evaluation"]
    if e["games_per_seat"] != e["control_games_per_seat"] or e["checkpoints"] != [
        "initial",
        f"update-{t['updates']}",
    ]:
        raise ValueError(
            "Evaluation must match fixed final update and paired control counts"
        )
    if not isinstance(t["ppo"], dict) or t["ppo"].keys() != asdict(PPOConfig()).keys():
        raise ValueError("Invalid PPO settings")
    PPOConfig(**t["ppo"])
    return raw, settings


def _encode(payload: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _atomic_bytes(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".connect-four-run-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _code_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            result = subprocess.run(["git", *args], capture_output=True, check=False)
        except OSError as error:
            raise ValueError(
                "Cannot launch Git for code provenance. Make git available on PATH "
                "(check: git --version), then rerun --check-only. "
                "No self-play or optimization has started."
            ) from error
        if result.returncode != 0:
            raise ValueError(
                "Cannot read Git provenance; run from the repository root with "
                "a valid Git checkout. No self-play or optimization has started."
            )
        return result.stdout.decode("utf-8", errors="replace").strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": digest(git("diff", "HEAD").encode()),
        "source_sha256": _source_digest(),
    }


def _source_digest() -> str:
    source = Path(__file__).parent.parent
    return digest(
        b"".join(
            str(p.relative_to(source)).encode() + p.read_bytes()
            for p in sorted(source.rglob("*.py"))
        )
    )


def _budget(directory: Path, identity: str) -> dict[str, Any]:
    ledger = json.loads((directory / "budget.json").read_bytes())
    if (
        not isinstance(ledger, dict)
        or set(ledger) != {"settings_sha256", "training", "evaluation"}
        or ledger["settings_sha256"] != identity
    ):
        raise ValueError("Budget identity mismatch")
    if any(
        type(ledger[k]) not in (int, float)
        or not math.isfinite(ledger[k])
        or ledger[k] < 0
        for k in ("training", "evaluation")
    ):
        raise ValueError("Invalid cumulative budget")
    return ledger


def load_run(
    directory: Path, identity: str, settings: dict[str, Any]
) -> tuple[ConnectFourPPOTrainer, dict[str, Any]]:
    payload = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema_version", "settings_sha256", "trainer", "games", "elapsed_seconds"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["settings_sha256"] != identity
    ):
        raise ValueError("Invalid Connect Four run checkpoint identity")
    trainer = trainer_from_connect_four_payload(
        payload["trainer"], expected_config=PPOConfig(**settings["training"]["ppo"])
    )
    games = payload["games"]
    elapsed = payload["elapsed_seconds"]
    if (
        type(games) is not int
        or games != trainer.update_count * settings["training"]["games_per_update"]
        or trainer.update_count > settings["training"]["updates"]
        or not 7 * games <= trainer.samples_seen <= 42 * games
    ):
        raise ValueError("Invalid run game/decision counters")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid run elapsed budget")
    return trainer, payload


def validate_run(
    directory: Path, raw: bytes, settings: dict[str, Any]
) -> tuple[ConnectFourPPOTrainer, dict[str, Any], dict[str, Any]]:
    identity = digest(raw)
    manifest = json.loads((directory / "training.json").read_bytes())
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "settings_sha256",
            "protocol_sha256",
            "initial_sha256",
            "code",
            "runtime",
            "parameter_count",
        }
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["code"], dict)
        or manifest["code"].get("source_sha256") != _source_digest()
        or not isinstance(manifest["runtime"], dict)
        or manifest["runtime"].get("python") != platform.python_version()
        or manifest["runtime"].get("torch") != str(torch.__version__)
        or manifest["runtime"].get("device") != "cpu"
        or manifest["runtime"].get("threads") != 1
        or manifest["parameter_count"] != 20872
        or manifest["settings_sha256"] != identity
        or (directory / "settings.json").read_bytes() != raw
        or digest((directory / "protocol.md").read_bytes())
        != manifest["protocol_sha256"]
    ):
        raise ValueError("Saved protocol provenance mismatch")
    initial = (directory / "initial.pt").read_bytes()
    if digest(initial) != manifest["initial_sha256"]:
        raise ValueError("Initial snapshot mismatch")
    original = trainer_from_connect_four_payload(
        torch.load(io.BytesIO(initial), map_location="cpu", weights_only=True),
        expected_config=PPOConfig(**settings["training"]["ppo"]),
    )
    if original.update_count != 0:
        raise ValueError("Initial snapshot is not initial")
    trainer, payload = load_run(directory, identity, settings)
    ledger = _budget(directory, identity)
    if ledger["training"] < payload["elapsed_seconds"]:
        raise ValueError("Budget precedes saved boundary")
    final = directory / "final.pt"
    if final.exists():
        expected = _encode(payload["trainer"])
        if (
            trainer.update_count != settings["training"]["updates"]
            or final.read_bytes() != expected
        ):
            raise ValueError("Final snapshot mismatch")
    samples = recover_metrics_log(
        directory / "updates.jsonl", trainer.update_count, repair=False
    )
    if samples is not None and samples != trainer.samples_seen:
        raise ValueError("Metrics samples disagree with checkpoint")
    return trainer, payload, ledger


def _target_updates(settings: dict[str, Any], stop_after: int | None) -> int:
    total = settings["training"]["updates"]
    target = total if stop_after is None else stop_after
    if type(target) is not int or not 0 <= target <= total:
        raise ValueError("Stop boundary must fit the declared update budget")
    return int(target)


def _check_new_output(directory: Path) -> None:
    if (
        directory.is_symlink()
        or directory.exists()
        and (not directory.is_dir() or any(directory.iterdir()))
    ):
        raise ValueError("New output must be absent or empty, without a symlink")
    parent = directory.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError("Output parent must be a directory")


def check_connect_four_run(
    output: str | Path,
    *,
    settings_path: str | Path,
    protocol_path: str | Path,
    resume: bool = False,
    evaluate: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Read-only preflight: never collect, optimize, publish or repair artifacts.

    New runs validate paths/settings without constructing a model or trainer.
    Resume/evaluation validate saved trainer state and metrics without changing
    RNG, threads, time ledgers or logs. This is a point-in-time check, not a file
    reservation or a guarantee of write permissions, timing or learning quality.
    """
    if resume and evaluate or evaluate and stop_after is not None:
        raise ValueError("Evaluation cannot combine with resume or a stop boundary")
    raw, settings = read_settings(settings_path)
    t, e = settings["training"], settings["evaluation"]
    target = _target_updates(settings, stop_after)
    directory = Path(output)
    completed = decisions = optimizer_steps = 0
    used_training = used_evaluation = 0.0
    metrics_complete = True
    if resume or evaluate:
        with isolated_rng():
            trainer, _, ledger = validate_run(directory, raw, settings)
        completed, decisions = trainer.update_count, trainer.samples_seen
        optimizer_steps = trainer.optimizer_steps
        used_training, used_evaluation = ledger["training"], ledger["evaluation"]
        if target < completed:
            raise ValueError("Stop boundary precedes checkpoint")
        metrics_complete = (
            recover_metrics_log(directory / "updates.jsonl", completed, repair=False)
            is not None
        )
        if evaluate:
            if completed != t["updates"]:
                raise ValueError("Evaluation requires the declared final boundary")
            if not (directory / "final.pt").is_file():
                raise ValueError(
                    "Final snapshot missing; resume the saved boundary first"
                )
            report = directory / "evaluation.json"
            if report.exists() or report.is_symlink():
                raise ValueError("Evaluation output already exists")
        protocol_hash = digest((directory / "protocol.md").read_bytes())
    else:
        _check_new_output(directory)
        protocol_hash = digest(Path(protocol_path).read_bytes())
        _code_identity()  # Check required executable/repository before reporting ready.
    training_remaining = max(0.0, t["wall_time_cap_seconds"] - used_training)
    evaluation_remaining = max(0.0, e["wall_time_cap_seconds"] - used_evaluation)
    complete = completed == t["updates"]
    ready = (
        evaluation_remaining > 0
        if evaluate
        else (target == completed or training_remaining > 0)
    )
    return {
        "check_only": True,
        "ready": ready,
        "reason": "ready" if ready else "cumulative_time_cap_exhausted",
        "operation": "evaluation" if evaluate else "resume" if resume else "new",
        "output": str(directory.resolve()),
        "protocol_id": settings["protocol_id"],
        "purpose": settings["purpose"],
        "settings_sha256": digest(raw),
        "protocol_sha256": protocol_hash,
        "source_sha256": _source_digest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "device": t["device"],
        "threads": t["threads"],
        "parameter_count": 20872,
        "completed_updates": completed,
        "target_updates": target,
        "scheduled_updates": t["updates"],
        "training_complete": complete,
        "completed_games": completed * t["games_per_update"],
        "completed_decisions": decisions,
        "completed_optimizer_steps": optimizer_steps,
        "next_game_index": completed * t["games_per_update"],
        "scheduled_training_games": t["updates"] * t["games_per_update"],
        "remaining_training_games_to_target": (target - completed)
        * t["games_per_update"],
        "scheduled_evaluation_games": 4 * e["games_per_seat"]
        + 2 * e["control_games_per_seat"],
        "training_seconds_remaining": training_remaining,
        "evaluation_seconds_remaining": evaluation_remaining,
        "metrics_complete": metrics_complete,
    }


def run_connect_four(
    output: str | Path,
    *,
    settings_path: str | Path,
    protocol_path: str | Path,
    resume: bool = False,
    stop_after: int | None = None,
) -> ConnectFourPPOTrainer:
    """Run only a declared schedule. stop_after pauses at an absolute boundary.

    CPU/one-thread protocol only. No implicit evaluation. Settings bytes, rather
    than current prose or Git state, define resume identity. Runtime is recorded;
    exact continuation requires the same runtime and source implementation.
    """
    raw, settings = read_settings(settings_path)
    t = settings["training"]
    target = _target_updates(settings, stop_after)
    directory, identity = Path(output), digest(raw)
    old_threads = torch.get_num_threads()
    with isolated_rng():
        torch.set_num_threads(t["threads"])
        try:
            preflight_start = monotonic()
            if resume:
                trainer, payload, ledger = validate_run(directory, raw, settings)
                if target < trainer.update_count:
                    raise ValueError("Stop boundary precedes checkpoint")
                recover_metrics_log(directory / "updates.jsonl", trainer.update_count)
                if target == trainer.update_count:
                    if target == t["updates"]:
                        _immutable_snapshot(
                            directory / "final.pt", _encode(payload["trainer"])
                        )
                    return trainer
                if ledger["training"] >= t["wall_time_cap_seconds"]:
                    return trainer
            else:
                _check_new_output(directory)
                # Read both provenance inputs before creating any output.
                prose = Path(protocol_path).read_bytes()
                code_identity = _code_identity()
                directory.mkdir(parents=True, exist_ok=True)
                ledger = {
                    "settings_sha256": identity,
                    "training": 0.0,
                    "evaluation": 0.0,
                }
                atomic_json(
                    directory / "budget.json",
                    {**ledger, "training": t["wall_time_cap_seconds"]},
                )
                trainer = ConnectFourPPOTrainer(
                    initial_connect_four_model(
                        derive_connect_four_seed(t["seed"], 0, "initialization")
                    ),
                    PPOConfig(**t["ppo"]),
                    shuffle_seed=derive_connect_four_seed(t["seed"], 0, "shuffle"),
                )
                initial = _encode(connect_four_payload(trainer))
                for name, data in (
                    ("settings.json", raw),
                    ("protocol.md", prose),
                    ("initial.pt", initial),
                ):
                    _immutable_snapshot(directory / name, data)
                atomic_json(
                    directory / "training.json",
                    {
                        "schema_version": 1,
                        "settings_sha256": identity,
                        "protocol_sha256": digest(prose),
                        "initial_sha256": digest(initial),
                        "code": code_identity,
                        "runtime": {
                            "python": platform.python_version(),
                            "torch": str(torch.__version__),
                            "platform": platform.platform(),
                            "processor": platform.processor(),
                            "device": "cpu",
                            "threads": t["threads"],
                        },
                        "parameter_count": sum(
                            p.numel() for p in trainer.model.parameters()
                        ),
                    },
                )
                _save_boundary(directory, trainer, identity, 0, 0.0)
                ledger["training"] = monotonic() - preflight_start
                atomic_json(directory / "budget.json", ledger)
                preflight_start = monotonic()
            start, used = preflight_start, ledger["training"]
            deadline = start + t["wall_time_cap_seconds"] - used
            # An unsettled reservation consumes the cap after abrupt termination.
            ledger["training"] = max(used, t["wall_time_cap_seconds"])
            atomic_json(directory / "budget.json", ledger)
            stop_reason = "boundary"
            discarded_games = discarded_decisions = 0
            try:
                for update in range(trainer.update_count, target):
                    if monotonic() >= deadline:
                        stop_reason = "time_cap"
                        break
                    collection_start = monotonic()
                    rollouts = []
                    for local in range(t["games_per_update"]):
                        index = update * t["games_per_update"] + local
                        rollouts.append(
                            collect_connect_four_self_play(
                                ConnectFourEngine(),
                                trainer.model,
                                engine_seed=derive_connect_four_seed(
                                    t["seed"], index, "engine"
                                ),
                                sampling_seeds=(
                                    derive_connect_four_seed(
                                        t["seed"], index, "seat:0"
                                    ),
                                    derive_connect_four_seed(
                                        t["seed"], index, "seat:1"
                                    ),
                                ),
                                deadline=deadline,
                            )
                        )
                        discarded_games = len(rollouts)
                        discarded_decisions = sum(r.num_steps for r in rollouts)
                    collection_seconds = monotonic() - collection_start
                    batch = prepare_connect_four_batch(
                        tuple(p for r in rollouts for p in r.players), trainer.config
                    )
                    optimization_start = monotonic()
                    ppo = trainer.update(batch, deadline=deadline)
                    optimization_seconds = monotonic() - optimization_start
                    if monotonic() >= deadline:
                        raise TimeoutError("Deadline before checkpoint publication")
                    save_start = monotonic()
                    games = trainer.update_count * t["games_per_update"]
                    _save_boundary(
                        directory, trainer, identity, games, used + monotonic() - start
                    )
                    discarded_games = discarded_decisions = 0
                    checkpoint_seconds = monotonic() - save_start
                    row = {
                        "update": trainer.update_count,
                        "games": games,
                        "decisions": trainer.samples_seen,
                        "optimizer_steps": trainer.optimizer_steps,
                        "sample_presentations": trainer.samples_seen
                        * trainer.config.epochs,
                        "replay_reuse": 0,
                        "ppo": asdict(ppo),
                        "outcomes": [list(r.total_rewards) for r in rollouts],
                        "unique_acting_positions": len(
                            {
                                p.observations[i].numpy().tobytes()
                                for r in rollouts
                                for p in r.players
                                for i in range(p.actions.numel())
                            }
                        ),
                        "action_frequencies": torch.bincount(
                            batch.actions, minlength=7
                        ).tolist(),
                        "collection_seconds": collection_seconds,
                        "optimization_seconds": optimization_seconds,
                        "checkpoint_seconds": checkpoint_seconds,
                        "elapsed_seconds": used + monotonic() - start,
                        "cap_overrun_seconds": max(0.0, monotonic() - deadline),
                    }
                    with (directory / "updates.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if trainer.update_count == t["updates"]:
                        _immutable_snapshot(
                            directory / "final.pt",
                            _encode(connect_four_payload(trainer)),
                        )
            except TimeoutError:
                stop_reason = "time_cap"
                # Return the saved boundary after a timeout during partial work.
                trainer, _ = load_run(directory, identity, settings)
            except KeyboardInterrupt:
                stop_reason = "KeyboardInterrupt"
                raise
            except Exception as error:
                stop_reason = type(error).__name__
                raise
            finally:
                ledger["training"] = used + monotonic() - start
                saved, _ = load_run(directory, identity, settings)
                atomic_json(
                    directory / "training-status.json",
                    {
                        "complete": saved.update_count == t["updates"],
                        "stop_reason": stop_reason,
                        "updates": saved.update_count,
                        "games": saved.update_count * t["games_per_update"],
                        "decisions": saved.samples_seen,
                        "optimizer_steps": saved.optimizer_steps,
                        "elapsed_seconds": ledger["training"],
                        "cap_overrun_seconds": max(
                            0.0, ledger["training"] - t["wall_time_cap_seconds"]
                        ),
                        "discarded_complete_games": discarded_games,
                        "discarded_complete_game_decisions": discarded_decisions,
                        "partial_game_decisions": "unknown on collector failure",
                    },
                )
                ledger["training"] = used + monotonic() - start
                atomic_json(directory / "budget.json", ledger)
            return trainer
        finally:
            torch.set_num_threads(old_threads)


def _save_boundary(
    directory: Path,
    trainer: ConnectFourPPOTrainer,
    identity: str,
    games: int,
    elapsed: float,
) -> None:
    _atomic_bytes(
        directory / "latest.pt",
        _encode(
            {
                "schema_version": 1,
                "settings_sha256": identity,
                "trainer": connect_four_payload(trainer),
                "games": games,
                "elapsed_seconds": elapsed,
            }
        ),
    )


def main() -> None:
    from pickle import UnpicklingError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    operation.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate saved initial/final snapshots only",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read-only preflight; no games, optimization, output files or log repair",
    )
    args = parser.parse_args()
    if args.evaluate and args.stop_after is not None:
        parser.error("--stop-after cannot combine with --evaluate")
    try:
        if args.check_only:
            report = check_connect_four_run(
                args.output,
                settings_path=args.settings,
                protocol_path=args.protocol,
                resume=args.resume,
                evaluate=args.evaluate,
                stop_after=args.stop_after,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            if not report["ready"]:
                parser.exit(1, "Cumulative cap exhausted; no work was started.\n")
        elif args.evaluate:
            from boardgame_ai.training.connect_four_evaluation import (
                evaluate_connect_four_run,
            )

            report = evaluate_connect_four_run(args.output, settings_path=args.settings)
            print(json.dumps(report, allow_nan=False))
            if not report["complete"]:
                parser.exit(1, "Evaluation incomplete; inspect evaluation.json.\n")
        else:
            trainer = run_connect_four(
                args.output,
                settings_path=args.settings,
                protocol_path=args.protocol,
                resume=args.resume,
                stop_after=args.stop_after,
            )
            raw, settings = read_settings(args.settings)
            ledger = _budget(args.output, digest(raw))
            target = _target_updates(settings, args.stop_after)
            reached = trainer.update_count == target
            print(
                json.dumps(
                    {
                        "updates": trainer.update_count,
                        "decisions": trainer.samples_seen,
                        "target_updates": target,
                        "scheduled_updates": settings["training"]["updates"],
                        "complete": trainer.update_count
                        == settings["training"]["updates"],
                        "target_reached": reached,
                        "elapsed_seconds": ledger["training"],
                    },
                    allow_nan=False,
                )
            )
            if not reached:
                parser.exit(
                    1,
                    "Training incomplete; inspect training-status.json "
                    "and budget.json.\n",
                )
    except (OSError, ValueError, RuntimeError, UnpicklingError, EOFError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        parser.exit(
            130, "Interrupted; retain output and inspect the last saved boundary.\n"
        )


if __name__ == "__main__":
    main()
