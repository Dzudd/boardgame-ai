"""Operator-run Phase 6 jobs: bounded training, fixed curves, atomic resume.

Single writer. Every invocation reserves the remaining cumulative allowance;
hard termination spends it conservatively. No mid-update or evaluation resume.
"""

import argparse
import io
import json
import math
import platform
import zipfile
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from boardgame_ai.models.multigame import (
    DEVELOPMENT_GAMES,
    MODEL_ID,
    REWARD_SCALES,
    TRANSFER_GAME,
    initial_multigame_model,
)
from boardgame_ai.training.connect_four_checkpoint import _same_types_and_values
from boardgame_ai.training.connect_four_run import (
    _check_new_output,
    _code_identity,
    _ConsoleProgress,
    _encode,
    _source_digest,
    device_details,
    digest,
    synchronize_device,
)
from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.metrics_log import recover_metrics_log
from boardgame_ai.training.monitoring import (
    _immutable_snapshot,
    isolated_rng,
)
from boardgame_ai.training.multigame import (
    MultiGameTrainer,
    collect_game,
    deadline_check,
    derive_seed,
)
from boardgame_ai.training.multigame_io import atomic_bytes as _atomic_bytes
from boardgame_ai.training.multigame_io import atomic_json


def read_protocol(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    settings = json.loads(raw)
    frozen = json.loads(
        (Path(__file__).parent / "protocols/multigame-v1.json").read_bytes()
    )
    if not _same_types_and_values(settings, frozen):
        raise ValueError("Settings differ from frozen Phase 6 v1 protocol")
    if (
        settings["model_id"] != MODEL_ID
        or settings["development_games"] != list(DEVELOPMENT_GAMES)
        or settings["transfer_game"] != TRANSFER_GAME
        or settings["reward_divisors"] != REWARD_SCALES
    ):
        raise ValueError("Protocol implementation mismatch")
    return raw, settings


def jobs(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for regime, budgets in settings["regimes"].items():
        for seed, evaluation_root in zip(
            settings["training_seeds"], settings["evaluation_roots"], strict=True
        ):
            shared = f"{regime}-{seed}-shared"
            for arm in (
                "shared",
                *(f"separate-{g}" for g in DEVELOPMENT_GAMES),
                "adapt-pretrained",
                "adapt-fresh",
            ):
                adaptation = arm.startswith("adapt-")
                game_ids = (
                    [TRANSFER_GAME]
                    if adaptation
                    else list(DEVELOPMENT_GAMES)
                    if arm == "shared"
                    else [arm.removeprefix("separate-")]
                )
                cap = (
                    budgets["adapt_seconds"]
                    if adaptation
                    else budgets[
                        "pretrain_shared_seconds"
                        if arm == "shared"
                        else "pretrain_separate_seconds"
                    ]
                )
                result[f"{regime}-{seed}-{arm}"] = {
                    "id": f"{regime}-{seed}-{arm}",
                    "regime": regime,
                    "seed": seed,
                    "evaluation_root": evaluation_root,
                    "arm": arm,
                    "games": game_ids,
                    "cap_seconds": cap,
                    "decision_cap": budgets["adapt_decisions"]
                    if adaptation
                    else budgets["pretrain_decisions_per_game"],
                    "source_job": shared if arm == "adapt-pretrained" else None,
                }
    return result


def trainer_payload(trainer: MultiGameTrainer) -> dict[str, Any]:
    if not trainer.valid_boundary:
        raise ValueError("Cannot save an incomplete optimizer update")
    optimizer = deepcopy(trainer.optimizer.state_dict())
    for state in optimizer["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.detach().cpu().clone()
    return {
        "model_id": MODEL_ID,
        "model": {
            k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()
        },
        "optimizer": optimizer,
        "updates": trainer.updates,
        "optimizer_steps": trainer.optimizer_steps,
    }


def restore_trainer(payload: Any, device: torch.device) -> MultiGameTrainer:
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "model_id",
            "model",
            "optimizer",
            "updates",
            "optimizer_steps",
        }
        or payload["model_id"] != MODEL_ID
        or type(payload["updates"]) is not int
        or payload["updates"] < 0
        or type(payload["optimizer_steps"]) is not int
        or payload["optimizer_steps"] != 2 * payload["updates"]
    ):
        raise ValueError("Invalid multi-game trainer identity/counters")
    model = initial_multigame_model(0)
    expected = model.state_dict()
    state = payload["model"]
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("Invalid model state")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[key].shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Invalid model tensor")
    model.load_state_dict(state)
    trainer = MultiGameTrainer(model.to(device))
    optimizer = payload["optimizer"]
    template = trainer.optimizer.state_dict()
    if (
        not isinstance(optimizer, dict)
        or optimizer.keys() != template.keys()
        or optimizer["param_groups"] != template["param_groups"]
        or not isinstance(optimizer["state"], dict)
    ):
        raise ValueError("Invalid Adam configuration")
    parameters = dict(enumerate(model.parameters()))
    recurrent_ids = {
        i
        for i, (name, _) in enumerate(model.named_parameters())
        if name.startswith("recurrent.")
    }
    if payload["updates"] == 0 and optimizer["state"]:
        raise ValueError("Initial checkpoint has Adam history")
    if payload["updates"] > 0 and not optimizer["state"]:
        raise ValueError("Missing Adam history")
    if payload["updates"] > 0 and not recurrent_ids.issubset(optimizer["state"]):
        raise ValueError("Missing recurrent Adam history")
    for index, moments in optimizer["state"].items():
        if index not in parameters or set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid Adam state")
        for name, tensor in moments.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all().item()
                or tensor.shape != (() if name == "step" else parameters[index].shape)
            ):
                raise ValueError("Invalid Adam tensor")
        step = float(moments["step"])
        if not 0 < step <= payload["optimizer_steps"] or step % 2:
            raise ValueError("Adam step mismatch")
        if index in recurrent_ids and step != payload["optimizer_steps"]:
            raise ValueError("Recurrent Adam step mismatch")
        if (moments["exp_avg_sq"] < 0).any().item():
            raise ValueError("Negative Adam second moment")
    trainer.optimizer.load_state_dict(optimizer)
    trainer.updates = payload["updates"]
    trainer.optimizer_steps = payload["optimizer_steps"]
    return trainer


def _load(path: Path) -> Any:
    return torch.load(
        io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True
    )


def _ledger(directory: Path, identity: str) -> dict[str, Any]:
    ledger = json.loads((directory / "budget.json").read_bytes())
    if (
        not isinstance(ledger, dict)
        or ledger.keys()
        != {
            "identity",
            "training",
            "evaluation",
        }
        or ledger["identity"] != identity
        or any(
            type(ledger[k]) not in (int, float)
            or not math.isfinite(ledger[k])
            or ledger[k] < 0
            for k in ("training", "evaluation")
        )
    ):
        raise ValueError("Invalid cumulative ledger")
    return ledger


def _runtime() -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": str(torch.__version__)}


def _compatible_source(saved: Any) -> bool:
    current = _source_digest()
    if saved == current:
        return True
    mapping = json.loads(
        (
            Path(__file__).parent / "protocols/multigame-source-compat-v1.json"
        ).read_bytes()
    )
    return bool(
        mapping.get("schema_version") == 1
        and saved in mapping.get("previous_source_sha256", [])
        and current == mapping.get("source_sha256")
    )


def recover_environment(path: Path, *, repair: bool = False) -> None:
    # The append log has no checkpoint authority and must have contiguous indices.
    # Its physical row count bounds the last index, including a possible torn tail.
    count = len(path.read_bytes().splitlines()) if path.exists() else 0
    recover_metrics_log(path, count, repair=repair)


def _source_archive() -> bytes:
    stream = io.BytesIO()
    source = Path(__file__).parent.parent
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*.py")):
            info = zipfile.ZipInfo(path.relative_to(source).as_posix())
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def validate_run(
    directory: Path,
    raw: bytes,
    settings: dict[str, Any],
    job: dict[str, Any],
    device: torch.device,
) -> tuple[MultiGameTrainer, dict[str, Any], dict[str, Any]]:
    manifest = json.loads((directory / "training.json").read_bytes())
    identity = digest(raw + job["id"].encode())
    if (
        manifest["identity"] != identity
        or manifest["job"] != job
        or not _compatible_source(manifest["source_sha256"])
        or manifest["runtime"] != _runtime()
        or (directory / "settings.json").read_bytes() != raw
        or digest((directory / "protocol.md").read_bytes())
        != manifest["protocol_sha256"]
        or digest((directory / "initial.pt").read_bytes()) != manifest["initial_sha256"]
        or digest((directory / "source.zip").read_bytes())
        != manifest["source_archive_sha256"]
    ):
        raise ValueError("Run provenance mismatch")
    payload = _load(directory / "latest.pt")
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "identity",
            "trainer",
            "per_game",
            "attempts",
            "elapsed",
            "curves",
            "complete",
            "exhausted",
        }
        or payload["identity"] != identity
        or type(payload["complete"]) is not bool
        or set(payload["per_game"]) != set(job["games"])
        or type(payload["attempts"]) is not int
        or payload["attempts"] < 0
        or not isinstance(payload["exhausted"], list)
        or len(set(payload["exhausted"])) != len(payload["exhausted"])
        or not set(payload["exhausted"]).issubset(job["games"])
    ):
        raise ValueError("Invalid run checkpoint")
    trainer = restore_trainer(payload["trainer"], device)
    original = restore_trainer(_load(directory / "initial.pt"), torch.device("cpu"))
    if original.updates != 0:
        raise ValueError("Initial snapshot is not initial")
    if manifest["parameters"] != sum(p.numel() for p in trainer.model.parameters()):
        raise ValueError("Parameter count mismatch")
    samples = 0
    for game, counts in payload["per_game"].items():
        if (
            set(counts) != {"games", "decisions", "presentations"}
            or any(type(v) is not int or v < 0 for v in counts.values())
            or counts["decisions"] < counts["games"]
            or counts["presentations"] != 2 * counts["decisions"]
            or counts["decisions"] > job["decision_cap"]
        ):
            raise ValueError(f"Invalid {game} counters")
        samples += counts["decisions"]
    if (
        type(payload["elapsed"]) not in (int, float)
        or not math.isfinite(payload["elapsed"])
        or payload["elapsed"] < 0
        or type(payload["curves"]) is not dict
    ):
        raise ValueError("Invalid boundary metadata")
    for name, info in payload["curves"].items():
        if name not in {"p0", "p25", "p50", "p75", "p100"}:
            raise ValueError("Invalid curve label")
        if (
            Path(info["file"]).name != info["file"]
            or digest((directory / info["file"]).read_bytes()) != info["sha256"]
        ):
            raise ValueError("Curve snapshot mismatch")
        if (
            type(info["elapsed"]) not in (int, float)
            or not math.isfinite(info["elapsed"])
            or info["elapsed"] < 0
            or set(info["per_game"]) != set(job["games"])
        ):
            raise ValueError("Invalid curve metadata")
    if (
        "p0" not in payload["curves"]
        or payload["curves"]["p0"]["sha256"] != manifest["initial_sha256"]
    ):
        raise ValueError("Initial curve mismatch")
    if payload["complete"] and (
        "p100" not in payload["curves"]
        or payload["curves"]["p100"]["sha256"] != digest(_encode(payload["trainer"]))
        or payload["curves"]["p100"]["per_game"] != payload["per_game"]
    ):
        raise ValueError("Final curve mismatch")
    count = recover_metrics_log(
        directory / "updates.jsonl", trainer.updates, repair=False
    )
    if count is not None and count != samples:
        raise ValueError("Metrics sample mismatch")
    ledger = _ledger(directory, identity)
    if ledger["training"] < payload["elapsed"]:
        raise ValueError("Ledger precedes checkpoint")
    recover_environment(directory / "environment.jsonl")
    recovery = directory / "finalization-recovery.json"
    if recovery.exists():
        receipt = json.loads(recovery.read_bytes())
        if (
            receipt.get("identity") != identity
            or receipt.get("trainer_sha256") != digest(_encode(payload["trainer"]))
            or Path(receipt["backup"]).name != receipt["backup"]
            or digest((directory / receipt["backup"]).read_bytes())
            != receipt["original_sha256"]
            or receipt.get("timing_uncertain") is not True
            or receipt.get("additional_training") is not False
        ):
            raise ValueError("Invalid finalization recovery receipt")
    return trainer, payload, ledger


def _curve(
    directory: Path,
    payload: dict[str, Any],
    label: str,
    elapsed: float,
) -> None:
    encoded = _encode(payload["trainer"])
    filename = f"{label}-u{payload['trainer']['updates']}.pt"
    _immutable_snapshot(directory / filename, encoded)
    payload["curves"][label] = {
        "sha256": digest(encoded),
        "file": filename,
        "elapsed": elapsed,
        "per_game": deepcopy(payload["per_game"]),
    }


def finalize_capped_job(
    root: Path,
    raw: bytes,
    settings: dict[str, Any],
    job: dict[str, Any],
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    """Explicit operator recovery: no games, optimizer, or ledger writes.

    A spent reservation is not measured time. Only nonzero checkpoints within
    min(5 seconds, 1%) of the wall cap qualify; reports flag cost uncertainty.
    """
    directory = root / job["id"]
    trainer, payload, ledger = validate_run(
        directory, raw, settings, job, torch.device("cpu")
    )
    if payload["complete"]:
        return {
            "ready": True,
            "complete": True,
            "preserved": True,
            "saved_updates": trainer.updates,
        }
    gap = job["cap_seconds"] - payload["elapsed"]
    if (
        job["regime"] != "wall"
        or ledger["training"] < job["cap_seconds"]
        or trainer.updates == 0
        or not 0 <= gap <= min(5.0, 0.01 * job["cap_seconds"])
    ):
        raise ValueError(
            "Finalization requires a spent wall cap and a nonzero "
            "saved checkpoint within 1%/5 seconds of that cap"
        )
    result = {
        "ready": True,
        "complete": False,
        "saved_updates": trainer.updates,
        "saved_elapsed_lower_bound": payload["elapsed"],
        "gap_seconds": gap,
        "budget": ledger,
        "timing_uncertain": True,
    }
    if check_only:
        return result
    original = (directory / "latest.pt").read_bytes()
    backup = f"recovery-before-{digest(original)}.pt"
    _immutable_snapshot(directory / backup, original)
    receipt = {
        "schema_version": 1,
        "identity": payload["identity"],
        "reason": "operator-finalized-near-cap-checkpoint",
        "backup": backup,
        "original_sha256": digest(original),
        "trainer_sha256": digest(_encode(payload["trainer"])),
        "saved_elapsed_lower_bound": payload["elapsed"],
        "charged_training_seconds": ledger["training"],
        "gap_seconds": gap,
        "timing_uncertain": True,
        "additional_training": False,
        "note": "Conservative budget reservation is not exact measured wall time",
    }
    _immutable_snapshot(
        directory / "finalization-recovery.json",
        (json.dumps(receipt, indent=2) + "\n").encode(),
    )
    _curve(directory, payload, "p100", ledger["training"])
    payload["complete"] = True
    _atomic_bytes(directory / "latest.pt", _encode(payload))
    status = {
        **result,
        "complete": True,
        "reason": receipt["reason"],
        "per_game": payload["per_game"],
        "additional_training": False,
    }
    atomic_json(directory / "status.json", status)
    return status


def run_job(
    root: Path,
    raw: bytes,
    settings: dict[str, Any],
    job: dict[str, Any],
    protocol: Path,
    *,
    resume: bool = False,
    check_only: bool = False,
    stop_after: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    invocation_start = monotonic()
    directory = root / job["id"]
    identity = digest(raw + job["id"].encode())
    device = resolve_training_device(settings["device"])
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("Invalid pause update")
    source: dict[str, Any] | None = None
    if resume:
        trainer, payload, ledger = validate_run(directory, raw, settings, job, device)
        manifest = json.loads((directory / "training.json").read_bytes())
        if manifest["device"] != device_details(device):
            raise ValueError("Training hardware differs from saved provenance")
        if stop_after is not None and stop_after < trainer.updates:
            raise ValueError("Pause precedes saved update")
    else:
        _check_new_output(directory)
        code = _code_identity()
        prose = protocol.read_bytes()
        if job["source_job"]:
            source_job = jobs(settings)[job["source_job"]]
            _, source_payload, _ = validate_run(
                root / source_job["id"],
                raw,
                settings,
                source_job,
                torch.device("cpu"),
            )
            if not source_payload["complete"]:
                raise ValueError("Shared pretraining must finish before adaptation")
            source = source_payload["trainer"]
        ledger = {"identity": identity, "training": 0.0, "evaluation": 0.0}
        if check_only:
            return {"ready": True, "job": job, "device": device_details(device)}
    if check_only:
        return {
            "ready": payload["complete"] or ledger["training"] < job["cap_seconds"],
            "job": job,
            "saved_updates": trainer.updates,
            "budget": ledger,
        }
    if resume and (payload["complete"] or stop_after == trainer.updates):
        return {
            "complete": payload["complete"],
            "saved_updates": trainer.updates,
            "budget": ledger,
            "paused": not payload["complete"],
        }
    previous_threads = torch.get_num_threads()
    start = invocation_start
    used = ledger["training"]
    remaining = max(0.0, job["cap_seconds"] - used)
    if remaining <= 0:
        return {"complete": False, "reason": "cap exhausted", "budget": ledger}
    deadline = start + remaining
    directory.mkdir(parents=True, exist_ok=True)
    # Publish reservation first: killed/failed work cannot reset the cap.
    ledger["training"] = job["cap_seconds"]
    atomic_json(directory / "budget.json", ledger)
    torch.set_num_threads(settings["threads"])
    progress = None if quiet else _ConsoleProgress()
    reason = "paused"
    discarded: dict[str, int] = {game: 0 for game in job["games"]}
    try:
        if not resume:
            trainer = MultiGameTrainer(
                initial_multigame_model(
                    derive_seed(job["seed"], 0, "initialization"),
                ).to(device)
            )
            if source is not None:
                trainer.model.load_state_dict(source["model"])
            # Fresh Adam in BOTH adaptation arms; no pretraining moments carry over.
            initial = _encode(trainer_payload(trainer))
            _immutable_snapshot(directory / "initial.pt", initial)
            _immutable_snapshot(directory / "settings.json", raw)
            _immutable_snapshot(directory / "protocol.md", prose)
            archive = _source_archive()
            _immutable_snapshot(directory / "source.zip", archive)
            manifest = {
                "identity": identity,
                "job": job,
                "source_sha256": _source_digest(),
                "code": code,
                "runtime": _runtime(),
                "device": device_details(device),
                "protocol_sha256": digest(prose),
                "initial_sha256": digest(initial),
                "source_checkpoint_sha256": digest(_encode(source)) if source else None,
                "source_archive_sha256": digest(archive),
                "parameters": sum(p.numel() for p in trainer.model.parameters()),
            }
            atomic_json(directory / "training.json", manifest)
            payload = {
                "identity": identity,
                "trainer": trainer_payload(trainer),
                "per_game": {
                    game: {"games": 0, "decisions": 0, "presentations": 0}
                    for game in job["games"]
                },
                "attempts": 0,
                "elapsed": 0.0,
                "curves": {},
                "complete": False,
                "exhausted": [],
            }
            _curve(directory, payload, "p0", 0.0)
            payload["elapsed"] = used + monotonic() - start
            _atomic_bytes(directory / "latest.pt", _encode(payload))
        else:
            recover_metrics_log(directory / "updates.jsonl", trainer.updates)
        recover_environment(directory / "environment.jsonl", repair=True)
        environment_path = directory / "environment.jsonl"
        environment_rows = (
            [
                json.loads(line)
                for line in environment_path.read_text(encoding="utf-8").splitlines()
            ]
            if environment_path.exists()
            else []
        )
        environment_index = environment_rows[-1]["update"] if environment_rows else 0
        if progress:
            progress(f"Initializing; target update 0/100; remaining {remaining:.1f}s")
        active = [game for game in job["games"] if game not in payload["exhausted"]]
        while active and not payload["complete"]:
            if stop_after is not None and trainer.updates >= stop_after:
                break
            deadline_check(deadline, device)
            batch = []
            collected_start = monotonic()
            for game in tuple(active):
                if job["decision_cap"] - payload["per_game"][game]["decisions"] < 1:
                    active.remove(game)
                    payload["exhausted"].append(game)
                    continue
                if progress:
                    progress(f"collecting; remaining {deadline - monotonic():.1f}s")
                rollout = collect_game(
                    trainer.model,
                    game,
                    root=job["seed"],
                    index=payload["per_game"][game]["games"],
                    deadline=deadline,
                )
                payload["attempts"] += 1
                environment_index += 1
                with environment_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "update": environment_index,
                                "game": game,
                                "game_index": payload["per_game"][game]["games"],
                                "ppo": {"num_samples": rollout.decisions},
                            }
                        )
                        + "\n"
                    )
                if (
                    rollout.decisions + payload["per_game"][game]["decisions"]
                    > job["decision_cap"]
                ):
                    discarded[game] += rollout.decisions
                    active.remove(game)
                    payload["exhausted"].append(game)
                else:
                    batch.append(rollout)
            collection_seconds = monotonic() - collected_start
            if not batch:
                payload["elapsed"] = used + monotonic() - start
                _atomic_bytes(directory / "latest.pt", _encode(payload))
                reason = "sample budget reached"
                break
            if progress:
                progress(f"optimizing; remaining {deadline - monotonic():.1f}s")
            optimization_start = monotonic()
            metrics = trainer.update(tuple(batch), deadline=deadline)
            optimization_seconds = monotonic() - optimization_start
            for rollout in batch:
                game = rollout.episodes[0].game
                counts = payload["per_game"][game]
                counts["games"] += 1
                counts["decisions"] += rollout.decisions
                counts["presentations"] += 2 * rollout.decisions
                metrics["per_game"][game]["raw_returns"] = rollout.raw_returns
            payload["trainer"] = trainer_payload(trainer)
            elapsed = used + monotonic() - start
            fraction = (
                elapsed / job["cap_seconds"]
                if job["regime"] == "wall"
                else min(
                    c["decisions"] / job["decision_cap"]
                    for c in payload["per_game"].values()
                )
            )
            for point in (25, 50, 75):
                if fraction >= point / 100 and f"p{point}" not in payload["curves"]:
                    _curve(directory, payload, f"p{point}", elapsed)
            deadline_check(deadline, device)
            payload["elapsed"] = used + monotonic() - start
            _atomic_bytes(directory / "latest.pt", _encode(payload))
            row = {
                "update": trainer.updates,
                "ppo": metrics,
                "per_game": deepcopy(payload["per_game"]),
                "collection_seconds": collection_seconds,
                "optimization_seconds": optimization_seconds,
                "elapsed": used + monotonic() - start,
            }
            with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if progress:
                progress(
                    f"Saved update {min(99, int(fraction * 100))}/100; "
                    f"remaining {max(0, deadline - monotonic()):.1f}s"
                )
        else:
            reason = "sample budget reached" if not active else "complete"
    except TimeoutError:
        reason = "time cap reached"
    except BaseException as error:
        if progress:
            progress(f"Stopped: {type(error).__name__}")
        try:
            saved = (
                _load(directory / "latest.pt")
                if (directory / "latest.pt").exists()
                else None
            )
            atomic_json(
                directory / "status.json",
                {
                    "complete": False,
                    "reason": type(error).__name__,
                    "saved_updates": saved["trainer"]["updates"] if saved else None,
                    "partial_decisions": (
                        "unknown after interrupted work; inspect environment.jsonl"
                    ),
                },
            )
        except (OSError, ValueError, RuntimeError):
            pass  # Retain the original failure and the authoritative budget ledger.
        raise
    finally:
        try:
            synchronize_device(device)
            ledger["training"] = used + monotonic() - start
            atomic_json(directory / "budget.json", ledger)
        finally:
            torch.set_num_threads(previous_threads)
    # Reserve through final validation/publication, including failed finalization.
    ledger["training"] = max(job["cap_seconds"], used + monotonic() - start)
    atomic_json(directory / "budget.json", ledger)
    try:
        # Never publish partially optimized in-memory weights after an expired update.
        trainer, payload, ledger = validate_run(directory, raw, settings, job, device)
        complete = (
            job["regime"] == "wall"
            and reason == "time cap reached"
            and monotonic() >= deadline
        ) or (job["regime"] == "samples" and reason == "sample budget reached")
        if complete:
            _curve(directory, payload, "p100", used + monotonic() - start)
            payload["complete"] = True
            _atomic_bytes(directory / "latest.pt", _encode(payload))
    finally:
        synchronize_device(device)
        ledger["training"] = used + monotonic() - start
        atomic_json(directory / "budget.json", ledger)
    status = {
        "complete": complete,
        "paused": reason == "paused",
        "reason": reason,
        "saved_updates": trainer.updates,
        "per_game": payload["per_game"],
        "budget": ledger,
        "overrun_seconds": max(0.0, ledger["training"] - job["cap_seconds"]),
        "discarded_known_decisions": discarded,
        "partial_decisions": "unknown after interrupted collection/update",
    }
    atomic_json(directory / "status.json", status)
    if progress:
        progress(f"Stopped: {reason}")
    ledger["training"] = used + monotonic() - start
    status["overrun_seconds"] = max(0.0, ledger["training"] - job["cap_seconds"])
    atomic_json(directory / "budget.json", ledger)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/phase6/MULTIGAME_PROTOCOL_V1.md")
    )
    parser.add_argument("--job")
    parser.add_argument(
        "--all", action="store_true", help="Run frozen jobs in dependency order"
    )
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--finalize-capped",
        action="store_true",
        help="Finalize a near-cap saved wall checkpoint without training",
    )
    args = parser.parse_args()
    try:
        raw, settings = read_protocol(args.settings)
        schedule = jobs(settings)
        with isolated_rng():
            if args.finalize_capped:
                if (
                    args.job not in schedule
                    or args.all
                    or args.resume
                    or args.evaluate
                    or args.report
                    or args.list_jobs
                    or args.stop_after is not None
                ):
                    raise ValueError(
                        "--finalize-capped requires only a named --job "
                        "and optional --check-only"
                    )
                print(
                    json.dumps(
                        finalize_capped_job(
                            args.root,
                            raw,
                            settings,
                            schedule[args.job],
                            check_only=args.check_only,
                        ),
                        indent=2,
                    )
                )
                return
            if args.list_jobs:
                print(json.dumps(schedule, indent=2))
                return
            if args.report:
                from boardgame_ai.training.multigame_report import report_suite

                print(json.dumps(report_suite(args.root, raw, settings), indent=2))
                return
            if args.all:
                if args.job or args.stop_after is not None:
                    raise ValueError("--all cannot combine with --job or --stop-after")
                if args.check_only and not args.resume and not args.evaluate:
                    _check_new_output(args.root)
                    device = resolve_training_device(settings["device"])
                    _code_identity()
                    args.protocol.read_bytes()
                    print(
                        json.dumps(
                            {
                                "ready": True,
                                "jobs": schedule,
                                "device": device_details(device),
                                "dependencies": "adaptation follows shared pretraining",
                            },
                            indent=2,
                        )
                    )
                    return
                results = {}
                for name, job in schedule.items():
                    if args.evaluate:
                        from boardgame_ai.training.multigame_evaluation import (
                            evaluate_job,
                        )

                        existing = args.root / name / "evaluation.json"
                        if existing.exists():
                            from boardgame_ai.training.multigame_report import (
                                report_suite,
                            )

                            state = report_suite(args.root, raw, settings)["jobs"][name]
                            if state["state"] != "evaluated":
                                raise ValueError(
                                    f"Preserve incomplete/rejected evaluation: {name}"
                                )
                            results[name] = {"complete": True, "preserved": True}
                            continue
                        result = evaluate_job(
                            args.root, raw, settings, job, check_only=args.check_only
                        )
                    else:
                        result = run_job(
                            args.root,
                            raw,
                            settings,
                            job,
                            args.protocol,
                            resume=args.resume and (args.root / name).exists(),
                            check_only=args.check_only,
                            quiet=args.quiet,
                        )
                    results[name] = result
                    if not result.get("ready", result.get("complete", False)):
                        print(json.dumps(results, indent=2))
                        parser.exit(1, f"Stopped at incomplete job {name}.\n")
                print(json.dumps(results, indent=2, allow_nan=False))
                return
            if args.job not in schedule:
                raise ValueError("Select a frozen --job from --list-jobs")
            if args.evaluate:
                if args.resume or args.stop_after is not None:
                    raise ValueError("Evaluation cannot combine with training options")
                from boardgame_ai.training.multigame_evaluation import evaluate_job

                result = evaluate_job(
                    args.root,
                    raw,
                    settings,
                    schedule[args.job],
                    check_only=args.check_only,
                )
            else:
                result = run_job(
                    args.root,
                    raw,
                    settings,
                    schedule[args.job],
                    args.protocol,
                    resume=args.resume,
                    check_only=args.check_only,
                    stop_after=args.stop_after,
                    quiet=args.quiet,
                )
            print(json.dumps(result, indent=2, allow_nan=False))
            if not result.get(
                "ready", result.get("complete", False)
            ) and not result.get("paused"):
                parser.exit(1, "Incomplete; inspect saved status and remaining caps.\n")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Phase 6 job rejected: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130, "Interrupted; last durable boundary and charged budget retained.\n"
        )


if __name__ == "__main__":
    main()
