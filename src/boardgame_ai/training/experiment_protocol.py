"""Immutable Phase 4.5 settings and explicitly verified legacy provenance."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

PROTOCOL_DIRECTORY = Path(__file__).with_name("protocols")
LEGACY_SHA256 = "9d55a0f8ae4c4a34fb69d7ce44508139390d5b9290fd846a79f8ef0b706f834f"
SETTINGS_SHA256 = "c34ead0a7071f9edb4a7793d7880d06fa7f8b14e0bec933172ca50b2593eb08f"


def settings() -> dict[str, Any]:
    """Versioned data, independent of defaults and editable explanatory prose."""
    data = (PROTOCOL_DIRECTORY / "phase45-v1.json").read_bytes()
    if sha256(data).hexdigest() != SETTINGS_SHA256:
        raise ValueError(
            "Immutable phase45-v1 settings changed; declare a new protocol version"
        )
    result: dict[str, Any] = json.loads(data)
    return result


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def legacy_manifest() -> dict[str, Any]:
    """Exact supported v1 manifest, bound to the archived original prose."""
    frozen = settings()
    return {
        "schema_version": 1,
        "protocol_sha256": LEGACY_SHA256,
        **{
            key: frozen[key]
            for key in (
                "seeds",
                "updates",
                "games_per_update",
                "variants",
                "monitoring",
                "final_seed",
                "final_games_per_seat",
                "device",
                "threads",
            )
        },
    }


def create_experiment_protocol(output: Path, prose: Path) -> dict[str, Any]:
    """New directory only; freeze both the machine settings and original prose."""
    frozen = settings()
    snapshot = prose.read_bytes()
    encoded = _canonical(frozen)
    manifest = {
        "schema_version": 2,
        "settings_sha256": sha256(encoded).hexdigest(),
        "protocol_sha256": sha256(snapshot).hexdigest(),
    }
    if output.exists():
        raise ValueError("Experiment output must be absent")
    output.mkdir(parents=True)
    for name, data in (
        ("settings.json", encoded),
        ("protocol.md", snapshot),
        ("experiment.json", _canonical(manifest)),
    ):
        with (output / name).open("xb") as stream:
            stream.write(data)
    return frozen


def validate_experiment_protocol(output: Path) -> dict[str, Any]:
    """Read-only preflight. Never loads weights, trains, evaluates, or migrates.

    One known legacy identity is supported after exact manifest verification.
    Unknown provenance must be restored/reviewed explicitly, never relabelled.
    """
    manifest = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
    ):
        raise ValueError("Unsupported experiment manifest schema")
    frozen = settings()
    if manifest.get("schema_version") == 1:
        snapshot = (PROTOCOL_DIRECTORY / "phase45-original.md").read_bytes()
        if sha256(snapshot).hexdigest() != LEGACY_SHA256 or _canonical(
            manifest
        ) != _canonical(legacy_manifest()):
            raise ValueError(
                "Unverified historical protocol: restore the original manifest and "
                "protocol snapshot, or review and add an explicit compatibility "
                "mapping; "
                "do not replace the recorded hash"
            )
        return {
            "settings": frozen,
            "provenance": manifest,
            "compatibility": "verified-v1",
        }
    if manifest.get("schema_version") != 2 or set(manifest) != {
        "schema_version",
        "settings_sha256",
        "protocol_sha256",
    }:
        raise ValueError("Unsupported experiment manifest schema")
    encoded = (output / "settings.json").read_bytes()
    snapshot = (output / "protocol.md").read_bytes()
    if (
        sha256(encoded).hexdigest() != manifest["settings_sha256"]
        or _canonical(json.loads(encoded)) != _canonical(frozen)
        or sha256(snapshot).hexdigest() != manifest["protocol_sha256"]
    ):
        raise ValueError("Experiment settings or original protocol snapshot mismatch")
    return {"settings": frozen, "provenance": manifest, "compatibility": "native-v2"}
