"""Shared destination policy for new frozen-policy evaluation reports."""

from pathlib import Path


def validate_evaluation_destination(path: str | Path) -> Path:
    """Reject managed names and all existing destinations, including aliases.

    Exclusive creation at the write boundary also handles races and hard links:
    no existing file identity can ever be opened for truncation.
    """
    requested = Path(path)
    destination = requested.resolve()
    for candidate in (requested, destination):
        name = candidate.name.casefold()
        if (
            candidate.suffix.casefold() == ".pt"
            or name
            in {"training.json", "updates.jsonl", "monitoring.json", "best.json"}
            or name.startswith("development-")
            or any(part.casefold() == "snapshots" for part in candidate.parts)
        ):
            raise ValueError("Evaluation output must not overwrite training artifacts")
    if requested.exists() or requested.is_symlink() or destination.exists():
        raise FileExistsError("Evaluation report must be new; choose another output")
    if not destination.parent.is_dir():
        raise ValueError("Evaluation output parent directory must exist")
    return destination
