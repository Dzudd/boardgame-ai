"""Bounded retries for transient Windows atomic-publication sharing failures."""

from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Any

from boardgame_ai.training.connect_four_run import _atomic_bytes as _publish_bytes
from boardgame_ai.training.monitoring import atomic_json as _publish_json


def _retry(operation: Callable[[], None]) -> None:
    for attempt in range(6):
        try:
            operation()
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            sleep(0.05 * 2**attempt)


def atomic_json(path: Path, payload: Any) -> None:
    _retry(lambda: _publish_json(path, payload))


def atomic_bytes(path: Path, data: bytes) -> None:
    _retry(lambda: _publish_bytes(path, data))
