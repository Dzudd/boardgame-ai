"""Checkpoint-authoritative, single-writer recovery of interrupted JSONL appends."""

import json
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from boardgame_ai.training.monitoring import _immutable_snapshot


class _Incomplete(ValueError):
    pass


def _unfinished_json(text: str) -> bool:
    """Recognize a strict JSON prefix, rather than arbitrary malformed last rows."""
    if not text.lstrip().startswith("{"):
        return False
    index = 0

    def peek() -> str:
        nonlocal index
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index == len(text):
            raise _Incomplete
        return text[index]

    def consume(expected: str) -> None:
        nonlocal index
        if peek() != expected:
            raise ValueError
        index += 1

    def string() -> None:
        nonlocal index
        consume('"')
        while True:
            if index == len(text):
                raise _Incomplete
            char = text[index]
            index += 1
            if char == '"':
                return
            if ord(char) < 32:
                raise ValueError
            if char == "\\":
                if index == len(text):
                    raise _Incomplete
                escape = text[index]
                index += 1
                if escape == "u":
                    for _ in range(4):
                        if index == len(text):
                            raise _Incomplete
                        if text[index] not in "0123456789abcdefABCDEF":
                            raise ValueError
                        index += 1
                elif escape not in '\\"/bfnrt':
                    raise ValueError

    def value() -> None:
        nonlocal index
        char = peek()
        if char == '"':
            string()
        elif char in "[{":
            index += 1
            end = "]" if char == "[" else "}"
            if peek() == end:
                index += 1
                return
            while True:
                if char == "{":
                    string()
                    consume(":")
                value()
                if peek() == end:
                    index += 1
                    return
                consume(",")
        elif char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            for letter in literal:
                if index == len(text):
                    raise _Incomplete
                if text[index] != letter:
                    raise ValueError
                index += 1
        else:
            match = re.match(
                r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", text[index:]
            )
            if match is None:
                if text[index:] == "-":
                    raise _Incomplete
                raise ValueError
            index += match.end()
            if re.fullmatch(r"\.|(?:\.[0-9]+)?[eE][+-]?", text[index:]):
                raise _Incomplete

    try:
        value()
    except _Incomplete:
        return True
    except (ValueError, RecursionError):
        return False
    return False


def recover_metrics_log(
    path: Path, checkpoint_updates: int, *, repair: bool = True
) -> int | None:
    """Validate history, back up torn bytes, repair append boundary, count samples.

    Only valid unfinished final JSON without LF is recoverable. All other
    corruption rejects before writes. Gaps leave counts unknown, never invented.
    repair=False performs the same validation/counting without modifying files.
    """
    original = path.read_bytes() if path.exists() else b""
    lines = original.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid = b""
    previous = 0
    for number, line in enumerate(lines):
        try:
            text = line.decode("utf-8")
            row = json.loads(text)
        except (UnicodeDecodeError, ValueError) as error:
            # Metrics emitted by json.dumps are ASCII (ensure_ascii=True).
            # Non-UTF-8 bytes therefore cannot be a normal torn metrics append.
            if (
                number != len(lines) - 1
                or line.endswith(b"\n")
                or not isinstance(error, json.JSONDecodeError)
                or not _unfinished_json(text)
            ):
                raise ValueError(
                    "Corrupt metrics history; restore from a verified backup"
                ) from error
            break
        if (
            not isinstance(row, dict)
            or type(row.get("update")) is not int
            or not previous < row["update"] <= checkpoint_updates
            or not isinstance(row.get("ppo"), dict)
            or type(row["ppo"].get("num_samples")) is not int
            or row["ppo"]["num_samples"] < 0
        ):
            raise ValueError("Invalid metrics history schema or checkpoint ordering")
        previous = row["update"]
        rows.append(row)
        valid += line if line.endswith(b"\n") else line + b"\n"
    if repair and valid != original:
        backup = path.with_name(
            f"{path.name}.recovery-{sha256(original).hexdigest()}.bak"
        )
        _immutable_snapshot(backup, original)
        fd, name = tempfile.mkstemp(prefix=".metrics-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(valid)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    return (
        sum(row["ppo"]["num_samples"] for row in rows)
        if [row["update"] for row in rows] == list(range(1, checkpoint_updates + 1))
        else None
    )
