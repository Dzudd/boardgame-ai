"""Verify that the installed package is discoverable."""

from importlib.metadata import version

import boardgame_ai


def test_installed_package() -> None:
    assert boardgame_ai.__name__ == "boardgame_ai"
    assert version("boardgame-ai") == "0.1.0"
