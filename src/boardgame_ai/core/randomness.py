"""Local engine or agent randomness, independent of global generators."""

from random import Random


def create_rng(seed: int | None = None) -> Random:
    """Create a fresh generator; None uses system entropy, integers reproduce play.

    Engines and agents keep separate private generators; engines deepcopy theirs
    when cloning. Exact sequences require the same code and runtime versions.
    """
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Seed must be an integer or None")
    return Random(seed)
