"""Fixed column action indices derived from supplied player choices."""

import torch

from boardgame_ai.core import Action


def encode_connect_four_action(action: Action) -> int:
    """Map a drop/column action to index 0 through 6, without certifying legality."""
    if (
        not isinstance(action, Action)
        or action.kind != "drop"
        or len(action.parameters) != 1
        or action.parameters[0][0] != "column"
    ):
        raise ValueError("Expected a Connect Four drop/column action")
    column = action.parameters[0][1]
    if type(column) is not int or not 0 <= column < 7:
        raise ValueError("Column must be an integer from 0 to 6")
    return column


def _indexed_actions(legal_actions: tuple[Action, ...]) -> dict[int, Action]:
    if not isinstance(legal_actions, tuple):
        raise ValueError("Legal actions must be a tuple")
    indexed: dict[int, Action] = {}
    for action in legal_actions:
        index = encode_connect_four_action(action)
        if index in indexed:
            raise ValueError("Legal actions must have unique column indices")
        indexed[index] = action
    return indexed


def encode_connect_four_legal_actions(
    legal_actions: tuple[Action, ...],
) -> torch.Tensor:
    """Return a fresh CPU bool mask (7,), True exactly at supplied columns.

    Empty choices produce an all-false mask, including for terminal/inactive
    views. The caller must supply the correct player's filtered choices.
    """
    indexed = _indexed_actions(legal_actions)
    return torch.tensor(
        [column in indexed for column in range(7)], dtype=torch.bool, device="cpu"
    )


def decode_connect_four_action(index: int, legal_actions: tuple[Action, ...]) -> Action:
    """Return the matching supplied action; reject unavailable/invalid indices."""
    if type(index) is not int or not 0 <= index < 7:
        raise ValueError("Action index must be an integer from 0 to 6")
    indexed = _indexed_actions(legal_actions)
    if index not in indexed:
        raise ValueError("Action index is not among supplied legal choices")
    return indexed[index]
