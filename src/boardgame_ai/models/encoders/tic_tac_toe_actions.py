"""Fixed row-major action indices derived from supplied player choices."""

import torch

from boardgame_ai.core import Action


def encode_tic_tac_toe_action(action: Action) -> int:
    """Map a place/cell action to index 0–8, without certifying legality."""
    if (
        not isinstance(action, Action)
        or action.kind != "place"
        or len(action.parameters) != 1
        or action.parameters[0][0] != "cell"
    ):
        raise ValueError("Expected a Tic-Tac-Toe place/cell action")
    cell = action.parameters[0][1]
    if type(cell) is not int or not 0 <= cell < 9:
        raise ValueError("Cell must be an integer from 0 to 8")
    return cell


def _indexed_actions(legal_actions: tuple[Action, ...]) -> dict[int, Action]:
    if not isinstance(legal_actions, tuple):
        raise ValueError("Legal actions must be a tuple")
    indexed: dict[int, Action] = {}
    for action in legal_actions:
        index = encode_tic_tac_toe_action(action)
        if index in indexed:
            raise ValueError("Legal actions must have unique cell indices")
        indexed[index] = action
    return indexed


def encode_tic_tac_toe_legal_actions(legal_actions: tuple[Action, ...]) -> torch.Tensor:
    """Return a fresh CPU bool mask (9,), True exactly at supplied cells.

    Empty choices produce an all-false mask, including for terminal/inactive
    views. The caller must supply the correct player's filtered choices.
    """
    indexed = _indexed_actions(legal_actions)
    return torch.tensor(
        [cell in indexed for cell in range(9)], dtype=torch.bool, device="cpu"
    )


def decode_tic_tac_toe_action(index: int, legal_actions: tuple[Action, ...]) -> Action:
    """Return the matching supplied action; reject unavailable/invalid indices."""
    if type(index) is not int or not 0 <= index < 9:
        raise ValueError("Action index must be an integer from 0 to 8")
    indexed = _indexed_actions(legal_actions)
    if index not in indexed:
        raise ValueError("Action index is not among supplied legal choices")
    return indexed[index]
