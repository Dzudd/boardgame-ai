"""Observer-relative Tic-Tac-Toe tensors derived only from public views."""

import torch

from boardgame_ai.core import Observation
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation


def encode_tic_tac_toe_observation(
    observation: Observation[TicTacToeObservation],
) -> torch.Tensor:
    """Return a fresh contiguous CPU float32 tensor of shape (6, 3, 3).

    Channels: own marks, opponent marks, empty cells, own turn, opponent turn,
    terminal. Turn/terminal channels are constant planes. Coordinates remain
    row-major for either observer. Finished views retain their board, with zero
    turn planes and a one terminal plane. There is no batch dimension or autograd
    history; callers can stack tensors or move them to their chosen device.

    Reject incompatible seats, cells, or phase/actor combinations with ValueError.
    Schema validation does not certify reachability or action legality.
    """
    if not isinstance(observation, Observation) or not isinstance(
        observation.data, TicTacToeObservation
    ):
        raise ValueError("Expected a Tic-Tac-Toe observation")
    player = observation.player_id
    actor = observation.current_player
    if type(player) is not int or player not in (0, 1):
        raise ValueError("Observer must be seat 0 or 1")
    if actor is not None and (type(actor) is not int or actor not in (0, 1)):
        raise ValueError("Actor must be seat 0, seat 1, or None")
    terminal = actor is None
    if observation.phase != ("finished" if terminal else "play"):
        raise ValueError("Phase and actor must describe play or finished")
    board = observation.data.board
    if (
        not isinstance(board, tuple)
        or len(board) != 9
        or any(
            cell is not None and (type(cell) is not int or cell not in (0, 1))
            for cell in board
        )
    ):
        raise ValueError("Board must contain nine empty or seat-valued cells")

    return torch.tensor(
        [
            [float(cell == player) for cell in board],
            [float(cell == 1 - player) for cell in board],
            [float(cell is None) for cell in board],
            [float(actor == player)] * 9,
            [float(actor == 1 - player)] * 9,
            [float(terminal)] * 9,
        ],
        dtype=torch.float32,
        device="cpu",
    ).reshape(6, 3, 3)
