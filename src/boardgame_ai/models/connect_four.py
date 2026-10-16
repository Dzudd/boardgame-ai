"""Feed-forward policy and value heads using player-visible encoded views."""

from typing import cast

import torch
from torch import nn


class ConnectFourActorCritic(nn.Module):
    """Shared 252→64→64 tanh body, seven policy logits, and one scalar value.

    Inputs have shape (..., 6, 6, 7). Outputs have shapes (..., 7) and (...).
    Values estimate future return for the observation's player, not the actor
    or an absolute X seat. Both heads receive the same permitted information.
    Values are unbounded estimates, not probabilities or exact game outcomes.

    Construction uses CPU float32 parameters and standard PyTorch initialization.
    Callers own initialization seeds and explicit dtype/device conversion.
    Forward uses no RNG and performs no masking, sampling, or engine queries.
    Terminal observations are supported, but outputs are still raw predictions:
    never select a terminal action; true termination must use zero bootstrap.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(252, 64, dtype=torch.float32, device="cpu"),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=torch.float32, device="cpu"),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, 7, dtype=torch.float32, device="cpu")
        self.value_head = nn.Linear(64, 1, dtype=torch.float32, device="cpu")

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict raw logits and values without changing the input or model.

        Reject incompatible shapes, empty batches, nonfinite inputs, and dtype
        or device mismatches with ValueError. Inputs must match the model's
        floating-point dtype/device. This validates tensors, not game legality.
        Use eval() and torch.no_grad() or inference_mode() for inference.
        """
        if not isinstance(observations, torch.Tensor):
            raise ValueError("Expected an encoded observation tensor")
        if (
            observations.ndim < 3
            or observations.shape[-3:] != (6, 6, 7)
            or observations.numel() == 0
        ):
            raise ValueError("Observations require nonempty shape (..., 6, 6, 7)")
        reference = self.policy_head.weight
        if (
            not observations.is_floating_point()
            or observations.dtype != reference.dtype
            or observations.device != reference.device
        ):
            raise ValueError("Observations must match the model dtype and device")
        if not torch.isfinite(observations).all().item():
            raise ValueError("Observations must be finite")

        features = cast(torch.Tensor, self.body(observations.flatten(start_dim=-3)))
        logits = cast(torch.Tensor, self.policy_head(features))
        values = cast(torch.Tensor, self.value_head(features)).squeeze(-1)
        return logits, values


MODEL_ID = "connect-four-mlp64x64-actions7-v1"
ENCODING_ID = "connect-four-relative-6x6x7-v1"


def initial_connect_four_model(seed: int) -> ConnectFourActorCritic:
    """Initialize on CPU with a private unsigned 64-bit seed, preserving RNG.

    Unlike direct construction, this helper does not advance the global CPU or
    CUDA generators. Forward uses no randomness. Seeds derived by the Connect
    Four training schedule can be passed directly, without a second transform.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Initialization seed must be an unsigned 64-bit integer")
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.random.set_rng_state(generator.get_state())
        return ConnectFourActorCritic()
