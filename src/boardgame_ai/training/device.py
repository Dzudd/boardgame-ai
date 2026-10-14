"""Explicit training-device selection without silent CPU fallback."""

import torch


def resolve_training_device(requested: str | torch.device) -> torch.device:
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Training device must be cpu, cuda, or cuda:N") from error
    if device.type == "cpu" and device.index is None:
        return device
    if device.type != "cuda":
        raise ValueError("Training device must be cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable. Install CUDA-enabled PyTorch and "
            "verify your NVIDIA driver; use --device cpu for explicit CPU execution."
        )
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device index is unavailable")
    return torch.device("cuda", index)
