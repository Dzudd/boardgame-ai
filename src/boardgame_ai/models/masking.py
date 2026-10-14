"""Differentiable logit masking using only supplied legal-action masks."""

import torch


def mask_action_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Return logits with illegal entries set to -inf, without mutating inputs.

    Inputs have identical shape (..., action_count) and device. Logits must be
    finite floating-point values; the mask must be bool. Each row must have at
    least one legal entry. No broadcasting, sampling, or legality inference is
    performed. Legal entries preserve dtype/device and autograd connectivity;
    illegal entries have zero gradient. Empty batches and all-false rows reject
    with ValueError, including terminal rows: do not select terminal actions.
    """
    if logits.ndim < 1 or logits.numel() == 0 or logits.shape != legal_mask.shape:
        raise ValueError("Logits and mask need identical nonempty action shapes")
    if not logits.is_floating_point() or legal_mask.dtype != torch.bool:
        raise ValueError("Expected floating-point logits and a boolean mask")
    if logits.device != legal_mask.device:
        raise ValueError("Logits and mask must be on the same device")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Logits must be finite")
    if not legal_mask.any(dim=-1).all().item():
        raise ValueError("Every logit row requires a legal action")
    return logits.masked_fill(~legal_mask, float("-inf"))
