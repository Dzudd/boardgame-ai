"""Exercise the numerical operations needed by future learning code."""

import numpy as np
import torch


def test_numpy_tensor_interoperability_and_autograd() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = torch.from_numpy(values).requires_grad_()

    loss = tensor.square().sum()
    # PyTorch exposes backward without a complete typed signature.
    loss.backward()  # type: ignore[no-untyped-call]

    assert loss.item() == 14.0
    assert tensor.grad is not None
    np.testing.assert_array_equal(tensor.grad.numpy(), 2 * values)
