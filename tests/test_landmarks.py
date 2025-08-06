import torch
import pytest
import numpy as np

from flooder import (
    generate_landmarks,
    generate_figure_eight_points_2d,
)


@pytest.mark.parametrize("n_lms", [64, 256, 1024])
def test_generate_landmarks(n_lms):
    """
    Test landmark generation via FPS.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X = generate_figure_eight_points_2d(10000)
    L = generate_landmarks(X, n_lms)

    assert L.shape == (n_lms, 2), f"Wrong shape {L.shape}"
    assert L.dtype == torch.float32, f"Wrong datatype {L.dtype}"
    assert L.device == X.device, f"Wrong device {L.device}"
