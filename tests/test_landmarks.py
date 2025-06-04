import torch
import pytest
import numpy as np

from flooder import (
    generate_landmarks,
    generate_figure_eight_2D_points,
)


@pytest.mark.parametrize("n_landmarks", [64, 256, 1024])
def test_generate_landmarks(n_landmarks):
    """
    Test landmark generation via FPS.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X = generate_figure_eight_2D_points(10000)
    L = generate_landmarks(X, n_landmarks)

    assert L.shape == (n_landmarks, 2), f"Wrong shape {L.shape}"
    assert L.dtype == torch.float32, f"Wrong datatype {L.dtype}"
    assert L.device == X.device, f"Wrong device {L.device}"
