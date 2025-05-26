import torch
from flooder.synthetic_data_generators import (
    generate_donut_points,
    generate_noisy_torus_points,
)


def test_donut():
    pts = generate_donut_points(100, torch.tensor([0.0, 0.0]), radius=1.0, width=0.2)
    assert pts.shape == (100, 2)


def test_noisy_torus():
    pts = generate_noisy_torus_points(1000)
    assert pts.shape == (1000, 3)
