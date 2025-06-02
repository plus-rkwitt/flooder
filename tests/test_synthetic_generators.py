import torch
from flooder.synthetic_data_generators import (
    generate_donut_points,
    generate_noisy_torus_points,
    generate_figure_eight_2D_points,
)


def test_generate_donut_points():
    pts = generate_donut_points(1000, torch.tensor([0.0, 0.0]), radius=1.0, width=0.2)
    assert pts.dtype == torch.float32
    assert pts.shape == (1000, 2)


def test_generate_noisy_torus_points():
    pts = generate_noisy_torus_points(1000)
    assert pts.dtype == torch.float32
    assert pts.shape == (1000, 3)


def test_generate_figure_eight_2D_points():
    pts = generate_figure_eight_2D_points(1000)
    assert pts.dtype == torch.float32
    assert pts.shape == (1000, 2)
