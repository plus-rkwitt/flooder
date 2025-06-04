import torch
from flooder.synthetic_data_generators import (
    generate_donut_points,
    generate_noisy_torus_points,
    generate_figure_eight_2D_points,
    generate_swiss_cheese_points,
)


def test_generate_donut_points():
    pts = generate_donut_points(1000, torch.tensor([0.0, 0.0]), radius=1.0, width=0.2)
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 2), f"Wrong shape {pts.shape}"


def test_generate_noisy_torus_points():
    pts = generate_noisy_torus_points(1000)
    assert pts.dtype == torch.float32
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 3), f"Wrong shape {pts.shape}"


def test_generate_figure_eight_2D_points():
    pts = generate_figure_eight_2D_points(1000)
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 2), f"Wrong shape {pts.shape}"


def test_generate_swiss_cheese_points():
    rect_min = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rect_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    void_radius_range = (0.1, 0.2)
    k = 6  # number of voids
    dim = 3  # swiss cheese dimension
    pts, _ = generate_swiss_cheese_points(
        1000, rect_min[:dim], rect_max[:dim], k, void_radius_range
    )
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 3), f"Wrong shape {pts.shape}"
