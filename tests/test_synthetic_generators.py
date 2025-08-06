import torch
from flooder.synthetic_data_generators import (
    generate_noisy_torus_points_3d,
    generate_figure_eight_points_2d,
    generate_annulus_points_2d,
    generate_swiss_cheese_points,
)


def test_generate_annulus_points_2d():
    pts = generate_annulus_points_2d(
        1000, torch.tensor([0.0, 0.0]), radius=1.0, width=0.2
    )
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 2), f"Wrong shape {pts.shape}"


def test_generate_noisy_torus_points_3d():
    pts = generate_noisy_torus_points_3d(1000)
    assert pts.dtype == torch.float32
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 3), f"Wrong shape {pts.shape}"


def test_generate_figure_eight_points_2d():
    pts = generate_figure_eight_points_2d(1000)
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 2), f"Wrong shape {pts.shape}"


def test_generate_swiss_cheese_points():
    rect_min = (0.0, 0.0, 0.0)
    rect_max = (1.0, 1.0, 1.0)
    void_radius_range = (0.1, 0.2)
    k = 6  # number of voids
    device = "cuda:0"
    pts, _, _ = generate_swiss_cheese_points(
        1000, rect_min, rect_max, k, void_radius_range, device=device
    )
    assert pts.dtype == torch.float32, f"Wrong datatype {pts.dtype}"
    assert pts.shape == (1000, 3), f"Wrong shape {pts.shape}"
