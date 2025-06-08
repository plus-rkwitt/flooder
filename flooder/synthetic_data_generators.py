"""Implementation of synthetic data generators.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Literal


def generate_figure_eight_2D_points(
    n_samples: int = 1000,
    r_bounds: Tuple[float, float] = (0.2, 0.3),
    centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.3, 0.5), (0.7, 0.5)),
    noise_std: float = 0.0,
    noise_kind: Literal["gaussian", "uniform"] = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> torch.tensor:
    """Uniformly sample a figure-eight shape in 2D and optionally add noise.

    Parameters
    ----------
    n_samples : int, optional
        Number of points to generate. Defaults to 1000.
    r_bounds : tuple(float, float)
        (inner_radius, outer_radius) shared by both parts of the figure-eight
    centers : tuple[(float, float), (float, float)]
        (x, y) coordinates of two figure-eight centers.
    noise_std : float
        Standard deviation (Gaussian) or half-width (uniform) of noise
        added independently to x and y.  Set to 0 for no noise.
    noise_kind : {"gaussian", "uniform"}
        Distribution of the noise.
    rng : int or numpy.random.Generator, optional
        Random-state for reproducibility.

    Returns
    -------
    pts : (n_samples, 2) ndarray
        Sampled coordinates.
    """
    rng = np.random.default_rng(rng)

    lobe_idx = rng.integers(0, 2, size=n_samples)
    cx, cy = np.asarray(centers).T  # shape (2,)
    cx = cx[lobe_idx]  # (n_samples,)
    cy = cy[lobe_idx]

    r_min, r_max = r_bounds
    r = np.sqrt(rng.uniform(r_min**2, r_max**2, size=n_samples))
    theta = rng.uniform(0.0, 2 * np.pi, size=n_samples)

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    if noise_std > 0:
        if noise_kind == "gaussian":
            x += rng.normal(0.0, noise_std, size=n_samples)
            y += rng.normal(0.0, noise_std, size=n_samples)
        elif noise_kind == "uniform":
            half = noise_std
            x += rng.uniform(-half, half, size=n_samples)
            y += rng.uniform(-half, half, size=n_samples)
        else:
            raise ValueError("noise_kind must be 'gaussian' or 'uniform'")

    return torch.tensor(np.stack((x, y), axis=1), dtype=torch.float32)


def generate_swiss_cheese_points(
    N: int = 1000,
    rect_min: torch.tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    rect_max: torch.tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    k: int = 6,
    void_radius_range: tuple = (0.1, 0.2),
    rng: int = None,
):
    """Generate points in a rectangular region with voids (like 3D Swiss cheese).

    Parameters
    ----------
    N : int, optional
        Number of points to generate. Defaults to 1000.
    rect_min : torch.tensor, optional
        Minimum coordinates of the rectangular region. Defaults to [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
    rect_max : torch.tensor, optional
        Maximum coordinates of the rectangular region. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0, 1.0].
    k : int, optional
        Number of voids to generate. Defaults to 6.
    void_radius_range : tuple, optional
        Range of radii for the voids (min, max). Defaults to (0.1, 0.2).
    rng : int, optional
        Random-state for reproducibility.
    Returns
    -------
    points : torch.tensor
        A tensor of shape (N, dim) containing the generated points.
    void_radii : torch.tensor
        A tensor of shape (k,) containing the radii of the voids.
    """
    if rng:
        torch.manual_seed(rng)

    void_centers = []
    void_radii = []
    for _ in range(k):
        while True:
            void_center = (rect_min + void_radius_range[1]) + (
                rect_max - rect_min - 2 * void_radius_range[1]
            ) * torch.rand(1, rect_min.shape[0])
            void_radius = void_radius_range[0] + (
                void_radius_range[1] - void_radius_range[0]
            ) * torch.rand(1)
            is_ok = True
            for i in range(len(void_centers)):
                if (
                    torch.norm(void_center - void_centers[i])
                    < void_radius + void_radii[i]
                ):
                    is_ok = False
            if is_ok:
                void_centers.append(void_center)
                void_radii.append(void_radius)
                break
    void_centers = torch.cat(void_centers)
    void_radii = torch.cat(void_radii)

    points = []
    while len(points) < N:
        # Generate a random point in the rectangular region
        point = rect_min + (rect_max - rect_min) * torch.rand(rect_min.shape[0])

        # Check if the point is inside any void
        distances = torch.norm(point - void_centers, dim=1)
        if not torch.any(distances < void_radii):
            points.append(point)

    return torch.stack(points, dim=0), void_radii


def generate_donut_points(
    N: int = 1000,
    center: torch.tensor = torch.tensor([0.0, 0.0]),
    radius: float = 1.0,
    width: float = 0.2,
    rng: int = None,
) -> torch.tensor:
    """Generate points uniformly distributed in a 2D annulus (donut shape).

    Parameters
    ----------
    N : int, optional
        Number of points to generate. Defaults to 1000.
    center : torch.tensor, optional
        Center of the annulus as a 2D tensor. Defaults to [0.0, 0.0].
    radius : float, optional
        Outer radius of the annulus. Defaults to 1.0.
    width : float, optional
        Width of the annulus. Defaults to 0.2.
    rng : int, optional
        Random-state for reproducibility.
    Returns
    -------
    points : torch.tensor
        A tensor of shape (N, 2) containing the generated points.
    """
    assert center.shape == (2,), "Center must be a 2D point."
    assert radius > 0 and width > 0, "Radius and width must be positive."

    if rng:
        torch.manual_seed(rng)

    angles = torch.rand(N) * 2 * torch.pi  # Random angles
    r = (
        radius - width + width * torch.sqrt(torch.rand(N))
    )  # Random radii (sqrt ensures uniform distribution in annulus)
    x = center[0] + r * torch.cos(angles)
    y = center[1] + r * torch.sin(angles)
    return torch.stack((x, y), dim=1)


def generate_noisy_torus_points(
    num_points=1000,
    R: float = 3.0,
    r: float = 1.0,
    noise_std: float = 0.02,
    rng: int = None,
):
    """Generate points on a torus in 3D with added Gaussian noise.

    Parameters
    ----------
    num_points : int, optional
        Number of points to generate. Defaults to 1000.
    R : int, optional
        Major radius of the torus. Defaults to 3.
    r : int, optional
        Minor radius of the torus. Defaults to 1.
    noise_std : float, optional
        Standard deviation of the Gaussian noise added to the points. Defaults to 0.02.
    rng : int or numpy.random.Generator, optional
        Random-state for reproducibility.
    Returns
    -------
    points : torch.Tensor
        A tensor of shape (num_points, 3) containing the generated points on the torus.
    """
    if rng:
        torch.manual_seed(rng)

    theta = torch.rand(num_points) * 2 * torch.pi
    phi = torch.rand(num_points) * 2 * torch.pi

    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)

    points = torch.stack((x, y, z), dim=1)

    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise
    return noisy_points
