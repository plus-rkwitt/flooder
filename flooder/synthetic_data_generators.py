"""Implementation of synthetic data generators.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import numpy as np
from typing import Tuple, Optional, Literal


def generate_figure_eight_2D_points(
    n_samples: int = 1000,
    r_bounds: Tuple[float, float] = (0.2, 0.3),
    centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.3, 0.5), (0.7, 0.5)),
    noise_std: float = 0.0,
    noise_kind: Literal["gaussian", "uniform"] = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> torch.tensor:
    """
    Generate 2D points uniformly sampled in a figure-eight shape, with optional noise.

    This function samples `n_samples` points distributed across two circular lobes 
    (forming a figure-eight shape) centered at specified coordinates. Optionally, 
    isotropic Gaussian or uniform noise can be added to the coordinates.

    Args:
        n_samples (int, optional): Number of 2D points to generate. Defaults to 1000.
        r_bounds (Tuple[float, float], optional): Tuple specifying the minimum and maximum 
            radius for sampling within each lobe. Defaults to (0.2, 0.3).
        centers (Tuple[Tuple[float, float], Tuple[float, float]], optional): Coordinates 
            of the centers of the two lobes. Defaults to ((0.3, 0.5), (0.7, 0.5)).
        noise_std (float, optional): Standard deviation (for Gaussian) or half-width 
            (for uniform) of noise to add to each point. Defaults to 0.0 (no noise).
        noise_kind (Literal["gaussian", "uniform"], optional): Type of noise distribution 
            to use if `noise_std > 0`. Defaults to "gaussian".
        rng (Optional[np.random.Generator], optional): Optional NumPy random number 
            generator for reproducibility. If None, a new default generator is used.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the sampled 2D points.
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate points in a high-dimensional rectangular region with randomly placed spherical voids,
    forming a "Swiss cheese" structure.

    Points are sampled uniformly within the bounding box defined by `rect_min` and `rect_max`,
    excluding k randomly positioned spherical voids with radii sampled from `void_radius_range`.

    Args:
        N (int, optional): Number of points to generate. Defaults to 1000.
        rect_min (torch.Tensor, optional): Minimum coordinates of the rectangular region. 
            Defaults to a tensor of six zeros.
        rect_max (torch.Tensor, optional): Maximum coordinates of the rectangular region. 
            Defaults to a tensor of six ones.
        k (int, optional): Number of spherical voids to generate. Defaults to 6.
        void_radius_range (Tuple[float, float], optional): Range `(min_radius, max_radius)` 
            for the void radii. Defaults to (0.1, 0.2).
        rng (int, optional): Random seed for reproducibility. If None, randomness is not seeded.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - `points` (torch.Tensor): Tensor of shape (N, dim) with generated sample points.
            - `void_radii` (torch.Tensor): Tensor of shape (k,) with the radii of the voids.
    
    Examples:    
        >>> rect_min = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> rect_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        >>> void_radius_range = (0.1, 0.2)
        >>> k = 5
        >>> points, _ = generate_swiss_cheese_points(
        ...     1000000, rect_min[:3], rect_max[:3], k, void_radius_range
        ... )
        >>> points.shape
        torch.Size([1000000, 3])
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
    """
    Generate 2D points uniformly distributed in a circular annulus (donut shape).

    Points are sampled uniformly within a ring defined by an outer `radius` and 
    an inner radius of `radius - width`, centered at a specified 2D location.

    Args:
        N (int, optional): Number of points to generate. Defaults to 1000.
        center (torch.Tensor, optional): Center of the annulus as a tensor of shape (2,). 
            Defaults to [0.0, 0.0].
        radius (float, optional): Outer radius of the annulus. Must be positive. Defaults to 1.0.
        width (float, optional): Thickness of the annulus. Must be positive and less than `radius`.
            Defaults to 0.2.
        rng (int, optional): Random seed for reproducibility. If None, randomness is not seeded.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the sampled 2D points.

    Examples:
        >>> center = torch.tensor([0.0, 0.0])
        >>> points = generate_donut_points(N=500, center=center, radius=1.0, width=0.3, rng=42)
        >>> points.shape
        torch.Size([500, 2])
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
) -> torch.tensor:
    """
    Generate 3D points on a torus with added Gaussian noise.

    Points are uniformly sampled on the surface of a torus defined by a major radius `R`
    and a minor radius `r`. Gaussian noise with standard deviation `noise_std` is added
    to each point independently in x, y, and z dimensions.

    Args:
        num_points (int, optional): Number of points to generate. Defaults to 1000.
        R (float, optional): Major radius of the torus (distance from the center of the tube 
            to the center of the torus). Must be positive. Defaults to 3.0.
        r (float, optional): Minor radius of the torus (radius of the tube). Must be positive. 
            Defaults to 1.0.
        noise_std (float, optional): Standard deviation of the Gaussian noise added to the 
            points. Defaults to 0.02.
        rng (int, optional): Random seed for reproducibility. If None, randomness is not seeded.

    Returns:
        torch.Tensor: A tensor of shape (num_points, 3) containing the generated noisy 3D points.

    Examples:
        >>> points = generate_noisy_torus_points(num_points=500, R=3.0, r=1.0, noise_std=0.05, rng=123)
        >>> points.shape
        torch.Size([500, 3])
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
