"""Implementation of synthetic data generators.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

from typing import Tuple, Literal

import torch
import numpy as np


def generate_figure_eight_points_2d(
    n: int = 1000,
    r_bounds: Tuple[float, float] = (0.2, 0.3),
    centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.3, 0.5), (0.7, 0.5)),
    noise_std: float = 0.0,
    noise_kind: Literal["gaussian", "uniform"] = "gaussian",
    seed: int = None,
) -> torch.tensor:
    """
    Generate 2D points uniformly sampled in a figure-eight shape, with optional noise.

    This function samples `n_samples` points distributed across two circular lobes
    (forming a figure-eight shape) centered at specified coordinates. Optionally,
    isotropic Gaussian or uniform noise can be added to the coordinates.

    Args:
        n (int, optional): Number of 2D points to generate. Defaults to 1000.
        r_bounds (Tuple[float, float], optional): Tuple specifying the minimum and maximum
            radius for sampling within each lobe. Defaults to (0.2, 0.3).
        centers (Tuple[Tuple[float, float], Tuple[float, float]], optional): Coordinates
            of the centers of the two lobes. Defaults to ((0.3, 0.5), (0.7, 0.5)).
        noise_std (float, optional): Standard deviation (for Gaussian) or half-width
            (for uniform) of noise to add to each point. Defaults to 0.0 (no noise).
        noise_kind (Literal["gaussian", "uniform"], optional): Type of noise distribution
            to use if `noise_std > 0`. Defaults to "gaussian".
        seed (int, optional): Random seed for reproducibility. If None, randomness is not seeded.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the sampled 2D points.
    """
    if seed is not None:
        np.random.seed(seed)

    lobe_idx = np.random.randint(0, 2, size=n)
    cx, cy = np.asarray(centers).T  # shape (2,)
    cx = cx[lobe_idx]  # (n_samples,)
    cy = cy[lobe_idx]

    r_min, r_max = r_bounds
    r = np.sqrt(np.random.uniform(r_min**2, r_max**2, size=n))
    theta = np.random.uniform(0.0, 2 * np.pi, size=n)

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    if noise_std > 0:
        if noise_kind == "gaussian":
            x += np.random.normal(0.0, noise_std, size=n)
            y += np.random.normal(0.0, noise_std, size=n)
        elif noise_kind == "uniform":
            half = noise_std
            x += np.random.uniform(-half, half, size=n)
            y += np.random.uniform(-half, half, size=n)
        else:
            raise ValueError("noise_kind must be 'gaussian' or 'uniform'")

    return torch.tensor(np.stack((x, y), axis=1), dtype=torch.float32)


@torch.no_grad()
def generate_swiss_cheese_points(
    n: int = 1000,
    rect_min: tuple = (0.0, 0.0, 0.0),
    rect_max: tuple = (1.0, 1.0, 1.0),
    k: int = 6,
    void_radius_range: tuple = (0.1, 0.2),
    seed: int = None,
    *,
    device="cpu",
    batch_factor=4,  # how many candidates to shoot each round
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate points in a high-dimensional rectangular region with randomly placed spherical voids,
    forming a "Swiss cheese" structure.

    Points are sampled uniformly within the bounding box defined by `rect_min` and `rect_max`,
    excluding k randomly positioned spherical voids with radii sampled from `void_radius_range`.

    Args:
        n (int, optional): Number of points to generate. Defaults to 1000.
        rect_min (tuple, optional): Minimum coordinates of the rectangular region.
            Defaults to a tuple of three zeros.
        rect_max (tuple, optional): Maximum coordinates of the rectangular region.
            Defaults to a tuple of three ones.
        k (int, optional): Number of spherical voids to generate. Defaults to 6.
        void_radius_range (Tuple[float, float], optional): Range `(min_radius, max_radius)`
            for the void radii. Defaults to (0.1, 0.2).
        seed (int, optional): Random seed for reproducibility. If None, randomness is not seeded.
        device (torch.device, optional): Device to perform computations on. Defaults to 'cpu'.
        batch_factor (int, optional): How many candidates to shoot each round. Defaults to 4.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - `points` (torch.Tensor): Tensor of shape (N, dim) with generated sample points.
            - `void_radii` (torch.Tensor): Tensor of shape (k,) with the radii of the voids.

    Examples:
        >>> rect_min = (0.0, 0.0, 0.0)
        >>> rect_max = (1.0, 1.0, 1.0)
        >>> void_radius_range = (0.1, 0.2)
        >>> k = 5
        >>> points, _ = generate_swiss_cheese_points(
        ...     1000000, rect_min, rect_max, k, void_radius_range
        ... )
        >>> points.shape
        torch.Size([1000000, 3])
    """
    if seed:
        torch.manual_seed(seed)

    assert len(rect_min) == len(
        rect_max
    ), "rect_min and rect_max must have the same dimension."
    d = len(rect_min)
    r_min, r_max = void_radius_range

    # --- 1.  build non-overlapping voids ------------------------------------
    centres = torch.empty((0, d), device=device)
    radii = torch.empty((0,), device=device)
    rect_min = torch.tensor(rect_min, dtype=torch.float32, device=device)
    rect_max = torch.tensor(rect_max, dtype=torch.float32, device=device)

    while centres.shape[0] < k:
        # shoot a small batch of candidate voids
        B = max(8, 2 * (k - centres.shape[0]))  # a handful is enough
        cand_centres = (rect_min + r_max) + (
            rect_max - rect_min - 2 * r_max
        ) * torch.rand(B, d, device=device)
        cand_radii = r_min + (r_max - r_min) * torch.rand(B, device=device)

        if centres.numel() == 0:
            ok = torch.ones(B, dtype=torch.bool, device=device)
        else:
            dist = torch.cdist(cand_centres, centres)  # B × |centres|
            ok = (dist >= (cand_radii[:, None] + radii[None, :])).all(dim=1)

        # keep as many as we still need
        keep = ok.nonzero(as_tuple=False).squeeze()[: k - centres.shape[0]]
        centres = torch.cat([centres, cand_centres[keep]], dim=0)
        radii = torch.cat([radii, cand_radii[keep]], dim=0)

    # --- 2.  rejection sample points in large vectorised batches ------------
    pts = torch.empty((0, d), dtype=rect_min.dtype, device=device)
    todo = n
    while todo:
        B = batch_factor * todo  # adaptive batch
        cand = rect_min + (rect_max - rect_min) * torch.rand(B, d, device=device)

        # distance of every candidate to every void centre:  B × k
        if k:
            dist = torch.cdist(cand, centres)
            good = (dist >= radii[None, :]).all(dim=1)
        else:  # no holes at all
            good = torch.ones(B, dtype=torch.bool, device=device)

        accepted = cand[good][:todo]  # at most 'todo'
        pts = torch.cat([pts, accepted], dim=0)
        todo = n - pts.shape[0]

    return pts, centres, radii


def generate_annulus_points_2d(
    n: int = 1000,
    center: torch.tensor = torch.tensor([0.0, 0.0]),
    radius: float = 1.0,
    width: float = 0.2,
    seed: int = None,
) -> torch.tensor:
    """
    Generate 2D points uniformly distributed in the region between two concentric circles.

    In particulr, points are sampled uniformly within a ring defined by an outer `radius` and an inner radius of `radius - width`, centered at a specified 2D location.

    Args:
        n (int, optional): Number of points to generate. Defaults to 1000.
        center (torch.Tensor, optional): Center of the annulus as a tensor of shape (2,).
            Defaults to [0.0, 0.0].
        radius (float, optional): Outer radius of the annulus. Must be positive. Defaults to 1.0.
        width (float, optional): Thickness of the annulus. Must be positive and less than `radius`.
            Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. If None, randomness is not seeded.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the sampled 2D points.

    Examples:
        >>> center = torch.tensor([0.0, 0.0])
        >>> points = generate_annulus_points_2d(n=500, center=center, radius=1.0, width=0.3, seed=42)
        >>> points.shape
        torch.Size([500, 2])
    """
    assert center.shape == (2,), "Center must be a 2D point."
    assert radius > 0 and width > 0, "Radius and width must be positive."

    if seed is not None:
        torch.manual_seed(seed)

    angles = torch.rand(n) * 2 * torch.pi  # Random angles
    r = (
        radius - width + width * torch.sqrt(torch.rand(n))
    )  # Random radii (sqrt ensures uniform distribution in annulus)
    x = center[0] + r * torch.cos(angles)
    y = center[1] + r * torch.sin(angles)
    return torch.stack((x, y), dim=1)


def generate_noisy_torus_points_3d(
    n=1000,
    R: float = 3.0,
    r: float = 1.0,
    noise_std: float = 0.02,
    seed: int = None,
) -> torch.tensor:
    """
    Generate 3D points on a torus with added Gaussian noise.

    Points are uniformly sampled on the surface of a torus defined by a major radius `R`
    and a minor radius `r`. Gaussian noise with standard deviation `noise_std` is added
    to each point independently in x, y, and z dimensions.

    Args:
        n (int, optional): Number of points to generate. Defaults to 1000.
            R (float, optional): Major radius of the torus (distance from the center of
            the tube to the center of the torus). Must be positive. Defaults to 3.0.
        r (float, optional): Minor radius of the torus (radius of the tube).
            Must be positive. Defaults to 1.0.
        noise_std (float, optional): Standard deviation of the Gaussian noise added to
            the points. Defaults to 0.02.
        seed (int, optional): Random seed for reproducibility. If None, randomness
            is not seeded.

    Returns:
        torch.Tensor: A tensor of shape (num_points, 3) containing the generated
            noisy 3D points.

    Examples:
        >>> points = generate_noisy_torus_points_3d(
                n=500, R=3.0, r=1.0, noise_std=0.05, seed=123)
        >>> points.shape
        torch.Size([500, 3])
    """
    if seed is not None:
        torch.manual_seed(seed)

    theta = torch.rand(n) * 2 * torch.pi
    phi = torch.rand(n) * 2 * torch.pi

    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)

    points = torch.stack((x, y, z), dim=1)

    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise
    return noisy_points
