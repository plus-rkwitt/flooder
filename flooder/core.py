"""Implementation of flooder core functionality.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import gudhi
import fpsample
import numpy as np
from math import sqrt
from typing import Union

from .triton_kernel_flood_filtered import flood_triton_filtered

BLOCK_W = 64
BLOCK_R = 64


def generate_landmarks(points: torch.Tensor, N_l: int) -> torch.Tensor:
    """
    Selects landmarks using Farthest-Point Sampling (bucket FPS).

    This method implements a variant of Farthest-Point Sampling from 
    [here](https://dl.acm.org/doi/abs/10.1109/TCAD.2023.3274922).

    Args:
        points (torch.Tensor):
            A (P, d) tensor representing a point cloud. The tensor may reside on any device
            (CPU or GPU) and be of any floating-point dtype.
        N_l (int):
            The number of landmarks to sample (must be <= P and > 0).

    Returns:
        torch.Tensor:
            A (N_l, d) tensor containing a subset of the input `points`, representing the
            sampled landmarks. Returned tensor is on the same device and has the same dtype
            as the input.
    """
    assert N_l > 0, "Number of landmarks must be positive."
    index_set = torch.tensor(
        fpsample.bucket_fps_kdline_sampling(points.cpu(), N_l, h=5).astype(np.int64)
    ).to(points.device)
    return points[index_set]


def flood_complex(
    landmarks: Union[int, torch.Tensor],
    witnesses: torch.Tensor,
    dim: int = 1,
    N: int = 512,  # needs to be a multiple of BLOCK_R
    batch_size: int = 32,
    BATCH_MULT: int = 32,
    disable_kernel: bool = False,
    do_second_stage: bool = False,
) -> dict:
    """
    Constructs a Flood complex from a set of landmark and witness points.

    Args:
        landmarks (Union[int, torch.Tensor]):
            Either an integer indicating the number of landmarks to randomly sample
            from `witnesses`, or a tensor of shape (N_l, d) specifying explicit landmark coordinates.
        witnesses (torch.Tensor):
            A (N, d) tensor containing witness points used as sources in the flood process.
        dim (int, optional):
            The top dimension of the simplices to construct (e.g., 1 for edges, 2 for triangles). Defaults to 1.
        N (int, optional):
            Number of random points to sample for each simplex. This value MUST be a multiple 
            of `BLOCK_R`. Defaults to 512.
        batch_size (int, optional):
            Number of simplices to process per batch. Defaults to 32.
        BATCH_MULT (int, optional):
            Batch size multiplier, used to control kernel tile granularity. Defaults to 32.
        disable_kernel (bool, optional):
            If True, disables the use of the Triton kernel and uses the CPU fallback method. 
            Defaults to False.
        do_second_stage (bool, optional):
            If True, performs a secondary refinement step to improve the accuracy 
            of the covering radii. Defaults to False.

    Returns:
        dict:
            A dictionary mapping simplices to their estimated covering radii (i.e., filtration
            value). Each key is a tuple of landmark indices (e.g., (i, j) for an edge), and 
            each value is a float radius.

    Notes:
        Triton kernel launches may fail if grid dimensions exceed hardware limits.
        Typical constraints include:
            - grid_x <= 2**31 - 1
            - grid_y <= 65535
        To avoid such issues, reduce `batch_size` and/or `BATCH_MULT` if necessary.
    """

    RADIUS_FACTOR = 1.4

    assert N % BLOCK_R == 0, "N must be a multiple of BLOCK_R."

    max_range_dim = torch.argmax(
        witnesses.max(dim=0).values - witnesses.min(dim=0).values
    ).item()
    witnesses = witnesses[torch.argsort(witnesses[:, max_range_dim])].contiguous()
    witnesses_search = witnesses[:, max_range_dim].contiguous()

    if isinstance(landmarks, int):
        landmarks = generate_landmarks(witnesses, min(landmarks, witnesses.shape[0]))
    resolution = torch.cdist(landmarks[-1:], landmarks[:-1]).min().item()
    resolution = 9.0 * resolution * resolution + 1e-3

    dc = gudhi.AlphaComplex(landmarks).create_simplex_tree()

    # For now, the landmark points are always born at time 0.
    out_complex = {}
    idxs = list(range(landmarks.shape[0]))
    for idx in idxs:
        out_complex[(idx,)] = 0.0

    list_simplexes = [[] for _ in range(1, dim + 1)]
    for simplex, filtration in dc.get_simplices():
        if len(simplex) == 1 or len(simplex) > dim + 1:
            continue

        if filtration > resolution:
            out_complex[tuple(simplex)] = sqrt(filtration)
        else:
            list_simplexes[len(simplex) - 2].append(tuple(simplex))

    for d in range(1, dim + 1):
        if len(list_simplexes[d - 1]) == 0:
            continue
        # precompute simplex centers
        all_simplex_points = landmarks[
            torch.tensor(list_simplexes[d - 1], device=landmarks.device)
        ]
        max_flat_idx = torch.argmax(
            torch.cdist(all_simplex_points, all_simplex_points).view(
                all_simplex_points.shape[0], -1
            ),
            dim=1,
        )
        simplex_centers_vec = (
            all_simplex_points[
                torch.arange(all_simplex_points.shape[0]),
                max_flat_idx // all_simplex_points.shape[1],
            ]
            + all_simplex_points[
                torch.arange(all_simplex_points.shape[0]),
                max_flat_idx % all_simplex_points.shape[1],
            ]
        ) / 2.0
        simplex_radii_vec = (
            all_simplex_points - simplex_centers_vec.unsqueeze(1)
        ).norm(dim=2).max(dim=1)[0] * (RADIUS_FACTOR if d > 1 else 1.0)

        splx_idx = torch.argsort(simplex_centers_vec[:, max_range_dim])
        all_simplex_points = all_simplex_points[splx_idx]
        simplex_centers_vec = simplex_centers_vec[splx_idx]
        simplex_radii_vec = simplex_radii_vec[splx_idx]
        list_simplexes[d - 1] = [list_simplexes[d - 1][ii] for ii in splx_idx]

        # Precompute random weights
        num_rand = N
        weights = -torch.log(torch.rand(num_rand, d + 1, device=landmarks.device))
        weights = weights / weights.sum(dim=1, keepdim=True)
        all_random_points = weights.unsqueeze(0) @ all_simplex_points
        del weights

        # If triton kernel is disabled or we are not on the GPU, run CPU computation
        if disable_kernel or (not landmarks.is_cuda):
            for i, simplex in enumerate(list_simplexes[d - 1]):
                valid_witnesses_mask = (
                    torch.cdist(simplex_centers_vec[i : i + 1], witnesses)
                    < simplex_radii_vec[i] + 1e-3
                )
                dists_valid = torch.cdist(
                    all_random_points[i], witnesses[valid_witnesses_mask[0]]
                )
                out_complex[tuple(simplex)] = torch.min(dists_valid, dim=1).values.max()
        # Run triton kernel
        else:
            start = 0
            while start < len(list_simplexes[d - 1]):
                end = min(len(list_simplexes[d - 1]), start + batch_size * BATCH_MULT)
                vmin = (
                    simplex_centers_vec[start:end, max_range_dim]
                    - simplex_radii_vec[start:end]
                ).min() - 1e-3
                vmax = (
                    simplex_centers_vec[start:end, max_range_dim]
                    + simplex_radii_vec[start:end]
                ).max() + 1e-3
                imin = torch.searchsorted(witnesses_search, vmin, right=False)
                imax = torch.searchsorted(witnesses_search, vmax, right=True)

                valid_witnesses_mask = (
                    torch.cdist(simplex_centers_vec[start:end], witnesses[imin:imax])
                    < simplex_radii_vec[start:end].unsqueeze(1) + 1e-3
                )
                valid = torch.cat(
                    [
                        valid_witnesses_mask,
                        torch.arange(BLOCK_W, device=landmarks.device).unsqueeze(0)
                        < ((-valid_witnesses_mask.sum(dim=1)) % BLOCK_W).unsqueeze(1),
                    ],
                    dim=1,
                )
                for b in range(BATCH_MULT):
                    start2 = start + b * batch_size
                    end2 = min(end, start2 + batch_size)

                    random_points = all_random_points[start2:end2]
                    row_idx, col_idx = torch.nonzero(
                        valid[start2 - start : end2 - start], as_tuple=True
                    )
                    row_idx = row_idx[::BLOCK_W]

                    # make sure indexing is contiguous and of type int32 for triton
                    row_idx = row_idx.contiguous().to(torch.int32)
                    col_idx = col_idx.contiguous().to(torch.int32)

                    min_covering_radius, idx = flood_triton_filtered(
                        random_points,
                        witnesses[imin:imax],
                        row_idx,
                        col_idx,
                        BLOCK_W=BLOCK_W,
                        BLOCK_R=BLOCK_R,
                    )

                    if do_second_stage:
                        random_points = (
                            random_points
                            - random_points[
                                torch.arange(random_points.shape[0]), idx, :
                            ].unsqueeze(1)
                        ) / 10.0 + random_points[
                            torch.arange(random_points.shape[0]), idx, :
                        ].unsqueeze(
                            1
                        )
                        min_covering_radius, idx = flood_triton_filtered(
                            random_points,
                            witnesses[imin:imax],
                            row_idx,
                            col_idx,
                            BLOCK_W=BLOCK_W,
                            BLOCK_R=BLOCK_R,
                        )

                    for i in range(end2 - start2):
                        out_complex[list_simplexes[d - 1][start2 + i]] = (
                            min_covering_radius[i]
                        )

                start = end

    return out_complex
