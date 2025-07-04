"""Implementation of flooder core functionality.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import gudhi
import fpsample
import numpy as np
from typing import Union
from scipy.spatial import KDTree

from .triton_kernels import compute_mask, compute_filtration

BLOCK_W = 512
BLOCK_R = 16
BLOCK_N = 16
BLOCK_M = 512


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
    if N_l > points.shape[0]:
        N_l = points.shape[0]
    index_set = torch.tensor(
        fpsample.bucket_fps_kdline_sampling(points.cpu(), N_l, h=5).astype(np.int64),
        device=points.device,
    )
    return points[index_set]


def flood_complex(
    landmarks: Union[int, torch.Tensor],
    witnesses: torch.Tensor,
    dim: Union[None, int] = None,
    num_rand: int = 512,
    batch_size: int = 256,
    use_triton: bool = True,
    return_simplex_tree: bool = False
) -> Union[dict, gudhi.SimplexTree]:
    """
    Constructs a Flood complex from a set of landmark and witness points.

    Args:
        landmarks (Union[int, torch.Tensor]):
            Either an integer indicating the number of landmarks to randomly sample
            from `witnesses`, or a tensor of shape (N_l, d) specifying explicit landmark coordinates.
        witnesses (torch.Tensor):
            A (N, d) tensor containing witness points used as sources in the flood process.
        dim (Union[None, int], optional):
            The top dimension of the simplices to construct.
            Defaults to None resulting in the dimension of the ambient space.
        num_rand (int, optional):
            Number of random points to sample for each simplex.
            Defaults to 512.
        batch_size (int, optional):
            Number of simplices to process per batch. Defaults to 32.
        use_triton (bool, optional):
            If True, Triton kernel is used
            Defaults to True.
        return_simplex_tree (bool, optional):
            I true, a gudhi.SimplexTree is returned, else a dictionary.
            Defaults to False

    Returns:
        Union[dict, gudhi.SimplexTree]
            Depending on the return_simplex_tree argument either a
            gudhi.SimplexTree or a dictionary is returned,
            mapping simplices to their estimated covering radii (i.e., filtration
            value). Each key is a tuple of landmark indices (e.g., (i, j) for an edge), and
            each value is a float radius.
    """
    if dim is None:
        dim = witnesses.shape[1]

    max_range_dim = torch.argmax(
        witnesses.max(dim=0).values - witnesses.min(dim=0).values
    ).item()
    witnesses = witnesses[torch.argsort(witnesses[:, max_range_dim])].contiguous()
    witnesses_search = witnesses[:, max_range_dim].contiguous()

    if isinstance(landmarks, int):
        landmarks = generate_landmarks(witnesses, min(landmarks, witnesses.shape[0]))
    assert (
        landmarks.device == witnesses.device
    ), f"landmarks.device ({landmarks.device}) != witnesses.device {witnesses.device}"
    device = landmarks.device

    if not landmarks.is_cuda:
        kdtree = KDTree(np.asarray(witnesses))

    dc = gudhi.DelaunayComplex(landmarks).create_simplex_tree()

    out_complex = {}

    # For now, the landmark points are always born at time 0.
    out_complex.update(((i,), 0.0) for i in range(len(landmarks)))

    list_simplices = [[] for _ in range(dim)]
    for simplex, _ in dc.get_simplices():
        if len(simplex) == 1 or len(simplex) > dim + 1:
            continue
        list_simplices[len(simplex) - 2].append(tuple(simplex))

    for d in range(1, dim + 1):
        d_simplices = list_simplices[d - 1]
        num_simplices = len(d_simplices)
        if num_simplices == 0:
            continue
        # precompute simplex centers
        all_simplex_points = landmarks[[d_simplices]]
        max_flat_idx = torch.argmax(
            torch.cdist(all_simplex_points, all_simplex_points).flatten(1),
            dim=1,
        )
        idx0, idx1 = torch.unravel_index(max_flat_idx, [d + 1, d + 1])
        simplex_centers_vec = (
            all_simplex_points[torch.arange(num_simplices), idx0]
            + all_simplex_points[torch.arange(num_simplices), idx1]
        ) / 2.0
        simplex_radii_vec = torch.amax(
            (all_simplex_points - simplex_centers_vec.unsqueeze(1)).norm(dim=2), dim=1
        ) * (1.42 if d > 1 else 1.01) + 1e-3

        splx_idx = torch.argsort(simplex_centers_vec[:, max_range_dim])
        all_simplex_points = all_simplex_points[splx_idx]
        simplex_centers_vec = simplex_centers_vec[splx_idx]
        simplex_radii_vec = simplex_radii_vec[splx_idx]
        d_simplices = [d_simplices[ii] for ii in splx_idx]

        # Precompute random weights
        weights = -torch.log(
            1 - torch.rand(num_rand, d + 1).to(device)
        )  # Random points are created on cpu for seed for consistency across devices, use 1 - torch.rand(..) to exclude 0.
        weights = weights / weights.sum(dim=1, keepdim=True)
        all_random_points = weights.unsqueeze(0) @ all_simplex_points
        del weights

        if landmarks.is_cpu:
            nn_dists, _ = kdtree.query(np.asarray(all_random_points))
            filt = np.max(nn_dists, axis=1)
            out_complex.update(zip(d_simplices, filt))

        elif landmarks.is_cuda and use_triton:
            for start in range(0, len(d_simplices), batch_size):
                end = min(len(d_simplices), start + batch_size)
                vmin = (
                    simplex_centers_vec[start:end, max_range_dim]
                    - simplex_radii_vec[start:end]
                ).min()
                vmax = (
                    simplex_centers_vec[start:end, max_range_dim]
                    + simplex_radii_vec[start:end]
                ).max()
                imin = torch.searchsorted(witnesses_search, vmin, right=False)
                imax = torch.searchsorted(witnesses_search, vmax, right=True)

                if use_triton:
                    valid = compute_mask(
                        witnesses[imin:imax],
                        simplex_centers_vec[start:end],
                        simplex_radii_vec[start:end],
                        BLOCK_N,
                        BLOCK_M,
                        BLOCK_W
                    )
                    row_idx, col_idx = torch.nonzero(
                        valid, as_tuple=True
                    )
                    min_covering_radius = compute_filtration(
                        all_random_points[start:end],
                        witnesses[imin:imax],
                        row_idx,
                        col_idx,
                        BLOCK_W=BLOCK_W,
                        BLOCK_R=BLOCK_R,
                    )
                    out_complex.update(zip(
                        d_simplices[start:end],
                        min_covering_radius.tolist())
                    )
        elif landmarks.is_cuda and not use_triton:
            for i, simplex in enumerate(d_simplices):
                valid_witnesses_mask = (
                    torch.cdist(simplex_centers_vec[i : i + 1], witnesses)
                    < simplex_radii_vec[i]
                )
                dists_valid = torch.cdist(
                    all_random_points[i], witnesses[valid_witnesses_mask[0]]
                )
                out_complex[tuple(simplex)] = torch.amin(dists_valid, dim=1).max()
        else:
            raise RuntimeError("device not supported.")

    stree = gudhi.SimplexTree()
    for simplex in out_complex:
        stree.insert(simplex, float("inf"))
        stree.assign_filtration(simplex, out_complex[simplex])
    stree.make_filtration_non_decreasing()

    if return_simplex_tree:
        return stree

    out_complex = {}
    out_complex.update(
        (tuple(simplex), filtr) for (simplex, filtr) in stree.get_simplices()
    )
    return out_complex
