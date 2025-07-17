"""Implementation of flooder core functionality.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import gudhi
import fpsample
import numpy as np
import itertools
from typing import Union
from scipy.spatial import KDTree

from .triton_kernels import compute_mask, compute_filtration

BLOCK_W = 512
BLOCK_R = 16
BLOCK_N = 16
BLOCK_M = 512


def generate_landmarks(points: torch.Tensor, N_l: int, fps_h: Union[None, int] = None) -> torch.Tensor:
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
        fps_h (Union[None, int], optional):
            h parameter (depth of kdtree) that is used for farthest point sampling to select the landmarks.
            If None, then h is selected based on the size of the point cloud.
            Defaults to None

    Returns:
        torch.Tensor:
            A (N_l, d) tensor containing a subset of the input `points`, representing the
            sampled landmarks. Returned tensor is on the same device and has the same dtype
            as the input.
    """
    assert N_l > 0, "Number of landmarks must be positive."
    if N_l > points.shape[0]:
        N_l = points.shape[0]
    N_p = len(points)
    if fps_h == None:
        if N_p > 200_000:
            fps_h = 9
        elif N_p > 80_000:
            fps_h = 7
        else:
            fps_h = 5

    index_set = torch.tensor(
        fpsample.bucket_fps_kdline_sampling(points.cpu(), N_l, h=fps_h, start_idx=0).astype(np.int64),
        device=points.device,
    )
    return points[index_set]


def flood_complex(
    landmarks: Union[int, torch.Tensor],
    witnesses: torch.Tensor,
    dim: Union[None, int] = None,
    num_rand: int = 512,
    points_per_edge: Union[None, int] = None,
    batch_size: int = 256,
    use_triton: bool = True,
    return_simplex_tree: bool = False,
    fps_h: Union[None, int] = None
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
        points_per_edge (int, optional):
            If specified, filtration values will be computed from a grid instead of random points.
            Defaults to None.
        batch_size (int, optional):
            Number of simplices to process per batch. Defaults to 32.
        use_triton (bool, optional):
            If True, Triton kernel is used
            Defaults to True.
        fps_h (Union[None, int], optional):
            h parameter (depth of kdtree) that is used for farthest point sampling to select the landmarks.
            If None, then h is selected based on the size of the point cloud.
            Defaults to None
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
        landmarks = generate_landmarks(witnesses, min(landmarks, witnesses.shape[0]), fps_h)
    assert (
        landmarks.device == witnesses.device
    ), f"landmarks.device ({landmarks.device}) != witnesses.device {witnesses.device}"
    if points_per_edge:
        assert use_triton, "points_per_edge requires use_triton or cpu tensors"
    device = landmarks.device
    if landmarks.is_cuda:
        torch.cuda.set_device(device)
    else:
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
        if points_per_edge is not None and d < dim:  # If grid is used, filtration values of faces can be computed together with max dim simplices.
            continue
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
        if points_per_edge is not None:
            d_simplices = torch.tensor(d_simplices, device=device)
            weights, vertex_ids, face_ids = generate_grid(points_per_edge, dim, device)
        else:
            weights = generate_uniform_weights(num_rand, d, device)
        all_random_points = weights.unsqueeze(0) @ all_simplex_points

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
                    distances = compute_filtration(
                        all_random_points[start:end],
                        witnesses[imin:imax],
                        row_idx,
                        col_idx,
                        BLOCK_W=BLOCK_W,
                        BLOCK_R=BLOCK_R
                    )
                    if points_per_edge is None:
                        min_covering_radius = torch.amax(
                            distances, dim=1
                        )
                        out_complex.update(zip(
                            d_simplices[start:end],
                            min_covering_radius.tolist())
                        )
                    else:
                        for face_id, vertex_id in zip(face_ids, vertex_ids):
                            faces = d_simplices[start:end][
                                :, vertex_id
                            ].flatten(0, 1)
                            distances_face = distances[:, face_id]
                            min_covering_radius_faces = torch.amax(
                                distances_face, dim=2
                            ).flatten()
                            out_complex.update(zip(map(tuple, faces.tolist()), min_covering_radius_faces.tolist()))  # By construction, each face gets the same filtration value irrespective of the simplex it was computed from. If this is violated (by modifying the grid), the code needs to be adapted to sort the simplex faces along axis 1 and take the maximum filtration value when updating the dictionary.
        elif landmarks.is_cuda and not use_triton:
            for i, simplex in enumerate(d_simplices):
                valid_witnesses_mask = (
                    torch.cdist(simplex_centers_vec[i : i + 1], witnesses)
                    < simplex_radii_vec[i]
                )
                distances = torch.cdist(
                    all_random_points[i], witnesses[valid_witnesses_mask[0]]
                )
                out_complex[tuple(simplex)] = torch.amin(distances, dim=1).max()
        else:
            raise RuntimeError("Device not supported.")

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


def generate_grid(n, dim, device):
    """Generates a grid of points on the unit simplex based on the number of points per edge.

    Args:
        n (int):
            Number of points per edge.
        dim (int): 
            Dimension of the simplex.
        device (torch.device):
            Device to create the tensors on.

    Returns:
        tuple:
            - grid (torch.Tensor): A tensor of shape [C, dim + 1] containing the grid points (coordinate weights).
            - vertex_ids (list): A list of tensors, each containing the vertex indices for each face.
            - face_ids (list): A list of tensors, each containing the face indices for each face.
    """

    combs = torch.tensor(list(itertools.combinations(range(n + dim), dim)), device=device)  # shape [C, dim]
    padded = torch.cat([
        torch.full((combs.shape[0], 1), -1, device=device), 
        combs,
        torch.full((combs.shape[0], 1), n + dim, device=device) 
    ], dim=1)  # shape [C, dim + 2]
    grid = torch.diff(padded, dim=1) - 1  # shape [C, dim + 1]

    face_ids = []
    vertex_ids = []
    all_axes = torch.arange(dim + 1, device=device)

    for k in range(dim + 1):
        face_ids_k = []
        vertex_ids_k = []
        for comb in itertools.combinations(range(dim + 1), k):
            comb_tensor = torch.tensor(comb, device=device)
            if len(comb) == 0:
                mask = torch.ones(len(grid), dtype=bool, device=device)
            else:
                mask = (grid[:, comb_tensor] == 0).all(dim=1)
            face_ids_k.append(torch.nonzero(mask).flatten())
            idx = all_axes[~torch.isin(all_axes, comb_tensor)]
            vertex_ids_k.append(idx)
        face_ids.append(torch.stack(face_ids_k))
        vertex_ids.append(torch.stack(vertex_ids_k))
    grid = grid / n
    return grid, vertex_ids, face_ids


def generate_uniform_weights(num_rand, dim, device):
    """Generates num_rand points from a uniform distribution on the unit simplex.
    Args:
        num_rand (int):
            Number of random points to generate.
        dim (int):
            Dimension of the simplex.
        device (torch.device):
            Device to create the tensor on.
    Returns:
        torch.Tensor:
            A tensor of shape [num_rand, dim + 1] containing the random points (coordinate weights).
    """
    weights = -torch.log(1 - torch.rand(num_rand, dim + 1)).to(device)  # For consistency with the cpu version, random points are generated on the CPU and then moved to the device.
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights
