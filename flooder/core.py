"""Implementation of flooder core functionality.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import itertools
from typing import Union
from typing import List, Tuple

import torch
import gudhi
import fpsample
import numpy as np
from numbers import Integral
from scipy.spatial import KDTree

from .triton_kernels import compute_mask, compute_filtration

BLOCK_W = 512
BLOCK_R = 16
BLOCK_N = 16
BLOCK_M = 512


def flood_complex(
    points: torch.Tensor,
    landmarks: Union[int, torch.Tensor],
    max_dimension: Union[None, int] = None,
    points_per_edge: Union[None, int] = 30,
    num_rand: int = None,
    batch_size: Union[None, int] = 64,
    use_triton: bool = True,
    return_simplex_tree: bool = False,
    fps_h: Union[None, int] = None,
    start_idx: Union[int, None] = 0,
) -> Union[dict, gudhi.SimplexTree]:  # pylint: disable=no-member
    """
    Constructs a Flood complex from a set of witness points and landmarks.

    Args:
        points (torch.Tensor):
            A (N, d) tensor containing witness points used as sources in the flood process.
        landmarks (Union[int, torch.Tensor]):
            Either an integer indicating the number of landmarks to randomly sample from
            `points`, or a tensor of shape (N_l, d) specifying explicit landmark
            coordinates.
        max_dimension (Union[None, int], optional):
            The top dimension of the simplices to construct.
            Defaults to None resulting in the dimension of the ambient space.
        points_per_edge (Union[None, int], optional):
            Specifies resolution on simplices used for computing filtration values.
            Tradeoff in accuracy vs. speed. Defaults to 30.
        num_rand (Union[None, int], optional):
            If specified, filtration values are computed from a fixed number of
            random points per simplex. Defaults to None.
        batch_size (int, optional):
            Number of simplices to process per batch. Defaults to 32.
        use_triton (bool, optional):
            If True, Triton kernel is used.
            Defaults to True.
        fps_h (Union[None, int], optional):
            h parameter (depth of kdtree) that is used for farthest point sampling to
            select the landmarks. If None, then h is selected based on the size of the
            point cloud. Defaults to None.
        return_simplex_tree (bool, optional):
            I true, a gudhi.SimplexTree is returned, else a dictionary.
            Defaults to False
        start_idx (int | None, optional):
            If provided, FPS starts from this index in the point cloud. If not,
            the start index will be randomly picked from the point cloud. Defaults to 0.

    Returns:
        Union[dict, gudhi.SimplexTree]
            Depending on the return_simplex_tree argument either a
            gudhi.SimplexTree or a dictionary is returned,
            mapping simplices to their estimated covering radii (i.e., filtration
            value). Each key is a tuple of landmark indices (e.g., (i, j) for an edge), and
            each value is a float radius.
    """
    if max_dimension is None:
        max_dimension = points.shape[1]
    if isinstance(landmarks, Integral):
        landmarks = generate_landmarks(
            points, min(landmarks, points.shape[0]), fps_h, start_idx=start_idx
        )
    assert (
        landmarks.device == points.device
    ), f"landmarks.device ({landmarks.device}) != points.device {points.device}"
    device = landmarks.device
    if landmarks.is_cuda:
        torch.cuda.set_device(device)
    else:
        kdtree = KDTree(np.asarray(points))

    stree = gudhi.DelaunayComplex(  # pylint: disable=no-member
        landmarks
    ).create_simplex_tree()
    out_complex = {}

    simplices = [[] for _ in range(max_dimension + 1)]
    for simplex, _ in stree.get_simplices():
        if len(simplex) <= max_dimension + 1:
            simplices[len(simplex) - 1].append(tuple(simplex))

    max_range_dim = torch.argmax(
        points.max(dim=0).values - points.min(dim=0).values
    ).item()
    points = points[torch.argsort(points[:, max_range_dim])].contiguous()
    points_search = points[:, max_range_dim].contiguous()

    for d in range(max_dimension + 1):
        # If grid is used, filtration values of faces can be computed
        # together with max dim simplices.
        if num_rand is None and d < max_dimension:
            continue
        d_simplices = torch.tensor(simplices[d], device=device)
        num_simplices = len(d_simplices)
        if num_simplices == 0:
            continue
        # precompute simplex centers
        simplex_vertices = landmarks[d_simplices]
        max_flat_idx = torch.argmax(
            torch.cdist(simplex_vertices, simplex_vertices).flatten(1),
            dim=1,
        )
        idx0, idx1 = torch.unravel_index(max_flat_idx, [d + 1, d + 1])
        simplex_centers = (
            simplex_vertices[torch.arange(num_simplices), idx0]
            + simplex_vertices[torch.arange(num_simplices), idx1]
        ) / 2.0
        simplex_radii = (
            torch.amax(
                (simplex_vertices - simplex_centers.unsqueeze(1)).norm(dim=2), dim=1
            )
            * (1.42 if d > 1 else 1.01)
            + 1e-3
        )

        # sort by coordinate in max_range_dim
        splx_idx = torch.argsort(simplex_centers[:, max_range_dim])
        simplex_vertices = simplex_vertices[splx_idx]
        simplex_centers = simplex_centers[splx_idx]
        simplex_radii = simplex_radii[splx_idx]
        d_simplices = d_simplices[splx_idx]

        # generate points on simplices
        if num_rand is None:
            weights, vertex_idxs, face_idxs = generate_grid(
                points_per_edge, max_dimension, device
            )
        else:
            weights = generate_uniform_weights(num_rand, d, device)
        points_on_simplex = weights.unsqueeze(0) @ simplex_vertices

        if landmarks.is_cpu or not use_triton:
            batch_size = num_simplices  # no batching needed

        for start in range(0, num_simplices, batch_size):
            end = min(num_simplices, start + batch_size)

            #  Compute distances
            if landmarks.is_cpu:
                distances, _ = kdtree.query(np.asarray(points_on_simplex[start:end]))
                distances = torch.as_tensor(distances)
            elif landmarks.is_cuda:
                vmin = (
                    simplex_centers[start:end, max_range_dim] - simplex_radii[start:end]
                ).min()
                vmax = (
                    simplex_centers[start:end, max_range_dim] + simplex_radii[start:end]
                ).max()
                imin = torch.searchsorted(points_search, vmin, right=False)
                imax = torch.searchsorted(points_search, vmax, right=True)
                if use_triton:
                    valid = compute_mask(
                        points[imin:imax],
                        simplex_centers[start:end],
                        simplex_radii[start:end],
                        BLOCK_N,
                        BLOCK_M,
                        BLOCK_W,
                    )
                    row_idx, col_idx = torch.nonzero(valid, as_tuple=True)
                    distances = compute_filtration(
                        points_on_simplex[start:end],
                        points[imin:imax],
                        row_idx,
                        col_idx,
                        BLOCK_W=BLOCK_W,
                        BLOCK_R=BLOCK_R,
                    )
                else:
                    distances = torch.full(
                        (end - start, len(weights)),
                        torch.inf,
                        device=device,
                        dtype=torch.float32,
                    )

                    for i in range(start, end):
                        # Surprisingly the loop is faster than scatter_reduce or
                        # segment_coo for large numbers of points.
                        # avoid torch.cdist for numerical stability
                        valid = (simplex_centers[i] - points[imin:imax]).norm(
                            dim=1
                        ) < simplex_radii[i]
                        inter = (
                            points_on_simplex[i : i + 1]
                            - points[imin:imax][valid, None]
                        ).norm(dim=2)
                        distances[i - start] = torch.amin(inter, dim=0)
            else:
                raise RuntimeError("Device not supported.")

            # Extract filtration values
            if num_rand is None:
                for face_idx, vertex_idx in zip(face_idxs, vertex_idxs):
                    faces = d_simplices[start:end][:, vertex_idx].flatten(0, 1)
                    distances_face = distances[:, face_idx]
                    min_covering_radius_faces = torch.amax(
                        distances_face, dim=2
                    ).flatten()
                    out_complex.update(
                        zip(
                            map(tuple, faces.tolist()),
                            min_covering_radius_faces.tolist(),
                        )
                    )
                    # By construction, each face gets the same filtration value
                    # irrespective of the simplex it was computed from. If this is violated
                    # (by modifying the grid), the code needs to be adapted to sort the
                    # simplex faces along axis 1 and take the maximum filtration value
                    # when updating the dictionary.
            else:
                min_covering_radius = torch.amax(distances, dim=1)
                out_complex.update(
                    zip(
                        map(tuple, d_simplices[start:end].tolist()),
                        min_covering_radius.tolist(),
                    )
                )

    for simplex, filtration_val in out_complex.items():
        stree.assign_filtration(simplex, filtration_val)
    stree.make_filtration_non_decreasing()
    torch.cuda.empty_cache()
    if return_simplex_tree:
        return stree

    out_complex = dict(
        (tuple(simplex), filtr) for (simplex, filtr) in stree.get_simplices()
    )
    return out_complex


def generate_landmarks(
    points: torch.Tensor,
    n_lms: int,
    fps_h: Union[None, int] = None,
    start_idx: Union[int, None] = None,
) -> torch.Tensor:
    """
    Selects landmarks using Farthest-Point Sampling (bucket FPS).

    This method implements a variant of Farthest-Point Sampling from
    [here](https://dl.acm.org/doi/abs/10.1109/TCAD.2023.3274922).

    Args:
        points (torch.Tensor):
            A (P, d) tensor representing a point cloud. The tensor may reside on any
            device (CPU or GPU) and be of any floating-point dtype.
        n_lms (int):
            The number of landmarks to sample (must be <= P and > 0).
        fps_h (Union[None, int], optional):
            h parameter (depth of kdtree) that is used for farthest point sampling to
            select the landmarks. If None, then h is selected based on the size of the
            point cloud. Defaults to None.
        start_idx (int | None, optional):
            If provided, the sampling starts from this index in the point cloud. If not,
            the start index will be randomly picked from the point cloud.

    Returns:
        torch.Tensor:
            A (n_l, d) tensor containing a subset of the input `points`, representing the
            sampled landmarks. Returned tensor is on the same device and has the same dtype
            as the input.
    """
    assert n_lms > 0, "Number of landmarks must be positive."
    n_pts = len(points)
    n_lms = min(n_lms, n_pts)
    if fps_h is None:
        if n_pts > 200_000:
            fps_h = 9
        elif n_pts > 80_000:
            fps_h = 7
        else:
            fps_h = 5

    index_set = torch.tensor(
        fpsample.bucket_fps_kdline_sampling(
            points.cpu(), n_lms, h=fps_h, start_idx=start_idx
        ).astype(np.int64),
        device=points.device,
    )
    return points[index_set]


def generate_grid(
    n: int, dim: int, device: torch.device
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
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
            grid (torch.Tensor):
                Tensor of shape (C, dim + 1), containing the grid points (coordinate weights).
            vertex_ids (list of torch.Tensor):
                A list of tensors, each containing the vertex indices for each face.
            face_ids (list of torch.Tensor):
                A list of tensors, each containing the face indices for each face.
    """

    combs = torch.tensor(
        list(itertools.combinations(range(n + dim - 1), dim)), device=device
    )  # shape [C, dim]
    padded = torch.cat(
        [
            torch.full((combs.shape[0], 1), -1, device=device),
            combs,
            torch.full((combs.shape[0], 1), n + dim - 1, device=device),
        ],
        dim=1,
    )  # shape [C, dim + 2]
    grid = torch.diff(padded, dim=1) - 1  # shape [C, dim + 1]

    face_idxs = []
    vertex_idxs = []
    all_axes = torch.arange(dim + 1, device=device)

    for k in range(dim + 1):
        face_idxs_k = []
        vertex_idxs_k = []
        for comb in itertools.combinations(range(dim + 1), k):
            comb_tensor = torch.tensor(comb, device=device)
            if len(comb) == 0:
                mask = torch.ones(len(grid), dtype=bool, device=device)
            else:
                mask = (grid[:, comb_tensor] == 0).all(dim=1)
            face_idxs_k.append(torch.nonzero(mask).flatten())
            idx = all_axes[~torch.isin(all_axes, comb_tensor)]
            vertex_idxs_k.append(idx)
        face_idxs.append(torch.stack(face_idxs_k))
        vertex_idxs.append(torch.stack(vertex_idxs_k))
    grid = grid / (n - 1)
    return grid, vertex_idxs, face_idxs


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
            A tensor of shape [num_rand, dim + 1] containing the random points
            (coordinate weights).
    """
    if dim == 0:
        weights = torch.ones((num_rand, 1), device=device)
    else:
        # For consistency with the cpu version, random points are generated on the
        # CPU and then moved to the device.
        weights = -torch.log(1 - torch.rand(num_rand, dim + 1)).to(device)
        weights = weights / weights.sum(dim=1, keepdim=True)
    return weights
