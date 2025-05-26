import torch
import gudhi
import fpsample
from math import sqrt

from .triton_kernel_flood_filtered import flood_triton_filtered

BLOCK_W = 64
BLOCK_R = 64


def generate_landmarks(points, N_l):
    import numpy as np

    index_set = torch.tensor(
        fpsample.bucket_fps_kdline_sampling(points.cpu(), N_l, h=5).astype(np.int64)
    ).to(points.device)
    return points[index_set]


def flood_complex(
    landmarks,
    witnesses,
    dim=1,
    N=512,
    batch_size=32,
    BATCH_MULT=32,
    disable_kernel=False,
    do_second_stage=False,
):
    RADIUS_FACTOR = 1.4

    max_range_dim = torch.argmax(
        witnesses.max(dim=0).values - witnesses.min(dim=0).values
    ).item()
    witnesses = witnesses[torch.argsort(witnesses[:, max_range_dim])].contiguous()
    witnesses_search = witnesses[:, max_range_dim].contiguous()

    if type(landmarks) == int:
        landmarks = generate_landmarks(witnesses, min(landmarks, witnesses.shape[0]))
    resolution = torch.cdist(landmarks[-1:], landmarks[:-1]).min().item()
    resolution = 9.0 * resolution * resolution + 1e-3

    dc = gudhi.AlphaComplex(landmarks).create_simplex_tree()

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

        # precompute random weights
        num_rand = N
        weights = -torch.log(torch.rand(num_rand, d + 1, device=landmarks.device))
        weights = weights / weights.sum(dim=1, keepdim=True)
        all_random_points = weights.unsqueeze(0) @ all_simplex_points
        del weights

        if disable_kernel == True or (not landmarks.is_cuda):
            # naive
            for i, simplex in enumerate(list_simplexes[d - 1]):
                valid_witnesses_mask = (
                    torch.cdist(simplex_centers_vec[i : i + 1], witnesses)
                    < simplex_radii_vec[i] + 1e-3
                )
                dists_valid = torch.cdist(
                    all_random_points[i], witnesses[valid_witnesses_mask[0]]
                )
                out_complex[tuple(simplex)] = torch.min(dists_valid, dim=1).values.max()
        else:
            # triton kernel
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
