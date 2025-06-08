"""Implementation of the triton kernel.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flood_kernel(
    x_ptr,  # pointer to x, shape (S, R, d)
    y_ptr,  # pointer to y, shape (W, d)
    s_idx_ptr,
    w_idx_ptr,
    inter_ptr,  # pointer to intermediate output
    R,  # total number of rows per sample in x
    W,  # number of y vectors
    d: tl.constexpr,  # feature dimension
    BLOCK_R: tl.constexpr,  # block size (tile size) for R dimension (must divide R)
    BLOCK_W: tl.constexpr,  # block size for the W dimension per tile
):
    pid_r = tl.program_id(0)  # tile index for R dimension
    pid_w = tl.program_id(1)  # tile index for W dimension
    id_s = tl.load(s_idx_ptr + pid_w)

    w_idx = tl.load(w_idx_ptr + pid_w * BLOCK_W + tl.arange(0, BLOCK_W))
    x_idx = id_s * R * d + pid_r * BLOCK_R * d + tl.arange(0, BLOCK_R) * d

    # Initialize the squared-distance accumulator for this (BLOCK_R x BLOCK_W) tile.
    dist2 = tl.zeros((BLOCK_R, BLOCK_W), dtype=tl.float32)
    for i in range(d):
        x_vals = tl.load(x_ptr + x_idx + i)
        y_vals = tl.load(y_ptr + w_idx * d + i, mask=(w_idx < W), other=float("inf"))
        diff = x_vals[:, None] - y_vals[None, :]
        dist2 += diff * diff

    # Use tl.min with axis=1 to compute the minimum along the BLOCK_W (tile) dimension.
    tile_min = tl.sqrt(tl.min(dist2, axis=1))
    tile_min = (tile_min * 1e6).to(tl.int32)  # cast to int

    tl.atomic_min(
        inter_ptr + id_s * R + pid_r * BLOCK_R + tl.arange(0, BLOCK_R), tile_min
    )


def flood_triton_filtered(
    x: torch.Tensor,
    y: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    BLOCK_W,
    BLOCK_R,
) -> torch.Tensor:
    S, R, d = x.shape
    W, d_y = y.shape
    num_valid = col_idx.shape[0]
    assert d == d_y, "Feature dimensions of x and y must match."

    T = num_valid // BLOCK_W  # Number of tiles along the W dimension.
    R_tiles = R // BLOCK_R  # Number of tiles in the R dimension.

    # Allocate an intermediate tensor of shape (S, R) on the GPU.
    inter = torch.full((S, R), 2e9, device=x.device, dtype=torch.int32)

    try:
        grid = lambda meta: (R_tiles, T)
        row_idx = row_idx + 0  # this is needed
        flood_kernel[grid](
            x, y, row_idx, col_idx, inter, R, W, d, BLOCK_R=BLOCK_R, BLOCK_W=BLOCK_W
        )
    except RuntimeError:
        raise RuntimeError(
            "Memory error in CUDA, try lowering the batch size or setting disable_kernel=True"
        )

    out, idx = (inter.to(torch.float32) / 1e6).max(dim=1)
    return out, idx
