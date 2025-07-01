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
    pid_w = tl.program_id(0)  # tile index for W dimension
    pid_r = tl.program_id(1)  # tile index for R dimension
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
    inter = torch.full((S, R), torch.inf, device=x.device, dtype=torch.float32)

    # Bounds check
    assert row_idx.shape ==  col_idx.shape, f"row_idx.shape ({row_idx.shape}) does not match col_idx.shape ({col_idx.shape}"
    assert col_idx.shape[0] == T * BLOCK_W, f"col_idx.shape[0] {col_idx.shape[0]} does not match T * BLOCK_W ({T} * {BLOCK_W} = {T * BLOCK_W})"    

    row_idx = row_idx[::BLOCK_W] # consecutive row_indices need to be constant in blocks of length BLOCK_W
    # make sure indexing is contiguous and of type int32 for triton
    row_idx = row_idx.to(torch.int32).contiguous()
    col_idx = col_idx.to(torch.int32).contiguous()

    """Important: run ./deviceQuery from 
    
    https://github.com/NVIDIA/cuda-samples/
    
    and check for "Max dimension size of a grid size" as the grid may become 
    too large for the GPU to handle. If this is the case, you can set
    disable_kernel=True in the flood_complex function to use the CPU fallback, or
    reduce the batch size.
    """

    try:
        x = x.contiguous().view(-1)  # Make sure indexing math (later) matches layout
        flood_kernel[(T, R_tiles)](
            x, y, row_idx, col_idx, inter, R, W, d, BLOCK_R=BLOCK_R, BLOCK_W=BLOCK_W
        )
    except RuntimeError:
        raise RuntimeError(
            "Memory/Grid size error in CUDA, try lowering the batch size or setting disable_kernel=True"
        )

    out, idx = inter.max(dim=1)
    return out, idx
