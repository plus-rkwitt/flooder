"""Implementation of the Triton kernels.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import triton
import triton.language as tl


@triton.jit
def compute_filtration_kernel(
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

    r_offset = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offset < R
    x_idx = id_s * R * d + r_offset * d  # offset into x_ptr for each row
    w_idx = tl.load(w_idx_ptr + pid_w * BLOCK_W + tl.arange(0, BLOCK_W))

    # Initialize the squared-distance accumulator for this (BLOCK_R x BLOCK_W) tile.
    dist2 = tl.zeros((BLOCK_R, BLOCK_W), dtype=tl.float32)
    for i in range(d):
        x_vals = tl.load(x_ptr + x_idx + i, mask=r_mask, other=0.0)
        y_vals = tl.load(y_ptr + w_idx * d + i, mask=(w_idx < W), other=float("inf"))
        diff = x_vals[:, None] - y_vals[None, :]
        dist2 += diff * diff

    # Use tl.min with axis=1 to compute the minimum along the BLOCK_W (tile) dimension.
    tile_min = tl.sqrt(tl.min(dist2, axis=1))
    tl.atomic_min(inter_ptr + id_s * R + r_offset, tile_min, mask=r_mask)


def compute_filtration(
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
    R_tiles = triton.cdiv(R, BLOCK_R)
    # Number of tiles in the R dimension.

    # Allocate an intermediate tensor of shape (S, R) on the GPU.
    inter = torch.full((S, R), torch.inf, device=x.device, dtype=torch.float32)

    # Bounds check
    assert (
        row_idx.shape == col_idx.shape
    ), f"row_idx.shape ({row_idx.shape}) does not match col_idx.shape ({col_idx.shape}"
    assert (
        col_idx.shape[0] == T * BLOCK_W
    ), f"col_idx.shape[0] {col_idx.shape[0]} does not match T * BLOCK_W ({T} * {BLOCK_W} = {T * BLOCK_W})"

    row_idx = row_idx[
        ::BLOCK_W
    ]  # consecutive row_indices need to be constant in blocks of length BLOCK_W
    # make sure indexing is contiguous and of type int32 for triton
    row_idx = row_idx.to(torch.int32).contiguous()
    col_idx = col_idx.to(torch.int32).contiguous()

    try:
        x = x.contiguous().view(-1)  # Make sure indexing math (later) matches layout
        compute_filtration_kernel[(T, R_tiles)](
            x, y, row_idx, col_idx, inter, R, W, d, BLOCK_R=BLOCK_R, BLOCK_W=BLOCK_W
        )
    except RuntimeError:
        raise RuntimeError(
            "Memory/Grid size error in CUDA, try lowering the batch size or setting disable_kernel=True"
        )
    return inter


@triton.jit
def compute_mask_kernel(
    points_ptr,  # (m, d), row-major
    mask_ptr,  # (n, m), flat index
    counts_ptr,  # (n), Trues per row
    cent_ptr,  # (n, d), center positions
    radi_ptr,  # (n, 1) or (n,), radius
    n_32b,
    m_32b,
    d_32b,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_W: tl.constexpr,
):

    # According to https://triton-lang.org/main/python-api/triton-semantics.html, any
    # operation with where one tensor is of a dtype of a higher kind, the other tensor
    # is promoted to this dtype
    n = tl.full((), n_32b, tl.int64)
    m = tl.full((), m_32b, tl.int64)
    d = tl.full((), d_32b, tl.int64)

    pid_m = tl.program_id(0)  # points
    pid_n = tl.program_id(1)  # centers

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offs_n < n
    mask_m = offs_m < m

    pt_stride = d
    cent_stride = d

    radi = tl.load(radi_ptr + offs_n, mask=mask_n, other=0.0)
    sq_radi = radi * radi  # [BLOCK_N]

    sq_dist = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(d):
        pt_i = tl.load(
            points_ptr + offs_m * pt_stride + i, mask=mask_m, other=0.0
        )  # [BLOCK_M]
        cent_i = tl.load(
            cent_ptr + offs_n * cent_stride + i, mask=mask_n, other=0.0
        )  # [BLOCK_N]
        diff_i = pt_i[None, :] - cent_i[:, None]  # [BLOCK_N, BLOCK_M]
        sq_dist += diff_i * diff_i  # [BLOCK_N, BLOCK_M]

    inside = sq_dist <= sq_radi[:, None]  # [BLOCK_N, BLOCK_M]

    stride_mask = m + BLOCK_W
    out_idx = offs_n[:, None] * stride_mask + offs_m[None, :]
    write_mask = (offs_n[:, None] < n) & (offs_m[None, :] < m)  # [BLOCK_N, BLOCK_M]

    tl.store(mask_ptr + out_idx, inside, mask=write_mask)
    counts_tile = tl.sum((inside * write_mask).to(tl.int32), axis=1)  # [BLOCK_N]

    # Atomically add counts_tile to global counts_ptr at offsets offs_n
    tl.atomic_add(counts_ptr + offs_n, counts_tile, mask=mask_n)


def compute_mask(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    BLOCK_N,
    BLOCK_M,
    BLOCK_W,
) -> torch.Tensor:
    """
    Check which points are inside Euclidean balls with given radii.

    Args:
        points (torch.Tensor):
            Tensor of shape (m, d), tensor with points to test.
        centers (torch.Tensor):
            Tensor of shape (n, d), tensor with centers of balls.
        radii (torch.Tensor):
            Tensor of shape (n, d), tensor with radii of balls.
        BLOCK_N (int):
            Block size along balls axis
        BLOCK_M (int):
            Block size along points axis
        BLOCk_W (int):
            Only used for padding

    Returns:
        mask (torch.Tensor):
            Boolean tensor of shape (n, m + BLOCK_W), mask[i,j] = True if points[j] inside simplices[i].
            Last (n, BLOCK_W) block is padded so that number of Trues per row is divisible by BLOCK_W
    """
    n, d = centers.shape
    m = points.shape[0]

    centers_flat = centers.view(n, -1).contiguous()
    radii_flat = radii.view(-1).contiguous()
    mask = torch.zeros((n, m + BLOCK_W), dtype=torch.bool, device=points.device)
    counts = torch.zeros(n, dtype=torch.int32, device=points.device)

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    compute_mask_kernel[grid](
        points,
        mask,
        counts,
        centers_flat,
        radii_flat,
        n,
        m,
        d,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_W=BLOCK_W,
    )
    extra = ((-counts) % BLOCK_W).unsqueeze(1)  # [n, 1]
    extra_range = torch.arange(BLOCK_W, device=counts.device).unsqueeze(
        0
    )  # [1, BLOCK_W]
    mask[:, m : m + BLOCK_W] = extra_range < extra  # [n, BLOCK_W]
    return mask
