"""Example 03: Flood PH of a noisy figure-eight sample (1M points)"""

import time
import gudhi
import torch
import numpy as np

from flooder import (
    generate_landmarks,
    flood_complex,
    generate_figure_eight_2D_points,
    save_to_disk,
)

device = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def top_k_longest(bd: np.array, k: int = 10):
    """
    Sort (N,2) array of persistence (birth, death) pairs by lifetime

    Parameters
    ----------
    bd : np.array
        Array of shape (N, 2) containing birth and death times.
    k : int
        Number of longest bars to return.
    Returns
    -------
    np.array
        Array of shape (k, 2) containing the k longest bars sorted by lifetime.
    """
    lifetimes = bd[:, 1] - bd[:, 0]
    idx = np.argsort(lifetimes)[-k:][::-1]
    return bd[idx]


def main():

    torch.cuda.reset_peak_memory_stats()

    N_w = 40_000_000  # Number of points sampled from figure-eight
    N_l = 2000  # Number of landmarks for Flood complex

    print(f"{YELLOW}Flood PH of a noisy figure-eight sample 40M points)")
    print(f"{YELLOW}---------------------------------------------------")

    pts = generate_figure_eight_2D_points(
        N_w, noise_std=0.02, noise_kind="gaussian", rng=42
    )

    t0_fps = time.perf_counter()
    lms = generate_landmarks(pts, N_l)
    t1_fps = time.perf_counter()

    lms = lms.to(device)
    pts = pts.to(device)

    t0_complex = time.perf_counter()
    out_complex = flood_complex(lms, pts, batch_size=8)
    t1_complex = time.perf_counter()

    t0_ph = time.perf_counter()
    st = gudhi.SimplexTree()
    for simplex in out_complex:
        st.insert(simplex, out_complex[simplex])
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    t1_ph = time.perf_counter()

    print(
        f"{BLUE}{N_w:8d} points ({N_l} landmarks) | "
        f"Complex (Flood): {(t1_complex - t0_complex):6.2f} sec | "
        f"PH (Flood): {t1_ph - t0_ph:6.2f} sec | "
        f"FPS: {t1_fps - t0_fps:6.2f} sec{RESET}"
    )

    diags = [st.persistence_intervals_in_dimension(i) for i in range(2)]
    for i in range(2):
        print(f"{RED}10 longest bars (sorted by lifetime) in dimension {i}: {RESET}")
        topk_Hi = top_k_longest(diags[i], k=10)
        for j, (b, d) in enumerate(topk_Hi):
            print(
                f"{BLUE}  {j + 1:2d}: (birth, death)=({b:.4f}, {d:.4f}), lifetime={(d - b):.4f} {RESET}"
            )

    # Save the output to disk
    save_to_disk(
        {
            "pts": pts.cpu().numpy(),
            "lms": lms.cpu().numpy(),
            "complex": out_complex,
            "diags": diags,
        },
        "/tmp/example_03_figure_eight_2D_out.pt",
        True,
        True,
    )
    print(
        f"{RED}Peak memory: {torch.cuda.max_memory_allocated() / 1024**2} MiB {RESET}"
    )


if __name__ == "__main__":
    main()
