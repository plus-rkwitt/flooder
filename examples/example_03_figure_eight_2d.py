"""Example 03: Flood PH of a noisy figure-eight sample (1M points)

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import time
import gudhi
import torch
import numpy as np

from flooder import generate_landmarks, flood_complex, generate_figure_eight_points_2d

device = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def top_k_longest(bd: np.array, k: int = 10):
    """Return the top-k longest persistence bars based on lifetime.

    Sorts a (N, 2) array of (birth, death) pairs by their lifetime (death - birth)
    and returns the `k` longest bars.

    Args:
        bd (np.array): A NumPy array of shape (N, 2) containing birth and death times.
        k (int, optional): Number of longest bars to return. Defaults to 10.

    Returns:
        np.array: A NumPy array of shape (k, 2) containing the `k` longest bars,
                sorted in descending order of lifetime.
    """
    lifetimes = bd[:, 1] - bd[:, 0]
    idx = np.argsort(lifetimes)[-k:][::-1]
    return bd[idx]


def main():  # pylint: disable=missing-function-docstring
    n_pts = 40_000_000  # Number of points sampled from figure-eight
    n_lms = 2000  # Number of landmarks for Flood complex

    print(f"{YELLOW}Flood PH of a noisy figure-eight sample 40M points)")
    print(f"{YELLOW}---------------------------------------------------")

    pts = generate_figure_eight_points_2d(n_pts, noise_std=0.02, noise_kind="gaussian")

    t0_fps = time.perf_counter()
    lms = generate_landmarks(pts, n_lms)
    t1_fps = time.perf_counter()

    lms = lms.to(device)
    pts = pts.to(device)

    t0_complex = time.perf_counter()
    out_complex = flood_complex(pts, lms, batch_size=8)
    t1_complex = time.perf_counter()

    t0_ph = time.perf_counter()
    st = gudhi.SimplexTree()  # pylint: disable=no-member
    for simplex in out_complex:
        st.insert(simplex, out_complex[simplex])
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    t1_ph = time.perf_counter()

    print(
        f"{BLUE}{n_pts:8d} points ({n_lms} landmarks) | "
        f"Complex (Flood): {(t1_complex - t0_complex):6.2f} sec | "
        f"PH (Flood): {t1_ph - t0_ph:6.2f} sec | "
        f"FPS: {t1_fps - t0_fps:6.2f} sec{RESET}"
    )

    diags = [st.persistence_intervals_in_dimension(i) for i in range(2)]
    for i in range(2):
        print(f"{RED}10 longest bars (sorted by lifetime) in dimension {i}: {RESET}")
        topk_hi = top_k_longest(diags[i], k=10)
        for j, (b, d) in enumerate(topk_hi):
            print(
                f"{BLUE}  {j + 1:2d}: (birth, death)=({b:.4f}, {d:.4f}), \
                    lifetime={(d - b):.4f} {RESET}"
            )


if __name__ == "__main__":
    main()
