"""Example 01: Runtime measurements for Alpha PH vs. Flood PH on 3D cheese data.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import time
import torch
import pandas as pd
from gudhi import AlphaComplex, SimplexTree  # pylint: disable=no-name-in-module

from flooder import generate_swiss_cheese_points, flood_complex

DEVICE = torch.device("cuda")

# Custom colors for terminal output
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():  # pylint: disable=missing-function-docstring
    n_pts_list = [
        10000,
        100000,
        1000000,
        10000000,
    ]  # Number of flood sources / data points
    n_lms = 1000  # Number of landmarks to use
    batch_sizes = [1024, 1024, 32, 2]  # Batch sizes for flood complex computation

    rect_min = (0.0, 0.0, 0.0)
    rect_max = (1.0, 1.0, 1.0)
    void_radius_range = (0.1, 0.2)
    k = 6  # Number of voids
    dim = len(rect_min)

    results = []
    pdiagram_flood_s = []
    pdiagram_alpha_s = []

    print(f"{YELLOW}Alpha PH vs. Flood PH timing on cheese")
    print(f"{YELLOW}--------------------------------------")
    for i, n_pts in enumerate(n_pts_list):
        for rep in range(5):
            points, _, _ = generate_swiss_cheese_points(
                n_pts, rect_min, rect_max, k, void_radius_range, device=DEVICE
            )

            startt = time.perf_counter()
            alpha = AlphaComplex(points).create_simplex_tree(
                output_squared_values=False
            )
            t1 = time.perf_counter() - startt

            alpha.compute_persistence()
            t2 = time.perf_counter() - startt
            print(
                f"{RED}{n_pts:8d} points (try {rep}) | "
                f"Complex (Alpha): {t1:6.2f} sec | "
                f"PH (Alpha): {t2:6.2f} sec{RESET}"
            )
            results.append(
                {
                    "rep": rep,
                    "n_pts": n_pts,
                    "method": "Alpha",
                    "complex_time": t1,
                    "ph_time": t2,
                }
            )

            pdiagram_alpha_s.append(alpha.persistence_intervals_in_dimension(dim - 1))

            points = points.to(DEVICE)
            # GPU warmup
            out_complex = flood_complex(
                points[:10000], n_lms, batch_size=batch_sizes[i]
            )
            torch.cuda.synchronize()

            startt = time.perf_counter()
            out_complex = flood_complex(points, n_lms, batch_size=batch_sizes[i])

            st = SimplexTree()
            for simplex in out_complex:
                st.insert(simplex, out_complex[simplex])
            st.make_filtration_non_decreasing()
            torch.cuda.synchronize()
            t1 = time.perf_counter() - startt

            st.compute_persistence()
            t2 = time.perf_counter() - startt
            print(
                f"{BLUE}{n_pts:8d} points (try {rep}) | "
                f"Complex (Flood): {t1:6.2f} sec | "
                f"PH (Flood): {t2:6.2f} sec{RESET}"
            )
            results.append(
                {
                    "rep": rep,
                    "n_pts": n_pts,
                    "method": "Flood",
                    "complex_time": t1,
                    "ph_time": t2,
                }
            )
            pdiagram_flood_s.append(st.persistence_intervals_in_dimension(dim - 1))

    df = pd.DataFrame(results)
    summary = (
        df.groupby(["n_pts", "method"])
        .agg(
            complex_mean=("complex_time", "mean"),
            complex_std=("complex_time", "std"),
            ph_mean=("ph_time", "mean"),
            ph_std=("ph_time", "std"),
        )
        .reset_index()
    )

    summary["Complex Time (s)"] = summary.apply(
        lambda row: f"{row['complex_mean']:.2f} ± {row['complex_std']:.2f}", axis=1
    )
    summary["PH Time (s)"] = summary.apply(
        lambda row: f"{row['ph_mean']:.2f} ± {row['ph_std']:.2f}", axis=1
    )
    print(f"\n{YELLOW}Summary of Timings (mean ± std over 5 repetitions){RESET}")
    print(
        summary[["n_pts", "method", "Complex Time (s)", "PH Time (s)"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
