"""Example 02: Flood PH of a noisy torus sample (1M points)

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import time
import torch
import gudhi
import pandas as pd

from flooder import generate_noisy_torus_points_3d, flood_complex, generate_landmarks


DEVICE = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():  # pylint: disable=missing-function-docstring
    print(f"{YELLOW}Flood PH of a noisy torus sample (1M points)")
    print(f"{YELLOW}--------------------------------------------")
    results = []
    for rep in range(5):  # Repeat the experiment 5 times
        n_pts = 1_000_000  # Number of points sampled from torus
        n_lms = 2000  # Number of landmarks for Flood complex

        pts = generate_noisy_torus_points_3d(n_pts)

        t0_fps = time.perf_counter()
        lms = generate_landmarks(pts, n_lms)
        t1_fps = time.perf_counter()

        # GPU warmup
        out_complex = flood_complex(
            pts[:10000].to(DEVICE), lms.to(DEVICE), use_triton=True
        )
        torch.cuda.synchronize()

        t0_complex = time.perf_counter()
        out_complex = flood_complex(
            pts.to(DEVICE), lms.to(DEVICE), batch_size=64, use_triton=True
        )
        t1_complex = time.perf_counter()

        t0_ph = time.perf_counter()
        st = gudhi.SimplexTree()  # pylint: disable=no-member
        for simplex, filtration_value in out_complex.items():
            st.insert(simplex, filtration_value)
        st.make_filtration_non_decreasing()
        st.compute_persistence()
        t1_ph = time.perf_counter()

        print(
            f"{BLUE}{n_pts:8d} points ({n_lms} landmarks) | "
            f"Complex (Flood): {(t1_complex - t0_complex):6.2f} sec | "
            f"PH (Flood): {t1_ph - t0_ph:6.2f} sec | "
            f"FPS: {t1_fps - t0_fps:6.2f} sec{RESET}"
        )

        results.append(
            {
                "rep": rep,
                "n_pts": n_pts,
                "n_lms": n_lms,
                "method": "Flood",
                "complex_time": t1_complex - t0_complex,
                "fps_time": t1_fps - t0_fps,
                "ph_time": t1_ph - t0_ph,
            }
        )

    df = pd.DataFrame(results)
    summary = (
        df.groupby(["n_pts", "method"])
        .agg(
            fps_time_mean=("fps_time", "mean"),
            fps_time_std=("fps_time", "std"),
            complex_time_mean=("complex_time", "mean"),
            complex_time_std=("complex_time", "std"),
            ph_time_mean=("ph_time", "mean"),
            ph_time_std=("ph_time", "std"),
        )
        .reset_index()
    )

    summary["FPS Time (s)"] = summary.apply(
        lambda row: f"{row['fps_time_mean']:.2f} ± {row['fps_time_std']:.2f}", axis=1
    )
    summary["Complex Time (s)"] = summary.apply(
        lambda row: f"{row['complex_time_mean']:.2f} ± {row['complex_time_std']:.2f}",
        axis=1,
    )
    summary["PH Time (s)"] = summary.apply(
        lambda row: f"{row['ph_time_mean']:.2f} ± {row['ph_time_std']:.2f}", axis=1
    )

    print(f"\n{YELLOW}Summary of Timings (mean ± std over 5 repetitions){RESET}")
    print(
        summary[
            ["n_pts", "method", "FPS Time (s)", "Complex Time (s)", "PH Time (s)"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
