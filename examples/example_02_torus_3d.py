"""Example 02: Flood PH of a noisy torus sample (1M points)

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import time
import torch
import gudhi

from flooder import generate_noisy_torus_points, flood_complex, generate_landmarks


DEVICE = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():  # pylint: disable=missing-function-docstring
    print(f"{YELLOW}Flood PH of a noisy torus sample (1M points)")
    print(f"{YELLOW}--------------------------------------------")
    for _ in range(3):
        n_p = 1_000_000  # Number of points sampled from torus
        n_l = 2000  # Number of landmarks for Flood complex

        pts = generate_noisy_torus_points(n_p)

        t0_fps = time.perf_counter()
        lms = generate_landmarks(pts, n_l)
        t1_fps = time.perf_counter()

        t0_complex = time.perf_counter()
        out_complex = flood_complex(
            lms.to(DEVICE), pts.to(DEVICE), batch_size=64, use_triton=True
        )
        t1_complex = time.perf_counter()

        t0_ph = time.perf_counter()
        st = gudhi.SimplexTree()  # pylint: disable=no-member
        for simplex in out_complex:
            st.insert(simplex, out_complex[simplex])
        st.make_filtration_non_decreasing()
        st.compute_persistence()
        t1_ph = time.perf_counter()

        print(
            f"{BLUE}{n_p:8d} points ({n_l} landmarks) | "
            f"Complex (Flood): {(t1_complex - t0_complex):6.2f} sec | "
            f"PH (Flood): {t1_ph - t0_ph:6.2f} sec | "
            f"FPS: {t1_fps - t0_fps:6.2f} sec{RESET}"
        )


if __name__ == "__main__":
    main()
