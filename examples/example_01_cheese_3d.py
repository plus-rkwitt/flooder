"""Example 01: Runtime measurements for Alpha PH vs. Flood PH on 3D cheese data.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

from timeit import default_timer as timer

import torch
from gudhi import AlphaComplex, SimplexTree  # pylint: disable=no-name-in-module

from flooder import generate_swiss_cheese_points, flood_complex

DEVICE = torch.device("cuda")

# Custom colors for terminal output
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():  # pylint: disable=missing-function-docstring
    n_ps = [10000, 100000, 1000000, 10000000]  # Number of flood sources / data points
    n_l = 1000  # Number of landmarks to use
    b_sizes = [1024, 1024, 32, 2]  # Batch sizes for flood complex computation

    rect_min = [0.0, 0.0, 0.0]
    rect_max = [1.0, 1.0, 1.0]
    void_radius_range = (0.1, 0.2)
    k = 6  # Number of voids
    dim = len(rect_min)

    results = []
    pdiagram_flood_s = []
    pdiagram_alpha_s = []

    print(f"{YELLOW}Alpha PH vs. Flood PH timing on cheese")
    print(f"{YELLOW}--------------------------------------")
    for i, n_p in enumerate(n_ps):
        for rep in range(5):
            points, _, _ = generate_swiss_cheese_points(
                n_p, rect_min, rect_max, k, void_radius_range, device=DEVICE
            )

            startt = timer()
            alpha = AlphaComplex(points).create_simplex_tree(
                output_squared_values=False
            )
            t1 = timer() - startt

            alpha.compute_persistence()
            t2 = timer() - startt
            print(
                f"{RED}{n_p:8d} points (try {rep}) | "
                f"Complex (Alpha): {t1:6.2f} sec | "
                f"PH (Alpha): {t2:6.2f} sec{RESET}"
            )
            results.append(
                {"rep": rep, "N_p": n_p, "method": "Alpha", "tA": t1, "tB": t2}
            )

            pdiagram_alpha_s.append(alpha.persistence_intervals_in_dimension(dim - 1))

            points = points.to(DEVICE)
            # GPU warmup
            out_complex = flood_complex(n_l, points[:10000], batch_size=b_sizes[i])
            torch.cuda.synchronize()

            startt = timer()
            out_complex = flood_complex(n_l, points, batch_size=b_sizes[i])

            st = SimplexTree()
            for simplex in out_complex:
                st.insert(simplex, out_complex[simplex])
            st.make_filtration_non_decreasing()
            torch.cuda.synchronize()
            t1 = timer() - startt

            st.compute_persistence()
            t2 = timer() - startt
            print(
                f"{BLUE}{n_p:8d} points (try {rep}) | "
                f"Complex (Flood): {t1:6.2f} sec | "
                f"PH (Flood): {t2:6.2f} sec{RESET}"
            )
            results.append(
                {
                    "rep": rep,
                    "n_p": n_p,
                    "method": "Flood",
                    "complex_time": t1,
                    "ph_time": t2,
                }
            )
            pdiagram_flood_s.append(st.persistence_intervals_in_dimension(dim - 1))


if __name__ == "__main__":
    main()
