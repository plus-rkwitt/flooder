"""Example 01: Runtime measurements for Alpha PH vs. Flood PH on 3D cheese data."""

from timeit import default_timer as timer

import torch
import numpy as np
from gudhi import AlphaComplex, SimplexTree

from flooder import generate_swiss_cheese_points, flood_complex, save_to_disk

DEVICE = torch.device("cuda")
OUT_DIR = "/tmp/"
OUT_FILE = "flooder_cheese_timing_for_varying_number_of_points.pt"

# Custom colors for terminal output
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    n_ws = [10000, 100000, 1000000, 10000000]  # Number of flood sources / data points
    n_l = 1000  # Number of landmarks to use
    b_sizes = [1024, 1024, 32, 2]  # Batch sizes for flood complex computation

    rect_min = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rect_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    void_radius_range = (0.1, 0.2)
    k = 6  # Number of voids
    dim = 3  # Swiss cheese dimension

    results = []
    pdiagram_land_flood_s = []
    pdiagram_land_alpha_s = []

    print(f"{YELLOW}Alpha PH vs. Flood PH timing on cheese")
    print(f"{YELLOW}--------------------------------------")
    for i, n_w in enumerate(n_ws):
        for rep in range(5):
            points, _ = generate_swiss_cheese_points(
                n_w, rect_min[:dim], rect_max[:dim], k, void_radius_range
            )

            startt = timer()
            alpha = AlphaComplex(points).create_simplex_tree()
            t1 = timer() - startt

            alpha.compute_persistence()
            t2 = timer() - startt
            print(
                f"{RED}{n_w:8d} points (try {rep}) | "
                f"Complex (Alpha): {t1:6.2f} sec | "
                f"PH (Alpha): {t2:6.2f} sec{RESET}"
            )
            results.append(
                {"rep": rep, "W": n_w, "method": "Alpha", "tA": t1, "tB": t2}
            )

            pdiagram_land2_alpha = np.sqrt(
                alpha.persistence_intervals_in_dimension(dim - 1)
            )
            pdiagram_land_alpha_s.append(pdiagram_land2_alpha)

            points = points.to(DEVICE)
            # GPU warmup
            out_complex = flood_complex(
                n_l, points[:10000], batch_size=b_sizes[i]
            )
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
                f"{BLUE}{n_w:8d} points (try {rep}) | "
                f"Complex (Flood): {t1:6.2f} sec | "
                f"PH (Flood): {t2:6.2f} sec{RESET}"
            )
            results.append(
                {"rep": rep, "W": n_w, "method": "Flood", "tA": t1, "tB": t2}
            )

            pdiagram_land2 = st.persistence_intervals_in_dimension(dim - 1)
            pdiagram_land_flood_s.append(pdiagram_land2)

    save_to_disk(
        {
            "results": results,
            "pdiagram_land_flood_s": pdiagram_land_flood_s,
            "pdiagram_land_alpha_s": pdiagram_land_alpha_s,
            "n_ws": n_ws,
            "n_l": n_l,
            "b_sizes": b_sizes,
            "rect_min": rect_min.cpu().numpy(),
            "rect_max": rect_max.cpu().numpy(),
        },
        "/tmp/example_01_cheese_3D_out.pt",
    )


if __name__ == "__main__":
    main()
