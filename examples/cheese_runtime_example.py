"""Example of Alpha PH vs. Flood PH on cheese data."""

import torch
import numpy as np
import gudhi
from timeit import default_timer as timer
import pandas as pd

from flooder import generate_swiss_cheese_points, flood_complex, generate_landmarks

device = torch.device("cuda")
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


N_ws = [10000, 100000]  # 1000000, 10000000]  # number of flood sources / data points
N_l = 1000  # number of landmarks
b_sizes = [1024, 1024, 32, 2]

rect_min = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
rect_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
void_radius_range = (0.1, 0.2)
k = 6  # number of voids
dim = 3  # swiss cheese dimension

results = []

pdiagram_land_flood_s = []
pdiagram_land_alpha_s = []

print(f"{YELLOW}Alpha PH vs. Flood PH timing on cheese")
print(f"{YELLOW}-----------------------------------")
for i, N_w in enumerate(N_ws):
    for rep in range(5):
        points, radii = generate_swiss_cheese_points(
            N_w, rect_min[:dim], rect_max[:dim], k, void_radius_range
        )

        startt = timer()
        alpha = gudhi.AlphaComplex(points).create_simplex_tree()
        t1 = timer() - startt

        alpha.compute_persistence()
        t2 = timer() - startt
        print(
            f"{RED}{N_w:8d} points (try {rep}) | "
            f"Complex (Alpha): {t1:6.2f} sec | "
            f"PH (Alpha): {t2:6.2f} sec{RESET}"
        )
        results.append({"rep": rep, "W": N_w, "method": "Alpha", "tA": t1, "tB": t2})

        pdiagram_land2_alpha = np.sqrt(
            alpha.persistence_intervals_in_dimension(dim - 1)
        )
        pdiagram_land_alpha_s.append(pdiagram_land2_alpha)

        points = points.to(device)
        out_complex = flood_complex(N_l, points[:10000], dim=3, batch_size=b_sizes[i])
        torch.cuda.synchronize()

        startt = timer()
        out_complex = flood_complex(N_l, points, dim=3, batch_size=b_sizes[i])

        st = gudhi.SimplexTree()
        for simplex in out_complex:
            st.insert(simplex, out_complex[simplex])
        st.make_filtration_non_decreasing()
        torch.cuda.synchronize()
        t1 = timer() - startt

        st.compute_persistence()
        t2 = timer() - startt
        print(
            f"{BLUE}{N_w:8d} points (try {rep}) | "
            f"Complex (Flood): {t1:6.2f} sec | "
            f"PH (Flood): {t2:6.2f} sec{RESET}"
        )
        results.append({"rep": rep, "W": N_w, "method": "Flood", "tA": t1, "tB": t2})

        pdiagram_land2 = st.persistence_intervals_in_dimension(dim - 1)
        pdiagram_land_flood_s.append(pdiagram_land2)


df = pd.DataFrame(results)
torch.save((df, pdiagram_land_flood_s, pdiagram_land_alpha_s), "cheese_w.pt")
