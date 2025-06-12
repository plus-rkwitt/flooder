"""Example 02: Flood PH of a noisy torus sample (1M points)"""

import time
import torch
import gudhi

from flooder import (
    generate_noisy_torus_points,
    flood_complex,
    generate_landmarks,
    save_to_disk,
)


OUT_DIR = "/tmp/"
OUT_FILE = "flooder_example_02_torus_3D_diagrams.pt"
DEVICE = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    print(f"{YELLOW}Flood PH of a noisy torus sample (1M points)")
    print(f"{YELLOW}--------------------------------------------")

    N_w = 1000000  # Number of points sampled from torus
    N_l = 1000  # Number of landmarks for Flood complex

    pts = generate_noisy_torus_points(N_w)

    t0_fps = time.perf_counter()
    lms = generate_landmarks(pts, N_l)
    t1_fps = time.perf_counter()

    t0_complex = time.perf_counter()
    out_complex = flood_complex(lms.to(DEVICE), pts.to(DEVICE), dim=3, batch_size=16)
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
        f"Complex (Flood): {(t1_complex-t0_complex):6.2f} sec | "
        f"PH (Flood): {t1_ph-t0_ph:6.2f} sec | "
        f"FPS: {t1_fps-t0_fps:6.2f} sec{RESET}"
    )

    diags = [st.persistence_intervals_in_dimension(d) for d in range(3)]

    save_to_disk(
        {
            "pts": pts.cpu().numpy(),
            "lms": lms.cpu().numpy(),
            "complex": out_complex,
            "diags": diags,
        },
        "/tmp/example_02_torus_3D_out.pt",
    )


if __name__ == "__main__":
    main()
