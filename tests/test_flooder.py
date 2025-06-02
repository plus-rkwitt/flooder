import torch
import numpy as np
import gudhi
import matplotlib.pyplot as plt

from flooder import flood_complex, generate_landmarks, generate_noisy_torus_points


def test_flooder():
    N_w = 1000000
    N_l = 1000

    pts = generate_noisy_torus_points(N_w)  # use default parameters for the noisy torus
    lms = generate_landmarks(pts, N_l)

    device = torch.device("cuda")
    out_complex = flood_complex(lms.to(device), pts.to(device), dim=3, batch_size=16)

    st = gudhi.SimplexTree()
    for simplex in out_complex:
        st.insert(simplex, out_complex[simplex])
    st.make_filtration_non_decreasing()
    st.compute_persistence()
