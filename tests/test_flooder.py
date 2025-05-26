import torch
import numpy as np
import gudhi
import matplotlib.pyplot as plt

from flooder import flood_complex, generate_landmarks, generate_noisy_torus_points


def test_flooder():
    N_w = 1000000
    N_l = 1000
    rect_min = torch.tensor([0.0, 0.0, 0.0])
    rect_max = torch.tensor([5.0, 5.0, 5.0])
    void_radius_range = (0.1, 0.5)
    k = 10

    pts = generate_noisy_torus_points(N_w)
    lms = generate_landmarks(pts, N_l)

    device = torch.device("cuda")
    out_complex = flood_complex(lms.to(device), pts.to(device), dim=3, batch_size=16)

    st = gudhi.SimplexTree()
    for simplex in out_complex:
        st.insert(simplex, out_complex[simplex])
    # print([len([i for i in st.get_simplices() if len(i[0]) == k]) for k in range(0, 6)])
    st.make_filtration_non_decreasing()
    st.compute_persistence()

    pdiagram_land0 = st.persistence_intervals_in_dimension(0)
    pdiagram_land1 = st.persistence_intervals_in_dimension(1)
    pdiagram_land2 = st.persistence_intervals_in_dimension(2)
