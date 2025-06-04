import torch
import numpy as np
import gudhi
import matplotlib.pyplot as plt

from flooder import (
    flood_complex,
    generate_landmarks,
    generate_noisy_torus_points,
    generate_figure_eight_2D_points,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vs_alpha_1():
    """Test case of homotopy equivalence for Alpha and Flood(X,X), i.e., when L=X"""
    X = generate_figure_eight_2D_points(1000)
    L = X

    X = X.to(DEVICE)
    L = L.to(DEVICE)

    fc = flood_complex(L, X, dim=2, N=1024, batch_size=16, disable_kernel=False)

    st = gudhi.SimplexTree()
    for simplex in fc:
        st.insert(simplex, fc[simplex])
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    flood_complex_diags = [st.persistence_intervals_in_dimension(i) for i in range(2)]

    alpha_complex = gudhi.AlphaComplex(X.cpu().numpy()).create_simplex_tree(
        output_squared_values=False
    )
    alpha_complex.compute_persistence()
    alpha_complex_diags = [
        alpha_complex.persistence_intervals_in_dimension(i) for i in range(2)
    ]

    assert (
        gudhi.bottleneck_distance(flood_complex_diags[0], alpha_complex_diags[0]) < 1e-3
    )
    assert (
        gudhi.bottleneck_distance(flood_complex_diags[1], alpha_complex_diags[1]) < 1e-3
    )
