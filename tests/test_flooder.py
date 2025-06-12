import torch
import gudhi
import pytest
import numpy as np

from flooder import (
    flood_complex,
    generate_figure_eight_2D_points,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("disable_kernel", [True, False])
@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("N", [64 * 16, 64 * 32, 64 * 64])
def test_vs_alpha_1(disable_kernel, batch_size, N):
    """
    Test the homotopy equivalence of the Alpha complex and the Flood complex
    when landmarks L are set equal to the dataset X.clear
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X = generate_figure_eight_2D_points(1000)
    L = X

    X = X.to(DEVICE)
    L = L.to(DEVICE)

    # Test w and w/o kernel
    fc = flood_complex(
        L, X, dim=2, N=N, batch_size=batch_size, disable_kernel=disable_kernel
    )

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

    for dim in range(2):
        dist = gudhi.bottleneck_distance(
            flood_complex_diags[dim], alpha_complex_diags[dim]
        )
        assert dist < 1e-3, (
            f"Bottleneck distance too high in dimension {dim} "
            f"with disable_kernel={disable_kernel}: {dist}"
        )

    assert (
        gudhi.bottleneck_distance(flood_complex_diags[0], alpha_complex_diags[0]) < 1e-3
    )
    assert (
        gudhi.bottleneck_distance(flood_complex_diags[1], alpha_complex_diags[1]) < 1e-3
    )
