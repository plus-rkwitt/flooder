import torch
import gudhi
import pytest
import numpy as np

from flooder import (
    flood_complex,
    generate_figure_eight_2D_points,
    generate_noisy_torus_points,
    generate_landmarks,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("disable_kernel", [True, False])
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("N", [64 * 16, 64 * 32])
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



@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("num_witnesses", [1000, 10_000, 50_000])
@pytest.mark.parametrize("num_landmarks", [20, 701, 1000, 2000]) 
def test_cpu_vs_gpu(batch_size, num_witnesses, num_landmarks):
    """
    Test consistency of the Flood complex between CPU and GPU computation
    for different batch sizes, number of witnesses and number of landmarks. 
    Tests also number of landmars being equal or larger than number of witnesses.
    """

    assert DEVICE.type == 'cuda'

    X = generate_noisy_torus_points(num_witnesses).to(DEVICE)
    L = generate_landmarks(X, num_landmarks)

    # Test w kernel 
    torch.manual_seed(42)
    np.random.seed(42)
    fc_cuda = flood_complex(
        L, X, dim=3, batch_size=batch_size
    )
    st_cuda = gudhi.SimplexTree()
    for simplex in fc_cuda:
        st_cuda.insert(simplex, fc_cuda[simplex])
    st_cuda.make_filtration_non_decreasing()
    st_cuda.compute_persistence()
    cuda_diags = [st_cuda.persistence_intervals_in_dimension(i) for i in range(2)]

    # Test w/o kernel
    torch.manual_seed(42)
    np.random.seed(42)
    fc_cpu = flood_complex(
        L, X, dim=3, batch_size=batch_size, disable_kernel=True
    )
    st_cpu = gudhi.SimplexTree()
    for simplex in fc_cpu:
        st_cpu.insert(simplex, fc_cpu[simplex])
    st_cpu.make_filtration_non_decreasing()
    st_cpu.compute_persistence()
    cpu_diags = [st_cpu.persistence_intervals_in_dimension(i) for i in range(2)]

    
    for simplex in fc_cpu:
        assert simplex in fc_cuda
        assert abs(fc_cpu[simplex] - fc_cuda[simplex]) < 1e-4, \
        f"Simplex {simplex}: CPU {fc_cpu[simplex]:.5f} and CUDA {fc_cuda[simplex]:.5f}"

