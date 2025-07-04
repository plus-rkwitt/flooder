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


@pytest.mark.parametrize("use_triton", [True, False])
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("N", [64 * 16, 64 * 32])
def test_vs_alpha_1(use_triton, batch_size, N):
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
        L, X, dim=2, N=N, batch_size=batch_size, use_triton=use_triton
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
            f"with use_triton={use_triton}: {dist}"
        )

    assert (
        gudhi.bottleneck_distance(flood_complex_diags[0], alpha_complex_diags[0]) < 1e-3
    )
    assert (
        gudhi.bottleneck_distance(flood_complex_diags[1], alpha_complex_diags[1]) < 1e-3
    )



@pytest.mark.parametrize("num_witnesses", [1000, 10_000])
@pytest.mark.parametrize("num_landmarks", [20, 701, 1000, 2000]) 
def test_triton(num_witnesses, num_landmarks):
    """
    Test consistency of the Flood complex between using and not using Triton
    kernels for different batch sizes, number of witnesses and number of landmarks.
    Tests also number of landmars being equal or larger than number of witnesses.
    """

    assert DEVICE.type == 'cuda'

    torch.manual_seed(42)
    np.random.seed(42)
    
    X = generate_noisy_torus_points(num_witnesses).to(DEVICE)
    L = generate_landmarks(X, num_landmarks)

    # Test w kernel 
    torch.manual_seed(42)
    np.random.seed(42)
    fc_triton = flood_complex(
        L, X, dim=3, batch_size=32, use_triton=True
    )

    # Test w/o kernel
    torch.manual_seed(42)
    np.random.seed(42)
    fc_no_triton = flood_complex(
        L, X, dim=3, batch_size=32, use_triton=False
    )
    
    for simplex in fc_no_triton:
        assert simplex in fc_triton
        assert abs(fc_no_triton[simplex] - fc_triton[simplex]) < 1e-4, \
        f"Simplex {simplex}: Naive {fc_no_triton[simplex]:.5f} and Triton {fc_triton[simplex]:.5f}"

@pytest.mark.parametrize("num_witnesses", [1000, 10_000])
@pytest.mark.parametrize("num_landmarks", [20, 701, 1000, 2000]) 
def test_kdtree_vs_triton(num_witnesses, num_landmarks):
    """
    Test consistency of the Flood complex between kdtree and Triton computation
    for different batch sizes, number of witnesses and number of landmarks. 
    Tests also number of landmars being equal or larger than number of witnesses.
    """

    assert DEVICE.type == 'cuda'

    torch.manual_seed(42)
    np.random.seed(42)
    
    X = generate_noisy_torus_points(num_witnesses).to(DEVICE)
    L = generate_landmarks(X, num_landmarks)

    # Test using triton kernel 
    torch.manual_seed(42)
    np.random.seed(42)
    fc_triton = flood_complex(
        L, X, dim=3, batch_size=32
    )

    # Test cpu version (kd-tree)
    torch.manual_seed(42)
    np.random.seed(42)
    fc_naive = flood_complex(
        L.cpu(), X.cpu(), dim=3, batch_size=32
    )
    
    for simplex in fc_naive:
        assert simplex in fc_triton
        assert abs(fc_naive[simplex] - fc_triton[simplex]) < 1e-4, \
        f"Simplex {simplex}: Naive {fc_naive[simplex]:.5f} and Triton {fc_triton[simplex]:.5f}"


@pytest.mark.parametrize("num_witnesses", [1000, 10_000])
@pytest.mark.parametrize("num_landmarks", [20, 1000]) 
@pytest.mark.parametrize("mode", ['CPU', 'dist', 'Triton']) 
@pytest.mark.parametrize("return_simplex_tree", [True, False]) 
def test_filtration_condition(num_witnesses, num_landmarks, mode, return_simplex_tree):
    """
    Test that the Flood complex is a filtered complex.
    """

    if mode == 'CPU':
        device = 'cpu'
        use_triton = False
    else:
        assert DEVICE.type == 'cuda'
        device = DEVICE
        if mode == 'dist':
            use_triton = False
        elif mode == 'Triton':
            use_triton = True
        else:
            raise RuntimeError('Mode not implemented')

    torch.manual_seed(42)
    np.random.seed(42)
    X = generate_noisy_torus_points(num_witnesses).to(device)
    L = generate_landmarks(X, num_landmarks)
    
    if return_simplex_tree == False:
        fc = flood_complex(
            L, X, dim=3, batch_size=32, use_triton=use_triton, return_simplex_tree=False
        )
        st = gudhi.SimplexTree()
        for simplex in fc:
            st.insert(simplex, float('inf'))
            st.assign_filtration(simplex, fc[simplex])
    else:
        st = flood_complex(
            L, X, dim=3, batch_size=32, use_triton=use_triton, return_simplex_tree=True
        )

    for simplex, filtration in st.get_simplices():
        faces = [ _ for _ in st.get_boundaries(simplex)]
        if len(simplex) > 1:
            assert len(faces) == len(simplex), f"Simplex {simplex} has {len(faces)} faces"
        else:
            assert len(simplex) == 1 and len(faces) == 0, f"Simplex {simplex} has {len(faces)} faces"
        
        for face, face_filtration in faces:
            assert face_filtration <= filtration, f"Simplex {simplex} has filtr. value {filtration:.5f} and its face {face} has {face_filtration:.5f}"


# test_kdtree_vs_triton(1000,100000)