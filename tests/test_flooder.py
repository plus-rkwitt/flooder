"""Test cases for the Flooder library, which implements the Flood complex.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import gudhi
import pytest
import numpy as np

from flooder import (
    flood_complex,
    generate_figure_eight_points_2d,
    generate_noisy_torus_points_3d,
    generate_landmarks,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("use_triton", [True, False])
@pytest.mark.parametrize("batch_size", [8, 23])
@pytest.mark.parametrize("use_rand", [True, False])
def test_vs_alpha(use_triton, batch_size, use_rand):
    """
    Test the homotopy equivalence of the Alpha complex and the Flood complex
    when landmarks L are set equal to the dataset X.clear
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X = generate_figure_eight_points_2d(1000)
    L = X

    X = X.to(DEVICE)
    L = L.to(DEVICE)
    num_rand = 20_000
    points_per_edge = 130
    if use_rand:
        kwargs = {"num_rand": num_rand, "points_per_edge": None}
    else:
        kwargs = {"num_rand": None, "points_per_edge": points_per_edge}

    stree = flood_complex(
        X,
        L,
        use_triton=use_triton,
        return_simplex_tree=True,
        batch_size=batch_size,
        **kwargs,
    )
    stree.compute_persistence()
    flood_complex_diags = [
        stree.persistence_intervals_in_dimension(i) for i in range(2)
    ]

    alpha_complex = gudhi.AlphaComplex(  # pylint: disable=no-member
        X.cpu().numpy()
    ).create_simplex_tree(output_squared_values=False)
    alpha_complex.compute_persistence()
    alpha_complex_diags = [
        alpha_complex.persistence_intervals_in_dimension(i) for i in range(2)
    ]

    for dim in range(2):
        dist = gudhi.bottleneck_distance(  # pylint: disable=no-member
            flood_complex_diags[dim], alpha_complex_diags[dim]
        )
        assert dist < 5e-4, (
            f"Bottleneck distance too high in dimension {dim} "
            f"with use_rand={use_rand} and use_triton={use_triton}: {dist}"
        )


@pytest.mark.parametrize("num_witnesses", [1000, 10_000])
@pytest.mark.parametrize("num_landmarks", [20, 701, 2000])
@pytest.mark.parametrize("use_rand", [True, False])
def test_triton(num_witnesses, num_landmarks, use_rand):
    """
    Test consistency of the Flood complex between using and not using Triton
    kernels for different batch sizes, number of witnesses and number of landmarks.
    Tests also number of landmars being equal or larger than number of witnesses.
    """

    assert DEVICE.type == "cuda"
    num_rand = 512
    points_per_edge = 20
    if use_rand:
        kwargs = {"num_rand": num_rand, "points_per_edge": None}
    else:
        kwargs = {"num_rand": None, "points_per_edge": points_per_edge}
    torch.manual_seed(42)
    np.random.seed(42)

    X = generate_noisy_torus_points_3d(num_witnesses).to(DEVICE)
    L = generate_landmarks(X, num_landmarks)

    # Test w kernel
    torch.manual_seed(42)
    np.random.seed(42)
    fc_triton = flood_complex(X, L, use_triton=True, **kwargs)

    # Test w/o kernel
    torch.manual_seed(42)
    np.random.seed(42)
    fc_no_triton = flood_complex(X, L, use_triton=False, **kwargs)

    for simplex in fc_no_triton:
        assert simplex in fc_triton
        assert (
            abs(fc_no_triton[simplex] - fc_triton[simplex]) < 1e-4
        ), f"Simplex {simplex}: Naive {fc_no_triton[simplex]:.5f} \
            and Triton {fc_triton[simplex]:.5f}"


@pytest.mark.parametrize("num_witnesses", [1000, 10_000])
@pytest.mark.parametrize("num_landmarks", [20, 701, 2000])
@pytest.mark.parametrize("use_rand", [True, False])
def test_kdtree_vs_triton(num_witnesses, num_landmarks, use_rand):
    """
    Test consistency of the Flood complex between kdtree and Triton computation
    for different batch sizes, number of witnesses and number of landmarks.
    Tests also number of landmars being equal or larger than number of witnesses.
    """
    num_rand = 512
    points_per_edge = 20
    if use_rand:
        kwargs = {"num_rand": num_rand, "points_per_edge": None}
    else:
        kwargs = {"num_rand": None, "points_per_edge": points_per_edge}

    assert DEVICE.type == "cuda"

    torch.manual_seed(42)
    np.random.seed(42)

    X = generate_noisy_torus_points_3d(num_witnesses).to(DEVICE)
    L = generate_landmarks(X, num_landmarks)

    # Test using triton kernel
    torch.manual_seed(42)
    np.random.seed(42)
    fc_triton = flood_complex(X, L, **kwargs)

    # Test cpu version (kd-tree)
    torch.manual_seed(42)
    np.random.seed(42)
    fc_cpu = flood_complex(X.cpu(), L.cpu(), **kwargs)

    for simplex in fc_cpu:
        assert simplex in fc_triton
        assert (
            abs(fc_cpu[simplex] - fc_triton[simplex]) < 1e-4
        ), f"Simplex {simplex}: Naive {fc_cpu[simplex]:.5f} and Triton {fc_triton[simplex]:.5f}"


@pytest.mark.parametrize("num_witnesses", [1000, 10_000])
@pytest.mark.parametrize("num_landmarks", [20, 1000])
@pytest.mark.parametrize("mode", ["CPU", "dist", "Triton"])
@pytest.mark.parametrize("return_simplex_tree", [True, False])
def test_filtration_condition(num_witnesses, num_landmarks, mode, return_simplex_tree):
    """
    Test that the Flood complex is a filtered complex.
    """

    if mode == "CPU":
        device = "cpu"
        use_triton = False
    else:
        assert DEVICE.type == "cuda"
        device = DEVICE
        if mode == "dist":
            use_triton = False
        elif mode == "Triton":
            use_triton = True
        else:
            raise RuntimeError("Mode not implemented")

    torch.manual_seed(42)
    np.random.seed(42)
    X = generate_noisy_torus_points_3d(num_witnesses).to(device)
    L = generate_landmarks(X, num_landmarks)

    if not return_simplex_tree:
        fc = flood_complex(X, L, use_triton=use_triton, return_simplex_tree=False)
        st = gudhi.SimplexTree()  # pylint: disable=no-member
        for simplex in fc:
            st.insert(simplex, float("inf"))
            st.assign_filtration(simplex, fc[simplex])
    else:
        st = flood_complex(X, L, use_triton=use_triton, return_simplex_tree=True)

    for simplex, filtration in st.get_simplices():
        faces = list(st.get_boundaries(simplex))
        if len(simplex) > 1:
            assert len(faces) == len(
                simplex
            ), f"Simplex {simplex} has {len(faces)} faces"
        else:
            assert (
                len(simplex) == 1 and len(faces) == 0
            ), f"Simplex {simplex} has {len(faces)} faces"

        for face, face_filtration in faces:
            assert (
                face_filtration <= filtration
            ), f"Simplex {simplex} has filtr. value {filtration:.5f} \
                and its face {face} has {face_filtration:.5f}"
