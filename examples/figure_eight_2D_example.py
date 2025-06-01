import torch
import time
import numpy as np
import gudhi
from timeit import default_timer as timer

from flooder import generate_landmarks, flood_complex
from persim import plot_diagrams
import matplotlib.pyplot as plt

device = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def sample_figure_eight(
    n_samples,
    r_bounds=(0.2, 0.3),
    centers=((0.3, 0.5), (0.7, 0.5)),
    noise_std=0.0,
    noise_kind="gaussian",
    rng=None,
):
    """
    Uniformly sample a figure-eight in 2D and optionally add noise.

    Parameters
    ----------
    n_samples : int
        Number of points to generate.
    r_bounds : tuple(float, float)
        (inner_radius, outer_radius) shared by both lobes.
    centers : tuple[(float, float), (float, float)]
        (x, y) coordinates of lobe centres.
    noise_std : float
        Standard deviation (Gaussian) or half-width (uniform) of noise
        added independently to x and y.  Set to 0 for no noise.
    noise_kind : {"gaussian", "uniform"}
        Distribution of the jitter.
    rng : int or numpy.random.Generator, optional
        Random-state for reproducibility.

    Returns
    -------
    pts : (n_samples, 2) ndarray
        Sampled coordinates.
    """
    rng = np.random.default_rng(rng)

    lobe_idx = rng.integers(0, 2, size=n_samples)
    cx, cy = np.asarray(centers).T  # shape (2,)
    cx = cx[lobe_idx]  # (n_samples,)
    cy = cy[lobe_idx]

    r_min, r_max = r_bounds
    r = np.sqrt(rng.uniform(r_min**2, r_max**2, size=n_samples))
    theta = rng.uniform(0.0, 2 * np.pi, size=n_samples)

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    if noise_std > 0:
        if noise_kind == "gaussian":
            x += rng.normal(0.0, noise_std, size=n_samples)
            y += rng.normal(0.0, noise_std, size=n_samples)
        elif noise_kind == "uniform":
            half = noise_std
            x += rng.uniform(-half, half, size=n_samples)
            y += rng.uniform(-half, half, size=n_samples)
        else:
            raise ValueError("noise_kind must be 'gaussian' or 'uniform'")

    return np.stack((x, y), axis=1)


def top_k_longest(bd, k=10):
    lifetimes = bd[:, 1] - bd[:, 0]
    idx = np.argsort(lifetimes)[-k:][::-1]
    return bd[idx]


def main():

    print(f"{YELLOW}Flood PH of a noisy figure-eight sample (1M points)")
    print(f"{YELLOW}---------------------------------------------------")

    N_w = 1000000  # Number of points sampled from figure-eight
    N_l = 1000  # Number of landmarks for Flood complex

    pts = torch.tensor(
        sample_figure_eight(
            100000, noise_std=0.02, noise_kind="gaussian", rng=42  # tweak this to taste
        ),
        dtype=torch.float32,
    )

    t0_fps = time.perf_counter()
    lms = generate_landmarks(pts, N_l)
    t1_fps = time.perf_counter()

    t0_complex = time.perf_counter()
    out_complex = flood_complex(lms.to(device), pts.to(device), dim=3, batch_size=16)
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

    diags = [st.persistence_intervals_in_dimension(i) for i in range(2)]
    for i in range(2):
        print(f"{RED}10 longest bars (by lifetime) in dimension {i}: {RESET}")
        print(top_k_longest(diags[i], k=10))


if __name__ == "__main__":
    main()
