import torch
import gudhi
import time
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flooder import generate_noisy_torus_points, flood_complex, generate_landmarks

device = torch.device("cuda")

RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def plot_persistence_diagrams(pdiagram0, pdiagram1, pdiagram2, outfile="diagrams.png"):
    all_pts = np.vstack([pdiagram0, pdiagram1, pdiagram2])
    min_val = np.min(all_pts)
    max_val = np.max(all_pts)
    buffer = 0.05 * (max_val - min_val)
    plot_range = [min_val - buffer, max_val + buffer]

    # Create subplots with shared axes off
    fig = make_subplots(rows=1, cols=3, subplot_titles=("H0", "H1", "H2"))

    # Helper to add a subplot
    def add_diagram(fig, diagram, row, col, name):
        fig.add_trace(
            go.Scatter(
                x=diagram[:, 0],
                y=diagram[:, 1],
                mode="markers",
                marker=dict(size=6),
                name=f"{name} features",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add diagonal
        fig.add_trace(
            go.Scatter(
                x=plot_range,
                y=plot_range,
                mode="lines",
                line=dict(color="black", dash="dash"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(range=plot_range, title_text="Birth", row=row, col=col)
        fig.update_yaxes(range=plot_range, title_text="Death", row=row, col=col)

    # Add each persistence diagram
    add_diagram(fig, pdiagram0, 1, 1, "H0")
    add_diagram(fig, pdiagram1, 1, 2, "H1")
    add_diagram(fig, pdiagram2, 1, 3, "H2")

    # Layout tweaks
    fig.update_layout(width=1000, height=350, margin=dict(t=40))
    fig.write_image(outfile)


def main():

    N_w = 1000000  # nr. of points on the torus
    N_l = 1000  # nr. of landmarks for Flood complex

    pts = generate_noisy_torus_points(N_w)

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

    # extact persistence diagrams and write to PNG file
    pdiagram0 = st.persistence_intervals_in_dimension(0)
    pdiagram1 = st.persistence_intervals_in_dimension(1)
    pdiagram2 = st.persistence_intervals_in_dimension(2)
    # plot_persistence_diagrams(pdiagram0, pdiagram1, pdiagram2)


if __name__ == "__main__":
    main()
