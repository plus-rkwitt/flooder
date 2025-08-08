---
hide:
  - navigation
---

# Overview

This is the project page of **flooder**, an easy-to-use Python package for constructing a lightweight simplicial complex (i.e., the FLood complex) on top of (low-dimensional) Euclidean point cloud data and subsequent persistent homology[^1] (PH) computation. Algorithmically, **flooder** is designed to take full advantage of state-of-the-art GPU computing frameworks (via PyTorch) to enable computation of a filtered simplicial complex on millions of points in seconds. Based on the Flood complex, we use the awesome [gudhi](https://gudhi.inria.fr/) library for PH computation.

[^1]: Edelsbrunner, Letscher & Zomorodian. Topological Persistence and Simplification. Discrete Comput Geom 28, 511–533 (2002). [DOI](https://doi.org/10.1007/s00454-002-2885-2)

## Illustration

Below is an illustrative animation on how the (filtered) *Flood complex* is built (on a noisy sample of a figure-eight shape in 2D), starting from a collection of points and a Delaunay triangulation of a small subset of *landmarks* (in yellow). In short, a simplex of the Delaunay triangulation is added to the Flood complex at time $t\geq 0$, if it is fully covered by balls of radius $t$, centered at *all* data points.

<style>
  #canvas-container canvas {
    width: auto;
    height: 400px;
    max-width: 100%;
  }
  #radiusSlider {
  accent-color: rgb(160, 170, 180); 
  }
</style>
<body>
  <div style="text-align:center;">
    <div id="canvas-container"></div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
    <button id="playPauseBtn" class="btn">
      <i class="fa fa-play"></i>
    </button>
    <input type="range" id="radiusSlider" min="0" max="4" step="0.01" value="0" style="vertical-align: middle; width: 300px;" />
  </div>
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.0/lib/p5.min.js"></script>
  <script src="animation/flood_triangle.js"></script>
</body>


## Runtime teaser

Below you find a **runtime comparison** of computing zero-, one-, and two-dimensional Flood and Alpha PH on 1M points in 3D (using a NVIDIA H100 NVL GPU). The data used in this comparison is a synthetic (3D) *swiss cheese* created by uniformly drawing points in the unit cube and introducing a certain number of voids. Note that on point clouds of this size (in fact, even at much lower scale), computing Vietoris-Rips PH is intractable (beyond 0-dimensional features). The table lists runtime in seconds (for creating the filtered Flood and Alpha complexes and subsequent PH computation).

| **Complex**      | :octicons-stopwatch-16: Runtime (in s) |
| ----------- | ------------------------------------ |
| Alpha complex (via `gudhi.AlphaComplex`)       | 141.8 $\pm$ 1.5  |
| Flood complex (via **flooder**)                | 1.4 $\pm$ 0.3  |

Please see our `examples` folder in the **flooder** [GitHub repository](https://github.com/plus-rkwitt/flooder/) to run your own runtime comparison.

!!! note "Caution"
    **flooder** is still under active development and it's usage API might change
    significantly over the next months.
