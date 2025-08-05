---
hide:
  - navigation
---

# Examples

## Flood PH of noisy torus points

In the following example, we sample `n_pts=1000000` points from a noisy torus in 3D, then
construct the Flood complex on top of `n_lms=1000` landmarks (see [Illustration](index.md#illustration)
) and compute persistent
homology (up to dimension 3) using `gudhi` based on the constructed filtered Flood simplicial complex.

``` py linenums="1"
from flooder import (
    generate_noisy_torus_points, 
    flood_complex, 
    generate_landmarks)

DEVICE = "cuda"
n_pts = 1_000_000  # Number of points to sample from torus
n_lms = 1_000      # Number of landmarks for Flood complex

pts = generate_noisy_torus_points(n_pts).to(DEVICE)
lms = generate_landmarks(pts, n_lms)

stree = flood_complex(pts, lms, return_simplex_tree=True)
stree.compute_persistence()
ph = [stree.persistence_intervals_in_dimension(i) for i in range(3)]
```

Importantly, one can either call `flood_complex` with the already pre-selected
(here via FPS) landmarks, or one can just specify the number of desired landmarks, e.g.,
via

```py linenums="1"
stree = flood_complex(pts, n_lms, return_simplex_tree=True)
```

in which case FPS is called internally.
