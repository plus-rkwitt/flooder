---
hide:
  - navigation
---

# Examples

## Flood PH of noisy torus points

In the following example, we sample `N_w=1000000` points from a noisy torus in 3D, then
construct the Flood complex on top of `N_l=1000` landmarks (see [Illustration](index.md#illustration)
) and compute persistent
homology (up to dimension 3) using `gudhi` based on the constructed filtered Flood simplicial complex.

``` py linenums="1"
from flooder import (
    generate_noisy_torus_points, 
    flood_complex, 
    generate_landmarks)

DEVICE = "cuda"
N_p = 1_000_000  # Number of points to sample from torus
N_l = 1_000      # Number of landmarks for Flood complex

pts = generate_noisy_torus_points(N_w).to(DEVICE)
lms = generate_landmarks(pts, N_l)

stree = flood_complex(lms, pts, return_simplex_tree=True)
stree.compute_persistence()
ph = [stree.persistence_intervals_in_dimension(i) for i in range(3)]
```

Importantly, one can either call `flood_complex` with the already pre-selected
(here via FPS) landmarks, or one can just specify the number of desired landmarks, e.g.,
via

```py linenums="1"
stree = flood_complex(1_000, pts, return_simplex_tree=True)
```

in which case FPS is called internally.
