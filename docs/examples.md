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
    generate_noisy_torus_points_3d, 
    flood_complex, 
    generate_landmarks)

DEVICE = "cuda"
n_pts = 1_000_000  # Number of points to sample from torus
n_lms = 1_000      # Number of landmarks for Flood complex

pts = generate_noisy_torus_points_3d(n_pts).to(DEVICE)
lms = generate_landmarks(pts, n_lms)

stree = flood_complex(pts, lms, return_simplex_tree=True)
stree.compute_persistence()
ph_diags = [stree.persistence_intervals_in_dimension(i) for i in range(3)]
```

Importantly, one can either call `flood_complex` with the already pre-selected
(here via FPS) landmarks, or one can just specify the number of desired landmarks, e.g.,
via

```py linenums="1"
stree = flood_complex(pts, n_lms, return_simplex_tree=True)
```

in which case FPS is called internally.

## Flooder CLI

If you installed flooder via `pip install flooder`, you not only have the API 
available but also a command-line interface (CLI), which you can execute as 
follows: for demonstration, we will download a PLY file of the *Lucy* angel (>14M points, available 
from the *Stanford 3D Scanning Repository*) 

``` bash linenums="1"
wget https://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz
tar xvfz lucy.tar.gz
```

and first convert it (using the Open3D library; install via `pip install open3d`) to a 
`(N,3)` numpy array:

```py linenums="1"
import numpy as np
import open3d as o3d
X_ply = o3d.io.read_point_cloud("lucy.ply")
X_np = np.asarray(X_ply.points, dtype=np.float32)
np.save("lucy.npy", X_np)
```

Finally, we execute the Flooder CLI, using 5k landmarks, 30 points per edge and a batch size of 64:

``` bash linenums="1"
flooder \
  --input-file lucy.npy \
  --output-file lucy-diagrams.pkl \
  --num-landmarks 5000 \
  --batch-size 64 \
  --max-dimension 3 \
  --points-per-edge 30 \
   --device cuda:0 \
  --cuda-events \
  --stats-json lucy.json
```
