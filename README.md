# Flooder

`flooder` is a Python package for constructing a lightweight filtered simplicial complex-*the Flood complex*-on Euclidean point cloud data, leveraging state-of-the-art GPU computing hardware, for subsequent persistent homology (PH) computation (using `gudhi`).

Currently, `flooder` allows computing *Flood PH* on millions of points in 3D (see [Usage](#usage)), enabling previously computationally infeasible large-scale applications of PH on point clouds. While `flooder` is primarily intended for 3D Euclidean point cloud data, it also works with Euclidean point cloud data of moderate dimension (e.g., 4,5,6). For theoretical guarantees of the Flood complex, including algorithmic details, see [Citing](#citing).

## Related Projects

If you are looking for fast implementations of (Vietoris-)Rips PH, see 
[ripser](https://github.com/ripser/ripser), or the GPU-accelerated [ripser++](https://github.com/simonzhang00/ripser-plusplus), respectively. In addition [gudhi](https://pypi.org/project/gudhi/) supports, e.g., computing Alpha PH also on fairly large point clouds (see the `examples/example_01_cheese_3D.py` for a runtime comparison).

## Setup

Currently, `flooder` is available on `pypi` with wheels for Unix-based platforms. To install, type the following command into your environment (we do recommend a clean new Anaconda environment, e.g., created via `conda create -n flooder-env python=3.9 -y`):

```bash
pip install flooder
```

### Local/Development build

In case you want to contribute to the project, we recommend checking out the `flooder` GitHub repository, and setting up the environment as follows:

```bash
git clone https://github.com/plus-rkwitt/flooder
conda create -n flooder-env python=3.9 -y
conda activate flooder-env
pip install -r requirements.txt
```

The previous commands will install all dependencies, such as `torch`, `gudhi`, `numpy`, `fpsample` and `scipy`. Once installed, you can run our examples from within the top-level `flooder` folder (i.e., the directory created when doing `git clone`) via 

```bash
PYTHONPATH=. python examples/example_01_cheese_3D.py
```

Alternatively, you can also do a `pip install -e .` for a local [editable](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) build. Note that the latter command will already install all required dependencies (so, there is no need to do a `pip install -r requirements.txt`).

### Optional dependencies

In case you want to plot persistence diagrams, we recommend using `persim`, which can be installed via

```bash
pip install persim
```

## Usage

In the following example, we compute **Flood PH** on 2M points from a standard multivariate Gaussian in 3D, using 1k landmarks, and finally plot the diagrams up to dimension 2. You could, e.g., just copy-paste the following code into a Jupyter notebook (note that, in case you just checked out the GitHub repository and did not do a `pip install flooder`, the notebook would need to be in the top-level directory for all imports to work).

```python
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
n_lms = 1_000
stree = flood_complex(pts, n_lms, return_simplex_tree=True)
```

## License

The code is licensed under an MIT license.

## Citing

Please cite the following arXiv preprint in case you `flooder` useful for your applications.

```bibtex
coming soon!
```





