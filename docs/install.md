---
hide:
  - navigation
---

# Installation

## Environment setup

We recommend installing **flooder** in a clean new [Anaconda](https://www.anaconda.com/) environment
as described below:

``` bash linenums="1"
conda create -n flooder-env python=3.9 -y
conda activate flooder-env
conda install pip git -y
```

## Via Pip (recommended)

**flooder** is available on PyPi ([here](https://pypi.org/project/flooder/)) and can be installed via:

```bash linenums="1"
pip install flooder
```

To test the command-line interface (CLI) use 

```bash linenums="1"
flooder --help 
```

## Development installation

In case you want to contribute to **flooder**, clone the [GitHub repo](https://github.com/plus-rkwitt/flooder) and run

```bash linenums="1"
git clone https://github.com/plus-rkwitt/flooder
cd flooder
pip install -e .
```

In case you do not want to install anything, you can also execute examples or tests
from within the checked-out folder by specifying `PYTHONPATH`as

```bash linenums="1"
git clone https://github.com/plus-rkwitt/flooder
cd flooder
PYTHONPATH=. python examples/example_01_cheese_3d.py
```

## GPU requirements

Our implementation relies heavily on custom Triton kernels (although we support CPU computation as well) for maximum performance. According to the official [Triton compatibility](https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility) page, you need a NVIDIA GPU with compute capabilty (check  
[here](https://developer.nvidia.com/cuda-gpus)) of at least 7.5 (e.g., GTX 3080, etc.).