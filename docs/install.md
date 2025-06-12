---
hide:
  - navigation
---

# Installation

## Environment setup

We recommend installing **flooder** in a clean new Anaconda environment
as described below:

``` bash linenums="1"
conda create -n flooder-env python=3.9 -y
conda activate flooder-env
conda install pip git -y
```

## Via Pip (recommended)

**flooder** is available on PyPi and can be installed via:

```bash linenums="1"
pip install flooder
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
PYTHONPATH=. python examples/example_01_cheese_3D.py
```
