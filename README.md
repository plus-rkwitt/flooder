# Flood complex 

**Source code** for *— The Flood Complex —
Large-Scale Persistent Homology on Millions of Points*, Anonymous NeurIPS 2025 submission

## Installation

We do recommend running our code in a new clean Anaconda environment, which can
be set up as follows:

```bash
conda create -n flooder-env python=3.9 -y
conda activate flooder-env
conda install pip git -y
```

Then, within the `flooder` top level directory we execute

```bash
pip install -e .
```
to install in editable/development mode. Once completed, check if `flooder` can be imported:

```bash
python -c "import flooder"
```

## Running examples

To run the examples (from within the `flooder` main folder):

```bash
PYTHONPATH=. python examples/cheese_runtime_example.py
PYTHONPATH=. python examples/torus_toy_example.py
```

## Hardware requirements

The code has been tested on an Ubuntu Linux 24.04 system with one NVIDIA H100 GPU
and 512GB of RAM (using PyTorch 2.7).

## Animation

An animation of Flood complex construction can be found in `assets/flooder.mp4`.



