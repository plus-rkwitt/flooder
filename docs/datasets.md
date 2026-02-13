---
hide:
  - navigation
---

# Datasets 


<div align="center">

<img src="../visualization/pointclouds.png" width=70% height=70%>

</div>

<br>

The `flooder` package includes a curated collection of point cloud datasets, including variations of classical ones
as well as new ones, designed to be geometrically challinging and to test topological ML methods. 
The package allows to import and use the datasets with only few lines of code:


```python
from flooder.datasets import (
    CoralDataset, MCBDataset, RocksDataset,
    ModelNet10Dataset, SwisscheeseDataset, LargePointCloudDataset,
)

dataset = CoralDataset('./coral_dataset/') 
for data in dataset:
    print(data.x.shape, data.y)
```

In particular, we provide the following datasets:

- **CoralDataset**: 
    this dataset consists of 81 point clouds, each comprising 1 million points
    uniformly sampled from surface meshes of coral specimens provided by the
    Smithsonian 3D Digitization program (https://3d.si.edu/corals). Labels
    correspond to the coral's genus, with 31 Acroporidae samples (label 0) and
    52 Poritidae samples (label 1). Due to its small size, it is challenging for 
    geometric neural networks, while topology-based methods are often at an advantage. 
- **RocksDataset**: 
    this synthetic dataset consists of 1000 point clouds representing symulated
    rock samples from two classes. The voxel grids are produced by the PoreSpy
    library (https://porespy.org/) with classes corresponding to the generation
    method, fractal noise and blobs, each with 500 samples. It offers also a regression
    task for the estimation of the surface of the material. 
- **ModelNet10Dataset**:
    this dataset consists of 4899 point clouds, each comprising 250k points
    uniformly sampled from surface meshes from the ModelNet10 dataset
    (Wu et al., 3D ShapeNets: A Deep Representation for Volumetric Shapes,
    CVPR 2015). This dataset contains topologically simple objects, and since it 
    is a high-resolution version of a well-studied dataset, it is useful to compare 
    methods with the existing literature. 
- **MCBDataset**:
    this dataset consists of 1745 point clouds, each comprising 1 million
    points uniformly sampled from surface meshes from a *subset* of the MCB
    dataset (Kim et al., A large-scale annotated mechanical components benchmark for
    classification and retrieval tasks with deep neural networks, ECCV, 2020), where we 
    only retain geometrically complex objects. Therefore, it is useful to probe the 
    capabilities of topological ML methods. 
- **SwisscheeseDataset**:
    this dataset class allows to procedurally generate a dataset of point clouds with multiple
    spherical voids. In particular, each sample consists
    of points uniformly sampled from a 3D axis-aligned box with multiple
    spherical voids removed ("Swiss cheese"). The number of voids defines the
    class label. 
- **LargePointCloudDataset**:
    this dataset contains two point clouds (a virus and a coral) with more than 10M points each.
    It can be used to benchmark the speed and memory usage of geometric algorithms 
    (e.g. simplicial complex constructions) on real-world data. 
    The *EMD-50844* point cloud is obtained from the protein structure of the RV-A89 virus, available at [www.ebi.ac.uk/emdb/EMD-50844](https://www.ebi.ac.uk/emdb/EMD-50844). 
    The *Leptoseris paschalensis* point cloud comprises the vertices of the mesh of the USNM 53156 coral obtained from the [Smithsonian 3D Digitization](https://3d.si.edu/) initiative.
