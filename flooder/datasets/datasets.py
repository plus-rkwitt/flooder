"""Implementation of datasets used in the original Flooder paper.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import copy
import hashlib
import os
import os.path as osp
import tarfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Union, Tuple

import yaml
import zstandard as zstd
import gdown
import numpy as np
import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm


IndexType = Union[slice, Tensor, np.ndarray, Sequence]


@dataclass
class FlooderData:
    x: torch.Tensor
    y: Union[int, torch.Tensor]
    name: str


@dataclass
class FlooderRocksData(FlooderData):
    surface: float
    volume: float


class BaseDataset(torch.utils.data.Dataset):  # Follows torch_geometric.data.dataset
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the ``self.raw_dir`` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the ``self.processed_dir`` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the ``self.raw_dir`` folder."""
        raise NotImplementedError

    def process(self) -> None:
        r"""Processes the dataset to the ``self.processed_dir`` folder."""
        raise NotImplementedError

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        raise NotImplementedError

    def get(self, idx: int) -> FlooderData:
        r"""Gets the data object at index ``idx`."""
        raise NotImplementedError

    def __init__(self, root, fixed_transform=None, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self._indices = None

        self._download()
        self._process()

        self.data = torch.load(self.processed_paths[0], weights_only=False)
        self.splits = torch.load(self.processed_paths[1], weights_only=False)
        self.classes = sorted({int(data.y) for data in self})
        self.num_classes = len(self.classes)

        if fixed_transform is not None:
            old_data = self.data
            self.data = [fixed_transform(d) for d in old_data]
            del old_data

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_paths(self) -> List[str]:
        files = self.raw_file_names
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        files = self.processed_file_names
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if all([osp.exists(f) for f in self.raw_paths]):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def _process(self):
        if all([osp.exists(f) for f in self.processed_paths]):
            return
        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ):
        r"""In case ``idx`` is of type integer, will return the data object
        at index ``idx`` (and transforms it in case ``transform`` is
        present). In case ``idx`` is a slicing object, *e.g.*, ``[2:5]``, a list, a
        tuple, or a ``torch.Tensor`` or ``np.ndarray`` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def index_select(self, idx: IndexType):
        r"""Creates a subset of the dataset from specified indices ``idx``.
        Indices ``idx`` can be a slicing object, *e.g.*, ``[2:5]``, a
        list, a tuple, or a ``torch.Tensor`` or ``np.ndarray`` of type
        long or bool.

        Args:
            idx (slice, list, tuple, torch.Tensor, np.ndarray): The indices to
                select.
        Returns:
            BaseDataset: A subset of the dataset at the specified indices.
        """
        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ):
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to ``True``, will also
                return the random permutation used to shuffle the dataset.
                (default: ``False``)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        def _safe_len(x, default="?"):
            try:
                return len(x)
            except Exception:
                return default

        def _short_path(p: str) -> str:
            try:
                p = str(p)
                # show last 2 components to keep it readable
                parts = p.replace("\\", "/").rstrip("/").split("/")
                return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            except Exception:
                return "?"

        def _has_all(paths):
            try:
                return all(osp.exists(p) for p in paths)
            except Exception:
                return False

        # Size: current view vs total
        n_view = _safe_len(self.indices())
        n_total = "?"
        try:
            # If this is a subset, the "full" dataset is its stored data length (if available)
            if getattr(self, "_indices", None) is not None and hasattr(self, "data"):
                n_total = _safe_len(self.data)
            else:
                n_total = n_view
        except Exception:
            pass

        is_subset = getattr(self, "_indices", None) is not None

        # Root / dirs
        root = _short_path(getattr(self, "root", "?"))

        raw_ok = _has_all(getattr(self, "raw_paths", []))
        proc_ok = _has_all(getattr(self, "processed_paths", []))

        num_classes = getattr(self, "num_classes", None)
        classes = getattr(self, "classes", None)
        class_part = ""
        if num_classes is not None:
            if classes is not None and _safe_len(classes) != "?":
                # show at most first 5
                cls_preview = list(classes)[:5]
                suffix = ", ..." if len(classes) > 5 else ""
                class_part = f", num_classes={num_classes}, classes={cls_preview}{suffix}"
            else:
                class_part = f", num_classes={num_classes}"

        split_part = ""
        splits = getattr(self, "splits", None)
        if isinstance(splits, dict):
            split_part = f", splits={list(splits.keys())}"
        elif splits is not None:
            split_part = ", splits=yes"

        tfm = getattr(self, "transform", None)
        tfm_part = f", transform={tfm.__class__.__name__}" if tfm is not None else ""

        # Compose
        size_part = f"n={n_view}"
        if is_subset and n_total != "?":
            size_part += f"/{n_total}"

        subset_part = ", subset=yes" if is_subset else ""

        return (
            f"{cls}({size_part}, root='{root}', raw={'ok' if raw_ok else 'missing'}, "
            f"processed={'ok' if proc_ok else 'missing'}"
            f"{subset_part}{class_part}{split_part}{tfm_part})"
        )


class FlooderDataset(BaseDataset):
    @property
    def file_id(self):
        raise NotImplementedError

    @property
    def checksum(self):
        raise NotImplementedError

    @property
    def folder_name(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        return ['data.pt', 'splits.pt']

    def get(self, idx: int):
        return self.data[idx]

    def len(self) -> int:
        return len(self.data)

    def unzip_file(self):
        with open(self.raw_paths[0], 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    tar.extractall(path=self.raw_dir, filter='data')

    def process_file(self, file, ydata):
        """Specific logic to turn an .npy file into a FlooderData object"""
        raise NotImplementedError

    def get_split_indices(self, splits_data):
        """Override this if split extraction logic differs"""
        return splits_data["splits"]

    def process(self):
        extract_path = osp.join(self.raw_dir, self.folder_name)
        if not osp.isdir(extract_path):
            self.unzip_file()

        # Load Metadata
        with open(osp.join(extract_path, 'meta.yaml'), "r") as f:
            ydata = yaml.safe_load(f)

        with open(osp.join(extract_path, 'splits.yaml'), "r") as f:
            splits_data = yaml.safe_load(f)

        split_indices = self.get_split_indices(splits_data)

        # Process Files
        data_list = []
        in_path = Path(extract_path)
        sorted_files = sorted(in_path.glob("*.npy"))

        for file in tqdm(sorted_files, desc=f"Processing {self.folder_name}"):
            data = self.process_file(file, ydata)
            data_list.append(data)

        torch.save(data_list, self.processed_paths[0])
        torch.save(split_indices, self.processed_paths[1])

    def download(self):
        url = f'https://drive.google.com/uc?id={self.file_id}'

        # Use gdown to handle the download
        output = os.path.join(self.raw_dir, self.raw_file_names[0])
        gdown.download(url, output, quiet=False)
        self.validate(output)

    def validate(self, file_path: Path):
        h = hashlib.new('sha256')
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        if h.hexdigest() != self.checksum:
            warnings.warn(
                f"Warning: the downloaded file {file_path} did not match the expected checksum.\n"
                f"This may indicate that the file is corrupted, incomplete, or altered during download.\n"
                f"Expected SHA256: {self._checksum}\n"
                f"Actual SHA256:   {h.hexdigest()}\n"
                f"Please try re-downloading the dataset or contact the dataset maintainer if the problem persists.",
                UserWarning
            )


class SwisscheeseDataset(FlooderDataset):
    def __init__(self, root, ks=[10, 20], num_per_class=500, num_points=1000000, fixed_transform=None, transform=None):
        """
        Create a Swiss Cheese dataset with specified parameters.

        Args:
            root (str): Root directory where the dataset is stored.
            ks (list): List of integers representing the number of voids in each class.
            num_per_class (int): Number of samples to generate per class.
            num_points (int): Number of points to generate for each sample.
        """

        self.rng = np.random.RandomState(42)
        self.k, self.num_per_class, self.num_points = ks, num_per_class, num_points
        super().__init__(root, fixed_transform=fixed_transform, transform=transform)

    @property
    def folder_name(self):
        return 'swisscheese'

    @property
    def raw_file_names(self):
        return []

    @torch.no_grad()
    def generate_swiss_cheese_points_fast(
        self,
        N: int,
        rect_min: torch.Tensor,
        rect_max: torch.Tensor,
        k: int,
        void_radius_rng: tuple,
        batch_factor=4,  # how many candidates to shoot each round
    ):
        """
        N                 number of output points
        rect_min,rect_max d-vectors (same device & dtype) describing the box
        k                 number of spherical voids
        void_radius_rng   (r_min, r_max)
        """
        d = rect_min.numel()
        r_min, r_max = void_radius_rng

        # --- 1.  build non-overlapping voids ------------------------------------
        centres = torch.empty((0, d))
        radii = torch.empty((0,))

        while centres.shape[0] < k:
            # shoot a small batch of candidate voids
            B = max(8, 2 * (k - centres.shape[0]))  # a handful is enough
            cand_centres = (rect_min + r_max) + (
                rect_max - rect_min - 2 * r_max
            ) * torch.from_numpy(self.rng.rand(B, d))
            cand_radii = r_min + (r_max - r_min) * torch.from_numpy(self.rng.rand(B))

            if centres.numel() == 0:
                ok = torch.ones(B, dtype=torch.bool)
            else:
                dist = torch.cdist(cand_centres, centres)  # B × |centres|
                ok = (dist >= (cand_radii[:, None] + radii[None, :])).all(dim=1)

            # keep as many as we still need
            keep = ok.nonzero(as_tuple=False).squeeze()[: k - centres.shape[0]]
            centres = torch.cat([centres, cand_centres[keep]], dim=0)
            radii = torch.cat([radii, cand_radii[keep]], dim=0)

        # --- 2.  rejection sample points in large vectorised batches ------------
        pts = torch.empty((0, d), dtype=rect_min.dtype)
        todo = N
        while todo:
            B = batch_factor * todo  # adaptive batch
            cand = rect_min + (rect_max - rect_min) * torch.from_numpy(self.rng.rand(B, d))

            # distance of every candidate to every void centre:  B × k
            if k:
                dist = torch.cdist(cand, centres)
                good = (dist >= radii[None, :]).all(dim=1)
            else:  # no holes at all
                good = torch.ones(B, dtype=torch.bool)

            accepted = cand[good][:todo]  # at most 'todo'
            pts = torch.cat([pts, accepted], dim=0)
            todo = N - pts.shape[0]

        return pts, centres, radii

    def process(self):
        split_indices = {}
        n = len(self.k) * self.num_per_class
        for i in range(10):
            split = {}
            indices = self.rng.permutation(np.arange(n))
            split['trn'] = indices[:int(n * 0.72)]
            split['val'] = indices[int(n * 0.72):int(n * 0.80)]
            split['tst'] = indices[int(n * 0.80):]
            split_indices[i] = split

        ks, num_per_class, num_points = self.k, self.num_per_class, self.num_points
        rect_min = torch.tensor([0.0, 0.0, 0.0])
        rect_max = torch.tensor([5.0, 5.0, 5.0])

        data_list = []
        for ki, k in enumerate(ks):
            for r in tqdm(range(num_per_class)):
                void_radius_range = (0.1, 0.5)
                points, _, _ = self.generate_swiss_cheese_points_fast(
                    num_points, rect_min, rect_max, k, void_radius_range
                )
                data_list.append(FlooderData(x=points.to(torch.float32), y=ki, name=f'{k}voids_{r}'))

        torch.save(data_list, self.processed_paths[0])
        torch.save(split_indices, self.processed_paths[1])

    def download(self):
        pass


class ModelNet10Dataset(FlooderDataset):
    @property
    def file_id(self):
        return '180Gk0I_JYWkGNnLj5McI2P3zwhgGeVtM'

    @property
    def checksum(self):
        return '6f9504d5574224fdf5b9255d2b9d5f041540298c0241fc6abbbfedaf9e1f4280'

    @property
    def folder_name(self):
        return 'modelnet10_250k'

    @property
    def raw_file_names(self):
        return ['modelnet10_250k.tar.zst']

    def process_file(self, file, ydata):
        x = torch.from_numpy((np.load(file) / 32767).astype(np.float32))
        y = ydata['data'][file.name]['label']
        return FlooderData(x=x, y=y, name=file.stem)


class CoralDataset(FlooderDataset):
    @property
    def file_id(self):
        return '1g-n8ExkU6eOJLelIMeNaFRdqoEM8ZDry'

    @property
    def checksum(self):
        return 'e8b5ae6b22d03e0bcf118bb28b4d465f8ec5b308e038385879b98df3fed0150f'

    @property
    def folder_name(self):
        return 'corals'

    @property
    def raw_file_names(self):
        return ['corals.tar.zst']

    def process_file(self, file, ydata):
        x = torch.from_numpy((np.load(file) / 32767).astype(np.float32))
        y = ydata['data'][file.name]['label']
        return FlooderData(x=x, y=y, name=file.stem)


class MCBDataset(FlooderDataset):
    @property
    def file_id(self):
        return '19EP9DEOMoSj0YVa_pXnui3OR2JZHOgSY'

    @property
    def checksum(self):
        return 'dc36e1c5886e2d21a9f1dbaec084852dda2aab06fb7cd1c36e4403ac3e486a10'

    @property
    def folder_name(self):
        return 'mcb'

    @property
    def raw_file_names(self):
        return ['mcb.tar.zst']

    def process_file(self, file, ydata):
        x = torch.from_numpy((np.load(file) / 32767).astype(np.float32))
        y = ydata['data'][file.name]['label']
        return FlooderData(x=x, y=y, name=file.stem)


class RocksDataset(FlooderDataset):
    @property
    def file_id(self):
        return '1htI0eeON3RG3V_fShd8U8tZmJ1g6akEx'

    @property
    def checksum(self):
        return 'd635e6ae2e949075ae69b4397217bb2949c737126bbc23108fc48ec1a7aa5b00'

    def __init__(self, root, fixed_transform=None, transform=None):
        self.rng = np.random.RandomState(42)
        super().__init__(root, fixed_transform, transform)

    @property
    def folder_name(self):
        return 'rocks'

    @property
    def raw_file_names(self):
        return ['rocks.tar.zst']

    def process_file(self, file, ydata):
        loaded_data = np.load(file)
        bool_data = np.unpackbits(loaded_data).reshape((256, 256, 256)).astype(bool)
        pts = np.stack(np.where(bool_data), axis=1).astype(np.float32)
        pts += 0.1 * self.rng.rand(*pts.shape)

        return FlooderRocksData(
            x=torch.from_numpy(pts),
            y=ydata['data'][file.name]['label'],
            surface=ydata['data'][file.name]['target'],
            volume=ydata['data'][file.name]['volume'],
            name=file.stem
        )
