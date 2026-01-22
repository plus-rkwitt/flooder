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

from .. import generate_swiss_cheese_points


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
    """Base class for Flooder datasets with download/process/load lifecycle.

    This class provides a dataset API inspired by
    `torch_geometric.data.Dataset`, including:
      - a standard directory layout (`root/raw` and `root/processed`);
      - a lifecycle executed at construction time: download, process, load;
      - integer indexing to return items, and advanced indexing to return
        a subset "view" of the dataset;
      - optional per-sample transformations.

    Subclasses must implement the abstract properties/methods that specify
    file requirements and dataset-specific loading logic.

    Attributes:
        root (str): Root directory containing the dataset folders.
        fixed_transform (Callable[[FlooderData], FlooderData] | None):
            Optional transform applied once to each item during `_load()`
            (i.e., at dataset load time, before storing in memory).
        transform (Callable[[FlooderData], FlooderData] | None):
            Optional transform applied on-the-fly in `__getitem__` for
            individual samples.
        _indices (Sequence[int] | None): If not None, defines a subset view
            over the underlying dataset indices.

    Notes:
        - The constructor triggers `_download()`, `_process()`, and `_load()`.
          This means instantiation may perform I/O and compute.
        - Advanced indexing (`slice`, sequences, boolean masks) returns a
          shallow-copied dataset object sharing the same underlying storage
          (whatever the subclass uses), but with `_indices` set.
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Required raw files to consider the dataset downloaded.

        Subclasses should return the file name(s) expected to exist in
        `raw_dir`. If all such files exist, `_download()` is skipped.

        Returns:
            Union[str, list[str], tuple[str, ...]]: File name(s) expected
            inside `self.raw_dir`.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Required processed files to consider the dataset processed.

        Subclasses should return the file name(s) expected to exist in
        `processed_dir`. If all such files exist, `_process()` is skipped.

        Returns:
            Union[str, list[str], tuple[str, ...]]: File name(s) expected
            inside `self.processed_dir`.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def download(self) -> None:
        """Download the dataset into `raw_dir`.

        Subclasses must implement the dataset-specific download logic.
        This method is called by `_download()` only if the required files
        in `raw_paths` are not present.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def process(self) -> None:
        """Process raw files into `processed_dir`.

        Subclasses must implement the dataset-specific processing logic.
        This method is called by `_process()` only if the required files
        in `processed_paths` are not present.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def len(self) -> int:
        """Return the number of items in the full dataset.

        This method is analogous to `torch_geometric.data.Dataset.len()`.
        It should return the total number of items in the underlying dataset,
        not the size of a subset view created via `index_select`.

        Returns:
            int: Total number of data objects stored by the dataset.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def get(self, idx: int) -> FlooderData:
        """Return the data object at a given *global* index.

        `idx` refers to the underlying dataset index, not the subset-view
        index. The subset mapping is handled by `__getitem__` via `indices()`.

        Args:
            idx (int): Global index into the underlying dataset storage.

        Returns:
            FlooderData: The data object at the given index.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def __init__(self, root, fixed_transform=None, transform=None):
        """Initialize a dataset and execute the download/process/load lifecycle.

        Args:
            root (str): Root directory where raw and processed data are stored.
            fixed_transform (Callable[[FlooderData], FlooderData] | None):
                Optional transform applied once to each item during `_load()`.
            transform (Callable[[FlooderData], FlooderData] | None):
                Optional transform applied on-the-fly in `__getitem__`.

        Notes:
            Instantiation may perform I/O and compute by calling `_download()`,
            `_process()`, and `_load()`.
        """
        super().__init__()
        self.root = root
        self.fixed_transform = fixed_transform
        self.transform = transform
        self._indices = None

        self._download()
        self._process()
        self._load()

    def indices(self) -> Sequence:
        """Return the active index mapping for this dataset view.

        For a full dataset (no subset), this is `range(self.len())`.
        For a subset view created via `index_select`, this is the stored
        `_indices` sequence.

        Returns:
            Sequence[int]: Index mapping from view indices to global indices.
        """
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_dir(self) -> str:
        """Directory containing raw downloaded files.

        Returns:
            str: Path to `<root>/raw`.
        """
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        """Directory containing processed files.

        Returns:
            str: Path to `<root>/processed`.
        """
        return osp.join(self.root, 'processed')

    @property
    def raw_paths(self) -> List[str]:
        """Absolute paths to required raw files.

        Returns:
            list[str]: List of absolute paths for `raw_file_names` under
            `raw_dir`.
        """
        files = self.raw_file_names
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        """Absolute paths to required processed files.

        Returns:
            list[str]: List of absolute paths for `processed_file_names` under
            `processed_dir`.
        """
        files = self.processed_file_names
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        """Ensure raw files exist, downloading them if needed.

        If all paths in `raw_paths` exist, this is a no-op. Otherwise, it
        creates `raw_dir` and calls `download()`.

        Notes:
            If `raw_paths` is empty, `all([])` is True and this method will
            skip downloading. Subclasses that do not require raw downloads
            may intentionally return an empty list.
        """
        if all([osp.exists(f) for f in self.raw_paths]):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def _process(self):
        """Ensure processed files exist, processing them if needed.

        If all paths in `processed_paths` exist, this is a no-op. Otherwise,
        it creates `processed_dir` and calls `process()`.
        """
        if all([osp.exists(f) for f in self.processed_paths]):
            return
        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

    def _load(self):
        """Load processed data into memory.

        Subclasses implement the in-memory representation (e.g., `self.data`)
        and are responsible for applying `fixed_transform` if desired.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the number of examples in the current dataset view.

        For a full dataset this equals `self.len()`. For a subset view this
        equals `len(self._indices)`.

        Returns:
            int: Number of examples exposed by this dataset instance.
        """
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ):
        """Get an item or a subset of the dataset.

        Behavior depends on the type of `idx`:

        - If `idx` is an integer (Python `int`, `np.integer`, 0-dim `Tensor`,
          or scalar `np.ndarray`), returns a single `FlooderData` object
          corresponding to the *view* index `idx`.
        - Otherwise, returns a subset view of the dataset created via
          `index_select(idx)`.

        If `transform` is set, it is applied on-the-fly to single-item access.

        Args:
            idx (int | np.integer | slice | torch.Tensor | np.ndarray | Sequence):
                Index or indices selecting items.

        Returns:
            FlooderData | BaseDataset:
                A single data object if `idx` is scalar-like, otherwise a
                `BaseDataset` subset view.

        Raises:
            IndexError: If `idx` type is unsupported (delegated to `index_select`).
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
        """Iterate over items in the current dataset view.

        Yields:
            FlooderData: Items in order from `0` to `len(self) - 1`, with
            `transform` applied if configured.
        """
        for i in range(len(self)):
            yield self[i]

    def index_select(self, idx: IndexType):
        """Create a subset view of the dataset from specified indices.

        Supported index types:
        - `slice`: includes support for float boundaries, e.g. `dataset[:0.9]`,
          interpreted as a fraction of the current view length.
        - `torch.Tensor` of dtype `long`: treated as integer indices.
        - `torch.Tensor` of dtype `bool`: treated as a boolean mask.
        - `np.ndarray` of dtype `int64`: treated as integer indices.
        - `np.ndarray` of dtype `bool`: treated as a boolean mask.
        - `Sequence` (excluding `str`): treated as a list of integer indices.

        The returned dataset is a shallow copy of `self` with `_indices` set
        to map view indices to global indices.

        Args:
            idx (slice | Sequence[int] | torch.Tensor | np.ndarray):
                Indices specifying the subset.

        Returns:
            BaseDataset: A subset view of the dataset.

        Raises:
            IndexError: If `idx` is not one of the supported types or has an
                unsupported dtype.
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
        """Return a shuffled subset view of the dataset.

        This method generates a random permutation of the current dataset view
        and returns a subset view with that ordering.

        Args:
            return_perm (bool): If True, also return the permutation tensor.

        Returns:
            BaseDataset | tuple[BaseDataset, torch.Tensor]:
                If `return_perm` is False, returns the shuffled dataset view.
                If True, returns `(dataset, perm)` where `perm` is a 1D long
                tensor of indices into the current view.
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset


class FlooderDataset(BaseDataset):
    """Base class for Flooder paper datasets distributed as compressed archives.

    This dataset class implements a standard pipeline:

      1) Download a `.tar.zst` archive from Google Drive (via `gdown`),
         identified by `file_id`, and validate it with a SHA256 `checksum`.
      2) Decompress and extract the archive into `raw_dir/<folder_name>/`.
      3) Read dataset metadata (`meta.yaml`) and split definitions (`splits.yaml`)
         from the extracted raw folder.
      4) Convert each raw `.npy` file into a `FlooderData`-like object via
         `process_file(...)`, and store it as a `.pt` file in `processed_dir`.
      5) Load all `.pt` files into memory (`self.data`) in `_load()`, optionally
         applying `fixed_transform` once per sample at load time.

    Subclasses are expected to define:
      - `file_id`: Google Drive file id
      - `checksum`: expected SHA256 checksum of the downloaded archive
      - `folder_name`: name of the extracted folder under `raw_dir`
      - `raw_file_names`: name(s) of the downloaded raw archive file(s)
      - `process_file(...)`: conversion logic from a `.npy` file and metadata

    Attributes:
        data (list[FlooderData]): In-memory list of processed examples loaded from
            `.pt` files. Populated by `_load()`.
        splits (dict): Split definitions loaded from `processed_dir/splits.yaml`.
            The structure depends on the dataset, but is typically a mapping from
            split identifier (e.g., fold index) to dicts containing keys like
            `'trn'`, `'val'`, `'tst'` with integer indices.
        classes (list[int]): Sorted list of unique class labels observed across the
            dataset (computed after loading).
        num_classes (int): Number of unique classes.
    """

    @property
    def file_id(self):
        """Google Drive file id for the dataset archive.

        Subclasses must provide the id used to construct the download URL:
        `https://drive.google.com/uc?id=<file_id>`.

        Returns:
            str: The Google Drive file id.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def checksum(self):
        """Expected SHA256 checksum of the downloaded archive.

        The checksum is used by `validate(...)` after download. If the computed
        SHA256 does not match, a warning is emitted.

        Returns:
            str: Lowercase hex-encoded SHA256 digest of the expected archive.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def folder_name(self):
        """Name of the extracted folder under `raw_dir`.

        After extraction, the expected raw folder is `raw_dir/<folder_name>/`
        containing `meta.yaml`, `splits.yaml`, and the `.npy` files.

        Returns:
            str: Folder name within `raw_dir`.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self):
        """Processed-file sentinel list for determining whether processing is done.

        The default convention for Flooder datasets is:
          - `_done`: an empty file indicating processing completion
          - `splits.yaml`: split definitions copied to the processed directory

        Returns:
            list[str]: List of required processed file names.
        """
        return ['_done', 'splits.yaml']

    def get(self, idx: int):
        """Return the in-memory data object at the given global index.

        This implementation assumes `_load()` has populated `self.data` with
        objects saved in `processed_dir` as `.pt` files.

        Args:
            idx (int): Global index into `self.data`.

        Returns:
            FlooderData: The data item at `idx`.
        """
        return self.data[idx]

    def len(self) -> int:
        """Return the number of examples in the full dataset.

        Returns:
            int: Total number of examples, equal to `len(self.data)` after `_load()`.
        """
        return len(self.data)

    def unzip_file(self):
        """Decompress and extract the dataset archive into `raw_dir`.

        This method reads the first file in `raw_paths` as a `.tar.zst` archive,
        decompresses it using `zstandard`, and extracts it using `tarfile`.

        Extraction behavior depends on Python version:
          - Python >= 3.12: uses `tar.extractall(..., filter='data')` to apply
            tarfile's safety filter.
          - Older versions: falls back to `tar.extractall(...)`.

        Raises:
            FileNotFoundError: If `raw_paths[0]` does not exist.
            tarfile.TarError: If the archive is invalid or cannot be read.
            zstandard.ZstdError: If decompression fails.
            OSError: For I/O errors during reading or extraction.

        Security:
            Be careful extracting archives from untrusted sources. While Python
            3.12's `filter='data'` mitigates some risks, older versions extract
            without filtering.
        """
        with open(self.raw_paths[0], 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    if 'filter' in tarfile.TarFile.extractall.__code__.co_varnames:  # Python >= 3.12
                        tar.extractall(path=self.raw_dir, filter='data')
                    else:
                        tar.extractall(path=self.raw_dir)

    def process_file(self, file, ydata):
        """Convert a raw `.npy` file into a `FlooderData`-like object.

        Subclasses must implement the dataset-specific logic for reading `file`
        (typically via `numpy.load`) and for producing an instance of `FlooderData`
        (or a subclass like `FlooderRocksData`).

        Args:
            file (pathlib.Path): Path to a `.npy` file inside the extracted raw folder.
            ydata (dict): Metadata loaded from `meta.yaml`. The structure is dataset-
                dependent but typically contains labels and other targets keyed by
                file name.

        Returns:
            FlooderData: Processed data object to be saved as a `.pt` file.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def get_split_indices(self, splits_data):
        """Extract split indices from raw `splits.yaml` content.

        The default behavior expects the raw `splits.yaml` to contain a top-level
        key `"splits"` holding the split definitions.

        Subclasses may override this method if their `splits.yaml` uses a different
        schema.

        Args:
            splits_data (dict): Parsed YAML content from `splits.yaml`.

        Returns:
            Any: The split indices structure to be saved into
            `processed_dir/splits.yaml`. Typically a dict mapping fold id to split
            dicts (`'trn'`, `'val'`, `'tst'`), but may vary by dataset.
        """
        return splits_data["splits"]

    def process(self):
        """Process the extracted raw dataset into serialized `.pt` files.

        Processing performs the following steps:

          1) Ensure the archive has been extracted into `raw_dir/<folder_name>/`.
          2) Load metadata from `meta.yaml` and split definitions from `splits.yaml`.
          3) Save extracted split indices into `processed_dir/splits.yaml`.
          4) Iterate over all `.npy` files in the extracted folder, sorted by name.
          5) For each `.npy`, call `process_file(file, ydata)` and save the returned
             object as `<stem>.pt` in `processed_dir`.
          6) Create the `_done` sentinel file in `processed_dir`.

        Raises:
            FileNotFoundError: If required raw files (`meta.yaml`, `splits.yaml`,
                or `.npy` files) are missing.
            yaml.YAMLError: If YAML parsing fails.
            OSError: For I/O errors reading raw files or writing processed files.
            RuntimeError: If `torch.save` fails for a produced object.
        """
        extract_path = osp.join(self.raw_dir, self.folder_name)
        if not osp.isdir(extract_path):
            self.unzip_file()

        # Load Metadata
        with open(osp.join(extract_path, 'meta.yaml'), "r") as f:
            ydata = yaml.safe_load(f)

        with open(osp.join(extract_path, 'splits.yaml'), "r") as f:
            splits_data = yaml.safe_load(f)

        split_indices = self.get_split_indices(splits_data)
        with open(osp.join(self.processed_dir, "splits.yaml"), "w") as f:
            yaml.safe_dump(split_indices, f)

        # Process Files
        in_path = Path(extract_path)
        sorted_files = sorted(in_path.glob("*.npy"))

        for file in tqdm(sorted_files, desc=f"Processing {self.folder_name}"):
            data = self.process_file(file, ydata)
            stem = Path(file).stem
            out_file = osp.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, out_file)
        Path(self.processed_dir, "_done").touch()

    def _load(self):
        """Load processed `.pt` files and dataset metadata into memory.

        This method:
          - Loads all `.pt` files in `processed_dir` (sorted by filename) into
            `self.data`.
          - Applies `fixed_transform` (if provided) once per loaded sample.
          - Loads split definitions from `processed_dir/splits.yaml` into `self.splits`.
          - Computes `self.classes` and `self.num_classes` from observed labels.

        Raises:
            FileNotFoundError: If `processed_dir/splits.yaml` is missing.
            RuntimeError: If `torch.load` fails for any `.pt` file.
            yaml.YAMLError: If `splits.yaml` cannot be parsed.
        """
        self.data = []
        in_path = Path(self.processed_dir)
        sorted_files = sorted(in_path.glob("*.pt"))
        for file in tqdm(sorted_files, desc=f"Loading {self.folder_name}"):
            data_i = torch.load(file, weights_only=False)
            if self.fixed_transform is not None:
                data_i = self.fixed_transform(data_i)
            self.data.append(data_i)

        with open(osp.join(self.processed_dir, "splits.yaml"), "r") as f:
            self.splits = yaml.safe_load(f)
        self.classes = sorted({int(data.y) for data in self})
        self.num_classes = len(self.classes)

    def download(self):
        """Download the dataset archive from Google Drive into `raw_dir`.

        Constructs a Google Drive download URL from `file_id` and downloads
        into `raw_dir/<raw_file_names[0]>` using `gdown`. After downloading,
        calls `validate(...)` to check integrity.

        Raises:
            IndexError: If `raw_file_names` is empty.
            OSError: If the destination file cannot be written.
            Exception: Propagates errors from `gdown.download(...)`.
        """
        url = f'https://drive.google.com/uc?id={self.file_id}'

        # Use gdown to handle the download
        output = os.path.join(self.raw_dir, self.raw_file_names[0])
        gdown.download(url, output, quiet=False)
        self.validate(output)

    def validate(self, file_path: Path):
        """Validate a downloaded archive against the expected SHA256 checksum.

        Computes the SHA256 digest of `file_path` and compares it to `self.checksum`.
        If they do not match, emits a `UserWarning`.

        Args:
            file_path (pathlib.Path | str): Path to the downloaded archive.

        Warns:
            UserWarning: If the computed checksum differs from the expected checksum.

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            OSError: For I/O errors reading the file.
        """
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
                if not paths:
                    return None
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

    def process(self):
        split_indices = {}
        n = len(self.k) * self.num_per_class
        for i in range(10):
            split = {}
            indices = self.rng.permutation(np.arange(n))
            split['trn'] = indices[:int(n * 0.72)].tolist()
            split['val'] = indices[int(n * 0.72):int(n * 0.80)].tolist()
            split['tst'] = indices[int(n * 0.80):].tolist()
            split_indices[i] = split
        with open(osp.join(self.processed_dir, "splits.yaml"), "w") as f:
            yaml.safe_dump(split_indices, f)

        ks, num_per_class, num_points = self.k, self.num_per_class, self.num_points
        rect_min = [0.0, 0.0, 0.0]
        rect_max = [5.0, 5.0, 5.0]

        for ki, k in enumerate(ks):
            for r in tqdm(range(num_per_class)):
                void_radius_range = (0.1, 0.5)
                points, _, _ = generate_swiss_cheese_points(
                    num_points, rect_min, rect_max, k, void_radius_range
                )                
                # points, _, _ = self.generate_swiss_cheese_points_fast(
                #    num_points, rect_min, rect_max, k, void_radius_range
                # )
                data = FlooderData(x=points.to(torch.float32), y=ki, name=f'{k}voids_{r}')
                file_id = hashlib.sha256(points.numpy().tobytes()).hexdigest()[:10]
                torch.save(data, osp.join(self.processed_dir, f"{file_id}.pt"))
        Path(self.processed_dir, "_done").touch()

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
