"""IO functionality (for consistent saving).

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import datetime
from pathlib import Path
from typing import Any, Union

import torch


def save_to_disk(
    obj: Any,
    path: Union[str, Path],
    metadata: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Save an object to disk using `torch.save`.

    This function saves any Python object to a specified path. If the object is a
    dictionary and `metadata=True`, a `_meta` entry is added to the copy with
    timestamp and key information. If the file already exists and `overwrite=False`,
    an error is raised.

    Args:
        obj (Any): The Python object to save (e.g., a tensor, dictionary, or model).
        path (Union[str, Path]): The file path where the object will be saved.
        metadata (bool, optional): Whether to include metadata in the saved file
            (only applicable if `obj` is a dictionary). Defaults to True.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.
            Defaults to False.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is False.

    Returns:
        None
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    to_save = obj

    if metadata and isinstance(obj, dict):
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "keys": list(obj.keys()),
        }
        # Avoid overwriting an existing "_meta" key
        to_save = obj.copy()
        to_save.setdefault("_meta", meta)

    torch.save(to_save, path)
