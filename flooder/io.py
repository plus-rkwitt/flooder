"""IO functionality (for consistent saving).

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

import torch
import datetime
from pathlib import Path
from typing import Any, Union


def save_to_disk(
    obj: Any,
    path: Union[str, Path],
    metadata: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Save any Python object to disk using torch.save().

    Args:
        obj: Any Python object (tensor, dict, custom class, etc.)
        path: Destination file path.
        metadata: If True and obj is a dict, add _meta with timestamp and key info.
        overwrite: If False, raise error if file already exists.
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
