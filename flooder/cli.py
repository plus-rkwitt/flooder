#!/usr/bin/env python3
"""
Implementation of flooder CLI.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf,
Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import re
import argparse
import pickle
import json
import time
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from rich.console import Console
from rich_argparse import RichHelpFormatter

try:
    import psutil
except Exception:
    psutil = None

from flooder import flood_complex

console = Console()


@dataclass
class StepStats:
    name: str
    wall_s: float
    cpu_s: float
    ram_delta_mib: Optional[float]
    gpu_peak_mib: Optional[float]
    cuda_ms: Optional[float]


# Meta data for writing to output file
@dataclass
class RunMeta:
    input_file: str
    output_file: Optional[str]
    num_landmarks: int
    max_dimension: int
    fps_height: int
    batch_size: int
    device: str
    points_per_edge: Optional[int]
    num_rand: Optional[int]
    seed: Optional[int]
    use_triton: bool
    n_points: int
    ambient_dim: int


class StepTimer:
    def __init__(self, name: str, device: torch.device, use_cuda_events: bool = False):
        self.name = name
        self.device = device
        self.use_cuda_events = use_cuda_events and (device.type == "cuda")
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._ram_before = None
        self.cuda_start = None
        self.cuda_end = None

    def __enter__(self):
        # CPU/Wall
        self._t0_wall = time.perf_counter()
        self._t0_cpu = time.process_time()

        # RAM
        if self._proc:
            try:
                self._ram_before = self._proc.memory_info().rss
            except Exception:
                self._ram_before = None

        # GPU peak mem + CUDA events
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.use_cuda_events:
                self.cuda_start = torch.cuda.Event(enable_timing=True)
                self.cuda_end = torch.cuda.Event(enable_timing=True)
                self.cuda_start.record()

        return self

    def __exit__(self, exc_type, exc, tb):
        # CPU/Wall
        wall = time.perf_counter() - self._t0_wall
        cpu = time.process_time() - self._t0_cpu

        # RAM delta
        ram_delta_mib = None
        if self._proc and self._ram_before is not None:
            try:
                ram_after = self._proc.memory_info().rss
                ram_delta_mib = (ram_after - self._ram_before) / (1024**2)
            except Exception:
                ram_delta_mib = None

        # GPU peak
        gpu_peak_mib = None
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device)
            gpu_peak_mib = peak / (1024**2)
            if self.use_cuda_events and self.cuda_start and self.cuda_end:
                self.cuda_end.record()
                torch.cuda.synchronize(self.device)  # ensure timing complete
                self.cuda_ms = self.cuda_start.elapsed_time(self.cuda_end)
            else:
                self.cuda_ms = None
        else:
            self.cuda_ms = None

        self.stats = StepStats(
            name=self.name,
            wall_s=wall,
            cpu_s=cpu,
            ram_delta_mib=ram_delta_mib,
            gpu_peak_mib=gpu_peak_mib,
            cuda_ms=self.cuda_ms,
        )


def print_stats_table(steps: List[StepStats], console):
    tbl = Table(title="Flooder runtime statistics", box=box.SIMPLE_HEAVY)
    tbl.add_column("Step", justify="left")
    tbl.add_column("Wall (s)", justify="right")
    tbl.add_column("CPU (s)", justify="right")
    tbl.add_column("GPU peak (MiB)", justify="right")
    tbl.add_column("RAM Δ (MiB)", justify="right")
    tbl.add_column("CUDA (ms)", justify="right")

    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{x:.{nd}f}"

    for s in steps:
        tbl.add_row(
            s.name,
            fmt(s.wall_s),
            fmt(s.cpu_s),
            fmt(s.gpu_peak_mib),
            fmt(s.ram_delta_mib),
            fmt(s.cuda_ms),
        )
    console.print(tbl)


def device_type(value: str) -> str:
    if value == "cpu":
        return value
    match = re.fullmatch(r"cuda:\d+", value)
    if match:
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid device '{value}'. Must be 'cpu' or 'cuda:<id>' with <id> an integer."
    )


def dump_stats_json(steps: List[StepStats], out_path: Optional[str]):
    if not out_path:
        return
    payload = [s.__dict__ for s in steps]
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(payload, f, indent=2)


def setup_cmdline_parsing() -> argparse.ArgumentParser:
    description = """**Flooder options**"""
    p = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=lambda prog: RichHelpFormatter(
            prog,
            max_help_position=35,
            width=100,
        ),
    )

    g0 = p.add_argument_group("Flooder options")
    g0.add_argument(
        "--num-landmarks",
        metavar="INT",
        type=int,
        default=2000,
        help="Number of landmarks for Flood complex (default: %(default)s)",
    )
    g0.add_argument(
        "--max-dimension",
        metavar="INT",
        type=int,
        default=None,
        help="Compute PH up to max. dimension (exclusive) (default: ambient dim)",
    )
    g0.add_argument(
        "--fpsh",
        dest="fps_height",
        metavar="INT",
        type=int,
        default=9,
        help="Farthest-Point Sampling height (default: %(default)s)",
    )
    g0.add_argument(
        "--batch-size",
        metavar="INT",
        type=int,
        default=64,
        help="Batch size for Flood complex (default: %(default)s)",
    )
    g0.add_argument(
        "--device",
        type=device_type,
        default="cuda:0",
        help='Device: "cpu", or "cuda:N" (default: %(default)s)',
    )
    g0.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=None,
        help="Random seed (only used when --num-rand is set)",
    )
    g0.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (enabled by default)",
    )
    mex = g0.add_mutually_exclusive_group(required=False)
    mex.add_argument(
        "--points-per-edge",
        metavar="INT",
        type=int,
        default=None,
        help="Points per edge for Flood PH (default: 30 if neither option given)",
    )
    mex.add_argument(
        "--num-rand",
        metavar="INT",
        type=int,
        default=None,
        help="Number of random points per simplex (default: None)",
    )
    g1 = p.add_argument_group("Input/Output options")
    g1.add_argument(
        "--input-file",
        metavar="FILE",
        type=str,
        required=True,
        help="NumPy .npy file with a (N, D) point cloud",
    )
    g1.add_argument(
        "--output-file",
        metavar="FILE",
        type=str,
        default=None,
        help="Output pickle (.pkl) with persistence diagrams + metadata",
    )
    g1.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print parsed arguments",
    )
    g1.add_argument(
        "--stats-json",
        metavar="FILE",
        type=str,
        default=None,
        help="Write runtime statistics to JSON",
    )
    g1.add_argument(
        "--cuda-events",
        action="store_true",
        help="Also measure CUDA kernel time with CUDA events",
    )
    return p


def validate_device(device_str: str) -> torch.device:
    """Validate device availability and capabilities.

    Args:
        device_str (str): Device string, e.g., "cpu", "cuda", "cuda:0".

    Raises:
        RuntimeError: If the requested device is not available.
        RuntimeError: If the CUDA compute capability is insufficient.

    Returns:
        torch.device: Device object.
    """
    dev = torch.device(device_str)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu.")
        major, minor = torch.cuda.get_device_capability(dev)
        if (major, minor) <= (7, 5):
            raise RuntimeError(
                f"CUDA compute capability {major}.{minor} detected; " "requires >= 7.5."
            )
        torch.cuda.set_device(dev)
    return dev


def load_point_cloud(path: Path) -> Tuple[torch.Tensor, int, int]:
    """Load a point cloud from a NumPy file.

    Args:
        path (Path): Path to the input .npy file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded.
        ValueError: If the array is not (n, d)

    Returns:
        Tuple[torch.Tensor, int, int]: A tuple containing the point cloud tensor,
        the number of points (n), and the ambient dimensionality (d).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as e:
        raise ValueError(f"Failed to load NumPy file '{path}': {e}") from e
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array (N, D); got shape {arr.shape}")

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    tensor = torch.from_numpy(arr.copy())
    n, d = tensor.shape
    return tensor, n, d


def effective_max_dim(user_max: Optional[int], ambient_dim: int) -> int:
    """Determine the effective maximum PH dimension.

    Args:
        user_max (Optional[int]): User-specified maximum dimension (can be None).
        ambient_dim (int): Ambient dimensionality of the point cloud.

    Raises:
        ValueError: If user_max is not positive.
        ValueError: If user_max exceeds ambient_dim.

    Returns:
        int: Effective maximum PH dimension.
    """
    if user_max is None:
        return ambient_dim
    if user_max < 1:
        raise ValueError("--max-dimension must be >= 1")
    if user_max > ambient_dim:
        raise ValueError(
            f"--max-dimension ({user_max}) cannot exceed ambient dimension ({ambient_dim})"
        )
    return user_max


def resolve_simplex_representation(
    points_per_edge: Optional[int], num_rand: Optional[int]
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve simplex representation either via grid or random sampling. When nothing
    is specified, default to 30 points per edge.

    Args:
        points_per_edge (Optional[int]): Number of points to sample per edge.
        num_rand (Optional[int]): Number of random points to sample.

    Returns:
        Tuple[Optional[int], Optional[int]]: Resolved points_per_edge and num_rand.
    """
    if points_per_edge is None and num_rand is None:
        return 30, None
    return points_per_edge, num_rand


def maybe_seed(seed: Optional[int]) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (Optional[int]): Random seed value.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_output(path: Path, diagrams, meta: RunMeta) -> None:
    """Save the output diagrams and metadata to a file.

    Args:
        path (Path): Path to the output file.
        diagrams (List[numpy.ndarray]): List of GUDHI persistence diagrams to save.
        meta (RunMeta): Metadata about the run.
    """
    if path.suffix == "":
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"diagrams": diagrams, "meta": asdict(meta)}
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def main() -> None:
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    if args.verbose:
        console.print(vars(args))

    if args.device == "cuda":
        args.device = "cuda:0"
    device = validate_device(args.device)

    stats: List[StepStats] = []

    with console.status("Loading point cloud...", spinner="dots"):
        with StepTimer("Loading", device, use_cuda_events=args.cuda_events) as t:
            pc_cpu, n_pts, dim = load_point_cloud(Path(args.input_file))
        stats.append(t.stats)
    console.print(f"✓ Loading point cloud ({n_pts},{dim}) done")

    max_dim = effective_max_dim(args.max_dimension, dim)
    points_per_edge, num_rand = resolve_simplex_representation(
        args.points_per_edge, args.num_rand
    )
    maybe_seed(args.seed if num_rand is not None else None)

    with console.status("Building Flood complex...", spinner="dots"):
        with StepTimer("Flood complex", device, use_cuda_events=args.cuda_events) as t:
            pc = pc_cpu.to(device, non_blocking=True)
            use_triton = not args.no_triton
            fc_st = flood_complex(
                pc,
                args.num_landmarks,
                max_dimension=max_dim,
                points_per_edge=points_per_edge,
                batch_size=args.batch_size,
                fps_h=args.fps_height,
                use_triton=use_triton,
                return_simplex_tree=True,
                num_rand=num_rand,
            )
        stats.append(t.stats)
    console.print(
        f"✓ Building Flood complex with {fc_st.num_simplices()} simplices done"
    )

    with console.status("Computing persistence...", spinner="dots"):
        with StepTimer("Persistence", device, use_cuda_events=args.cuda_events) as t:
            fc_st.compute_persistence()
            diagrams = [
                fc_st.persistence_intervals_in_dimension(i) for i in range(max_dim)
            ]
    console.print(f"✓ Computing persistence up to max. dim {max_dim} done")
    print()
    stats.append(t.stats)

    if args.output_file:
        meta = RunMeta(
            input_file=args.input_file,
            output_file=args.output_file,
            num_landmarks=args.num_landmarks,
            max_dimension=max_dim,
            fps_height=args.fps_height,
            batch_size=args.batch_size,
            device=str(device),
            points_per_edge=points_per_edge,
            num_rand=num_rand,
            seed=args.seed if num_rand is not None else None,
            use_triton=use_triton,
            n_points=n_pts,
            ambient_dim=dim,
        )
        save_output(Path(args.output_file), diagrams, meta)

    print_stats_table(stats, console)
    dump_stats_json(stats, args.stats_json)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
