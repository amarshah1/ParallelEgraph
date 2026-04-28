"""Shared helpers for the bench driver scripts."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
SYNTHETIC_DIR = REPO_ROOT / "examples" / "synthetic"


def ensure_built(binary: str) -> Path:
    """Resolve a bench binary by name under build/, exit if missing."""
    p = BUILD_DIR / binary
    if not p.exists():
        sys.exit(
            f"missing binary: {p}\n"
            f"build first:  cmake --build {BUILD_DIR.relative_to(REPO_ROOT)}"
        )
    return p


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def run_capture(
    cmd: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    append: bool = False,
) -> int:
    """Run cmd with env overrides; pipe stdout/stderr to files (or inherit)."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    mode = "ab" if append else "wb"
    out_f = open(stdout_path, mode) if stdout_path else None
    err_f = open(stderr_path, mode) if stderr_path else None
    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=out_f if out_f else None,
            stderr=err_f if err_f else None,
        )
    finally:
        if out_f:
            out_f.close()
        if err_f:
            err_f.close()
    return result.returncode


def threads_default() -> list[int]:
    """Default thread sweep: powers of two up to logical CPU count, plus the cap."""
    nproc = os.cpu_count() or 8
    seq = []
    t = 1
    while t < nproc:
        seq.append(t)
        t *= 2
    seq.append(nproc)
    # de-dup while preserving order
    seen = set()
    out = []
    for t in seq:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def write_csv_header_once(path: Path, header: str) -> None:
    """If `path` doesn't exist, write the header line; otherwise leave alone."""
    if not path.exists():
        path.write_text(header + "\n")


def append_csv(path: Path, src: Path) -> None:
    """Append rows from `src` to `path` (assumes both have the same schema)."""
    with open(src, "rb") as fsrc, open(path, "ab") as fdst:
        shutil.copyfileobj(fsrc, fdst)
