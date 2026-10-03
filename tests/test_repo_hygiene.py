"""Guards against the repository ever shipping data or machine-specific paths.

Runs over every file in the working tree that git would not ignore. Cheap, and
the first line of defence if a data file is added by accident (it will fail
here even before it reaches a commit).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DATA_SUFFIXES = {
    ".parquet",
    ".feather",
    ".arrow",
    ".h5",
    ".hdf5",
    ".h5ad",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
    ".sdf",
    ".smi",
    ".a3m",
    ".fasta",
    ".fa",
    ".pdb",
    ".cif",
    ".mmcif",
    ".zip",
    ".tar",
    ".gz",
    ".csv",
    ".tsv",
}
MAX_BYTES = 1_000_000
# Machine-specific absolute paths and credentials-like strings.
FORBIDDEN_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"(?<![A-Za-z0-9_.])/u/[a-z0-9]+/"),
    re.compile(r"/scratch/[A-Za-z0-9._-]+"),
    re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret[_-]?token)\s*[:=]\s*['\"]?\w{8,}"),
]
SELF = Path(__file__).resolve()


def _visible_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    return [ROOT / p for p in out if (ROOT / p).is_file()]


def test_no_data_files_or_oversized_files():
    offenders = []
    for path in _visible_files():
        if path.suffix.lower() in DATA_SUFFIXES or path.stat().st_size > MAX_BYTES:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"data-like or oversized files would be committed: {offenders}"


def test_no_machine_specific_paths_or_secrets():
    offenders = []
    for path in _visible_files():
        if path == SELF or path.suffix.lower() in {".png", ".jpg"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_PATTERNS:
            m = pattern.search(text)
            if m:
                offenders.append(f"{path.relative_to(ROOT)}: {m.group(0)!r}")
    assert not offenders, f"machine-specific paths or secret-like strings found: {offenders}"
