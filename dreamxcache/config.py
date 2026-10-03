"""Filesystem layout and third-party locations.

Nothing in this repository ships data. Every path below is resolved at call
time from an environment variable, falling back to a directory relative to
the current working directory (run scripts from the repository root).

    DREAMXCACHE_DATA_DIR         default ./data
        raw/         files you obtained yourself (see DATA.md)
        cache/       fingerprints, score arrays and other derived artifacts
        external/    public structures and ligand tables (PDB, ChEMBL, ...)
        submissions/ generated submission files
    DREAMXCACHE_RESULTS_DIR      default ./results-local (run outputs)
    DREAMXCACHE_THIRD_PARTY_DIR  default ./third_party (see scripts/third_party.sh)

`data/`, `results-local/` and `third_party/` are git-ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def data_dir() -> Path:
    return _env_path("DREAMXCACHE_DATA_DIR", Path.cwd() / "data")


def raw_dir() -> Path:
    return data_dir() / "raw"


def cache_dir() -> Path:
    return data_dir() / "cache"


def external_dir() -> Path:
    return data_dir() / "external"


def submissions_dir() -> Path:
    return data_dir() / "submissions"


def results_dir() -> Path:
    return _env_path("DREAMXCACHE_RESULTS_DIR", Path.cwd() / "results-local")


def third_party_dir() -> Path:
    return _env_path("DREAMXCACHE_THIRD_PARTY_DIR", Path.cwd() / "third_party")


def timestamped_results_subdir(label: str) -> Path:
    """Create and return a fresh `<results_dir>/<UTC timestamp>-<label>` directory.

    Runs never overwrite earlier runs: a new invocation always gets a new
    directory.
    """
    from datetime import UTC, datetime

    out = results_dir() / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{label}"
    out.mkdir(parents=True, exist_ok=False)
    return out
