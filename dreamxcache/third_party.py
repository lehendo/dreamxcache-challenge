"""Locating third-party code that is referenced, not copied.

Three external repositories are used and none is redistributed here; they
are cloned at pinned commits by ``scripts/third_party.sh`` into
``DREAMXCACHE_THIRD_PARTY_DIR`` (default ``./third_party``). The imports below
are lazy so that the rest of the package works without them.

    aircheck_utils   official evaluator and fingerprint extraction
    bitbirch         BitBIRCH clustering (negative-sampling stratification)
    dedup            reference SMILES deduplication procedure (reference only)
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from dreamxcache.config import third_party_dir


def require_dir(name: str) -> Path:
    path = third_party_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/third_party.sh (or set DREAMXCACHE_THIRD_PARTY_DIR); see DATA.md."
        )
    return path


def is_available(name: str) -> bool:
    return (third_party_dir() / name).exists()


def load_module_from_file(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_to_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
