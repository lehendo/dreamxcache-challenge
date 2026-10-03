"""Boltz-2 co-folding affinity harness: inputs, pre-flight checks, aggregation.

Boltz-2 co-folds a protein sequence with a ligand and predicts a binding
affinity. It is far too slow to score a full library (on the order of 150
complexes per H100-hour), so it is used as a *second-stage* channel on a
shortlist built from cheaper channels, or on a test shortlist.

This module covers what surrounds the model call:

- `check_a3m` / `strip_nul_bytes`: **a pre-flight check every MSA file must
  pass.** MSAs fetched through Boltz's MSA-server path can end with a stray
  NUL byte. The ``boltz predict`` run then fails to parse *every* input with
  ``Failed to process <file>. Skipping. Error: '\\x00'`` and exits with
  status 0. A chunk "completes" in seconds with no predictions. A job that
  finishes far faster than the model can run is a symptom, not a success.
- `write_yamls`: one input YAML per compound, MSA-only (no templates), skipping
  compounds above the affinity module's 128-heavy-atom limit.
- `aggregate_affinities`: per-compound affinity JSON files from any number of
  output directories into one NaN-safe score array.

Settings used with the CLI (``boltz predict <dir>``): ``--sampling_steps 50
--diffusion_samples 1 --sampling_steps_affinity 50
--diffusion_samples_affinity 1 --recycling_steps 1``. Ranking uses
``affinity_probability_binary``.

Templates are deliberately not used: in Boltz 2.2.1, supplying a template
structure raised ``IndexError`` inside ``parse_polymer`` for the structures
tried, across raw, cleaned and renumbered inputs. MSA-based co-folding alone is
the default mode. Boltz-2 has no input for retaining crystallographic waters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

MAX_HEAVY_ATOMS = 128  # affinity module's hard limit

YAML_TEMPLATE = """version: 1
sequences:
  - protein:
      id: A
      sequence: "{sequence}"
      msa: {msa_path}
  - ligand:
      id: B
      smiles: "{smiles}"
properties:
  - affinity:
      binder: B
"""


class NulByteInMsaError(ValueError):
    pass


def check_a3m(path: Path) -> None:
    """Raise `NulByteInMsaError` if the MSA file contains any NUL byte."""
    n_nul = path.read_bytes().count(b"\x00")
    if n_nul:
        raise NulByteInMsaError(
            f"{path} contains {n_nul} NUL byte(s). Boltz would skip every input that uses it "
            "and still exit 0. Run `python scripts/boltz2_generate_yamls.py --fix-msa` or "
            "dreamxcache.rankers.boltz2.strip_nul_bytes(path) first."
        )


def strip_nul_bytes(path: Path) -> int:
    """Remove NUL bytes from an MSA file in place; returns how many were removed."""
    data = path.read_bytes()
    n_nul = data.count(b"\x00")
    if n_nul:
        path.write_bytes(data.replace(b"\x00", b""))
    return n_nul


def write_yamls(
    compounds: pd.DataFrame,
    id_col: str,
    smiles_col: str,
    sequence: str,
    msa_path: Path,
    out_dir: Path,
) -> tuple[int, list[tuple[str, str]]]:
    """Write ``<id>.yaml`` for every compound. Returns ``(n_written, skipped)``
    where `skipped` lists ``(id, reason)`` for unparseable or oversized
    compounds. Refuses to run if the MSA file fails `check_a3m`."""
    check_a3m(msa_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    skipped: list[tuple[str, str]] = []
    for _, row in compounds.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            skipped.append((str(row[id_col]), "unparseable"))
            continue
        n_heavy = mol.GetNumAtoms()
        if n_heavy > MAX_HEAVY_ATOMS:
            skipped.append((str(row[id_col]), f"{n_heavy} heavy atoms > {MAX_HEAVY_ATOMS}"))
            continue
        content = YAML_TEMPLATE.format(sequence=sequence, msa_path=msa_path.resolve(), smiles=row[smiles_col])
        (out_dir / f"{row[id_col]}.yaml").write_text(content)
        n_written += 1
    return n_written, skipped


def split_into_chunks(yaml_dir: Path, out_root: Path, n_chunks: int) -> list[Path]:
    """Partition the YAMLs in `yaml_dir` into `n_chunks` directories under
    `out_root` (``chunk_00``, ``chunk_01``, ...) of near-equal size. One
    ``boltz predict <chunk_dir>`` job per chunk reuses a single loaded model
    across all its compounds. Keep chunks small (around 100 compounds). In my
    runs Boltz started one preprocessing thread per input compound ("Processing
    94 inputs with 94 threads"), and 600-compound chunks failed on shared
    nodes while 94-compound chunks completed; give each job generous memory
    (a 94-compound chunk was killed at 16 GB on one cluster)."""
    import math
    import shutil

    files = sorted(p for p in yaml_dir.iterdir() if p.suffix == ".yaml")
    chunk_size = math.ceil(len(files) / n_chunks)
    dirs: list[Path] = []
    for i in range(n_chunks):
        chunk_files = files[i * chunk_size : (i + 1) * chunk_size]
        if not chunk_files:
            continue
        d = out_root / f"chunk_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for f in chunk_files:
            shutil.copy(f, d / f.name)
        dirs.append(d)
    return dirs


def aggregate_affinities(
    out_dirs: list[Path], ids: list[str], score_field: str = "affinity_probability_binary"
) -> tuple[np.ndarray, dict[str, int]]:
    """Per-compound affinity JSONs (``affinity_<id>.json``, found recursively
    under each output directory) -> a score array aligned to `ids`.

    Compounds with no result get NaN, **not** 0: ``affinity_probability_binary``
    is itself a probability, so a genuine low-confidence score and "never ran"
    are both near zero and must not be interchangeable. Downstream ranking must
    drop NaN *before* taking a top-N (``np.argsort(-scores)`` ranks NaN last, but
    a top-N slice that runs past the finite scores would include them, and
    ``np.argsort(scores)[::-1]`` would put them first).

    Output directories may overlap (redundant copies of the same chunk run on
    different hardware); a compound found twice keeps its last value.
    """
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    scores = np.full(len(ids), np.nan, dtype=np.float32)
    stats = {"n_found": 0, "n_missing_field": 0, "n_unmatched": 0}
    for out_dir in out_dirs:
        for affinity_json in out_dir.rglob("affinity_*.json"):
            compound_id = affinity_json.stem.removeprefix("affinity_")
            if compound_id not in id_to_row:
                stats["n_unmatched"] += 1
                continue
            data = json.loads(affinity_json.read_text())
            if score_field not in data:
                stats["n_missing_field"] += 1
                continue
            scores[id_to_row[compound_id]] = data[score_field]
            stats["n_found"] += 1
    return scores, stats
