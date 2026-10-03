"""BitBIRCH clustering of binary fingerprints, used to partition a compound
pool for stratified negative sampling.

BitBIRCH (https://github.com/mqcomplab/bitbirch) is referenced, not copied:
clone it with ``scripts/third_party.sh``. It is loaded through ``sys.path``
rather than pip-installed because its own packaging pins ``numpy<2``, while
the parts used here (``BitBirch.fit``, ``get_cluster_mol_ids``) only need
numpy and scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dreamxcache.third_party import add_to_sys_path, require_dir

_loaded: tuple[type, Any] | None = None


def _load_bitbirch() -> tuple[type, Any]:
    global _loaded
    if _loaded is None:
        add_to_sys_path(require_dir("bitbirch"))
        from bitbirch import bitbirch as bb_module  # type: ignore[import-not-found]

        bitbirch_cls: type = bb_module.BitBirch
        _loaded = (bitbirch_cls, bb_module)
    return _loaded


def cluster_fingerprints(
    fps: np.ndarray, threshold: float, branching_factor: int = 50, merge_criterion: str = "diameter"
) -> tuple[list[list[int]], np.ndarray]:
    """Cluster an ``(n, d)`` binary fingerprint array with BitBIRCH.

    **The input is upcast to int32 before fitting, whatever its dtype.**
    ``np.dot`` on ``int8`` arrays does not widen the accumulator, so the
    dot product of two 2048-bit fingerprints (their shared-bit count, easily
    above 127) silently wraps modulo 256: ``np.dot(np.ones(2048, np.int8),
    np.ones(2048, np.int8))`` is ``0``, not ``2048``. BitBIRCH runs all of its
    internal similarity arithmetic in whatever dtype it is given, so int8
    input corrupts every similarity it computes, independent of the threshold
    or branching factor. Compact int8 fingerprint caches are fine on disk;
    only the one call that does dot products needs the wider type. (int32
    rather than int64: at tens of millions of rows by 2048 bits, int64 doubles
    memory for no benefit, since 2^31 is far beyond any dot product or linear
    sum of binary 2048-bit vectors.)

    ``set_merge`` sets a module-global merge function that the library reads
    at fit time, so it is called before every ``fit()`` rather than once at
    import.

    `threshold` has no cross-dataset default: it controls how many clusters
    result.

    Returns ``(clusters, cluster_id)``: `clusters` is a list of clusters, each
    a list of row indices, largest first; `cluster_id` is an ``(n,)`` integer
    array mapping each row to its cluster.
    """
    bitbirch_cls, bb_module = _load_bitbirch()
    fps_safe = fps.astype(np.int32)
    bb_module.set_merge(merge_criterion)
    model = bitbirch_cls(threshold=threshold, branching_factor=branching_factor)
    model.fit(fps_safe)
    clusters: list[list[int]] = model.get_cluster_mol_ids()
    cluster_id: np.ndarray = model.get_assignments(n_mols=len(fps)) - 1  # library ids are 1-indexed
    return clusters, cluster_id
