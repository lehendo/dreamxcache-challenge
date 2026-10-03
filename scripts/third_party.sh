#!/usr/bin/env bash
# Clone the third-party repositories this project uses, at pinned commits.
# They are referenced, not redistributed: nothing from them is copied into this
# repository, and third_party/ is git-ignored.
#
#   ./scripts/third_party.sh            # into ./third_party (or $DREAMXCACHE_THIRD_PARTY_DIR)
set -euo pipefail

DEST="${DREAMXCACHE_THIRD_PARTY_DIR:-$(pwd)/third_party}"
mkdir -p "$DEST"

clone_pinned() {
  local url="$1" dir="$2" sha="$3"
  local target="$DEST/$dir"
  if [ -d "$target" ]; then
    echo "skip: $target already exists"
    return
  fi
  git clone "$url" "$target"
  git -C "$target" checkout "$sha"
  echo "pinned $dir @ $sha"
}

# Official DREAM x CACHE Target2035 evaluator and fingerprint extraction (MIT).
# Used unmodified: dreamxcache.eval.scorer wraps the evaluator,
# dreamxcache.features.ecfp4 calls the fingerprint extraction.
clone_pinned \
  "https://github.com/StructuralGenomicsConsortium/Target2035_Aircheck_Utils.git" \
  "aircheck_utils" \
  "c3472a4c6185e5dc74a6da9d28d4f80aeb4ebfcf"

# BitBIRCH clustering (LGPL-3.0), loaded via sys.path by
# dreamxcache.negatives.bitbirch_clustering. Needed only for BitBIRCH-based
# stratification; its own packaging pins numpy<2, which is why it is not pip-installed.
clone_pinned \
  "https://github.com/mqcomplab/bitbirch.git" \
  "bitbirch" \
  "311fbf7dcc8ba019a53f58e75d40ebce6a3bc76f"

# Reference SMILES deduplication procedure. Reference only: dreamxcache re-implements
# the same Stouffer combination (sum(z) / sqrt(n)) in dreamxcache.ingest.load, so nothing
# imports this checkout. Cloned so the two can be compared.
clone_pinned \
  "https://github.com/VonBoss/DREAM_DEL_deduplication.git" \
  "dedup" \
  "a81b7dc4940fed6178ac723ffe8f66d7a7568bd4"
