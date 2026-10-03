# DREAM x CACHE Target2035 DEL-ML Challenge (Phase 1)

A pipeline, and the lessons from running it, for the **DREAM x CACHE Target2035
DEL-ML challenge**: predict binders of PGK2 from a DNA-encoded-library (DEL)
selection, submitting 50 compounds per test split, scored on held-out
mass-spectrometry hits.

**Both of my blind-test submissions scored 0 hits.**

This repository is organized around what I learned, not around features. The code
(labeling recipes, ranking channels, fusion, evaluation) is here so the findings can
be checked and the methods reused. No data is included; see [DATA.md](DATA.md).

## Findings

### 1. Both blind-test submissions scored zero hits

| submission | method | blind-test hits | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| 1 | similarity to two known ligands (ErG), top 40 + top 10 | **0** | 0.695 | 0.002 |
| 2 | rank fusion of Boltz-2 co-folding, a water-mediated pharmacophore and a DEL classifier, with diversity selection | **0** | 0.705 | 0.002 |

The first submission had scored 6 chemical series (7 hits) on the validation split
before it was submitted. Numbers: [`results/blind_test_outcomes.json`](results/blind_test_outcomes.json).

### 2. Adaptive overfitting: about 60 validation submissions, 6 series on validation, 0 on test

Validation submissions returned scores as feedback but did not count toward the final
result, which made them tempting to use as an optimization signal. The validation split
was queried about 60 times, each query chosen after seeing the last answer. The challenge deliberately makes validation and test scaffold-disjoint, which
is the case where adaptive reuse hurts most: a method tuned until it scores well on one
set of scaffolds has been selected for *that* set, and a single-anchor similarity
method has no reason to transfer to scaffolds it does not resemble. The configuration
that looked best on validation scored zero on test.

This is the failure that the Ladder (Blum and Hardt 2015) and the reusable holdout
(Dwork et al. 2015) are designed to prevent: answering adaptive queries about a
held-out set erodes its validity unless the answers are limited or noised. I used
neither mechanism, and the validation score leaked into every decision. After diagnosing
this I stopped sweeping fusion weights and ratios against validation and fixed them in
advance.

### 3. ROC-AUC 0.695 and 0.705 with PR-AUC 0.002 on both: global signal, no early recognition

Both submissions have a respectable ROC-AUC and a PR-AUC at the level of an uninformed
ranking (hits are on the order of 0.1% of the test compounds). The metrics disagree
because they look at different parts of the ranking.

ROC-AUC integrates over *all* ~185,000 compounds. A model can move hits from the median
of the ranking to the 70th percentile, which is real signal and an ROC-AUC near 0.7,
while placing none of them in the top 50, which is the only part that is ever scored.
PR-AUC and the early-recognition metrics (enrichment factor, LogAUC) look at the head
of the ranking. A decent ROC-AUC with a no-skill PR-AUC is the signature of a method with
no early recognition (Truchon and Bayly 2007). I compared channels with the wrong
metric for the question until the blind test showed it. Everything in
[`dreamxcache/eval/early_recognition.py`](dreamxcache/eval/early_recognition.py) exists
because of this.

### 4. Channel validation against a permutation null

The ranking channels in the second submission were each tested on a *known-answer*
library: a target with experimentally confirmed hits that the channel was never tuned
against. Each result is read as a **percentile of that library's own permutation null**
(1,000 noise rankings; seed 42), not against a theoretical baseline.

| channel | library (compounds, hits) | LogAUC (null percentile) | EF@1% (null percentile) | EF@50 (null percentile) |
|---|---|---|---|---|
| Boltz-2 co-folding affinity | LRRK2 WD40 shortlist (1,500; 9) | 0.311 (**99.7**) | 0.0 (91.4) | 9.99 (99.8) |
| Structure-based pharmacophore | WDR91 library (15,419; 19) | 0.218 (**100.0**) | 10.54 (99.7) | 16.23 (94.3) |
| ErG similarity to a co-crystal ligand | LRRK2 shortlist (1,500; 9) | 0.039 (**0.2**) | 0.0 (91.4) | 0.0 (74.9) |
| ErG similarity to a co-crystal ligand | WDR91 library (15,419; 19) | 0.137 (89.1) | 0.0 (82.6) | 0.0 (94.3) |
| ErG similarity to a co-crystal ligand | DDB1 shortlist (1,500; 21) | 0.216 (99.1) | 0.0 (81.4) | 1.43 (85.3) |

The methodology is the contribution, because the null is what makes any of these numbers
readable:

* **The null mean matches the closed form but the spread does not.** A random ranking
  scores a LogAUC of `(1/ln 10) / log10(1/min_fpr)` in expectation (about 0.10 at 15,000
  negatives, about 0.14 at 1,500), not near 1. The permutation null reproduces that mean
  (0.136 vs 0.137 at 1,500 compounds) and also shows the spread, which is wide at a handful
  of hits. DDB1's ErG LogAUC of 0.216 is only 0.08 above a 0.14 baseline, a gap that is easy to
  dismiss as noise, yet it sits in the 99th percentile of its own null.
* **EF@50 is noise at low hit counts.** At 19 hits among 15,419 compounds, landing one hit
  in the top 50 by chance happens about 6% of the time, which is exactly where the
  pharmacophore's EF@50 of 16.23 sits (94th percentile). Its LogAUC and EF@1% are at or above
  essentially every null draw. LogAUC is the metric to trust.
* **Shortlist construction does not bias the reading.** The shortlists are all known hits
  plus a random sample of non-hits, so their hit rate is inflated. The null is drawn on the
  same composition, so the inflation cancels.

Null distributions (all three libraries, 1,000 permutations): [`results/empirical_null.json`](results/empirical_null.json).
All results with caveats: [`results/testbed_results.json`](results/testbed_results.json).
Reproduce the null with no data: `python scripts/empirical_null.py --shortlist LRRK2:1500:9`.

These are small samples (9, 19 and 21 hits) and each channel was tested on one or two
targets. They support "not at chance on a target it was never tuned for", not an effect
size. They did not translate into blind-test hits.

### 5. Two bugs worth documenting

**int8 overflow in fingerprint dot products.** Fingerprints were cached as `int8`
(8x smaller than float64) and passed straight into BitBIRCH clustering. `np.dot` on
`int8` does not widen its accumulator, so the shared-bit count of two 2048-bit
fingerprints wraps modulo 256: `np.dot(np.ones(2048, np.int8), np.ones(2048, np.int8))`
is `0`, not `2048`. Every similarity BitBIRCH computed was corrupted, independent of
threshold or branching factor, which is why nothing helped across two full-pool runs and two
calibration sweeps. All four were invalid. The bug surfaced only when a clustering-free
check contradicted them: compounds sharing two of three building blocks have a median ECFP4
Tanimoto of 0.64 (random pairs: 0.17), yet BitBIRCH merged 0 of 30 such pairs at any setting
(22 to 25 of 30 after the fix). The existing regression test used 64-dimensional vectors,
too short to overflow, which is exactly why it had not caught it. The fix is to upcast to
int32 inside the one function that takes dot products
([`dreamxcache/negatives/bitbirch_clustering.py`](dreamxcache/negatives/bitbirch_clustering.py));
the test now uses 2048-dimensional vectors.

**OpenBabel mis-extracting a quinazoline co-crystal ligand as a non-aromatic
dihydroquinazoline.** A PDB file has coordinates and no bond orders, so a ligand's
chemistry must be perceived. OpenBabel's bond-order perception assigned the central
pyrimidine ring of a quinazoline ligand as a non-aromatic dihydro form with a spurious
stereocentre. The SMILES was valid and passed an eyeball check. It was caught by
comparing against the crystal itself: the ring was planar with symmetric, double-bond-length
C-N distances on both sides, and an independent re-derivation with RDKit's
`rdDetermineBonds` (reliable when the record has explicit hydrogens) agreed with the
geometry. The ranking built on the wrong structure shared only 25 of its top 50 compounds
with the corrected one. `check_ring_geometry_consistency` now automates the geometric test
(an sp3-assigned ring carbon whose own coordinates are planar), so this kind of silent
mis-assignment is flagged instead of needing another manual look
([`dreamxcache/structure/ligand_extraction.py`](dreamxcache/structure/ligand_extraction.py)).

### 6. A cautionary note on testbed selection: LRRK2's anchor was anti-correlated with its own hits

The ErG negative control scored at the 0.2nd percentile of its null on LRRK2, worse than
nearly every random ranking, which looks like a bug. It is not. Mean ECFP4 similarity of
LRRK2's real hits to the anchor ligand sits at the **1.3rd percentile** of a random-draw
null: the hits are *less* similar to the anchor than almost any random set of compounds. Any
ranking by similarity to that anchor fails by construction, so LRRK2 tests nothing about ErG
as a method. It is a poisoned testbed, and the "ErG is at chance" conclusion therefore rests
on WDR91 alone. Check an anchor against the hits before reading the control
([`scripts/anchor_similarity_test.py`](scripts/anchor_similarity_test.py)).

The same follow-ups on DDB1 narrowed a different question without closing it: DDB1's hits are
*more* diverse than random (3rd percentile on mutual similarity), and their mean similarity to
the anchor is unremarkable, so neither explains its elevated ErG LogAUC. Both tests use plain
ECFP4 and a mean, not ErG's representation and maximum, so the question stays open
([`results/hit_structure_followups.json`](results/hit_structure_followups.json)).

### Other failure modes the code guards against

Each has a regression test:

* **Silent success.** A Boltz-2 MSA ending in a NUL byte makes `boltz predict` skip every input
  and exit 0. 32 chunk jobs "completed" in 24 to 61 seconds with no predictions. The harness
  checks every MSA for NUL bytes, and `boltz2_guard.sh` fails a job whose result count is below
  its input count. A job that finishes far faster than the model can run is a symptom.
* **Threshold before dedup.** Labeling by thresholding raw rows undercounts positives, because
  two sub-threshold rows for one structure clear it once summed. Every recipe deduplicates first.
* **Summed z-scores.** A summed or maxed z-score still looks like a z-score. `dedup_by_smiles`
  refuses; use Stouffer combination.
* **A 2-valued stratum treated as clusters** silently collapses the "elsewhere" negatives to zero
  (`sample_negatives(..., binary_stratum=True)`).
* **NaN in a channel** (a compound a channel never scored) must be dropped *before* taking a
  channel's top-N, or compounds it never scored receive rank credit.

## Repository map

```
dreamxcache/
  ingest/       loaders, cleaning, descriptive audit
  label/        labeling recipes L0-L8, disynthon pooling
  features/     ECFP4 via the official extraction code
  negatives/    stratified negative sampling, BitBIRCH wrapper
  models/       LightGBM baseline
  rankers/      ErG, CATS-like, ECFP4/FCFP4, USRCAT, pharmacophore, Boltz-2 harness
  structure/    bound-ligand extraction and geometry checks, bridging waters
  fusion/       reciprocal rank fusion, diversity selection, proxy series clustering
  eval/         early-recognition metrics, permutation null, official-evaluator wrapper, testbeds
scripts/        command-line entry points; scripts/slurm/ holds cluster templates
tests/          100+ tests; run without any data (four skip without the third-party checkouts)
results/        the numbers cited above
examples/       a worked pharmacophore reference-point file derived from a public PDB entry
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -e '.[dev]'
pytest                                                 # no data needed
python scripts/empirical_null.py --shortlist LRRK2:1500:9 --shortlist WDR91:15419:19
```

To run the pipeline on real inputs, obtain them as described in [DATA.md](DATA.md), then
`./scripts/third_party.sh` for the pinned third-party checkouts. Every script prints its
usage with `--help`, writes a timestamped `report.json` (code version, parameters) under
`results-local/`, and never overwrites an earlier run. SLURM scripts under
`scripts/slurm/` are templates: set the account and partition for your cluster.

## License

MIT, see [LICENSE](LICENSE). Third-party code is referenced by URL and pinned commit, not copied,
and keeps its own license.

## References

* Blum, A. and Hardt, M. (2015). The Ladder: A reliable leaderboard for machine learning competitions. *ICML*.
* Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. and Roth, A. (2015). The reusable holdout: Preserving validity in adaptive data analysis. *Science* 349, 636-638.
* Truchon, J.-F. and Bayly, C. I. (2007). Evaluating virtual screening methods: good and bad metrics for the "early recognition" problem. *J. Chem. Inf. Model.* 47, 488-508.
* Mysinger, M. M. and Shoichet, B. K. (2010). Rapid context-dependent ligand desolvation in molecular docking. *J. Chem. Inf. Model.* 50, 1561-1573.
* McCloskey, K. et al. (2020). Machine learning on DNA-encoded libraries: a new paradigm for hit finding. *J. Med. Chem.* 63, 8857-8866.
* Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR*.
