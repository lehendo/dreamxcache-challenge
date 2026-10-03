# DREAM x CACHE Target2035 DEL-ML Challenge (Phase 1)

A pipeline for the **DREAM x CACHE Target2035 DEL-ML challenge**: predict binders of PGK2 from a DNA-encoded-library (DEL)
selection, submitting 50 compounds per test split, scored on held-out mass-spectrometry hits. This repository is organized around what I learned, not around features. The code (labeling recipes, ranking channels, fusion, evaluation) is here so the findings can be checked and the methods reused. No data is included; see [DATA.md](DATA.md).

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
