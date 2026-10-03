# Data

**This repository ships no data.** Every input is obtained from its original
source, under that source's terms, and placed under `data/` (git-ignored). The
`.gitignore` is written so that data files cannot be added by accident, and
`tests/test_repo_hygiene.py` fails if anything data-like or machine-specific
would be committed.

Never commit AIRCHECK data, or anything derived from it per compound (score
arrays, fingerprint matrices, shortlists, result tables, trained models). That
access is controlled by AIRCHECK (registration and approval); redistribution is
not mine to grant.

Paths are resolved from environment variables, defaulting to the current
working directory (run scripts from the repository root):

| variable | default | holds |
|---|---|---|
| `DREAMXCACHE_DATA_DIR` | `./data` | `raw/`, `cache/`, `external/`, `submissions/` |
| `DREAMXCACHE_RESULTS_DIR` | `./results-local` | timestamped run outputs |
| `DREAMXCACHE_THIRD_PARTY_DIR` | `./third_party` | pinned third-party checkouts |

## 1. Challenge and testbed data (AIRCHECK)

Obtain from [AIRCHECK](https://aircheck.ai) following its access process, and
follow its terms of use.

| file | expected location | needed for |
|---|---|---|
| PGK2 DEL selection table | `data/raw/PGK2_selection.parquet` | labeling recipes, baseline model |
| PGK2 no-target-control supplement (optional) | `data/raw/PGK2_NTC_supplement.parquet` | audit only |
| Validation and test split templates | `data/raw/Val-Test-set/PGK2_Validation_split.csv`, `PGK2_Test_split.csv` | scoring splits, fusion, submissions |
| OpenDEL enumeration parquets (optional) | `data/raw/OpeDELLibrary/<library>.enumeration.parquet` | enumeration-gap audit |
| E-ASMS testbed libraries | anywhere; pass with `--library` | known-answer testbeds |

The three E-ASMS testbed libraries used for the findings in the README are listed
in AIRCHECK's E-ASMS dataset catalogue as `WDR91_A4D1P6_392_747`,
`LRRK2_Q5S007_2141_2527` and `DDB1_Q16531_1_1140`. Each is a table with a `SMILES`
column and a 0/1 `LABEL` column of confirmed hits. Testbed scripts expect a
`COMPOUND_ID` column as well. Build the 1,500-compound shortlists with
`scripts/build_testbed_shortlist.py` (all known hits plus a seeded random sample
of non-hits).

The expected selection-table columns are the ones the organizers document:
`compound` (`library-bb1-bb2-bb3`), `SMILES`, `count_PGK2`,
`count_PGK2_with_inhibitor`, `count_NTC`, `historic_hits`, `zscore_PGK2`.

## 2. Public structures (PDB)

Download from the [RCSB PDB](https://www.rcsb.org) into `data/external/`:

| entry | content | used for |
|---|---|---|
| 8SHJ | WDR91 WD-repeat domain with ligand ZI8 (chain A) | WDR91 ErG anchor and pharmacophore reference (`examples/wdr91_8SHJ_reference_points.json`) |
| 9C61 | LRRK2 WD40 domain with ligand A1AUU | LRRK2 ErG anchor |
| 37MF | DDB1 with ligand A1DPN | DDB1 ErG anchor |

Newer entries may be available only as mmCIF, and their 5-character ligand codes do
not fit the legacy fixed-width PDB format. Take ligand SMILES from the RCSB chemical
component dictionary (`data.rcsb.org/rest/v1/core/chemcomp/<code>`, descriptor type
`SMILES_CANONICAL`) rather than from a converted PDB file.

Pharmacophore reference ligands are extracted from PDB files by
`dreamxcache.structure.ligand_extraction`; read its module docstring first, because
bond-order perception from coordinates can be silently wrong.

## 3. Public ligand data (ChEMBL)

`scripts/fetch_chembl_actives.py --target-chembl-id <ID> --out data/external/<file>.csv`
queries the public ChEMBL REST API once and saves the result. Anchor-based rankers
read a CSV with a `smiles` column and optional `source` and `detail` columns used for
filtering. Supply your own anchors; none are included.

Material on the companion CACHE challenge page that is restricted to challenge
participants is not used or included here.

## 4. Protein sequences and MSAs

Sequences come from UniProt, trimmed to the construct that was assayed:

| target | UniProt | residues |
|---|---|---|
| LRRK2 (WD40 domain) | Q5S007 | 2141-2527 |
| DDB1 | Q16531 | 1-1140 |
| WDR91 | A4D1P6 | 392-747 |

Fetch FASTA **directly** (for example `curl https://rest.uniprot.org/uniprotkb/<ID>.fasta`)
and check the length against the expected construct. Do not copy a sequence through any
tool that rewrites text: an LLM-summarized copy of the DDB1 sequence came back 7 residues
too long, and nothing about it looked wrong.

MSAs are computed once per target and referenced by every Boltz-2 input. Generating
one with `boltz predict --use_msa_server` needs internet access and writes an `.a3m`
that can end with a stray NUL byte. **Check every new `.a3m`** (`scripts/boltz2_generate_yamls.py`
does, and `--fix-msa` repairs it): a NUL byte makes Boltz skip every input yet exit 0.
MSA files are not included (they are large and regenerable).

## 5. Third-party code (referenced, not copied)

`scripts/third_party.sh` clones these at pinned commits into `third_party/`:

| name | repository | commit | used for |
|---|---|---|---|
| `aircheck_utils` | https://github.com/StructuralGenomicsConsortium/Target2035_Aircheck_Utils | `c3472a4c6185e5dc74a6da9d28d4f80aeb4ebfcf` | official evaluator, fingerprint extraction |
| `bitbirch` | https://github.com/mqcomplab/bitbirch | `311fbf7dcc8ba019a53f58e75d40ebce6a3bc76f` | BitBIRCH clustering |
| `dedup` | https://github.com/VonBoss/DREAM_DEL_deduplication | `a81b7dc4940fed6178ac723ffe8f66d7a7568bd4` | reference deduplication procedure (nothing imports it) |

## 6. Boltz-2

Install [Boltz](https://github.com/jwohlwend/boltz) in its **own** environment
(the versions used: boltz 2.2.1, torch 2.14, Python 3.11). Installing it into a
shared environment silently changed pinned numpy, scipy and scikit-learn versions.
Settings and failure modes are documented in `dreamxcache/rankers/boltz2.py` and
`scripts/slurm/boltz2_guard.sh`.

## What needs no data

The unit tests, the permutation null (`scripts/empirical_null.py`, which depends
only on the number of compounds and hits), and everything under `results/`.
