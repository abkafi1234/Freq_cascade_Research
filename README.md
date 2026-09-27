# Paper artifacts: resampling collapse and local decomposition at high class cardinality

Every measurement behind the paper

> **An Empirical Study of Resampling Collapse and Local Decomposition for Class Imbalance in
> High-Cardinality Text Classification**
> Md Abdullah Al Kafi, Raka Moni, Walayat Hussain

is released here: the raw per-fold result files, the tabular supporting information published with
the paper, the figure files as they appear in it, the cross-validation partitions every method
shared, and the exact software environment. Every measurement the paper reports is computed from
files released here. Its table of dataset statistics is computed from the corpora rather than from
a result file, so `env/dataset_fingerprints.json` records those statistics, and the corpus digests
behind them, in order that it can be checked here too.

This repository is the **data companion**. The software that produced the data is a separate
package:

| what | where |
|---|---|
| algorithms, experiment scripts, table and figure generators, unit tests | <https://github.com/abkafi1234/freqcascade> (`pip install freqcascade`) |
| result files, supporting-information tables, figures, folds, environment | this repository |

The split exists because the package is gitignored for result files: running an experiment writes
into `results/`, which never entered the package's history. Those files are the evidence, so they
are published here instead.

## Layout

```
results/            39 canonical result files (JSONL, one row per evaluated cell)
  _archive_*/       superseded runs, kept so that every reported change can be checked
  _invalid/         runs discarded as invalid, kept for the same reason
  _pre_config_fix/  runs that predate a configuration fix
SI/                 the 25 supporting-information tables (CSV) and their manifest
figures/            25 figure files: the 24 used in the paper and its supplement, plus one
                    produced but not used (EXPERIMENTS.md says which)
folds/              the cross-validation partitions (integer row indices), shared by every method
env/                pinned requirements, the recorded environment, and setup notes
tools/              scripts that check the paper's claims, regenerate SI/, the
                    inventory table, the checksums and the corpus fingerprints
EXPERIMENTS.md      every experiment: what it is, what wrote it, what it supports, its caveats
REPRODUCE.md        how to reproduce a number, a table or the whole campaign
REPRODUCTION_CHECK.md  the recorded output of re-running two stored cells
MANIFEST.sha256     checksum of every file here
```

## Check the paper in one command

No installation, no corpora, no GPU -- the standard library is enough:

```bash
python tools/verify_claims.py
```

It recomputes every headline number of the paper from `results/` and prints, for each, the value
printed in the paper beside the value in the data. Each claim names the section it comes from, so
a disagreement points at a specific sentence rather than at the repository as a whole. All of them
currently reproduce.

## Start here

- **To check a number in the paper**: run `tools/verify_claims.py`, or read `EXPERIMENTS.md`,
  which maps every table and figure to the result file behind it and to the script that turns one
  into the other.
- **To re-run something**: `REPRODUCE.md`.
- **To confirm these are the files the paper was written from**:
  `python tools/make_manifest.py --check`.
- **To see that the stored numbers recompute on real hardware**: `REPRODUCTION_CHECK.md`.

## What is deliberately not here, and why

**The corpora.** All eight benchmarks are public, but they are third-party data distributed under
their own terms, so none of the text is redistributed here. The loaders in `freqcascade.datasets`
fetch each one from its original source, and `folds/` fixes the partitions so that a re-run
evaluates on exactly the rows the paper evaluated on. To close the one gap that leaves -- whether
the copy you fetched is the copy we used -- `env/dataset_fingerprints.json` records the size, the
class structure and one-way digests of the text and labels of each corpus as the loader returns
them, and `tools/make_dataset_fingerprints.py --check` compares a local copy against that record.

| corpus | source |
|---|---|
| CLINC150 (imbalanced split) | `clinc/oos-eval`, `data/data_imbalanced.json` |
| 20 Newsgroups | `scikit-learn`, with the geometric imbalance injection in the loader |
| WOS46985 | Hugging Face `bakirgrbic/web-of-science` |
| OHSUMED-23 | Moschitti's `ohsumed-first-20000-docs` distribution |
| Drug Reviews | Hugging Face `lewtun/drug-reviews` (UCI / Drugs.com) |
| Reuters-21578 (ModApte) | NLTK `reuters` |
| Hallmarks of Cancer | `sb895/Hallmarks-of-Cancer` |
| LitCovid | Hugging Face `KushT/LitCovid_BioCreative` |

**Cached embeddings** (about 6.5 GB of frozen sentence-encoder output) are derived from that text
and are likewise not redistributed; `scripts/precompute_embeddings.py` rebuilds them.

**The package source**, because it lives in its own repository and is installable from PyPI.

**Run logs and internal working documents** (planning notes, the reviewer correspondence and the
responses drafted from it). The per-cell timing, peak GPU memory, error and abandonment records
that a reader would want from a log are fields in the result rows themselves.

**The manuscript PDF** is not included pending the publisher's sharing terms; the published version
will be linked from this README.

## Integrity and honesty notes

These are stated here rather than left to be discovered in the files.

1. **Superseded runs are kept, not deleted.** `results/_archive_ncap2000/` holds the main benchmark
   under the original fixed node cap of 2,000, which the paper replaced with an adaptive cap;
   `results/_archive_global_tfidf/` holds the pooled-vectoriser originals that the per-fold TF-IDF
   re-run replaced. Both are what the leakage and cap comparisons in the paper are computed
   against, so both are published.
2. **Failures are recorded as results.** Cells that could not run appear as rows with an `error`
   field rather than as gaps: RUSBoost aborting when its undersampled stump cannot beat chance,
   ADASYN refusing a ratio, and and nine capacity-ladder cells marked `abandoned: killed by the OS`,
   which means they exceeded the 6 GB memory limit imposed on that analysis (a Linux control
   group on a 15.7 GB host) and not that the method cannot run.
3. **Some analyses predate the final node cap** and use the earlier fixed cap of 2,000: the
   base-learner pilot, the WOS46985 diagnosis and remediation sweeps, the dependence measurements,
   the exposure study and the CLINC150 protocol check. Each is internally consistent, because every
   comparison inside it holds the cap fixed, but its absolute values differ from the main tables.
   The paper says so in its limitations section, and `EXPERIMENTS.md` marks each affected file.
4. **Neural results carry GPU nondeterminism.** A single-label cell reproduced to 4e-5 macro-F1 on a
   second GPU and a multi-label chain cell to 2e-3, but individual chain links differed by up to
   0.094, which is why the per-link study was recomputed in full on one device
   (`results/_archive_rtx4070/` keeps the earlier run). Do not mix devices within an experiment.
5. **One table in the paper has no generator**: the CLINC150 protocol check was assembled by hand
   from two result files. Its RFOED-NN row reports the earlier fixed cap in both of its columns; the
   paper's caption states this.

## Licence

- `tools/` and any other code here: MIT (`LICENSE`).
- Result files, supporting-information tables, figures and fold indices: CC BY 4.0
  (`LICENSE-DATA`). Please cite the paper if you use them.
- The corpora themselves are not covered by either licence; they remain under the terms of their
  original distributors.

## Citation

See `CITATION.cff`. The paper is under review; its citation and DOI will be added here on
publication.

Correspondence: Md Abdullah Al Kafi (<kafi.cse@diu.edu.bd>), Walayat Hussain
(<Walayat.Hussain@acu.edu.au>, corresponding author).
