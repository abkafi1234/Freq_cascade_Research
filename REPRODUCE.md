# Reproducing the paper from this repository

Three levels, in increasing cost: check a published number against the released data (seconds),
regenerate a table or figure from the released data (minutes), or re-run an experiment (hours to
days, GPU for the neural arms).

## 0. Confirm the data is intact

```bash
python tools/make_manifest.py --check      # SHA-256 of every file in this repository
python tools/verify_claims.py              # recompute the paper's numbers from results/
```

The second command is the short answer to "does the paper match its data": 65 claims, each naming
the section it comes from, recomputed from the result files with nothing installed.

## 1. Check a number without running anything

The result files are JSON Lines: one row per evaluated cell, with the metrics as fields. The mean
that the paper reports is the mean over the ten matched folds or seeds of that cell.

```python
import json, statistics
rows = [json.loads(l) for l in open("results/clinc150.jsonl")]
v = [r["macro_f1"] for r in rows if r.get("method") == "RFOED-NN" and "error" not in r]
print(len(v), statistics.mean(v))        # 10 0.9026...
```

`SI/*.csv` holds the same data flattened to one table per experiment, with a `source_file` column
naming the JSONL each row came from; `SI/README.md` gives row counts, checksums and the paper
element each table supports. `EXPERIMENTS.md` maps every table and figure to its data.

## 2. Regenerate the tables, figures or the SI tables

The generators are in the software repository. Point them at these result files by copying the
directory in:

```bash
git clone https://github.com/abkafi1234/freqcascade
cd freqcascade
pip install -e ".[all]"
cp -r ../freqcascade-paper-artifacts/results .        # results/ is gitignored there by design
cp ../freqcascade-paper-artifacts/folds/*.npz data/folds/

python scripts/latex_tables.py                 # the benchmark, ablation and RQ5 tables
python scripts/latex_round3_tables.py r5compact   # the capacity table; other keys in EXPERIMENTS.md
python scripts/make_figures.py                 # the numbered result figures
python scripts/kbs_figures.py                  # the remaining figures
```

`EXPERIMENTS.md` section 3 lists the generator for each table and figure. The generators print
LaTeX table bodies and write PDFs; all differences are computed from unrounded means.

To regenerate `SI/` itself (it is derived from `results/`):

```bash
python tools/make_si.py       # run from a directory whose ../freqcascade/results exists
```

## 3. Re-run an experiment

Follow `REPRODUCE.md` in the software repository, which lists every experiment script in the order
it was run. Two things matter for agreement with the published numbers:

**The environment.** `env/requirements.txt` pins the versions the results were produced with, and
`env/environment.txt` records the machines. The neural results used PyTorch 2.11.0 with CUDA 12.8
wheels on an 8 GB GPU; scikit-learn 1.9.0; imbalanced-learn 0.14.2. `env/SETUP.md` has the
installation steps.

**The partitions.** `folds/*.npz` holds the exact cross-validation indices every method shared
(`train_<repeat>_<fold>` and `test_<repeat>_<fold>`, as row indices into the corpus as the loader
returns it). Copy them into `data/folds/` before running; `scripts/make_folds.py` regenerates them
identically.

The corpora are not redistributed here. `python scripts/verify_datasets.py` fetches all eight from
their original sources and prints a summary to check against the paper's dataset table.

### What counts as a reproduction

CPU cells reproduce exactly. Neural cells carry GPU nondeterminism: on a second device a
single-label cell matched to 4e-5 macro-F1 and a multi-label chain cell to 2e-3, while individual
chain links differed by up to 0.094. Treat a single-label macro-F1 within about 1e-4 as a match, do
not mix devices inside one experiment, and use `scripts/check_reproduction.py`, which recomputes one
stored CPU cell and one stored GPU cell before a run resumes on a new machine.

Heavy scripts are resumable: each appends one row per finished cell and skips cells already present
in the output file. A re-run against the files in this repository therefore skips everything unless
those rows are removed first.
