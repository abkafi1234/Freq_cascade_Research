# The environments the results were produced in

Two machines were used. The allocation below is the one stated in the paper's supplement;
`environment.txt` is the original, verbatim record written on machine A when the campaign paused,
and is kept unedited for provenance.

## Machine A -- main campaign

| | |
|---|---|
| ran | the main single- and multi-label benchmark, the factorial ablations, the representation (encoder) study, the pilots and the first diagnostic sweeps |
| OS | Windows 11 Home 10.0.26200 |
| CPU | 28 hardware threads |
| GPU | NVIDIA GeForce RTX 4070, 12 GB, driver 616.64 |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |

Work on this machine stopped on 2026-09-16.

## Machine B -- revision campaign

| | |
|---|---|
| ran | the per-link exposure analysis, the adaptive rebalancing-cap study, the weak-learner capacity sweep, the fine-tuning grid, the leakage check, and the re-run of every neural main result under the adopted cap |
| OS | Fedora Linux 43, kernel 6.18.4 |
| CPU | Intel Core i7-14700K, 20 cores / 28 threads |
| RAM | 15.7 GB (the capacity sweep was additionally confined to 6 GB by a control group) |
| GPU | NVIDIA GeForce RTX 3050, 8 GB, driver 580.119.02 |
| Python | 3.12.12 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| scikit-learn / NumPy | 1.9.0 / 2.5.1 |

Before any experiment was resumed on machine B, stored cells were recomputed there
(`scripts/check_reproduction.py`): a CPU cell reproduced exactly and a GPU cell to within 4e-5
macro-F1. Because individual chain links differed by up to 0.094 between the two GPUs, the per-link
exposure analysis was recomputed in full on machine B rather than merged; the earlier run is kept
in `results/_archive_rtx4070/`.

## Settings

`environment.txt` records `n_cap = 2000`, which was the node cap in force on machine A **at the
time it was written**. That fixed cap was later replaced by the adaptive cap
`n_cap = round(gamma * n / K)` with `gamma = 128`, selected on validation data carved from the
training folds, and every neural result in the paper was re-run under it. Runs still on the fixed
cap are listed in `EXPERIMENTS.md`, section 4. The other settings in that file are unchanged:
M = 25 ensemble members, 150 epochs, 100 trees, TF-IDF with 10,000 features, CV seed 20260905.

`requirements.txt` pins the package versions; `SETUP.md` gives the installation steps for a fresh
machine, including how to choose a PyTorch build that matches the local driver.
