# Reproduction check

Evidence, not assertion: the output below is the unedited result of re-running two stored cells
from `results/` on the machine described in `env/environment.md` as machine B. Nothing was
written; both cells were recomputed in memory and compared with the stored values.

The two cells are chosen to separate the two kinds of agreement that matter. The CPU cell must be
bit-identical, because nothing in it is nondeterministic given the pinned library versions. The GPU
cell may differ slightly, because the batched neural ensemble is subject to kernel nondeterminism
across devices.

```
$ python scripts/check_reproduction.py                      # 2026-09-27

CPU cell  stored 0.8957966002  recomputed 0.8957966002  -> EXACT
GPU cell  stored 0.6503026130  recomputed 0.6503421862  |diff| 0.000040
          (tolerance 0.002, device: NVIDIA GeForce RTX 3050)  -> WITHIN TOLERANCE

REPRODUCTION: OK
```

- **CPU cell**: the weak-learner capacity sweep, RUSBoost with a decision stump on TF-IDF features,
  CLINC150 restricted to K = 5, seed 0 (`results/weaklearner_sweep.jsonl`).
- **GPU cell**: the cap study, RFOED-NN at the fixed cap of 2,000 on 20 Newsgroups, fold 0
  (`results/ncap_adaptive.jsonl`).

A reader can run the same check after installing the package and copying `results/` in; the
script is `scripts/check_reproduction.py` in the software repository. A CPU cell that is not exact
means the library versions differ from `env/requirements.txt`. A GPU cell outside tolerance means
the device produces different neural numbers, in which case results from the two devices should
not be mixed within one experiment -- which is why the per-link exposure analysis was recomputed in
full rather than merged (`results/_archive_rtx4070/` holds the earlier run).

## Checks that need no installation

These run against this repository alone, on the standard library:

```
$ python tools/verify_claims.py       # recompute the paper's numbers from results/
$ python tools/make_manifest.py --check   # confirm no file has changed
$ python tools/make_si.py             # regenerate SI/ from results/ (byte-for-byte)
```
