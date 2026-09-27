# Every experiment behind the paper

This file exists so that a reader can account for all of the computation the paper rests on,
including the parts that did not work. Section 1 describes each experiment, Section 2 is a
generated inventory of the files, Section 3 maps each table and figure to the file behind it, and
Section 4 lists the failures and the runs that were superseded.

Counts in Section 2 are computed by `tools/make_inventory.py` from the files themselves, and the
column that says what each file supports is read from `SI/README.md`, which is itself generated.
The prose is written by hand; the numbers in it are not.

`tools/verify_claims.py` goes the other way: it takes 65 numbers as printed in the paper and
recomputes each from these files, so a reader can see the agreement rather than take it on trust.
Three third-decimal disagreements were found and corrected in the manuscript that way.

## 1. The experiments

### 1.1 Main benchmark

*Single-label* (`clinc150`, `20newsgroups_ir50`, `wos46985`, `ohsumed_23`, `drug_reviews`). Every
method is evaluated on ten matched partitions: repeated stratified 5x2 cross-validation, except on
CLINC150, which keeps its published imbalanced train/validation/test split and repeats ten seeds on
it, so its ten repetitions vary training stochasticity rather than the partition. TF-IDF methods
fit the vectoriser inside each fold on that fold's training rows only.

*Multi-label* (`reuters21578`, `hoc`, `litcovid`). Repeated iterative-stratified 5x2
cross-validation, implemented in the package because the reference implementation ignores its
random seed.

Two configuration details that a reader comparing rows will notice:

- On Drug Reviews (K of about 356) the cascade is capped at the 50 most frequent classes with a
  flat MLP resolving the rest, and the rows are labelled `RFOED-NN (capped)`. A cascade of that
  many sequential neural nodes did not complete on the machine used for that run. `RFOED-RF` was
  not run on that corpus at all, on cost grounds.
- The cap-arm study (`ncap_adaptive.jsonl`) evaluates the *full-depth* cascade on Drug Reviews, so
  its rows there are not the main-benchmark rows. On the other seven corpora its `gamma=128` arm is
  the main run restricted to folds 0-4.

### 1.2 Ablations and drivers

`ablation_*.jsonl` is the 2x2x2 factorial over ordering, base learner and rebalancing (and its 2x2
multi-label counterpart). `rq5_*.jsonl` re-runs the neural instantiations under four frozen
encoders. `scaling_clinc150.jsonl` is the controlled class-count sweep on a single corpus.
`perclass_clinc150.jsonl` records per-class recall behind the rare-class discussion.

### 1.3 Pilots that fixed a setting

`unit_pilot.jsonl` compares node units (frozen embedding + MLP against a trainable TextCNN) and
sweeps the ensemble size; it fixed M = 25. `spot_trees.jsonl` checks 100 against 200 trees.
`probe_finetune_throughput.json` timed a fine-tuning cell to plan the grid. All three predate the
adaptive node cap and use the earlier fixed cap of 2,000.

### 1.4 Failure analysis

`wos46985_diagnosis.jsonl` assigns every test error on one 80/20 split to an own-node miss or an
early capture; `wos46985_sweep.jsonl` sweeps the two corrections the diagnosis implies;
`wos46985_remediated.jsonl` runs the selected configuration under cross-validation;
`wos46985_ncap.jsonl` is the control confirming that the node cap is inert for the Random Forest
learner. Configurations in the sweep were selected on the same split they are scored on, which the
paper states explicitly when reporting them.

### 1.5 Analyses added in revision

| file | what it establishes |
|---|---|
| `weaklearner_sweep.jsonl` (R5) | the collapse of the resampling ensembles is an interaction with weak-learner capacity, not class count alone: a ladder of six weak learners over six class counts and two feature sets |
| `cascade_dependence.jsonl`, `cascade_bound.jsonl` (R6) | the independence assumption behind Proposition 1, measured rather than assumed, and the bound checked per class |
| `finetune_grid.jsonl` (R7) | eight corpora x three losses x folds 0-2 with a fine-tuned DeBERTa-v3-base encoder |
| `ncap_adaptive.jsonl`, `ncap_gamma_selection.jsonl` (R8) | the fixed node cap replaced by an adaptive one, with gamma selected on validation data carved from the training folds only |
| `focc_perlink.jsonl`, `focc_exposure.jsonl` (R9) | per-link precision, recall and false-positive rate under deployed and oracle inference |
| `leak_check.jsonl` (B6) | every TF-IDF method re-run with a per-fold vectoriser, against the pooled originals in `results/_archive_global_tfidf/` |

`finetune_study.jsonl` is an earlier three-corpus fine-tuning pilot on a different encoder. It is
superseded by `finetune_grid.jsonl` and is published as `SI10_finetuning_pilot_superseded.csv`; no
claim in the paper rests on it.

### 1.6 Controls

`clinc150_cv.jsonl` re-runs CLINC150 under the cross-validation protocol used everywhere else, to
test whether its fixed split drives any conclusion. `threshold_flat_mlp.jsonl` is the
pre-registered threshold control behind the matched-recall comparison: it re-thresholds the flat
MLP three ways, one of which is a test-set diagnostic that is labelled as such.

## 2. Inventory

<!-- BEGIN INVENTORY -->

| result file | rows | error rows | written by | supporting data | backs |
|---|---:|---:|---|---|---|
| `20newsgroups_ir50.jsonl` | 122 | 2 | `run_single_label_benchmark.py` | `SI01_main_benchmark_single_label.csv` | Main single-label benchmark; Tables S1-S2 and the single-label results figure. |
| `ablation_20newsgroups_ir50.jsonl` | 48 | 0 | `run_factorial_ablation.py` | `SI05_factorial_ablation.csv` | Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7. |
| `ablation_clinc150.jsonl` | 48 | 0 | `run_factorial_ablation.py` | `SI05_factorial_ablation.csv` | Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7. |
| `ablation_hoc.jsonl` | 24 | 0 | `run_factorial_ablation.py --multilabel` | `SI05_factorial_ablation.csv` | Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7. |
| `ablation_litcovid.jsonl` | 24 | 0 | `run_factorial_ablation.py --multilabel` | `SI05_factorial_ablation.csv` | Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7. |
| `ablation_reuters21578.jsonl` | 24 | 0 | `run_factorial_ablation.py --multilabel` | `SI05_factorial_ablation.csv` | Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7. |
| `ablation_wos46985.jsonl` | 48 | 0 | `run_factorial_ablation.py` | `SI05_factorial_ablation.csv` | Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7. |
| `cascade_bound.jsonl` | 500 | 0 | `verify_cascade_bound.py` | `SI08_cascade_bound_R6.csv` | R6 empirical check of the cascade-admissibility bound; the bound table. |
| `cascade_dependence.jsonl` | 614 | 0 | `verify_cascade_dependence.py` | `SI07_cascade_dependence_R6.csv` | R6 node-decision dependence measurements; Table S13. |
| `clinc150.jsonl` | 140 | 0 | `run_single_label_benchmark.py` | `SI01_main_benchmark_single_label.csv` | Main single-label benchmark; Tables S1-S2 and the single-label results figure. |
| `clinc150_cv.jsonl` | 120 | 10 | `run_clinc150_cv.py` | `SI22_clinc150_protocol_check.csv` | CLINC150 re-run under cross-validation instead of its fixed split; Table S12. |
| `drug_reviews.jsonl` | 70 | 0 | `run_single_label_benchmark.py + run_drug_reviews_nn.py` | `SI01_main_benchmark_single_label.csv` | Main single-label benchmark; Tables S1-S2 and the single-label results figure. |
| `finetune_grid.jsonl` | 522 | 10 | `run_finetune_grid.py` | `SI09_finetuning_grid_R7.csv` | R7 DeBERTa-v3-base fine-tuning grid, 8 corpora x 3 losses x folds 0-2; fine-tuning table. |
| `finetune_study.jsonl` | 42 | 2 | `run_finetune_study.py` | `SI10_finetuning_pilot_superseded.csv` | Superseded MiniLM fine-tuning pilot (3 corpora, one split). Retained for provenance; no result in the paper rests on it. |
| `focc_exposure.jsonl` | 30 | 0 | `focc_exposure_bias.py` | `SI15_exposure_aggregate_R9.csv` | R9 aggregate deployed-versus-oracle gap per corpus; Table S14. |
| `focc_perlink.jsonl` | 2140 | 0 | `focc_perlink_exposure.py` | `SI14_perlink_exposure_R9.csv` | R9 per-link precision/recall/FPR under deployed and oracle inference; per-link figure. |
| `hoc.jsonl` | 70 | 0 | `run_multilabel_benchmark.py` | `SI02_main_benchmark_multi_label.csv` | Main multi-label benchmark; Table S3 and the multi-label results figure. |
| `leak_check.jsonl` | 286 | 6 | `run_leak_check.py` | `SI17_tfidf_leak_check.csv` | Per-fold TF-IDF refit against the pooled vectoriser; the leakage paragraph. |
| `litcovid.jsonl` | 70 | 0 | `run_multilabel_benchmark.py` | `SI02_main_benchmark_multi_label.csv` | Main multi-label benchmark; Table S3 and the multi-label results figure. |
| `ncap_adaptive.jsonl` | 240 | 0 | `run_ncap_adaptive.py --folds 5` | `SI11_rebalancing_cap_arms_R8.csv` | R8 fixed / adaptive-gamma / uncapped cap arms on all eight corpora; cap table. |
| `ncap_gamma_selection.jsonl` | 96 | 0 | `select_ncap_gamma.py` | `SI12_rebalancing_cap_selection_R8.csv` | R8 inner-validation gamma selection (training folds only); selects gamma=128. |
| `ncap_sweep.jsonl` | 320 | 0 | `run_ncap_sweep.py` | `SI13_rebalancing_cap_sweep_earlier.csv` | Earlier fixed-cap sweep that preceded the adaptive rule. |
| `ohsumed_23.jsonl` | 91 | 1 | `run_single_label_benchmark.py` | `SI01_main_benchmark_single_label.csv` | Main single-label benchmark; Tables S1-S2 and the single-label results figure. |
| `perclass_clinc150.jsonl` | 300 | 0 | `dump_per_class_recall.py` | `SI24_perclass_clinc150.csv` | Per-class CLINC150 detail behind the rare-class recall discussion. |
| `reuters21578.jsonl` | 70 | 0 | `run_multilabel_benchmark.py` | `SI02_main_benchmark_multi_label.csv` | Main multi-label benchmark; Table S3 and the multi-label results figure. |
| `rq5_clinc150.jsonl` | 50 | 0 | `run_representation_study.py` | `SI06_encoder_study_RQ5.csv` | RQ5 frozen-encoder comparison; Table S8 and the representation figure. |
| `rq5_hoc.jsonl` | 50 | 0 | `run_representation_study.py` | `SI06_encoder_study_RQ5.csv` | RQ5 frozen-encoder comparison; Table S8 and the representation figure. |
| `rq5_litcovid.jsonl` | 50 | 0 | `run_representation_study.py` | `SI06_encoder_study_RQ5.csv` | RQ5 frozen-encoder comparison; Table S8 and the representation figure. |
| `rq5_ohsumed_23.jsonl` | 50 | 0 | `run_representation_study.py` | `SI06_encoder_study_RQ5.csv` | RQ5 frozen-encoder comparison; Table S8 and the representation figure. |
| `scaling_clinc150.jsonl` | 90 | 1 | `scaling_curve.py` | `SI03_class_count_sweep.csv` | Controlled CLINC150 class-count sweep; the macro-F1 versus class-count figure. |
| `spot_trees.jsonl` | 20 | 0 | `spot_check_trees.py` | `SI25_spot_checks.csv` | Spot checks used while validating the tree-based configurations. |
| `threshold_flat_mlp.jsonl` | 120 | 0 | `threshold_flat_mlp.py` | `SI16_threshold_control.csv` | Pre-registered flat-MLP threshold control behind the matched-recall comparison. |
| `unit_pilot.jsonl` | 16 | 0 | `run_unit_pilot.py` | `SI23_base_learner_pilot.csv` | Base-learner unit comparison and ensemble-size sweep; Table S9. |
| `weaklearner_sweep.jsonl` | 648 | 58 | `run_weaklearner_sweep.py` | `SI04_weak_learner_capacity_R5.csv` | R5 weak-learner capacity ladder including the unbounded-tree rung; capacity table. |
| `wos46985.jsonl` | 100 | 0 | `run_single_label_benchmark.py` | `SI01_main_benchmark_single_label.csv` | Main single-label benchmark; Tables S1-S2 and the single-label results figure. |
| `wos46985_diagnosis.jsonl` | 3 | 0 | `diagnose_wos.py` | `SI18_wos46985_diagnosis.csv` | WOS46985 node-level error decomposition; Table S10. |
| `wos46985_ncap.jsonl` | 5 | 0 | `diagnose_ncap_wos.py` | `SI21_wos46985_cap_control.csv` | WOS46985 control confirming the node cap is inert for the Random-Forest learner. |
| `wos46985_remediated.jsonl` | 20 | 0 | `run_wos_remediation.py` | `SI20_wos46985_remediated_cv.csv` | WOS46985 remediated configuration under cross-validation. |
| `wos46985_sweep.jsonl` | 18 | 0 | `diagnose_wos_sweep.py` | `SI19_wos46985_remediation_sweep.csv` | WOS46985 threshold and depth-cap remediation sweep; Table S11. |
| `finetune_attempts.json` | 59 | 0 | `run_finetune_grid.py` | -- | per-cell attempt counter of the fine-tuning grid |
| `probe_finetune_throughput.json` | 5 | 0 | `probe_finetune_throughput.py` | -- | throughput probe used to plan the grid; no result rests on it |
| `weaklearner_attempts.json` | 83 | 0 | `run_weaklearner_sweep.py` | -- | per-cell attempt counter of the capacity ladder |

### Superseded and discarded runs, kept for the record

| directory | files | rows | what it is |
|---|---:|---:|---|
| `results/_archive_global_tfidf/` | 7 | 583 | pooled-corpus TF-IDF originals, before the per-fold refit |
| `results/_archive_ncap2000/` | 8 | 733 | main benchmark under the original fixed node cap of 2,000 |
| `results/_archive_rtx4070/` | 1 | 280 | per-link exposure run on the first GPU, before the recomputation |
| `results/_invalid/` | 2 | 35 | runs discarded as invalid (a NaN smoke test; an early-stopping configuration) |
| `results/_pre_config_fix/` | 4 | 314 | runs that predate a configuration fix (RUSBoost rounds) |

Totals: 61 JSONL files, 9244 rows, 3.1 MB. Generated by `tools/make_inventory.py` on 2026-09-27.

<!-- END INVENTORY -->

## 3. From a table or figure to its data

The generators live in the `freqcascade` repository, under `scripts/`. Floats that moved into the
supplement during shortening are numbered S1, S2, ... there; the labels below are the ones used in
the LaTeX source.

| paper element | generator | data |
|---|---|---|
| `tab:mainresults`, `tab:mainresultsmed`, `tab:focc` | `latex_tables.py` | the eight per-corpus benchmark files |
| `tab:anova`, `tab:foccablation` | `latex_tables.py` | `ablation_*.jsonl` |
| `tab:rq5` | `latex_tables.py` | `rq5_*.jsonl` |
| `tab:remediation`, `tab:diagnosis` | `latex_tables.py`, `latex_round3_tables.py diagnosis` | `wos46985_sweep.jsonl`, `wos46985_diagnosis.jsonl` |
| `tab:pilot` | `latex_pilot_table.py` | `unit_pilot.jsonl` |
| `tab:capacity` | `latex_round3_tables.py r5compact` | `weaklearner_sweep.jsonl` |
| `tab:cascadecheck` | `latex_reviewer_tables.py cascade` | `cascade_bound.jsonl` |
| `tab:dependence` | `latex_round3_tables.py r6` | `cascade_dependence.jsonl` |
| `tab:finetune` | `latex_reviewer_tables.py finetune` | `finetune_grid.jsonl` |
| `tab:ncap` | `latex_round3_tables.py r8`; gamma by `select_ncap_gamma.py --report` | `ncap_adaptive.jsonl`, `ncap_gamma_selection.jsonl` |
| `tab:perlinkorder` | `latex_round3_tables.py r9order` | `focc_perlink.jsonl` |
| `tab:threshold` | `threshold_flat_mlp.py` then `analyze_threshold.py --latex` | `threshold_flat_mlp.jsonl` |
| `tab:protocolcheck` | **no generator** (assembled by hand from two files) | `clinc150.jsonl`, `clinc150_cv.jsonl` |
| `figure2_rank_frequency`, `figure3_cd_diagram`, `figure5_focc_ablation`, `figure7_representation`, `figure8_scaling_curve` | `make_figures.py <key>` | the benchmark, ablation, RQ5 and scaling files |
| all other `figure_*.pdf`, `graphical_abstract` | `kbs_figures.py <key>` | as named in that script |
| `figure1_schematic` | drawn by hand; no generator | -- |
| `figure6_per_class_recall` | `kbs_figures.py` | `perclass_clinc150.jsonl`; produced but **not used** in the final paper, kept here for completeness |

`tab:protocolcheck` is the one table not produced by a script. Its RFOED-NN row reports the earlier
fixed node cap in both columns, because the protocol re-run predates the adoption of the adaptive
cap; both columns are therefore internally consistent and the comparison still isolates the
protocol. The paper's caption states this, and gives the value under the current cap.

## 4. Failures, superseded runs and known limits

**Failures are rows, not gaps.** Three kinds appear in the `error` field, and Section 2 gives the
count per file:

1. `BaseClassifier in AdaBoostClassifier ensemble is worse than random` -- the boosting loop of
   RUSBoost (and of EasyEnsemble, which boosts internally) aborting when the undersampled weak
   learner cannot beat chance. This is reported as the result rather than engineered around,
   because it is the behaviour of the canonical configuration at high class count.
2. `No samples will be generated with the provided ratio settings` -- ADASYN refusing a
   configuration whose bounded target falls below one synthetic example. It affects the pooled
   CLINC150 folds in particular.
3. `abandoned: killed by the OS` -- nine cells of the capacity ladder, all EasyEnsemble with an
   unbounded tree: TF-IDF at K = 100 and K = 150 and embeddings at K = 150, three seeds each. They
   exceeded the 6 GB limit under which that analysis was run. This is a limit of the host, not a
   property of the method, and the paper says so where the table is discussed.

**Superseded runs are kept.** `results/_archive_ncap2000/` (main benchmark under the fixed cap of
2,000), `results/_archive_global_tfidf/` (pooled-vectoriser TF-IDF originals),
`results/_archive_rtx4070/` (per-link exposure on the first GPU), `results/_pre_config_fix/` (runs
before a configuration fix) and `results/_invalid/` (a NaN smoke test and an early-stopping
configuration that halted training before the rare labels were learned). The leakage and cap
comparisons in the paper are differences against these files, so they have to be public for those
comparisons to be checkable.

**Analyses that use the earlier fixed cap of 2,000**, and whose absolute values therefore differ
from the main tables: `unit_pilot.jsonl`, `spot_trees.jsonl`, `wos46985_diagnosis.jsonl`,
`wos46985_sweep.jsonl`, `wos46985_remediated.jsonl`, `cascade_dependence.jsonl`,
`cascade_bound.jsonl`, `focc_exposure.jsonl`, `focc_perlink.jsonl`, `clinc150_cv.jsonl` and the
`rq5_*.jsonl` encoder study. Every comparison inside each of them holds the cap fixed.

**Device sensitivity.** Neural cells were reproduced across two GPUs to 4e-5 macro-F1 (single
label) and 2e-3 (a multi-label chain), but individual chain links differed by up to 0.094, so the
per-link study was recomputed in full on one device. CPU cells reproduce exactly.

**One loader caveat.** The per-class subsample counts of the original 20 Newsgroups configuration
were not archived, so its loader reproduces the target imbalance ratio and the geometric
construction rather than, necessarily, the first run's exact row count.
