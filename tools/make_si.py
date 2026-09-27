#!/usr/bin/env python3
"""Generate the Supporting Information data files from the canonical result JSONL.

One CSV per experiment, plus a manifest naming the paper element each file backs.
Every number is copied verbatim from freqcascade/results/*.jsonl by this script; nothing
is typed. List- and dict-valued fields (per-epoch loss histories, per-class arrays) are
dropped from the CSV and noted in the manifest -- the raw JSONL ships with the code.

    python tools/make_si.py        # from the root of the artifact repository
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In this artifact repository the results sit beside the script's parent; in the manuscript
# working copy they sit in the sibling package checkout. Prefer whichever exists.
_LOCAL = HERE.parent / "results"
RESULTS = _LOCAL if _LOCAL.is_dir() else HERE.parent / "KBS" / "freqcascade" / "results"
OUT = (HERE.parent / "SI") if _LOCAL.is_dir() else (HERE / "SI")

# (SI file stem, [source jsonl, ...], what it backs in the paper)
GROUPS = [
    # A file pools several sources ONLY when they are the same experiment split by corpus
    # (they then carry a `dataset` column, so pooling is unambiguous). Different experiments
    # are never pooled: an earlier draft put the retired MiniLM pilot in the same file as the
    # R7 DeBERTa grid, and filtering by dataset+method then silently averaged the two.
    ("SI01_main_benchmark_single_label",
     ["clinc150", "20newsgroups_ir50", "wos46985", "ohsumed_23", "drug_reviews"],
     "Main single-label benchmark; Tables S1-S2 and the single-label results figure."),
    ("SI02_main_benchmark_multi_label",
     ["reuters21578", "hoc", "litcovid"],
     "Main multi-label benchmark; Table S3 and the multi-label results figure."),
    ("SI03_class_count_sweep",
     ["scaling_clinc150"],
     "Controlled CLINC150 class-count sweep; the macro-F1 versus class-count figure."),
    ("SI04_weak_learner_capacity_R5",
     ["weaklearner_sweep"],
     "R5 weak-learner capacity ladder including the unbounded-tree rung; capacity table."),
    ("SI05_factorial_ablation",
     ["ablation_clinc150", "ablation_20newsgroups_ir50", "ablation_wos46985",
      "ablation_reuters21578", "ablation_hoc", "ablation_litcovid"],
     "Ordering x base learner x rebalancing factorial and the FOCC 2x2; Tables S6-S7."),
    ("SI06_encoder_study_RQ5",
     ["rq5_clinc150", "rq5_ohsumed_23", "rq5_hoc", "rq5_litcovid"],
     "RQ5 frozen-encoder comparison; Table S8 and the representation figure."),
    ("SI07_cascade_dependence_R6",
     ["cascade_dependence"],
     "R6 node-decision dependence measurements; Table S13."),
    ("SI08_cascade_bound_R6",
     ["cascade_bound"],
     "R6 empirical check of the cascade-admissibility bound; the bound table."),
    ("SI09_finetuning_grid_R7",
     ["finetune_grid"],
     "R7 DeBERTa-v3-base fine-tuning grid, 8 corpora x 3 losses x folds 0-2; fine-tuning table."),
    ("SI10_finetuning_pilot_superseded",
     ["finetune_study"],
     "Superseded MiniLM fine-tuning pilot (3 corpora, one split). Retained for provenance; "
     "no result in the paper rests on it."),
    ("SI11_rebalancing_cap_arms_R8",
     ["ncap_adaptive"],
     "R8 fixed / adaptive-gamma / uncapped cap arms on all eight corpora; cap table."),
    ("SI12_rebalancing_cap_selection_R8",
     ["ncap_gamma_selection"],
     "R8 inner-validation gamma selection (training folds only); selects gamma=128."),
    ("SI13_rebalancing_cap_sweep_earlier",
     ["ncap_sweep"],
     "Earlier fixed-cap sweep that preceded the adaptive rule."),
    ("SI14_perlink_exposure_R9",
     ["focc_perlink"],
     "R9 per-link precision/recall/FPR under deployed and oracle inference; per-link figure."),
    ("SI15_exposure_aggregate_R9",
     ["focc_exposure"],
     "R9 aggregate deployed-versus-oracle gap per corpus; Table S14."),
    ("SI16_threshold_control",
     ["threshold_flat_mlp"],
     "Pre-registered flat-MLP threshold control behind the matched-recall comparison."),
    ("SI17_tfidf_leak_check",
     ["leak_check"],
     "Per-fold TF-IDF refit against the pooled vectoriser; the leakage paragraph."),
    ("SI18_wos46985_diagnosis",
     ["wos46985_diagnosis"],
     "WOS46985 node-level error decomposition; Table S10."),
    ("SI19_wos46985_remediation_sweep",
     ["wos46985_sweep"],
     "WOS46985 threshold and depth-cap remediation sweep; Table S11."),
    ("SI20_wos46985_remediated_cv",
     ["wos46985_remediated"],
     "WOS46985 remediated configuration under cross-validation."),
    ("SI21_wos46985_cap_control",
     ["wos46985_ncap"],
     "WOS46985 control confirming the node cap is inert for the Random-Forest learner."),
    ("SI22_clinc150_protocol_check",
     ["clinc150_cv"],
     "CLINC150 re-run under cross-validation instead of its fixed split; Table S12."),
    ("SI23_base_learner_pilot",
     ["unit_pilot"],
     "Base-learner unit comparison and ensemble-size sweep; Table S9."),
    ("SI24_perclass_clinc150",
     ["perclass_clinc150"],
     "Per-class CLINC150 detail behind the rare-class recall discussion."),
    ("SI25_spot_checks",
     ["spot_trees"],
     "Spot checks used while validating the tree-based configurations."),
]


def rows(stem):
    p = RESULTS / (stem + ".jsonl")
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    OUT.mkdir(exist_ok=True)
    manifest = []
    for name, stems, what in GROUPS:
        recs, missing, dropped = [], [], set()
        for stem in stems:
            rs = rows(stem)
            if rs is None:
                missing.append(stem)
                continue
            for r in rs:
                flat = {"source_file": stem + ".jsonl"}
                for k, v in r.items():
                    if isinstance(v, (list, dict)):
                        dropped.add(k)
                        continue
                    flat[k] = v
                recs.append(flat)
        if not recs:
            continue
        keys = ["source_file"] + sorted({k for r in recs for k in r} - {"source_file"})
        path = OUT / (name + ".csv")
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        manifest.append((name, len(recs), len(keys), path.stat().st_size, digest,
                         stems, what, sorted(dropped), missing))

    lines = ["# Supporting Information: experimental data", "",
             "Generated by `make_si.py` from the canonical result files in "
             "`freqcascade/results/*.jsonl`; do not edit by hand. Every value is copied "
             "verbatim from those files.", "",
             "One CSV per experiment. A `source_file` column names the canonical JSONL each "
             "row came from, so a group that pools several corpora stays traceable. "
             "List- and dict-valued fields (per-epoch loss histories, per-class arrays) are "
             "omitted from the CSVs; the raw JSONL released with the code retains them.", "",
             "| file | rows | cols | size | sha256 (first 16) | backs |",
             "|---|---|---|---|---|---|"]
    for name, n, c, size, digest, stems, what, dropped, missing in manifest:
        lines.append("| `%s.csv` | %d | %d | %.0f kB | `%s` | %s |"
                     % (name, n, c, size / 1024, digest, what))
    lines += ["", "## Sources", ""]
    for name, n, c, size, digest, stems, what, dropped, missing in manifest:
        note = ""
        if dropped:
            note += "  Omitted non-scalar fields: %s." % ", ".join("`%s`" % d for d in dropped)
        if missing:
            note += "  Missing source(s): %s." % ", ".join(missing)
        lines.append("- **%s.csv** <- %s.%s"
                     % (name, ", ".join("`%s.jsonl`" % s for s in stems), note))
    lines += ["", "## Reproduction", "",
              "The package that produced these files, the dataset loaders and the "
              "cross-validation fold indices are released with the paper "
              "(`pip install freqcascade`). Regenerate this directory with:", "",
              "    cd KBS_shorten && python make_si.py", ""]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %d SI files to %s" % (len(manifest) + 1, OUT))


if __name__ == "__main__":
    main()
