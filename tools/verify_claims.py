#!/usr/bin/env python3
"""Recompute the paper's quantitative claims from the released result files.

Standard library only: no installation, no corpora, no GPU. Every claim below names where it
appears in the paper, states the value printed there, and recomputes it from `results/`. A
failure means the paper and the data disagree, which is exactly what this script is for.

    python tools/verify_claims.py            # check every claim
    python tools/verify_claims.py -v         # also show how each one is computed
    python tools/verify_claims.py -k rank    # only claims whose id matches a substring

Exit status is 0 only if every claim passes.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
_CACHE: dict = {}


# ---------------------------------------------------------------- data access

def rows(name: str) -> list:
    if name not in _CACHE:
        p = RESULTS / name
        _CACHE[name] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return _CACHE[name]


def _match(r: dict, filt: dict) -> bool:
    for k, v in filt.items():
        if callable(v):
            if not v(r.get(k)):
                return False
        elif r.get(k) != v:
            return False
    return True


def series(name: str, metric: str, **filt) -> dict:
    """{fold: value} for the rows matching ``filt``; error rows are skipped."""
    out = {}
    for r in rows(name):
        if r.get("error") or not _match(r, filt):
            continue
        v = r.get(metric)
        if v is None:
            continue
        key = r.get("fold", r.get("seed", len(out)))
        out[key] = v
    return out


def mean(name: str, metric: str, **filt) -> float:
    s = series(name, metric, **filt)
    if not s:
        raise AssertionError(f"no rows: {name} {metric} {filt}")
    return st.mean(s.values())


def gap(a: tuple, b: tuple) -> float:
    """Mean paired difference a - b, matched on fold; falls back to a difference of means."""
    (na, ma, fa), (nb, mb, fb) = a, b
    sa, sb = series(na, ma, **fa), series(nb, mb, **fb)
    common = sorted(set(sa) & set(sb))
    if not common:
        raise AssertionError(f"no matched folds between {na}{fa} and {nb}{fb}")
    return st.mean([sa[k] - sb[k] for k in common])


SL = {  # corpus -> (file, the strongest TF-IDF baseline, the strongest flat MLP variant)
    "CLINC150": ("clinc150.jsonl", "OVR-RF", "Flat-MLP"),
    "20 Newsgroups": ("20newsgroups_ir50.jsonl", "Flat-RF+SMOTE", "Flat-MLP (class-weighted)"),
    "Drug Reviews": ("drug_reviews.jsonl", "Flat Balanced-RF", "Flat-MLP"),
    "OHSUMED-23": ("ohsumed_23.jsonl", "Flat-RF+SMOTE", "Flat-MLP (class-weighted)"),
    "WOS46985": ("wos46985.jsonl", "Flat-RF+SMOTE", "Flat-MLP"),
}
CASCADE = {"Drug Reviews": "RFOED-NN (capped)"}


def sl(corpus: str, which: str) -> float:
    f, tfidf, mlp = SL[corpus]
    m = {"cascade": CASCADE.get(corpus, "RFOED-NN"), "tfidf": tfidf, "mlp": mlp}[which]
    return mean(f, "macro_f1", method=m)


def sl_gap(corpus: str, a: str, b: str) -> float:
    f, tfidf, mlp = SL[corpus]
    pick = {"cascade": CASCADE.get(corpus, "RFOED-NN"), "tfidf": tfidf, "mlp": mlp}
    return gap((f, "macro_f1", {"method": pick[a]}), (f, "macro_f1", {"method": pick[b]}))


def ml(corpus: str, metric: str, method: str) -> float:
    return mean({"Reuters-21578": "reuters21578.jsonl", "HoC": "hoc.jsonl",
                 "LitCovid": "litcovid.jsonl"}[corpus], metric, method=method)


def cap(corpus: str, arm: str) -> float:
    filt = {"fixed": {"rule": "fixed"}, "uncapped": {"rule": "uncapped"},
            "g128": {"rule": "adaptive", "gamma": lambda g: g is not None and float(g) == 128}}[arm]
    return mean("ncap_adaptive.jsonl", "score", dataset=corpus, **filt)


def ranks() -> dict:
    """Mean rank of each method over the four corpora that ran the full method set."""
    corpora = ["clinc150.jsonl", "20newsgroups_ir50.jsonl", "wos46985.jsonl", "ohsumed_23.jsonl"]
    per = []
    for f in corpora:
        got = {}
        for r in rows(f):
            if r.get("error") or r.get("macro_f1") is None:
                continue
            got.setdefault(r["method"], []).append(r["macro_f1"])
        per.append({m: st.mean(v) for m, v in got.items()})
    shared = set.intersection(*[set(p) for p in per])
    acc = {m: 0.0 for m in shared}
    for p in per:
        for i, m in enumerate(sorted(shared, key=lambda x: -p[x]), start=1):
            acc[m] += i
    return {m: acc[m] / len(per) for m in shared}


def friedman_chi2() -> float:
    r = ranks()
    k, n = len(r), 4
    return 12 * n / (k * (k + 1)) * sum(v * v for v in r.values()) - 3 * n * (k + 1)


def finetune_rare_gap(corpus: str) -> float:
    """FOCC-NN minus the flat MLP on bottom-quartile-label recall, over three losses x three folds."""
    ds = {"Reuters-21578": "reuters21578", "HoC": "hoc", "LitCovid": "litcovid"}[corpus]
    per = {}
    for r in rows("finetune_grid.jsonl"):
        if r.get("error") or r.get("dataset") != ds or r.get("arm") != "finetuned-features":
            continue
        v = r.get("bottom_quartile_label_recall")
        if v is None or r["method"] not in ("FOCC-NN", "Flat-MLP"):
            continue
        per.setdefault((r["loss"], r["fold"]), {})[r["method"]] = v
    d = [c["FOCC-NN"] - c["Flat-MLP"] for c in per.values() if len(c) == 2]
    if not d:
        raise AssertionError("no matched fine-tuning cells")
    return st.mean(d)


def leak_shift() -> tuple:
    """(mean, max |delta|) of the per-fold TF-IDF refit against the pooled originals."""
    d = []
    for ds in ("20newsgroups_ir50", "ohsumed_23", "wos46985", "drug_reviews"):
        new, old = {}, {}
        for r in rows("leak_check.jsonl"):
            if not r.get("error") and r.get("dataset") == ds:
                new.setdefault(r["method"], {})[r["fold"]] = r["macro_f1"]
        for r in rows(f"_archive_global_tfidf/{ds}.jsonl"):
            if not r.get("error") and r.get("macro_f1") is not None:
                old.setdefault(r["method"], {})[r["fold"]] = r["macro_f1"]
        for m in set(new) & set(old):
            f = sorted(set(new[m]) & set(old[m]))
            if f:
                d.append(st.mean([new[m][x] for x in f]) - st.mean([old[m][x] for x in f]))
    return st.mean(d), max(abs(x) for x in d)


def capacity(ensemble: str, learner: str, features: str, k: int) -> float:
    v = [r["macro_f1"] for r in rows("weaklearner_sweep.jsonl")
         if not r.get("error") and r.get("ensemble") == ensemble and r.get("learner") == learner
         and r.get("features") == features and r.get("K") == k and r.get("macro_f1") is not None]
    if not v:
        raise AssertionError(f"no capacity cells: {ensemble} {learner} {features} K={k}")
    return st.mean(v)


def corpus(loader: str, field: str):
    """A field of the recorded corpus fingerprint (env/dataset_fingerprints.json)."""
    doc = json.loads((Path(__file__).resolve().parent.parent / "env"
                      / "dataset_fingerprints.json").read_text(encoding="utf-8"))
    return doc["corpora"][loader][field]


# ---------------------------------------------------------------- the claims

def C(cid, where, text, expected, fn, tol=0.0005):
    return dict(id=cid, where=where, text=text, expected=expected, fn=fn, tol=tol)


CLAIMS = [
    # --- main single-label benchmark (Section 7.1, Tables S1-S2)
    C("sl.clinc.cascade", "7.1 / Table S1a", "RFOED-NN on CLINC150 = 0.903", 0.903,
      lambda: sl("CLINC150", "cascade")),
    C("sl.clinc.mlp", "7.1 / Table S1a", "flat MLP on CLINC150 = 0.948", 0.948,
      lambda: sl("CLINC150", "mlp")),
    C("sl.clinc.tfidf", "7.1 / Table S1a", "strongest TF-IDF baseline on CLINC150 = 0.863", 0.863,
      lambda: sl("CLINC150", "tfidf")),
    C("sl.20ng.cascade", "7.1 / Table S1b", "RFOED-NN on 20 Newsgroups = 0.663", 0.663,
      lambda: sl("20 Newsgroups", "cascade")),
    C("sl.20ng.mlp", "7.1 / Table S1b", "flat MLP on 20 Newsgroups = 0.689", 0.689,
      lambda: sl("20 Newsgroups", "mlp")),
    C("sl.20ng.tfidf", "7.1 / Table S1b", "strongest TF-IDF baseline on 20 Newsgroups = 0.622", 0.622,
      lambda: sl("20 Newsgroups", "tfidf")),
    C("sl.20ng.rfoedrf", "7.1 / Table S1b", "RFOED-RF on 20 Newsgroups = 0.401", 0.401,
      lambda: mean("20newsgroups_ir50.jsonl", "macro_f1", method="RFOED-RF")),
    C("sl.wos.cascade", "7.1 / Table S1c", "RFOED-NN on WOS46985 = 0.452", 0.452,
      lambda: sl("WOS46985", "cascade")),
    C("sl.wos.mlp", "7.1 / Table S1c", "flat MLP on WOS46985 = 0.589", 0.589,
      lambda: sl("WOS46985", "mlp")),
    C("sl.wos.tfidf", "7.1 / Table S1c", "best TF-IDF baseline on WOS46985 = 0.781", 0.781,
      lambda: sl("WOS46985", "tfidf")),
    C("sl.ohsumed.cascade", "7.1 / Table S2d", "RFOED-NN on OHSUMED-23 = 0.251", 0.251,
      lambda: sl("OHSUMED-23", "cascade")),
    C("sl.ohsumed.mlp", "7.1 / Table S2d", "flat MLP on OHSUMED-23 = 0.380", 0.380,
      lambda: sl("OHSUMED-23", "mlp")),
    C("sl.drug.cascade", "7.1 / Table S2e", "RFOED-NN (capped) on Drug Reviews = 0.263", 0.263,
      lambda: sl("Drug Reviews", "cascade")),
    C("sl.drug.mlp", "7.1 / Table S2e", "flat MLP on Drug Reviews = 0.333", 0.333,
      lambda: sl("Drug Reviews", "mlp")),

    # --- the decomposition of the margin (Section 8.1)
    C("margin.clinc", "8.1", "margin over the TF-IDF baselines, CLINC150 = +0.039", 0.039,
      lambda: sl_gap("CLINC150", "cascade", "tfidf")),
    C("margin.20ng", "8.1", "margin over the TF-IDF baselines, 20 Newsgroups = +0.041", 0.041,
      lambda: sl_gap("20 Newsgroups", "cascade", "tfidf")),
    C("margin.drug", "8.1", "margin over the TF-IDF baselines, Drug Reviews = +0.022", 0.022,
      lambda: sl_gap("Drug Reviews", "cascade", "tfidf")),
    C("margin.ohsumed", "8.1", "margin over the TF-IDF baselines, OHSUMED-23 = -0.004", -0.004,
      lambda: sl_gap("OHSUMED-23", "cascade", "tfidf")),
    C("margin.wos", "8.1", "margin over the TF-IDF baselines, WOS46985 = -0.329", -0.329,
      lambda: sl_gap("WOS46985", "cascade", "tfidf")),
    C("decomp.clinc", "8.1", "decomposition term, CLINC150 = -0.045", -0.045,
      lambda: sl_gap("CLINC150", "cascade", "mlp")),
    C("decomp.20ng", "8.1", "decomposition term, 20 Newsgroups = -0.027", -0.027,
      lambda: sl_gap("20 Newsgroups", "cascade", "mlp")),
    C("decomp.drug", "8.1", "decomposition term, Drug Reviews = -0.070", -0.070,
      lambda: sl_gap("Drug Reviews", "cascade", "mlp")),
    C("decomp.ohsumed", "8.1", "decomposition term, OHSUMED-23 = -0.128", -0.128,
      lambda: sl_gap("OHSUMED-23", "cascade", "mlp")),
    C("decomp.wos", "8.1", "decomposition term, WOS46985 = -0.137", -0.137,
      lambda: sl_gap("WOS46985", "cascade", "mlp")),
    C("repr.wos", "8.1", "representation term, WOS46985 = -0.192", -0.192,
      lambda: sl_gap("WOS46985", "mlp", "tfidf")),

    # --- multi-label benchmark (Section 7.1, Table S3)
    C("ml.reuters.focc", "7.1 / Table S3a", "FOCC-NN label-macro-F1, Reuters-21578 = 0.488", 0.488,
      lambda: ml("Reuters-21578", "label_macro_f1", "FOCC-NN")),
    C("ml.hoc.focc", "7.1 / Table S3b", "FOCC-NN label-macro-F1, HoC = 0.688", 0.688,
      lambda: ml("HoC", "label_macro_f1", "FOCC-NN")),
    C("ml.litcovid.focc", "7.1 / Table S3c", "FOCC-NN label-macro-F1, LitCovid = 0.767", 0.767,
      lambda: ml("LitCovid", "label_macro_f1", "FOCC-NN")),
    C("ml.reuters.rare", "7.1", "bottom-quartile-label recall, Reuters-21578: 0.319 vs 0.101", 0.218,
      lambda: gap(("reuters21578.jsonl", "bottom_quartile_label_recall", {"method": "FOCC-NN"}),
                  ("reuters21578.jsonl", "bottom_quartile_label_recall", {"method": "Flat-MLP"}))),
    C("ml.hoc.rare", "7.1", "bottom-quartile-label recall, HoC: 0.756 vs 0.681", 0.075,
      lambda: gap(("hoc.jsonl", "bottom_quartile_label_recall", {"method": "FOCC-NN"}),
                  ("hoc.jsonl", "bottom_quartile_label_recall", {"method": "Flat-MLP"}))),
    C("ml.litcovid.rare", "7.1", "bottom-quartile-label recall, LitCovid: 0.923 vs 0.829 (+0.093)", 0.093,
      lambda: gap(("litcovid.jsonl", "bottom_quartile_label_recall", {"method": "FOCC-NN"}),
                  ("litcovid.jsonl", "bottom_quartile_label_recall", {"method": "Flat-MLP"}))),
    C("ml.micro.reuters", "7.1", "micro-F1 cost on Reuters-21578 = -0.057", -0.057,
      lambda: gap(("reuters21578.jsonl", "micro_f1", {"method": "FOCC-NN"}),
                  ("reuters21578.jsonl", "micro_f1", {"method": "Flat-MLP"}))),
    C("ml.micro.hoc", "7.1", "micro-F1 cost on HoC = -0.011", -0.011,
      lambda: gap(("hoc.jsonl", "micro_f1", {"method": "FOCC-NN"}),
                  ("hoc.jsonl", "micro_f1", {"method": "Flat-MLP"}))),
    C("ml.micro.litcovid", "7.1", "micro-F1 cost on LitCovid = -0.004", -0.004,
      lambda: gap(("litcovid.jsonl", "micro_f1", {"method": "FOCC-NN"}),
                  ("litcovid.jsonl", "micro_f1", {"method": "Flat-MLP"}))),

    # --- pre-registered threshold control (Section 7.1)
    C("thr.reuters", "7.1 / Table 11", "at matched recall, Reuters-21578 advantage = 0.058", 0.058,
      lambda: gap(("reuters21578.jsonl", "bottom_quartile_label_recall", {"method": "FOCC-NN"}),
                  ("threshold_flat_mlp.jsonl", "bottom_quartile_label_recall",
                   {"dataset": "reuters21578", "arm": "C recall-matched"}))),
    C("thr.litcovid", "7.1 / Table 11", "at matched recall, LitCovid advantage = 0.044", 0.044,
      lambda: gap(("litcovid.jsonl", "bottom_quartile_label_recall", {"method": "FOCC-NN"}),
                  ("threshold_flat_mlp.jsonl", "bottom_quartile_label_recall",
                   {"dataset": "litcovid", "arm": "C recall-matched"}))),
    C("thr.hoc", "7.1 / Table 11", "at matched recall, HoC advantage = -0.001", -0.001,
      lambda: gap(("hoc.jsonl", "bottom_quartile_label_recall", {"method": "FOCC-NN"}),
                  ("threshold_flat_mlp.jsonl", "bottom_quartile_label_recall",
                   {"dataset": "hoc", "arm": "C recall-matched"}))),

    # --- the node cap (Section 4.3, Table 8)
    C("cap.clinc.fixed", "4.3 / Table 8", "CLINC150 under the fixed cap = 0.884", 0.884,
      lambda: cap("clinc150", "fixed")),
    C("cap.clinc.g128", "4.3 / Table 8", "CLINC150 at gamma=128 = 0.902", 0.902,
      lambda: cap("clinc150", "g128")),
    C("cap.wos.fixed", "4.3 / Table 8", "WOS46985 under the fixed cap = 0.332", 0.332,
      lambda: cap("wos46985", "fixed")),
    C("cap.wos.g128", "4.3 / Table 8", "WOS46985 at gamma=128 = 0.452", 0.452,
      lambda: cap("wos46985", "g128")),
    C("cap.near.uncapped", "4.3", "gamma=128 is within 0.004 of an uncapped draw on every corpus",
      True, lambda: all(abs(cap(d, "g128") - cap(d, "uncapped")) <= 0.004 for d in
                        sorted({r["dataset"] for r in rows("ncap_adaptive.jsonl")}))),

    # --- cross-dataset rank analysis (Section 7.1, Figure S6)
    C("rank.cascade", "7.1 / Figure S6", "mean rank of RFOED-NN = 4.00", 4.00,
      lambda: ranks()["RFOED-NN"], tol=0.005),
    C("rank.best", "7.1 / Figure S6", "best mean rank is the class-weighted flat MLP at 2.50", 2.50,
      lambda: ranks()["Flat-MLP (class-weighted)"], tol=0.005),
    C("rank.k", "7.1 / Figure S6", "nine methods enter the ranking", 9,
      lambda: len(ranks()), tol=0),
    C("rank.chi2", "7.1 / Figure S6", "Friedman chi-squared = 19.5", 19.5,
      lambda: friedman_chi2(), tol=0.05),

    # --- Proposition 1 (Section 8.3, Table 7)
    C("bound.within", "8.3 / Table 7", "every class lies within the interval of the bound", True,
      lambda: all(r["within_bounds_frac"] == 1.0 for r in rows("cascade_bound.jsonl")
                  if "within_bounds_frac" in r)),
    C("bound.rows", "8.3 / Table 7", "the bound is checked on five corpus-learner configurations", 5,
      lambda: len({(r["dataset"], r["base"]) for r in rows("cascade_bound.jsonl")
                   if "within_bounds_frac" in r}), tol=0),

    # --- fine-tuning (Section 7.4)
    C("ft.reuters", "7.4", "fine-tuned rare-label margin, Reuters-21578 = +0.071", 0.071,
      lambda: finetune_rare_gap("Reuters-21578")),
    C("ft.hoc", "7.4", "fine-tuned rare-label margin, HoC = +0.035", 0.035,
      lambda: finetune_rare_gap("HoC")),
    C("ft.litcovid", "7.4", "fine-tuned rare-label margin, LitCovid = +0.086", 0.086,
      lambda: finetune_rare_gap("LitCovid")),

    # --- weak-learner capacity (Section 6, abstract)
    C("cap5.easy.tfidf", "6 / Table 4", "EasyEnsemble with a depth-10 tree at K=150, TF-IDF >= 0.74",
      (0.735, 0.86), lambda: capacity("EasyEnsemble", "tree10", "tfidf", 150)),
    C("cap5.easy.emb", "6 / Table 4", "EasyEnsemble with a depth-10 tree at K=150, embeddings <= 0.85",
      (0.735, 0.86), lambda: capacity("EasyEnsemble", "tree10", "emb", 150)),
    C("cap5.rus.tfidf", "6 / Table 4", "RUSBoost with an unbounded tree at K=150, TF-IDF = 0.77",
      (0.755, 0.78), lambda: capacity("RUSBoost", "treefull", "tfidf", 150)),
    C("cap5.rus.emb", "6 / Table 4", "RUSBoost with an unbounded tree at K=150, embeddings = 0.76",
      (0.755, 0.78), lambda: capacity("RUSBoost", "treefull", "emb", 150)),

    # --- leakage control (Section 5.3)
    C("leak.mean", "5.3", "per-fold TF-IDF refit shifts macro-F1 by -0.0006 on average", -0.0006,
      lambda: leak_shift()[0], tol=0.00005),
    C("leak.max", "5.3", "the largest shift from the refit is 0.0167", 0.0167,
      lambda: leak_shift()[1], tol=0.00005),
    # --- dataset statistics (Section 5.1, Table 1), against the recorded corpus fingerprints
    C("data.clinc", "5.1 / Table 1", "CLINC150: 150 classes, 18,025 documents, IR 4.0", True,
      lambda: (corpus("clinc150", "n_classes"), corpus("clinc150", "n_documents"),
               round(corpus("clinc150", "imbalance_ratio"), 1)) == (150, 18025, 4.0)),
    C("data.20ng", "5.1 / Table 1", "20 Newsgroups: 20 classes, 5,272 documents, IR about 50", True,
      lambda: (corpus("20newsgroups", "n_classes"), corpus("20newsgroups", "n_documents")) == (20, 5272)
              and 49 <= corpus("20newsgroups", "imbalance_ratio") <= 51),
    C("data.wos", "5.1 / Table 1", "WOS46985: 134 classes, 46,985 documents, IR 17.4", True,
      lambda: (corpus("wos46985", "n_classes"), corpus("wos46985", "n_documents"),
               round(corpus("wos46985", "imbalance_ratio"), 1)) == (134, 46985, 17.4)),
    C("data.ohsumed", "5.1 / Table 1", "OHSUMED-23: 23 classes, 23,166 documents, IR 29.3", True,
      lambda: (corpus("ohsumed_23", "n_classes"), corpus("ohsumed_23", "n_documents"),
               round(corpus("ohsumed_23", "imbalance_ratio"), 1)) == (23, 23166, 29.3)),
    C("data.drug", "5.1 / Table 1", "Drug Reviews: about 356 classes, 49,995 documents, IR about 1830",
      True, lambda: corpus("drug_reviews", "n_documents") == 49995
              and 350 <= corpus("drug_reviews", "n_classes") <= 360
              and 1820 <= corpus("drug_reviews", "imbalance_ratio") <= 1840),
    C("data.reuters", "5.1 / Table 1", "Reuters-21578: 90 labels, 10,788 documents, IR 1982", True,
      lambda: (corpus("reuters21578", "n_classes"), corpus("reuters21578", "n_documents"),
               round(corpus("reuters21578", "imbalance_ratio"))) == (90, 10788, 1982)),
    C("data.hoc", "5.1 / Table 1", "Hallmarks of Cancer: 10 labels, 1,580 documents, IR 4.4", True,
      lambda: (corpus("hoc", "n_classes"), corpus("hoc", "n_documents"),
               round(corpus("hoc", "imbalance_ratio"), 1)) == (10, 1580, 4.4)),
    C("data.litcovid", "5.1 / Table 1", "LitCovid: 7 labels, 33,699 documents, IR 16.8", True,
      lambda: (corpus("litcovid", "n_classes"), corpus("litcovid", "n_documents"),
               round(corpus("litcovid", "imbalance_ratio"), 1)) == (7, 33699, 16.8)),
]



# ---------------------------------------------------------------- runner

def check(c) -> tuple:
    try:
        got = c["fn"]()
    except Exception as e:  # noqa: BLE001
        return False, f"ERROR {type(e).__name__}: {e}"
    exp, tol = c["expected"], c["tol"]
    if isinstance(exp, tuple):
        ok = exp[0] <= got <= exp[1]
        return ok, f"got {got:.4f}, expected within [{exp[0]}, {exp[1]}]"
    if isinstance(exp, bool):
        return got is exp or got == exp, f"got {got}, expected {exp}"
    if isinstance(exp, int) and tol == 0:
        return got == exp, f"got {got}, expected {exp}"
    ok = abs(got - exp) <= tol
    return ok, f"got {got:+.4f}, paper {exp:+.4f}, |diff| {abs(got - exp):.4f} (tolerance {tol})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-k", "--filter", default="", help="only claims whose id contains this text")
    a = ap.parse_args()

    claims = [c for c in CLAIMS if a.filter in c["id"]]
    failed = 0
    for c in claims:
        ok, detail = check(c)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {c['where']:20s} {c['text']}")
        if a.verbose or not ok:
            print(f"        {detail}")
    print(f"\n{len(claims) - failed}/{len(claims)} claims reproduce from results/.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
