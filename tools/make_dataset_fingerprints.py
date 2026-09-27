#!/usr/bin/env python3
"""Fingerprint the eight corpora, so a reader can prove they downloaded what we used.

None of the corpora is redistributed in this repository: each is fetched from its original
distributor by the loaders in ``freqcascade.datasets``. That leaves one gap a reader cannot
otherwise close -- whether the copy they fetched is the copy the paper was computed on. This
script closes it by recording, per corpus, the size, the class structure and one-way digests of
the text and of the labels, in the order the loader returns them.

A digest is not the data: it cannot be inverted into text, so publishing it redistributes
nothing.

Requires the package and the corpora:

    pip install freqcascade
    python tools/make_dataset_fingerprints.py        # writes env/dataset_fingerprints.json
    python tools/make_dataset_fingerprints.py --check # compare a local copy against the record
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "env" / "dataset_fingerprints.json"
CORPORA = ["clinc150", "20newsgroups", "wos46985", "ohsumed_23", "drug_reviews",
           "reuters21578", "hoc", "litcovid"]


def sha256_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def fingerprint(name: str) -> dict:
    from freqcascade.datasets import load
    ds = load(name)
    freqs = ds.class_frequencies
    return {
        "loader": name,
        "dataset_name": ds.name,
        "n_documents": int(ds.n_samples),
        "n_classes": int(ds.n_classes),
        "multilabel": bool(ds.is_multilabel),
        "imbalance_ratio": round(float(ds.imbalance_ratio), 4),
        "class_frequencies": [int(x) for x in freqs],
        "has_provided_split": ds.split is not None,
        "sha256_texts": sha256_texts(ds.texts),
        "sha256_target": hashlib.sha256(ds.target.tobytes()).hexdigest(),
        "sha256_label_names": hashlib.sha256("\x00".join(ds.label_names).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("corpora", nargs="*", default=None)
    a = ap.parse_args()
    names = a.corpora or CORPORA

    if a.check:
        stored = json.loads(OUT.read_text(encoding="utf-8"))["corpora"]
        bad = 0
        for n in names:
            want = stored.get(n)
            if want is None:
                print(f"SKIP  {n}: not in the record")
                continue
            try:
                got = fingerprint(n)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR {n}: {type(e).__name__}: {e}")
                bad += 1
                continue
            diff = [k for k in want if k != "note" and want[k] != got[k]]
            print(("MATCH " if not diff else "DIFFER") + f" {n}"
                  + (f"  ({', '.join(diff)})" if diff else ""))
            bad += bool(diff)
        raise SystemExit(1 if bad else 0)

    out = {}
    for n in names:
        print(f"fingerprinting {n} ...", flush=True)
        out[n] = fingerprint(n)
    doc = {
        "note": "Digests of the corpora as the loaders return them, in loader order. "
                "The corpora themselves are not redistributed; see README.md for their sources. "
                "A differing digest means the local copy is not the one the paper used.",
        "corpora": out,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
