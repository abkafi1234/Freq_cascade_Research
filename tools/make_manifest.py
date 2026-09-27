#!/usr/bin/env python3
"""Write or verify MANIFEST.sha256, a checksum of every file in this repository.

    python tools/make_manifest.py           # write MANIFEST.sha256
    python tools/make_manifest.py --check   # verify it; exit 1 on any mismatch

A reader who downloads a release can run --check to confirm that the result
files are bit-for-bit the ones the paper was written from.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"
SKIP_DIRS = {".git", "__pycache__"}


def files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == MANIFEST:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.check:
        if not MANIFEST.exists():
            sys.exit("MANIFEST.sha256 not found")
        expected = {}
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            h, name = line.split("  ", 1)
            expected[name] = h
        actual = {str(p.relative_to(ROOT)): digest(p) for p in files()}
        bad = [n for n, h in expected.items() if actual.get(n) != h]
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        for n in bad:
            print(f"CHANGED  {n}")
        for n in missing:
            print(f"MISSING  {n}")
        for n in extra:
            print(f"UNLISTED {n}")
        if bad or missing or extra:
            sys.exit(1)
        print(f"OK: {len(expected)} files match MANIFEST.sha256")
        return

    lines = [f"{digest(p)}  {p.relative_to(ROOT)}" for p in files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(lines)} files)")


if __name__ == "__main__":
    main()
