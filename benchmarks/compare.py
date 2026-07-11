"""Aggregate every results/*.json into one comparison table (sorted by KL).

    python compare.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import glob
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.results_dir}/*bit.json"))
    if not files:
        print(f"no result files in {args.results_dir}/ — run the method scripts first.")
        return
    rows = [json.load(open(f)) for f in files]

    # fp16 baseline ppl (same across methods) for a compression/quality frame.
    ppl_fp = next((r["ppl_fp"] for r in rows if r.get("ppl_fp")), None)

    cols = ["method", "nominal_bits", "size_mb", "mean_kl", "top1", "ppl_q", "seconds"]
    rows.sort(key=lambda r: r.get("mean_kl", 9e9))

    w = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
    header = "  ".join(c.ljust(w[c]) for c in cols)
    print(f"\nmodel: {rows[0].get('model')}   fp16 ppl: {ppl_fp}")
    print("(lower mean_kl / higher top1 / lower ppl_q = better)\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))

    # also emit CSV for spreadsheets
    import csv
    from pathlib import Path

    out = Path(args.results_dir) / "summary.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols + ["model", "ppl_fp", "tokens"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in cols + ["model", "ppl_fp", "tokens"]})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
