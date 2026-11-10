"""Write the index CSV (segment_id, category_id) for one institution of the Kaggle dataset.

Example:
    python scripts/build_index.py data/DatasetFnusa data/fnusa.csv --max-per-class 13784
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="institution folder containing class sub-folders 0..3")
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-per-class", type=int, default=None, help="random undersampling cap per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for class_dir in sorted(d for d in args.root.iterdir() if d.is_dir() and d.name in {"0", "1", "2", "3"}):
        files = sorted(class_dir.glob("*.mat"))
        if args.max_per_class and len(files) > args.max_per_class:
            files = [files[i] for i in sorted(rng.choice(len(files), args.max_per_class, replace=False))]
        rows += [(f"{class_dir.name}/{f.name}", int(class_dir.name)) for f in files]
        print(f"class {class_dir.name}: {len(files)} segments")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "category_id"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
