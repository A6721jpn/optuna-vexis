"""
教師データCSVマージユーティリティ

使用方法:
  conda run -n ai python merge_samples.py --inputs old.csv new.csv --output merged.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple training data CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs", type=Path, nargs="+", required=True,
        help="Input CSV files to merge"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output merged CSV file"
    )
    parser.add_argument(
        "--dedupe", action="store_true", default=True,
        help="Remove duplicate rows"
    )
    parser.add_argument(
        "--report", action="store_true", default=True,
        help="Print statistics report"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    dfs = []
    for path in args.inputs:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        print(f"  Loaded {path.name}: {len(df)} rows")
        dfs.append(df)
    
    if not dfs:
        print("Error: No valid input files")
        return
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Total after concat: {len(merged)} rows")
    
    if args.dedupe:
        before = len(merged)
        merged = merged.drop_duplicates()
        after = len(merged)
        print(f"After deduplication: {after} rows (removed {before - after})")
    
    if args.report and "safe" in merged.columns:
        safe_count = merged["safe"].sum()
        unsafe_count = len(merged) - safe_count
        safe_ratio = safe_count / len(merged) if len(merged) > 0 else 0
        print(f"\n=== Dataset Statistics ===")
        print(f"  Total samples: {len(merged)}")
        print(f"  Safe: {int(safe_count)} ({safe_ratio:.1%})")
        print(f"  Unsafe: {int(unsafe_count)} ({1 - safe_ratio:.1%})")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
