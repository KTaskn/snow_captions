#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
        return pd
    except Exception as exc:
        raise SystemExit(
            "pandas is required to read .xlsx files. Install with: pip install pandas openpyxl"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare train/valid JSONL from Excel files.")
    parser.add_argument(
        "--input-xlsx",
        nargs="+",
        required=True,
        help="One or more .xlsx files (e.g., dataset/T15-2020.1.7.xlsx).",
    )
    parser.add_argument(
        "--sheet",
        default="平易化コーパス",
        help="Sheet name or index (default: 平易化コーパス).",
    )
    parser.add_argument(
        "--source-column",
        default="#日本語(原文)",
        help="Column name for source text (original Japanese).",
    )
    parser.add_argument(
        "--target-column",
        default="#やさしい日本語",
        help="Column name for target text (easy Japanese).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05 = 1/20).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for JSONL files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pd = _require_pandas()

    src_col = args.source_column
    tgt_col = args.target_column
    frames = []
    file_stats = []
    for path in args.input_xlsx:
        try:
            df = pd.read_excel(path, sheet_name=args.sheet)
        except ValueError:
            # Fallback to the first sheet when the named sheet is missing.
            df = pd.read_excel(path, sheet_name=0)
        if src_col not in df.columns or tgt_col not in df.columns:
            raise SystemExit(
                f"Column(s) not found in {path}. Available columns: {list(df.columns)}"
            )
        subset = df[[src_col, tgt_col]].dropna().astype(str)
        passthrough = (subset[src_col] == subset[tgt_col]).mean() if len(subset) else 0.0
        file_stats.append(
            {
                "file": str(path),
                "rows": int(len(subset)),
                "passthrough_rate": float(passthrough),
            }
        )
        frames.append(subset)

    data = pd.concat(frames, ignore_index=True)

    # Shuffle
    data = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    total = len(data)
    valid_size = max(1, int(total * args.validation_ratio))
    valid = data.iloc[:valid_size]
    train = data.iloc[valid_size:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(df, path):
        with path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record = {"source": row[src_col], "target": row[tgt_col]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")

    passthrough = (data[src_col] == data[tgt_col]).mean()
    stats = {
        "total": total,
        "train": len(train),
        "valid": len(valid),
        "passthrough_rate": float(passthrough),
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with (out_dir / "stats_by_file.json").open("w", encoding="utf-8") as f:
        json.dump(file_stats, f, ensure_ascii=False, indent=2)

    print("Wrote:")
    print("-", out_dir / "train.jsonl")
    print("-", out_dir / "valid.jsonl")
    print("-", out_dir / "stats.json")
    print("-", out_dir / "stats_by_file.json")


if __name__ == "__main__":
    main()
