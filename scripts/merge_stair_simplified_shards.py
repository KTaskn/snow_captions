#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge sharded STAIR simplified caption outputs."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for shard JSONs (e.g. outputs/stair_simplified_train.shard*.json).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path for merged captions.",
    )
    parser.add_argument(
        "--sort-key",
        default="id",
        help="Annotation key to sort by before writing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    merged_data = None
    merged_annotations = []

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        annotations = data.get("annotations")
        if not isinstance(annotations, list):
            raise SystemExit(f"Missing annotations list in {path}")
        if merged_data is None:
            merged_data = data
        merged_annotations.extend(annotations)

    merged_annotations.sort(key=lambda ann: ann.get(args.sort_key))
    merged_data["annotations"] = merged_annotations
    if "simplification" in merged_data:
        merged_data["simplification"]["sharding"] = {
            "num_shards": len(input_paths),
            "shard_id": "merged",
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
