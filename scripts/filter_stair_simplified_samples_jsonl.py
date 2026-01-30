#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from check_tokens import (
    can_segment,
    extract_missing_tokens_mecab,
    load_tokens,
    make_tokenizer,
    normalize_text,
    try_make_fugashi_tagger,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter sampled captions by TOKENS and write STAIR JSON."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Original STAIR captions JSON.",
    )
    parser.add_argument(
        "--samples-jsonl",
        required=True,
        help="JSONL produced by generate_stair_simplified_samples_jsonl.py.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path for filtered captions.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the source caption.",
    )
    parser.add_argument(
        "--output-caption-field",
        default="caption",
        help="Field name to store the selected caption.",
    )
    parser.add_argument(
        "--source-caption-field",
        default="source_caption",
        help="Field name to keep the original caption (empty to skip).",
    )
    parser.add_argument(
        "--tokens-file",
        default="dataset/TOKENS.txt",
        help="Path to allowed tokens list (one token per line).",
    )
    parser.add_argument(
        "--tokenize",
        choices=["char", "fugashi"],
        default="fugashi",
        help="Tokenization mode for TOKENS filtering.",
    )
    parser.add_argument(
        "--mecabrc",
        default=None,
        help="Path to mecabrc (used when --tokenize fugashi).",
    )
    parser.add_argument(
        "--mecab-dicdir",
        default=None,
        help="Path to MeCab dictionary dir (used when --tokenize fugashi).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the dataset into N shards for parallel processing.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index to process (0-based).",
    )
    return parser.parse_args()


def build_token_checker(args):
    token_set, _, lengths = load_tokens(args.tokens_file)
    if not token_set:
        raise SystemExit("Tokens file is empty.")
    tokenizer = make_tokenizer(args.tokenize, args.mecabrc, args.mecab_dicdir)
    fallback_tagger = None

    def is_allowed(text):
        nonlocal fallback_tagger
        if tokenizer is None:
            normalized = normalize_text(text)
            ok = can_segment(normalized, token_set, lengths)
            if not ok:
                if fallback_tagger is None:
                    fallback_tagger = try_make_fugashi_tagger(
                        args.mecabrc, args.mecab_dicdir
                    )
                if fallback_tagger is not None:
                    missing = extract_missing_tokens_mecab(
                        text, token_set, fallback_tagger
                    )
                    ok = not missing
            return ok
        missing = extract_missing_tokens_mecab(text, token_set, tokenizer)
        return not missing

    return is_allowed


def load_samples(path):
    samples = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "index" not in record:
                raise SystemExit("Each JSONL record must include 'index'.")
            idx = record["index"]
            if idx in samples:
                raise SystemExit(f"Duplicate index in samples JSONL: {idx}")
            samples[idx] = record
    return samples


def select_best_candidate(candidates, is_allowed):
    best_text = None
    best_score = None
    for cand in candidates:
        text = cand.get("text")
        if text is None:
            continue
        if not is_allowed(text):
            continue
        score = cand.get("score")
        if best_text is None:
            best_text = text
            best_score = score
            continue
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_text = text
            best_score = score
    return best_text


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards).")

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise SystemExit("Input JSON does not contain an annotations list.")

    indexed = [(idx, ann) for idx, ann in enumerate(annotations)]
    if args.num_shards > 1:
        indexed = [pair for pair in indexed if pair[0] % args.num_shards == args.shard_id]
        data["annotations"] = [ann for _, ann in indexed]

    samples = load_samples(args.samples_jsonl)
    is_allowed = build_token_checker(args)

    for idx, ann in indexed:
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        record = samples.get(idx)
        if record is None:
            raise SystemExit(f"Missing samples for index {idx}.")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise SystemExit(f"Missing candidates for index {idx}.")
        best = select_best_candidate(candidates, is_allowed)
        if args.source_caption_field:
            ann.setdefault(args.source_caption_field, ann.get(args.caption_field, ""))
        ann[args.output_caption_field] = best

    data["sampling_filtering"] = {
        "samples_jsonl": str(args.samples_jsonl),
        "tokens_file": args.tokens_file,
        "tokenize": args.tokenize,
        "mecabrc": args.mecabrc,
        "mecab_dicdir": args.mecab_dicdir,
        "fallback_on_all_fail": None,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
