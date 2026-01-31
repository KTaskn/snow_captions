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
from count_unique_tokens_fugashi import normalize_token_with_tokens, segment_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split simplified STAIR captions by TOKENS coverage."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input STAIR captions JSON (e.g., dataset/extras/train_v11.json).",
    )
    parser.add_argument(
        "--output-pass-jsonl",
        default=None,
        help="Optional JSONL output for TOKENS-covered pairs.",
    )
    parser.add_argument(
        "--output-pass-csv",
        default=None,
        help="Optional CSV output for TOKENS-covered pairs.",
    )
    parser.add_argument(
        "--csv-source-column",
        default="#日本語(原文)",
        help="CSV column name for the source text.",
    )
    parser.add_argument(
        "--csv-target-column",
        default="#やさしい日本語",
        help="CSV column name for the target text.",
    )
    parser.add_argument(
        "--output-fail-json",
        required=True,
        help="Output STAIR JSON for captions that are outside TOKENS.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the simplified caption to check.",
    )
    parser.add_argument(
        "--source-caption-field",
        default="source_caption",
        help="Field name containing the source caption.",
    )
    parser.add_argument(
        "--tokens-file",
        default="dataset/TOKENS.txt",
        help="Path to allowed tokens list (one token per line).",
    )
    parser.add_argument(
        "--outside-tokens-json",
        default="dataset/extras/lemma_tokens_outside_tokens_txt_norm3.json",
        help=(
            "JSON list of known outside tokens to validate missing tokens against. "
            "Set to empty string to skip."
        ),
    )
    parser.add_argument(
        "--outside-tokens-output",
        default=None,
        help="Optional JSON output of outside tokens observed (list of {token,count}).",
    )
    parser.add_argument(
        "--outside-tokens-report",
        default=None,
        help="Optional JSON output for outside-tokens coverage summary.",
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
    return parser.parse_args()


def build_token_checker(args, token_set, lengths):
    if not token_set:
        raise SystemExit("Tokens file is empty.")
    tokenizer = make_tokenizer(args.tokenize, args.mecabrc, args.mecab_dicdir)
    fallback_tagger = None

    def check_text(text):
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

    return check_text


def load_outside_tokens(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            return set()
        if isinstance(data[0], dict) and "token" in data[0]:
            return {item["token"] for item in data if "token" in item}
        if isinstance(data[0], str):
            return set(data)
    raise SystemExit(f"Unexpected outside tokens JSON format: {path}")


def main():
    args = parse_args()

    if not args.output_pass_jsonl and not args.output_pass_csv:
        raise SystemExit("Either --output-pass-jsonl or --output-pass-csv must be set.")

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise SystemExit("Input JSON does not contain an annotations list.")

    token_set, _, lengths = load_tokens(args.tokens_file)
    check_text = build_token_checker(args, token_set, lengths)

    outside_tokens = None
    # if args.outside_tokens_json:
    #     outside_path = Path(args.outside_tokens_json)
    #     if not outside_path.exists():
    #         raise SystemExit(f"Outside tokens JSON not found: {outside_path}")
    #     outside_tokens = load_outside_tokens(outside_path)

    need_outside_counts = bool(args.outside_tokens_output or args.outside_tokens_report)
    report_tagger = None
    if need_outside_counts:
        report_tagger = make_tokenizer("fugashi", args.mecabrc, args.mecab_dicdir)
        if report_tagger is None:
            raise SystemExit("Failed to initialize fugashi tagger for report.")

    output_pass = Path(args.output_pass_jsonl) if args.output_pass_jsonl else None
    if output_pass is not None:
        output_pass.parent.mkdir(parents=True, exist_ok=True)
    output_pass_csv = Path(args.output_pass_csv) if args.output_pass_csv else None
    if output_pass_csv is not None:
        output_pass_csv.parent.mkdir(parents=True, exist_ok=True)

    failed_annotations = []
    report_token_counts = {}
    passed = 0
    failed = 0
    from tqdm import tqdm

    import csv

    csv_writer = None
    csv_file = None
    if output_pass_csv is not None:
        csv_file = output_pass_csv.open("w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([args.csv_source_column, args.csv_target_column])

    out_f = None
    if output_pass is not None:
        out_f = output_pass.open("w", encoding="utf-8")

    for idx, ann in enumerate(tqdm(annotations, desc="Checking", unit="ann")):
            if args.caption_field not in ann:
                raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
            if args.source_caption_field not in ann:
                raise SystemExit(
                    f"Missing field '{args.source_caption_field}' in annotations."
                )
            caption = ann.get(args.caption_field)
            source_caption = ann.get(args.source_caption_field)
            if caption is None or source_caption is None:
                failed_annotations.append(ann)
                failed += 1
                continue
            caption_text = str(caption)
            ok = check_text(caption_text)
            if ok:
                record = {"source": str(source_caption), "target": caption_text}
                if out_f is not None:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if csv_writer is not None:
                    csv_writer.writerow([record["source"], record["target"]])
                passed += 1
            else:
                failed_annotations.append(ann)
                failed += 1
                if need_outside_counts:
                    for token in report_tagger(caption_text):
                        token_text = normalize_token_with_tokens(
                            token,
                            use_lemma=True,
                            token_set=token_set,
                            use_kana=True,
                            normalize_numbers=True,
                        )
                        if not token_text:
                            continue
                        if token_text in token_set:
                            continue
                        segments = segment_text(token_text, token_set)
                        if segments:
                            # Segmentable tokens are considered covered by TOKENS.
                            continue
                        report_token_counts[token_text] = (
                            report_token_counts.get(token_text, 0) + 1
                        )

    output_fail = Path(args.output_fail_json)
    output_fail.parent.mkdir(parents=True, exist_ok=True)
    failed_data = dict(data)
    failed_data["annotations"] = failed_annotations
    with output_fail.open("w", encoding="utf-8") as f:
        json.dump(failed_data, f, ensure_ascii=False)

    total = passed + failed
    rate = (passed / total) if total else 0.0
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {rate:.4f}")
    if out_f is not None:
        out_f.close()
        print(f"Wrote: {output_pass}")
    print(f"Wrote: {output_fail}")
    if output_pass_csv is not None:
        print(f"Wrote: {output_pass_csv}")

    if args.outside_tokens_output and need_outside_counts:
        output_path = Path(args.outside_tokens_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        token_list = [
            {"token": token, "count": count}
            for token, count in sorted(
                report_token_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        output_path.write_text(
            json.dumps(token_list, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote: {output_path}")

    if outside_tokens is not None and args.outside_tokens_report and need_outside_counts:
        missing_unique = len(report_token_counts)
        missing_total_count = sum(report_token_counts.values())
        in_list_unique = sum(1 for token in report_token_counts if token in outside_tokens)
        in_list_total = sum(
            count for token, count in report_token_counts.items() if token in outside_tokens
        )
        not_in_list = [
            {"token": token, "count": count}
            for token, count in sorted(
                report_token_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if token not in outside_tokens
        ]
        not_in_list_total = missing_total_count - in_list_total

        print("Outside tokens coverage:")
        print(f"- missing unique: {missing_unique}")
        print(f"- missing in list unique: {in_list_unique}")
        print(f"- missing not in list unique: {missing_unique - in_list_unique}")
        print(f"- missing total: {missing_total_count}")
        print(f"- missing in list total: {in_list_total}")
        print(f"- missing not in list total: {not_in_list_total}")

        report = {
            "outside_tokens_json": args.outside_tokens_json,
            "missing_unique": missing_unique,
            "missing_total": missing_total_count,
            "missing_in_list_unique": in_list_unique,
            "missing_in_list_total": in_list_total,
            "missing_not_in_list_unique": missing_unique - in_list_unique,
            "missing_not_in_list_total": not_in_list_total,
            "missing_not_in_list_top": not_in_list[:50],
            "normalization": {
                "use_lemma": True,
                "use_kana": True,
                "normalize_numbers": True,
                "segment_to_tokens": True,
                "tokenize": "fugashi",
            },
        }
        if args.outside_tokens_report:
            report_path = Path(args.outside_tokens_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Wrote: {report_path}")

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
