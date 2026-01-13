#!/usr/bin/env python3
import argparse
import json
from collections import Counter
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify captions use only allowed tokens.")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input JSON containing annotations.",
    )
    parser.add_argument(
        "--tokens-file",
        default="dataset/TOKENS.txt",
        help="Path to allowed tokens list (one token per line).",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the caption to check.",
    )
    parser.add_argument(
        "--tokenize",
        choices=["char", "fugashi"],
        default="char",
        help="Tokenization mode for checking tokens.",
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
        "--failures-output",
        default=None,
        help="Optional JSONL output for failed captions.",
    )
    parser.add_argument(
        "--missing-tokens-output",
        default=None,
        help="Optional JSON output for missing tokens aggregated across failures.",
    )
    parser.add_argument(
        "--include-missing-tokens-in-failures",
        action="store_true",
        help="Include missing tokens list in failures output records.",
    )
    return parser.parse_args()


def load_tokens(path):
    tokens = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip("\n")
            if token:
                tokens.append(token)
    token_set = set(tokens)
    max_len = max((len(t) for t in token_set), default=0)
    lengths = sorted({len(t) for t in token_set}, reverse=True)
    return token_set, max_len, lengths


def normalize_text(text):
    return "".join(text.split())


def can_segment(text, token_set, lengths):
    if not text:
        return True
    n = len(text)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(n):
        if not dp[i]:
            continue
        for length in lengths:
            j = i + length
            if j > n:
                continue
            if text[i:j] in token_set:
                dp[j] = True
        if dp[n]:
            return True
    return dp[n]


def token_base_form(token):
    feature = token.feature
    for attr in ("lemma", "dictionary_form", "orthBase"):
        value = getattr(feature, attr, None)
        if value and value != "*":
            return value
    return token.surface


def strip_english_gloss(token):
    if "-" not in token:
        return token
    head, tail = token.split("-", 1)
    if not head or not tail:
        return token
    if all(ord(ch) < 128 for ch in tail):
        return head
    return token


def kata_to_hira(text):
    if not text:
        return text
    diff = ord("ぁ") - ord("ァ")
    result = []
    for ch in text:
        code = ord(ch)
        if ord("ァ") <= code <= ord("ヶ"):
            result.append(chr(code + diff))
        else:
            result.append(ch)
    return "".join(result)


def token_kana_base(token):
    feature = token.feature
    kana = getattr(feature, "kanaBase", None) or getattr(feature, "pronBase", None)
    if not kana or kana == "*":
        kana = getattr(feature, "kana", None) or getattr(feature, "pron", None)
    if not kana or kana == "*":
        return None
    return kata_to_hira(kana)


_KANJI_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_KANJI_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
}
_KANJI_LARGE_UNITS = {
    "万": 10**4,
    "億": 10**8,
}


def kanji_to_int(text):
    if not text or any(ch not in _KANJI_DIGITS and ch not in _KANJI_UNITS and ch not in _KANJI_LARGE_UNITS for ch in text):
        return None
    total = 0
    section = 0
    number = 0
    for ch in text:
        if ch in _KANJI_DIGITS:
            number = _KANJI_DIGITS[ch]
            continue
        if ch in _KANJI_UNITS:
            unit = _KANJI_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            continue
        if ch in _KANJI_LARGE_UNITS:
            unit = _KANJI_LARGE_UNITS[ch]
            section += number
            if section == 0:
                section = 1
            total += section * unit
            section = 0
            number = 0
    return total + section + number


def make_tokenizer(mode, mecabrc=None, mecab_dicdir=None):
    if mode == "char":
        return None
    if mode == "fugashi":
        try:
            import fugashi
        except Exception as exc:
            raise SystemExit(
                "fugashi is required for tokenization. Install with: pip install fugashi unidic-lite"
            ) from exc
        options = []
        mecabrc = mecabrc or os.environ.get("MECABRC")
        mecab_dicdir = mecab_dicdir or os.environ.get("MECAB_DICDIR")
        if not mecabrc:
            for candidate in ("/etc/mecabrc", "/usr/local/etc/mecabrc"):
                if Path(candidate).exists():
                    mecabrc = candidate
                    break
        if mecabrc:
            options.extend(["-r", mecabrc])
        if mecab_dicdir:
            options.extend(["-d", mecab_dicdir])
        tagger_args = " ".join(options)
        try:
            tagger = fugashi.Tagger(tagger_args) if tagger_args else fugashi.Tagger()
        except RuntimeError as exc:
            if "Unknown dictionary format" in str(exc):
                tagger = (
                    fugashi.GenericTagger(tagger_args)
                    if tagger_args
                    else fugashi.GenericTagger()
                )
            else:
                raise
        return tagger
    raise ValueError(f"Unknown mode: {mode}")


def extract_missing_tokens(text, token_set, lengths):
    if not text:
        return []
    n = len(text)
    inf = n + 1
    dp = [inf] * (n + 1)
    prev = [-1] * (n + 1)
    prev_unknown = [False] * (n + 1)
    dp[0] = 0
    for i in range(n):
        if dp[i] == inf:
            continue
        for length in lengths:
            j = i + length
            if j > n:
                continue
            if text[i:j] in token_set and dp[i] < dp[j]:
                dp[j] = dp[i]
                prev[j] = i
                prev_unknown[j] = False
        j = i + 1
        if j <= n and dp[i] + 1 < dp[j]:
            dp[j] = dp[i] + 1
            prev[j] = i
            prev_unknown[j] = True
    if dp[n] == inf:
        return []
    missing = []
    buffer = []
    idx = n
    while idx > 0:
        start = prev[idx]
        if start < 0:
            break
        if prev_unknown[idx]:
            buffer.append(text[start:idx])
        else:
            if buffer:
                missing.append("".join(reversed(buffer)))
                buffer = []
        idx = start
    if buffer:
        missing.append("".join(reversed(buffer)))
    missing.reverse()
    return missing


def extract_missing_tokens_mecab(text, token_set, tagger):
    missing = []
    for token in tagger(text):
        base = strip_english_gloss(token_base_form(token))
        if base in token_set:
            continue
        kana = token_kana_base(token)
        if kana and kana in token_set:
            continue
        number = kanji_to_int(base)
        if number is not None and str(number) in token_set:
            continue
        if base:
            missing.append(base)
    return missing


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise SystemExit("Input JSON does not contain an annotations list.")

    token_set, _, lengths = load_tokens(args.tokens_file)
    if not token_set:
        raise SystemExit("Tokens file is empty.")

    failures = []
    missing_tokens = Counter()
    tokenizer = make_tokenizer(args.tokenize, args.mecabrc, args.mecab_dicdir)
    for idx, ann in enumerate(annotations):
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        raw_text = str(ann[args.caption_field])
        if tokenizer is None:
            text = normalize_text(raw_text)
            ok = can_segment(text, token_set, lengths)
        else:
            text = raw_text
            missing = extract_missing_tokens_mecab(text, token_set, tokenizer)
            ok = not missing
        if not ok:
            record = {
                "index": idx,
                "id": ann.get("id"),
                "image_id": ann.get("image_id"),
                "caption": ann.get(args.caption_field),
            }
            if tokenizer is None:
                if args.missing_tokens_output or args.include_missing_tokens_in_failures:
                    missing = extract_missing_tokens(text, token_set, lengths)
                else:
                    missing = []
            if args.include_missing_tokens_in_failures:
                record["missing_tokens"] = missing
            failures.append(record)
            if args.missing_tokens_output:
                missing_tokens.update(missing)

    total = len(annotations)
    failed = len(failures)
    passed = total - failed
    rate = (passed / total) if total else 0.0
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {rate:.4f}")

    if args.failures_output:
        output_path = Path(args.failures_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in failures:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote: {output_path}")

    if args.missing_tokens_output:
        output_path = Path(args.missing_tokens_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        missing_list = [
            {"token": token, "count": count}
            for token, count in sorted(
                missing_tokens.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(missing_list, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
