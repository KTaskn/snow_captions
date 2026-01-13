#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace tokens to fit within a fixed vocabulary."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input JSON containing annotations.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON with replaced captions.",
    )
    parser.add_argument(
        "--tokens-file",
        default="dataset/TOKENS.txt",
        help="Path to allowed tokens list (one token per line).",
    )
    parser.add_argument(
        "--replacements",
        default="scripts/token_replacements.json",
        help="JSON mapping of token replacements.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the caption to replace.",
    )
    parser.add_argument(
        "--output-field",
        default=None,
        help="Optional field name to store the replaced caption.",
    )
    parser.add_argument(
        "--tokenize",
        choices=["char", "fugashi"],
        default="fugashi",
        help="Tokenization mode for replacements.",
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


def load_tokens(path):
    tokens = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip("\n")
            if token:
                tokens.append(token)
    return set(tokens)


def load_replacements(path, token_set):
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise SystemExit(f"Replacements file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    for key, value in mapping.items():
        if value not in token_set:
            raise SystemExit(f"Replacement target not in TOKENS: {key} -> {value}")
    return mapping


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


def token_base_form(token):
    feature = token.feature
    for attr in ("lemma", "dictionary_form", "orthBase"):
        value = getattr(feature, attr, None)
        if value and value != "*":
            return value
    return token.surface


def strip_gloss(token):
    if "-" not in token:
        return token
    head, tail = token.split("-", 1)
    if not head or not tail:
        return token
    return head


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
    if not text or any(
        ch not in _KANJI_DIGITS
        and ch not in _KANJI_UNITS
        and ch not in _KANJI_LARGE_UNITS
        for ch in text
    ):
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


def is_latin_token(text):
    if not text:
        return False
    for ch in text:
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            continue
        if "Ａ" <= ch <= "Ｚ" or "ａ" <= ch <= "ｚ":
            continue
        return False
    return True


def make_tagger(mode, mecabrc=None, mecab_dicdir=None):
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


def normalize_token(token, token_set, replacements):
    base = strip_gloss(token_base_form(token))
    kana = token_kana_base(token)
    number = kanji_to_int(base)

    # Keep original surface if the token can be validated via base/kana/number.
    if base in token_set:
        return token.surface
    if kana and kana in token_set:
        return token.surface
    if number is not None and str(number) in token_set:
        return token.surface

    if base in replacements:
        return replacements[base]
    if kana and kana in replacements:
        return replacements[kana]
    if number is not None and str(number) in replacements:
        return replacements[str(number)]
    if kana and kana in token_set:
        return kana
    if number is not None and str(number) in token_set:
        return str(number)
    if is_latin_token(base) and "文字" in token_set:
        return "文字"
    return base


def replace_caption(text, tagger, token_set, replacements):
    if tagger is None:
        return text
    parts = []
    for token in tagger(text):
        parts.append(normalize_token(token, token_set, replacements))
    return "".join(parts)


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

    token_set = load_tokens(args.tokens_file)
    if not token_set:
        raise SystemExit("Tokens file is empty.")
    replacements = load_replacements(args.replacements, token_set)

    tagger = make_tagger(args.tokenize, args.mecabrc, args.mecab_dicdir)

    output_field = args.output_field or args.caption_field
    for ann in annotations:
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        text = str(ann[args.caption_field])
        ann[output_field] = replace_caption(text, tagger, token_set, replacements)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
