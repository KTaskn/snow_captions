#!/usr/bin/env python3
import argparse
import json
import os
import re
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


def is_verb_token(token):
    feature = token.feature
    for attr in ("pos1", "pos"):
        value = getattr(feature, attr, None)
        if not value:
            continue
        if isinstance(value, (tuple, list)):
            value = value[0]
        if isinstance(value, str) and value.startswith("動詞"):
            return True
    return False


def _verb_class_from_tag(tag):
    if not is_verb_token(tag):
        return None
    ctype = getattr(tag.feature, "cType", None)
    if not ctype:
        return "godan"
    if "サ行変格" in ctype:
        return "suru"
    if "カ行変格" in ctype:
        return "kuru"
    if ctype.startswith("上一段") or ctype.startswith("下一段"):
        return "ichidan"
    if ctype.startswith("五段"):
        return "godan"
    return "godan"


_ICHIDAN_HINT = set("いきぎしじちぢにひびぴみりえけげせぜてでねへべぺめれ")


def get_verb_class(lemma, tagger, cache):
    if lemma in cache:
        return cache[lemma]
    if lemma in ("する", "為る"):
        cache[lemma] = "suru"
        return cache[lemma]
    if lemma in ("来る", "くる"):
        cache[lemma] = "kuru"
        return cache[lemma]
    if tagger is not None:
        tags = list(tagger(lemma))
        if len(tags) == 1:
            verb_class = _verb_class_from_tag(tags[0])
            cache[lemma] = verb_class
            return cache[lemma]
    if lemma.endswith("る") and len(lemma) >= 2:
        prev = lemma[-2]
        if prev in _ICHIDAN_HINT:
            cache[lemma] = "ichidan"
            return cache[lemma]
    cache[lemma] = "godan"
    return cache[lemma]


def conjugate_te_ta(lemma, form, tagger, cache):
    if form not in ("て", "で", "た", "だ"):
        return None
    verb_class = get_verb_class(lemma, tagger, cache)
    if not verb_class:
        return None
    if verb_class == "suru":
        return "して" if form in ("て", "で") else "した"
    if verb_class == "kuru":
        if lemma == "来る":
            return "来て" if form in ("て", "で") else "来た"
        return "きて" if form in ("て", "で") else "きた"
    if verb_class == "ichidan":
        stem = lemma[:-1]
        return stem + ("て" if form in ("て", "で") else "た")
    last = lemma[-1]
    stem = lemma[:-1]
    if lemma in ("行く", "いく"):
        return stem + ("って" if form in ("て", "で") else "った")
    mapping = {
        "う": ("って", "った"),
        "つ": ("って", "った"),
        "る": ("って", "った"),
        "む": ("んで", "んだ"),
        "ぶ": ("んで", "んだ"),
        "ぬ": ("んで", "んだ"),
        "く": ("いて", "いた"),
        "ぐ": ("いで", "いだ"),
        "す": ("して", "した"),
    }
    if last in mapping:
        te_form, ta_form = mapping[last]
        return stem + (te_form if form in ("て", "で") else ta_form)
    return None


def conjugate_negative(lemma, form, tagger, cache):
    if form not in ("ない", "ぬ"):
        return None
    verb_class = get_verb_class(lemma, tagger, cache)
    if not verb_class:
        return None
    if verb_class == "suru":
        return "し" + form
    if verb_class == "kuru":
        if lemma == "来る":
            return "来" + form
        return "こ" + form
    if verb_class == "ichidan":
        return lemma[:-1] + form
    last = lemma[-1]
    stem = lemma[:-1]
    mapping = {
        "う": "わ",
        "つ": "た",
        "る": "ら",
        "む": "ま",
        "ぶ": "ば",
        "ぬ": "な",
        "く": "か",
        "ぐ": "が",
        "す": "さ",
    }
    if last in mapping:
        return stem + mapping[last] + form
    return None


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

    if base in replacements:
        return replacements[base]
    if kana and kana in replacements:
        return replacements[kana]
    if number is not None and str(number) in replacements:
        return replacements[str(number)]

    # Keep original surface if the token can be validated via base/kana/number.
    if base in token_set:
        return token.surface
    if kana and kana in token_set:
        return token.surface
    if number is not None and str(number) in token_set:
        return token.surface

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
    tokens = list(tagger(text))
    verb_class_cache = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        base = strip_gloss(token_base_form(token))
        if base in replacements and is_verb_token(token) and i + 1 < len(tokens):
            next_surface = tokens[i + 1].surface
            replacement = replacements[base]
            conjugated = conjugate_te_ta(
                replacement, next_surface, tagger, verb_class_cache
            )
            if conjugated is None:
                conjugated = conjugate_negative(
                    replacement, next_surface, tagger, verb_class_cache
                )
            if conjugated is not None:
                parts.append(conjugated)
                i += 2
                continue
        parts.append(normalize_token(token, token_set, replacements))
        i += 1
    replaced = "".join(parts)
    # Avoid compound lemmas like "白き物" and repeated 物 artifacts.
    replaced = re.sub(r"([ぁ-ん一-龥]+い)物", r"\1もの", replaced)
    replaced = replaced.replace("物々", "もの")
    replaced = re.sub(r"物{2,}", "もの", replaced)
    replaced = replaced.replace("ものする", "する")
    return replaced


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
