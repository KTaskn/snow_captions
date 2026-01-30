#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count unique tokens in captions using fugashi tokenization."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input JSON containing annotations.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the caption to tokenize.",
    )
    parser.add_argument(
        "--use-lemma",
        action="store_true",
        help="Use lemma/base form instead of surface tokens.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize tokens (strip gloss, common lemma mapping).",
    )
    parser.add_argument(
        "--kana",
        action="store_true",
        help="Use kana base form when available (requires --normalize).",
    )
    parser.add_argument(
        "--normalize-numbers",
        action="store_true",
        help="Normalize kanji numerals to arabic digits (requires --normalize).",
    )
    parser.add_argument(
        "--tokens-file",
        default=None,
        help="Allowed tokens list; if set, apply kana/number only when it matches tokens.",
    )
    parser.add_argument(
        "--segment-to-tokens",
        action="store_true",
        help="If token is not in TOKENS but can be segmented, count its segments instead.",
    )
    parser.add_argument(
        "--mecabrc",
        default=None,
        help="Path to mecabrc for MeCab (optional).",
    )
    parser.add_argument(
        "--mecab-dicdir",
        default=None,
        help="Path to MeCab dictionary dir (optional).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path for token list.",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional JSON output path for token statistics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Print top-k most frequent tokens.",
    )
    return parser.parse_args()


def make_tagger(mecabrc=None, mecab_dicdir=None):
    try:
        import fugashi
    except Exception as exc:
        raise SystemExit(
            "fugashi is required. Install with: pip install fugashi unidic-lite"
        ) from exc
    options = []
    if mecabrc:
        options.extend(["-r", mecabrc])
    if mecab_dicdir:
        options.extend(["-d", mecab_dicdir])
    tagger_args = " ".join(options)
    if tagger_args:
        return fugashi.Tagger(tagger_args)
    return fugashi.Tagger()


def token_base_form(token):
    feature = token.feature
    for attr in ("lemma", "dictionary_form", "orthBase"):
        value = getattr(feature, attr, None)
        if value and value != "*":
            return value
    return token.surface


def strip_english_gloss(text):
    if "-" not in text:
        return text
    head, tail = text.split("-", 1)
    if not head or not tail:
        return text
    return head


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


COMMON_LEMMA_MAP = {
    "有る": "ある",
    "為る": "する",
    "居る": "いる",
    "成る": "なる",
}

COLOR_NOUN_TO_ADJ = {
    "白": "白い",
    "黒": "黒い",
    "赤": "赤い",
    "青": "青い",
}

ZERO_TO_DIGIT = {
    "ゼロ": "0",
    "０": "0",
    "〇": "0",
    "零": "0",
}


def normalize_token(token, use_lemma, use_kana, normalize_numbers):
    text = token_base_form(token) if use_lemma else token.surface
    text = strip_english_gloss(text)
    if text in COMMON_LEMMA_MAP:
        text = COMMON_LEMMA_MAP[text]
    if use_kana:
        kana = token_kana_base(token)
        if kana:
            text = kana
    if normalize_numbers:
        number = kanji_to_int(text)
        if number is not None:
            text = str(number)
    return text


def load_tokens(path):
    tokens = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip("\n")
            if token:
                tokens.append(token)
    return set(tokens)


def segment_text(text, token_set):
    if not text:
        return []
    lengths = sorted({len(t) for t in token_set}, reverse=True)
    n = len(text)
    inf = n + 1
    dp = [inf] * (n + 1)
    prev = [-1] * (n + 1)
    prev_len = [0] * (n + 1)
    dp[0] = 0
    for i in range(n):
        if dp[i] == inf:
            continue
        for length in lengths:
            j = i + length
            if j > n:
                continue
            if text[i:j] in token_set and dp[i] + 1 < dp[j]:
                dp[j] = dp[i] + 1
                prev[j] = i
                prev_len[j] = length
        if dp[n] != inf:
            break
    if dp[n] == inf:
        return None
    out = []
    j = n
    while j > 0:
        i = prev[j]
        length = prev_len[j]
        if i < 0 or length <= 0:
            return None
        out.append(text[i:j])
        j = i
    out.reverse()
    return out


def normalize_token_with_tokens(token, use_lemma, token_set, use_kana, normalize_numbers):
    text = token_base_form(token) if use_lemma else token.surface
    text = strip_english_gloss(text)
    if text in COMMON_LEMMA_MAP:
        text = COMMON_LEMMA_MAP[text]
    if text in COLOR_NOUN_TO_ADJ and COLOR_NOUN_TO_ADJ[text] in token_set:
        text = COLOR_NOUN_TO_ADJ[text]
    if text in ZERO_TO_DIGIT and ZERO_TO_DIGIT[text] in token_set:
        text = ZERO_TO_DIGIT[text]
    if token_set and text in token_set:
        return text
    if use_kana:
        kana = token_kana_base(token)
        if kana and kana in token_set:
            return kana
    if normalize_numbers:
        number = kanji_to_int(text)
        if number is not None:
            number_text = str(number)
            if number_text in token_set:
                return number_text
    return text


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

    tagger = make_tagger(args.mecabrc, args.mecab_dicdir)
    token_counts = {}
    total_captions = 0
    total_tokens = 0
    captions_with_outside = 0

    token_set = load_tokens(args.tokens_file) if args.tokens_file else None

    for ann in annotations:
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        text = str(ann[args.caption_field])
        total_captions += 1
        has_outside = False
        for token in tagger(text):
            if args.normalize:
                if token_set is not None:
                    token_text = normalize_token_with_tokens(
                        token,
                        use_lemma=args.use_lemma,
                        token_set=token_set,
                        use_kana=args.kana,
                        normalize_numbers=args.normalize_numbers,
                    )
                else:
                    token_text = normalize_token(
                        token,
                        use_lemma=args.use_lemma,
                        use_kana=args.kana,
                        normalize_numbers=args.normalize_numbers,
                    )
            else:
                token_text = token_base_form(token) if args.use_lemma else token.surface
            if not token_text:
                continue
            if args.segment_to_tokens and token_set is not None and token_text not in token_set:
                segments = segment_text(token_text, token_set)
                if segments:
                    for seg in segments:
                        total_tokens += 1
                        token_counts[seg] = token_counts.get(seg, 0) + 1
                    continue
            if token_set is not None and token_text not in token_set:
                has_outside = True
            total_tokens += 1
            token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if has_outside:
            captions_with_outside += 1

    print(f"Total captions: {total_captions}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(token_counts)}")
    if token_set is not None:
        print(f"Captions with outside tokens: {captions_with_outside}")
    if args.top_k and token_counts:
        top_items = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[: args.top_k]
        top_preview = ", ".join(f"{tok}:{cnt}" for tok, cnt in top_items)
        print(f"Top {args.top_k}: {top_preview}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            token_list = [
                {"token": token, "count": count}
                for token, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
            ]
            json.dump(token_list, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {output_path}")
    if args.stats_json:
        output_path = Path(args.stats_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "total_captions": total_captions,
            "total_tokens": total_tokens,
            "unique_tokens": len(token_counts),
        }
        if token_set is not None:
            stats["captions_with_outside_tokens"] = captions_with_outside
            stats["captions_with_outside_tokens_rate"] = (
                captions_with_outside / total_captions if total_captions else 0.0
            )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
