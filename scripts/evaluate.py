#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path


def _try_import(name, install_hint):
    try:
        module = __import__(name)
        return module
    except Exception:
        print(f"Warning: {name} not available. Install with: {install_hint}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a simplification model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or HF model name.",
    )
    parser.add_argument(
        "--data-file",
        default="data/processed/valid.jsonl",
        help="Validation JSONL with source/target fields.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 inference.")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes.",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action="store_true",
        help="Pin memory for faster host-to-device transfer.",
    )
    parser.add_argument(
        "--tokenize",
        choices=["fugashi", "whitespace", "char"],
        default="fugashi",
        help="Tokenization mode for metrics.",
    )
    parser.add_argument(
        "--predictions-path",
        default=None,
        help="Optional path to write predictions JSONL.",
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


def load_jsonl(path):
    data = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def make_tokenizer(mode, mecabrc=None, mecab_dicdir=None):
    if mode == "whitespace":
        return lambda s: s.split()
    if mode == "char":
        return list
    if mode == "fugashi":
        try:
            import fugashi
        except Exception as exc:
            raise SystemExit(
                "fugashi is required for tokenization. Install with: pip install fugashi ipadic"
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
        return lambda s: [token.surface for token in tagger(s)]
    raise ValueError(f"Unknown mode: {mode}")


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sari_sentence(source, prediction, reference, tokenizer):
    # Single-reference SARI implementation (n=1..4)
    s_tok = tokenizer(source)
    p_tok = tokenizer(prediction)
    r_tok = tokenizer(reference)

    if not s_tok or not p_tok or not r_tok:
        return 0.0

    scores = []
    for n in range(1, 5):
        s_ng = set(ngrams(s_tok, n))
        p_ng = set(ngrams(p_tok, n))
        r_ng = set(ngrams(r_tok, n))

        # Keep
        keep_ng = s_ng & p_ng
        keep_ref = s_ng & r_ng
        keep_good = keep_ng & keep_ref
        keep_precision = len(keep_good) / max(1, len(keep_ng))
        keep_recall = len(keep_good) / max(1, len(keep_ref))
        keep_f1 = f1(keep_precision, keep_recall)

        # Add
        add_ng = p_ng - s_ng
        add_ref = r_ng - s_ng
        add_good = add_ng & add_ref
        add_precision = len(add_good) / max(1, len(add_ng))
        add_recall = len(add_good) / max(1, len(add_ref))
        add_f1 = f1(add_precision, add_recall)

        # Delete
        del_ng = s_ng - p_ng
        del_ref = s_ng - r_ng
        del_good = del_ng & del_ref
        del_precision = len(del_good) / max(1, len(del_ng))
        del_recall = len(del_good) / max(1, len(del_ref))
        del_f1 = f1(del_precision, del_recall)

        scores.append((keep_f1 + add_f1 + del_f1) / 3.0)

    return sum(scores) / len(scores)


def f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def build_dataloader(data, batch_size, num_workers, pin_memory):
    from torch.utils.data import DataLoader

    def collate(rows):
        sources = [row["source"] for row in rows]
        targets = [row["target"] for row in rows]
        return sources, targets

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    data = load_jsonl(args.data_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    tok = make_tokenizer(args.tokenize, args.mecabrc, args.mecab_dicdir)

    predictions = []
    sources = []
    references = []

    if args.fp16 or args.bf16:
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        model = model.to(dtype=dtype)

    dataloader = build_dataloader(
        data,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    with torch.inference_mode():
        for texts, targets in dataloader:
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
            sources.extend(texts)
            references.extend(targets)

    # Metrics
    sari_scores = [
        sari_sentence(s, p, r, tok) for s, p, r in zip(sources, predictions, references)
    ]
    sari = sum(sari_scores) / max(1, len(sari_scores))

    sacrebleu = _try_import("sacrebleu", "pip install sacrebleu")
    rouge_score = _try_import("rouge_score", "pip install rouge-score")

    bleu = None
    if sacrebleu:
        hyp = [" ".join(tok(p)) for p in predictions]
        refs = [[" ".join(tok(r)) for r in references]]
        bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize="none").score

    rouge_l = None
    if rouge_score:
        try:
            from rouge_score import rouge_scorer as _rouge_scorer
        except Exception:
            _rouge_scorer = None
        if _rouge_scorer:
            scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            scores = [
                scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(references, predictions)
            ]
            rouge_l = sum(scores) / max(1, len(scores))

    copy_rate = sum(1 for s, p in zip(sources, predictions) if s == p) / max(1, len(sources))
    avg_len_pred = sum(len(tok(p)) for p in predictions) / max(1, len(predictions))
    avg_len_ref = sum(len(tok(r)) for r in references) / max(1, len(references))
    len_ratio = avg_len_pred / max(1.0, avg_len_ref)

    results = {
        "sari": sari,
        "bleu": bleu,
        "rougeL": rouge_l,
        "copy_rate": copy_rate,
        "avg_len_pred": avg_len_pred,
        "avg_len_ref": avg_len_ref,
        "len_ratio": len_ratio,
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.predictions_path:
        out_path = Path(args.predictions_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for s, r, p in zip(sources, references, predictions):
                f.write(json.dumps({"source": s, "target": r, "prediction": p}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
