#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score semantic similarity between source and target captions."
    )
    parser.add_argument("--model", required=True, help="HF model name or path.")
    parser.add_argument("--input-json", required=True, help="Input JSON annotations.")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path with similarity scores added.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON output with similarity thresholds.",
    )
    parser.add_argument(
        "--source-field",
        default="source_caption",
        help="Field name for the original caption.",
    )
    parser.add_argument(
        "--target-field",
        default="caption_replaced",
        help="Field name for the simplified caption.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 inference.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. cpu, cuda). Defaults to auto.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional similarity threshold to flag low-similarity pairs.",
    )
    parser.add_argument(
        "--percentiles",
        default="0.05,0.01",
        help="Comma-separated percentiles for similarity thresholds.",
    )
    return parser.parse_args()


def batched(items, batch_size):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def mean_pooling(model_output, attention_mask):
    import torch

    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(model, tokenizer, texts, device, max_length):
    import torch

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(outputs, attention_mask)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def percentile(values, p):
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)
    values = sorted(values)
    idx = p * (len(values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModel
    import torch
    from tqdm import tqdm

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise SystemExit("Input JSON does not contain an annotations list.")

    for field in (args.source_field, args.target_field):
        if any(field not in ann for ann in annotations):
            raise SystemExit(f"Missing field '{field}' in annotations.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model)

    if args.fp16 or args.bf16:
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        model = model.to(dtype=dtype)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    similarities = []

    with torch.inference_mode():
        total = len(annotations)
        for batch in tqdm(
            batched(annotations, args.batch_size), total=math.ceil(total / args.batch_size)
        ):
            source_texts = [str(ann[args.source_field]) for ann in batch]
            target_texts = [str(ann[args.target_field]) for ann in batch]

            source_emb = embed_texts(
                model, tokenizer, source_texts, device, args.max_length
            )
            target_emb = embed_texts(
                model, tokenizer, target_texts, device, args.max_length
            )
            sim = (source_emb * target_emb).sum(dim=1).detach().cpu().tolist()

            for ann, score in zip(batch, sim):
                similarities.append(score)
                ann["semantic_similarity"] = float(score)
                if args.threshold is not None:
                    ann["semantic_below_threshold"] = bool(score < args.threshold)

    percentiles = []
    for item in args.percentiles.split(","):
        item = item.strip()
        if not item:
            continue
        percentiles.append(float(item))

    summary = {
        "model": args.model,
        "source_field": args.source_field,
        "target_field": args.target_field,
        "max_length": args.max_length,
        "percentiles": percentiles,
        "similarity_thresholds": {str(p): percentile(similarities, p) for p in percentiles},
    }

    data["semantic_scoring"] = summary

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {output_path}")
    if args.summary_json:
        print(f"Wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
