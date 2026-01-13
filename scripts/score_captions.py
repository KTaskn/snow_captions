#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Score captions with a causal LM.")
    parser.add_argument(
        "--lm-model",
        required=True,
        help="Model path or HF model name for scoring.",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input JSON containing annotations.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path with score fields added.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the caption to score.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 inference.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. cpu, cuda). Defaults to auto.",
    )
    return parser.parse_args()


def batched(items, batch_size):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def score_batch(model, tokenizer, texts, device, max_length):
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
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * target_mask

    total_logprob = token_log_probs.sum(dim=-1)
    token_count = target_mask.sum(dim=-1)

    totals = total_logprob.detach().cpu().tolist()
    counts = token_count.detach().cpu().tolist()
    return totals, counts


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise SystemExit("Input JSON does not contain an annotations list.")

    texts = []
    for ann in annotations:
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        texts.append(str(ann[args.caption_field]))

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.lm_model)

    if args.fp16 or args.bf16:
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        model = model.to(dtype=dtype)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    totals = []
    counts = []
    with torch.inference_mode():
        for batch in batched(texts, args.batch_size):
            batch_totals, batch_counts = score_batch(
                model, tokenizer, batch, device, args.max_length
            )
            totals.extend(batch_totals)
            counts.extend(batch_counts)

    if len(totals) != len(annotations):
        raise SystemExit("Score count does not match annotation count.")

    for ann, total, count in zip(annotations, totals, counts):
        if count == 0:
            avg = float("inf")
            ppl = float("inf")
        else:
            avg = total / count
            ppl = math.exp(-avg)
        ann["lm_total_logprob"] = float(total)
        ann["lm_avg_logprob"] = float(avg)
        ann["lm_ppl"] = float(ppl)

    data["lm_scoring"] = {
        "model": args.lm_model,
        "caption_field": args.caption_field,
        "max_length": args.max_length,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
