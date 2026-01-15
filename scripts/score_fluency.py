#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score fluency with a causal LM and compare to source percentiles."
    )
    parser.add_argument("--lm-model", required=True, help="HF model name or path.")
    parser.add_argument("--input-json", required=True, help="Input JSON annotations.")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL path with per-caption scores.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON output with percentile thresholds.",
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 inference.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. cpu, cuda). Defaults to auto.",
    )
    parser.add_argument(
        "--percentiles",
        default="0.95,0.99",
        help="Comma-separated percentiles for source thresholds.",
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


def ppl_from_total(total, count):
    if count == 0:
        return float("inf")
    avg = total / count
    return math.exp(-avg)


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

    for field in (args.source_field, args.target_field):
        if any(field not in ann for ann in annotations):
            raise SystemExit(f"Missing field '{field}' in annotations.")

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

    source_ppls = []
    target_ppls = []

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        with torch.inference_mode():
            for batch in batched(annotations, args.batch_size):
                source_texts = [str(ann[args.source_field]) for ann in batch]
                target_texts = [str(ann[args.target_field]) for ann in batch]

                source_totals, source_counts = score_batch(
                    model, tokenizer, source_texts, device, args.max_length
                )
                target_totals, target_counts = score_batch(
                    model, tokenizer, target_texts, device, args.max_length
                )

                for ann, s_total, s_count, t_total, t_count in zip(
                    batch, source_totals, source_counts, target_totals, target_counts
                ):
                    s_ppl = ppl_from_total(s_total, s_count)
                    t_ppl = ppl_from_total(t_total, t_count)
                    source_ppls.append(s_ppl)
                    target_ppls.append(t_ppl)
                    record = {
                        "id": ann.get("id"),
                        "image_id": ann.get("image_id"),
                        "source_ppl": float(s_ppl),
                        "target_ppl": float(t_ppl),
                        "delta_ppl": float(t_ppl - s_ppl),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

    percentiles = []
    for item in args.percentiles.split(","):
        item = item.strip()
        if not item:
            continue
        percentiles.append(float(item))

    summary = {
        "model": args.lm_model,
        "source_field": args.source_field,
        "target_field": args.target_field,
        "max_length": args.max_length,
        "percentiles": percentiles,
        "source_thresholds": {
            str(p): percentile(source_ppls, p) for p in percentiles
        },
    }

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
