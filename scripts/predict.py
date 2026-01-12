#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate simplifications for input text.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or HF model name.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Input text (can be repeated).",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Text file with one input per line.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 inference.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. cpu, cuda). Defaults to auto.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSONL with source/prediction fields.",
    )
    return parser.parse_args()


def read_inputs(args):
    inputs = []
    if args.text:
        inputs.extend(args.text)
    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    inputs.append(line)
    if not inputs:
        raise SystemExit("Provide --text or --input-file")
    return inputs


def batched(items, batch_size):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    inputs = read_inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.fp16 or args.bf16:
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        model = model.to(dtype=dtype)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for batch in batched(inputs, args.batch_size):
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model.generate(
                **encoded,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for src, pred in zip(batch, preds):
                if args.json:
                    print(json.dumps({"source": src, "prediction": pred}, ensure_ascii=False))
                else:
                    print(pred)


if __name__ == "__main__":
    main()
