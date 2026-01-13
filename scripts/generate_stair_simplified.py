#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate simplified STAIR captions.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or HF model name for simplification.",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input STAIR captions JSON (e.g., STAIR-captions/stair_captions_v1.2_train.json).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path for simplified captions.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the source caption.",
    )
    parser.add_argument(
        "--output-caption-field",
        default="caption",
        help="Field name to store the generated caption.",
    )
    parser.add_argument(
        "--source-caption-field",
        default="source_caption",
        help="Field name to keep the original caption (empty to skip).",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes for tokenization.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each worker.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin memory for faster host-to-device transfer.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 inference.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. cpu, cuda). Defaults to auto.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Show a sample source/prediction in the progress bar.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the dataset into N shards for parallel generation.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index to process (0-based).",
    )
    return parser.parse_args()


def batched(items, batch_size):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards).")

    if args.num_shards > 1:
        sharded = []
        for idx, ann in enumerate(annotations):
            if idx % args.num_shards == args.shard_id:
                sharded.append(ann)
        annotations = sharded
        data["annotations"] = annotations

    captions = []
    for ann in annotations:
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        captions.append(str(ann[args.caption_field]))

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

    predictions = []
    total_batches = (len(captions) + args.batch_size - 1) // args.batch_size
    if args.num_workers > 0:
        from torch.utils.data import DataLoader

        def collate_fn(batch_texts):
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            return list(batch_texts), encoded

        batch_iterable = DataLoader(
            captions,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        def batch_iter():
            for batch in batched(captions, args.batch_size):
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                yield batch, encoded

        batch_iterable = batch_iter()

    with torch.inference_mode():
        progress = tqdm(batch_iterable, total=total_batches, desc="Generating")
        for batch_texts, encoded in progress:
            encoded = {
                k: v.to(device, non_blocking=args.pin_memory) for k, v in encoded.items()
            }
            outputs = model.generate(
                **encoded,
                max_length=args.max_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping if args.num_beams > 1 else False,
                do_sample=args.do_sample,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
            )
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)
            if args.show_samples and batch_preds:
                progress.set_postfix_str(
                    f"src={batch_texts[0][:20]} pred={batch_preds[0][:20]}"
                )

    if len(predictions) != len(annotations):
        raise SystemExit("Prediction count does not match annotation count.")

    for ann, pred in zip(annotations, predictions):
        if args.source_caption_field:
            ann.setdefault(args.source_caption_field, ann.get(args.caption_field, ""))
        ann[args.output_caption_field] = pred

    data["simplification"] = {
        "model": args.model,
        "caption_field": args.caption_field,
        "output_caption_field": args.output_caption_field,
        "source_caption_field": args.source_caption_field,
        "generation": {
            "max_length": args.max_length,
            "num_beams": args.num_beams,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "early_stopping": args.early_stopping,
            "do_sample": args.do_sample,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "pin_memory": args.pin_memory,
        },
        "sharding": {
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
