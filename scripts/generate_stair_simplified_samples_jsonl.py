#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple simplified caption samples and write JSONL."
    )
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
        "--output-jsonl",
        required=True,
        help="Output JSONL path for sampled candidates.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Field name containing the source caption.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=16)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.1)
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


def compute_sequence_logprobs(scores, sequences, eos_token_id):
    import torch

    if not scores:
        return None
    tokens = sequences[:, 1 : 1 + len(scores)]
    max_steps = tokens.size(1)
    lengths = torch.full(
        (tokens.size(0),), max_steps, device=tokens.device, dtype=torch.long
    )
    if eos_token_id is not None and max_steps > 0:
        eos_mask = tokens.eq(eos_token_id)
        if eos_mask.any():
            idxs = torch.arange(max_steps, device=tokens.device).unsqueeze(0)
            masked = torch.where(eos_mask, idxs, torch.full_like(idxs, max_steps))
            eos_pos = masked.min(dim=1).values
            lengths = torch.where(eos_mask.any(dim=1), eos_pos + 1, lengths)

    logprobs = torch.zeros(tokens.size(0), device=tokens.device)
    for step, step_scores in enumerate(scores[:max_steps]):
        step_logprobs = step_scores.log_softmax(dim=-1)
        step_tokens = tokens[:, step]
        step_lp = step_logprobs.gather(1, step_tokens.unsqueeze(1)).squeeze(1)
        mask = step < lengths
        logprobs += step_lp * mask
    return logprobs


def main():
    args = parse_args()

    if args.num_return_sequences < 1:
        raise SystemExit("--num-return-sequences must be >= 1.")
    if not args.do_sample and args.num_return_sequences > args.num_beams:
        raise SystemExit("--num-return-sequences must be <= num-beams when not sampling.")

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

    indexed = [(idx, ann) for idx, ann in enumerate(annotations)]
    if args.num_shards > 1:
        indexed = [pair for pair in indexed if pair[0] % args.num_shards == args.shard_id]

    index_to_ann = {idx: ann for idx, ann in indexed}
    records = []
    for idx, ann in indexed:
        if args.caption_field not in ann:
            raise SystemExit(f"Missing field '{args.caption_field}' in annotations.")
        records.append({"index": idx, "text": str(ann[args.caption_field])})

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

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_batches = (len(records) + args.batch_size - 1) // args.batch_size
    if args.num_workers > 0:
        from torch.utils.data import DataLoader

        def collate_fn(batch_records):
            indices = [item["index"] for item in batch_records]
            texts = [item["text"] for item in batch_records]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            return indices, texts, encoded

        batch_iterable = DataLoader(
            records,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
    else:

        def batch_iter():
            for batch in batched(records, args.batch_size):
                indices = [item["index"] for item in batch]
                texts = [item["text"] for item in batch]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                yield indices, texts, encoded

        batch_iterable = batch_iter()

    with output_path.open("w", encoding="utf-8") as out_f:
        with torch.inference_mode():
            progress = tqdm(batch_iterable, total=total_batches, desc="Sampling")
            for batch_indices, batch_texts, encoded in progress:
                encoded = {
                    k: v.to(device, non_blocking=args.pin_memory)
                    for k, v in encoded.items()
                }
                outputs = model.generate(
                    **encoded,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    length_penalty=args.length_penalty,
                    early_stopping=args.early_stopping if args.num_beams > 1 else False,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                sequences = outputs.sequences
                decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

                if getattr(outputs, "sequences_scores", None) is not None:
                    scores = outputs.sequences_scores
                else:
                    scores = compute_sequence_logprobs(
                        outputs.scores, sequences, tokenizer.eos_token_id
                    )

                if scores is not None:
                    scores = scores.detach().cpu().tolist()

                group_size = args.num_return_sequences
                if len(decoded) % group_size != 0:
                    raise SystemExit("Generated sequences do not align with return sequences.")

                for i, index in enumerate(batch_indices):
                    start = i * group_size
                    end = start + group_size
                    candidates = []
                    for j, text in enumerate(decoded[start:end]):
                        score = None if scores is None else scores[start + j]
                        candidates.append({"text": text, "score": score})
                    ann = index_to_ann[index]
                    record = {
                        "index": index,
                        "id": ann.get("id"),
                        "image_id": ann.get("image_id"),
                        "source": batch_texts[i],
                        "candidates": candidates,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.show_samples and decoded:
                    progress.set_postfix_str(
                        f"src={batch_texts[0][:20]} pred={decoded[0][:20]}"
                    )

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
