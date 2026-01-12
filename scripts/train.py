#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Japanese T5 model.")
    parser.add_argument(
        "--model-name",
        default="sonoisa/t5-base-japanese-v1.1",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing train.jsonl and valid.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/t5-simplify",
        help="Output directory for checkpoints.",
    )
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Max number of checkpoints to keep.",
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/logs",
        help="Directory to write loss curves.",
    )
    parser.add_argument(
        "--eval-metrics",
        choices=["none", "sari", "sari_bleu"],
        default="none",
        help="Compute SARI (and BLEU) during eval. Requires generation.",
    )
    parser.add_argument(
        "--metrics-tokenize",
        choices=["fugashi", "whitespace", "char"],
        default="fugashi",
        help="Tokenization mode for eval metrics.",
    )
    parser.add_argument(
        "--mecabrc",
        default=None,
        help="Path to mecabrc (used when metrics tokenization is fugashi).",
    )
    parser.add_argument(
        "--mecab-dicdir",
        default=None,
        help="Path to MeCab dictionary dir (used when metrics tokenization is fugashi).",
    )
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
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if eval loss does not improve for N evals.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit number of training samples for quick checks.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Limit number of validation samples for quick checks.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU training (slow, for debugging).",
    )
    return parser.parse_args()


def build_training_args(training_cls, kwargs):
    for _ in range(10):
        try:
            return training_cls(**kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            key = msg.split("unexpected keyword argument")[-1].strip().strip(":").strip()
            key = key.strip("'")
            if key in kwargs:
                kwargs.pop(key)
                if key == "evaluation_strategy":
                    kwargs.setdefault("evaluate_during_training", True)
                continue
            raise
        except ValueError as exc:
            msg = str(exc)
            if "load_best_model_at_end requires" in msg and kwargs.get("load_best_model_at_end"):
                kwargs["load_best_model_at_end"] = False
                continue
            raise
    raise RuntimeError("Failed to build Seq2SeqTrainingArguments with compatible kwargs") from None


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


def f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def sari_sentence(source, prediction, reference, tokenizer):
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

        keep_ng = s_ng & p_ng
        keep_ref = s_ng & r_ng
        keep_good = keep_ng & keep_ref
        keep_precision = len(keep_good) / max(1, len(keep_ng))
        keep_recall = len(keep_good) / max(1, len(keep_ref))
        keep_f1 = f1(keep_precision, keep_recall)

        add_ng = p_ng - s_ng
        add_ref = r_ng - s_ng
        add_good = add_ng & add_ref
        add_precision = len(add_good) / max(1, len(add_ng))
        add_recall = len(add_good) / max(1, len(add_ref))
        add_f1 = f1(add_precision, add_recall)

        del_ng = s_ng - p_ng
        del_ref = s_ng - r_ng
        del_good = del_ng & del_ref
        del_precision = len(del_good) / max(1, len(del_ng))
        del_recall = len(del_good) / max(1, len(del_ref))
        del_f1 = f1(del_precision, del_recall)

        scores.append((keep_f1 + add_f1 + del_f1) / 3.0)

    return sum(scores) / len(scores)


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    if not train_file.exists() or not valid_file.exists():
        raise SystemExit(
            f"Missing data files in {data_dir}. Run scripts/prepare_data.py first."
        )

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(valid_file)},
    )
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
    if args.max_valid_samples:
        dataset["validation"] = dataset["validation"].select(range(args.max_valid_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def preprocess(batch):
        inputs = tokenizer(
            batch["source"],
            max_length=args.max_source_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=args.max_target_length,
                truncation=True,
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_kwargs = dict(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=args.eval_metrics != "none",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        no_cuda=args.no_cuda,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
    )
    training_kwargs.setdefault("do_train", True)
    training_kwargs.setdefault("do_eval", True)

    training_args = build_training_args(Seq2SeqTrainingArguments, training_kwargs)

    def has_eval_strategy(args_obj):
        strategy = None
        if hasattr(args_obj, "evaluation_strategy"):
            strategy = getattr(args_obj, "evaluation_strategy")
        elif hasattr(args_obj, "eval_strategy"):
            strategy = getattr(args_obj, "eval_strategy")
        if strategy is None:
            return False
        strategy_text = str(strategy).lower()
        return not (strategy_text.endswith(".no") or strategy_text == "no")

    class LossLoggerCallback(TrainerCallback):
        def __init__(self, log_dir):
            self.log_dir = Path(log_dir)
            self.records = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {"step": int(state.global_step)}
            for key in ("loss", "eval_loss", "learning_rate", "epoch"):
                if key in logs:
                    record[key] = logs[key]
            if len(record) > 1:
                self.records.append(record)

        def on_train_end(self, args, state, control, **kwargs):
            if not self.records:
                return
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.log_dir / "loss_curve.json"
            out_path.write_text(
                "\n".join([json.dumps(r, ensure_ascii=False) for r in self.records]),
                encoding="utf-8",
            )

    callbacks = [LossLoggerCallback(args.log_dir)]
    if getattr(training_args, "do_eval", True) and has_eval_strategy(training_args):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    def build_compute_metrics():
        if args.eval_metrics == "none":
            return None
        tok = make_tokenizer(args.metrics_tokenize, args.mecabrc, args.mecab_dicdir)
        sacrebleu = None
        if args.eval_metrics == "sari_bleu":
            try:
                import sacrebleu as _sacrebleu
                sacrebleu = _sacrebleu
            except Exception:
                sacrebleu = None

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if predictions is None:
                return {}
            pred_ids = predictions[0] if isinstance(predictions, tuple) else predictions
            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            labels_ids = labels.copy()
            labels_ids = [[tid if tid != -100 else tokenizer.pad_token_id for tid in row] for row in labels_ids]
            label_texts = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            sources = dataset["validation"]["source"]
            sari_scores = [
                sari_sentence(s, p, r, tok)
                for s, p, r in zip(sources, pred_texts, label_texts)
            ]
            metrics = {"sari": sum(sari_scores) / max(1, len(sari_scores))}

            if sacrebleu:
                hyp = [" ".join(tok(p)) for p in pred_texts]
                refs = [[" ".join(tok(r)) for r in label_texts]]
                metrics["bleu"] = sacrebleu.corpus_bleu(hyp, refs, tokenize="none").score

            return metrics

        return compute_metrics

    compute_metrics = build_compute_metrics()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
