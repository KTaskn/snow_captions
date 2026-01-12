#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss curve.")
    parser.add_argument(
        "--log-file",
        default="outputs/logs/loss_curve.json",
        help="Path to loss_curve.json (JSONL).",
    )
    parser.add_argument(
        "--output",
        default="outputs/logs/loss_curve.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise SystemExit("No records found in log file.")

    steps = [r.get("step") for r in records if "loss" in r]
    losses = [r.get("loss") for r in records if "loss" in r]
    eval_steps = [r.get("step") for r in records if "eval_loss" in r]
    eval_losses = [r.get("eval_loss") for r in records if "eval_loss" in r]

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    plt.figure(figsize=(8, 4))
    if losses:
        plt.plot(steps, losses, label="train loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
