# Repository Guidelines

## Project Structure & Module Organization
- `README.md` outlines the project goal: build a Japanese text simplification model.
- `dataset/` contains the training data in Excel format (`T15-2020.1.7.xlsx`, `T23-2020.1.7.xlsx`).
- No source code directories are present yet; if you add code, keep it in a clear top-level folder such as `src/` and keep data artifacts in `dataset/`.

## Build, Test, and Development Commands
- No build, run, or test scripts are defined in this repository.
- If you add training or evaluation scripts, document the exact commands here (for example: `python -m train --data dataset/…`).
- Generate simplified STAIR captions:
  - `python scripts/generate_stair_simplified.py --model outputs/t5-simplify --input-json STAIR-captions/stair_captions_v1.2_train.json --output-json outputs/stair_simplified_train.json`
  - `python scripts/generate_stair_simplified.py --model outputs/t5-simplify --input-json STAIR-captions/stair_captions_v1.2_val.json --output-json outputs/stair_simplified_val.json`
- Merge sharded simplification outputs:
  - `python scripts/merge_stair_simplified_shards.py --input-glob outputs/stair_simplified_train.shard*.json --output-json outputs/stair_simplified_train.json`
- Score captions with a higher-accuracy LM:
  - `python scripts/score_captions.py --lm-model <lm-path-or-name> --input-json outputs/stair_simplified_train.json --output-json outputs/stair_simplified_train_scored.json`
- Check caption tokens against `dataset/TOKENS.txt`:
  - `python scripts/check_tokens.py --input-json outputs/stair_simplified_train_scored.json --failures-output outputs/stair_simplified_train_token_failures.jsonl`

## Coding Style & Naming Conventions
- No coding style tools are configured yet.
- When introducing code, define formatting and linting rules early (for example, `ruff`/`black` for Python), and document them in this file.
- Prefer descriptive module and function names that reflect NLP tasks (e.g., `dataset_loader.py`, `evaluate_simplification.py`).

## Testing Guidelines
- No testing framework is configured.
- If tests are added, keep them in a `tests/` directory and name files `test_*.py` to match common Python conventions.
- Document how to run tests once a framework is chosen.

## Commit & Pull Request Guidelines
- There is no Git history or commit convention documented in this repo.
- Use concise, imperative commit messages (e.g., "Add dataset loader") and include context in PR descriptions.
- For data or model changes, call out dataset versions, evaluation metrics, and any reproducibility steps.

## Data & Evaluation Notes
- Follow the README instructions for splitting data (1/20 validation) and evaluating predictions.
- Record evaluation results in the PR description or a dedicated results note so future contributors can compare runs.
