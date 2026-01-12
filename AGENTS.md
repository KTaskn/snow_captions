# Repository Guidelines

## Project Structure & Module Organization
- `README.md` outlines the project goal: build a Japanese text simplification model.
- `dataset/` contains the training data in Excel format (`T15-2020.1.7.xlsx`, `T23-2020.1.7.xlsx`).
- No source code directories are present yet; if you add code, keep it in a clear top-level folder such as `src/` and keep data artifacts in `dataset/`.

## Build, Test, and Development Commands
- No build, run, or test scripts are defined in this repository.
- If you add training or evaluation scripts, document the exact commands here (for example: `python -m train --data dataset/…`).

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
