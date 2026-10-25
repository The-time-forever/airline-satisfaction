# Repository Guidelines

## Project Structure & Module Organization

- `main.py` dispatches the command-line workflows.
- `airline_satisfaction/` contains preprocessing, cross-validation, baseline training, evaluation, feature experiments, Optuna tuning, and fusion utilities. Keep reusable estimators in importable modules so saved Pipelines reload correctly.
- `tests/` contains standard-library `unittest` tests named `test_*.py`.
- `analysis/eda.py` generates `analysis/EDA.md`.
- `data/` contains `train.csv`, `test.csv`, and `sample_submission.csv`; treat these inputs as read-only.
- `artifacts/` holds generated models, OOF predictions, reports, and experiment databases and is Git-ignored. `artifacts/selected_model.json` identifies the current model; `artifacts/submission.csv` is its submission file.

## Development & Validation Commands

The shared Conda `ml` environment interpreter is `D:\SoftWare\Anaconda\envs\ml\python.exe` (verified as Python 3.12.15). Run every Python command through this absolute path from the repository root; do not rely on the active environment or a `python` command resolved through `PATH`.

```bat
D:\SoftWare\Anaconda\envs\ml\python.exe -m pip install -r requirements.txt
D:\SoftWare\Anaconda\envs\ml\python.exe -m unittest discover -s tests -v
D:\SoftWare\Anaconda\envs\ml\python.exe analysis/eda.py
D:\SoftWare\Anaconda\envs\ml\python.exe main.py preprocess
D:\SoftWare\Anaconda\envs\ml\python.exe main.py cv
D:\SoftWare\Anaconda\envs\ml\python.exe main.py baseline
D:\SoftWare\Anaconda\envs\ml\python.exe main.py evaluate
```

These install project dependencies into `ml`, run tests, generate EDA, prepare inputs, establish folds, train baselines, and evaluate OOF predictions. There is no separate build step.

Additional experiments use `D:\SoftWare\Anaconda\envs\ml\python.exe main.py features`, `targetstats`, `explore`, or `ensemble`. Run `D:\SoftWare\Anaconda\envs\ml\python.exe main.py tune --trials 16` for Optuna; `--trials` specifies the target total, including pruned trials. Check each command's `--help` before expensive runs.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, and `PascalCase` estimator classes. Follow existing pandas/scikit-learn patterns. No formatter or linter is configured. Manage project dependencies in `requirements.txt`; install them into the shared Conda `ml` environment with `D:\SoftWare\Anaconda\envs\ml\python.exe -m pip install -r requirements.txt`.

## Testing & Experiment Integrity

Add focused tests for changed behavior, especially leakage prevention, unknown categories, probability bounds, and ID alignment. No numerical coverage threshold is configured.

Preserve fixed folds and seeds for comparisons. Fit preprocessing within each training fold; training target encodings require inner cross-fitting. Record paired ROC-AUC differences, fold counts, and confirmation results. Verify saved Pipelines reload and reproduce predictions. Use fresh output directories when changing experiment inputs or protocols.

## Commit & Pull Request Guidelines

The repository has no commits yet. Use concise imperative subjects, such as `Add cross-fitted route statistics`. Keep commits scoped; exclude credentials, competition data, and generated artifacts.

PRs should explain the change, list validation commands, and report experiment parameters, seeds, paired scores, and runtime when relevant. Link related issues and include charts when visual results change.

## Adoption & Submission Policy

Retain the single-LightGBM approach by default. Fusion is experimental and requires explicit authorization and `--adopt-blend` for adoption. Submit predictions to Kaggle only when explicitly requested.
