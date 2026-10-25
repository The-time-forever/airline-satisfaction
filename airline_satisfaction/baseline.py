"""Train fixed-parameter baselines on the project's saved folds."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .cross_validation import iter_preprocessed_folds, validate_folds
from .preprocessing import ROOT, TARGET, CATEGORICAL, validate_inputs


def model_specs(seed=42, threads=4):
    return {
        'constant': ('tree', lambda: DummyClassifier(strategy='prior')),
        'logistic_regression': ('linear', lambda: LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000, tol=1e-5, random_state=seed)),
        'lightgbm': ('tree', lambda: LGBMClassifier(
            objective='binary', n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            colsample_bytree=0.9, subsample=0.9, subsample_freq=1,
            reg_lambda=1.0, random_state=seed, n_jobs=threads,
            deterministic=True, force_col_wise=True, verbosity=-1)),
        'xgboost': ('tree', lambda: XGBClassifier(
            objective='binary:logistic', eval_metric='auc', tree_method='hist',
            enable_categorical=True, device='cpu', n_estimators=500,
            learning_rate=0.05, max_depth=6, min_child_weight=5,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=threads, verbosity=0)),
        'catboost': ('catboost', lambda: CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='AUC',
            cat_features=CATEGORICAL, random_seed=seed,
            thread_count=threads, task_type='CPU',
            allow_writing_files=False, verbose=False)),
    }


def positive_probabilities(model, X):
    column = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(column) != 1:
        raise ValueError('Positive class 1 is missing.')
    probabilities = np.asarray(model.predict_proba(X))[:, column[0]]
    if probabilities.shape != (X.shape[0],) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid probability shape or non-finite values.')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must lie in [0, 1].')
    return probabilities


def train_baselines(train, test, submission, folds, output_dir, seed=42, threads=4, models=None):
    validate_inputs(train, test, submission)
    validate_folds(train, folds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_test = test.drop(columns='id')
    y = train[TARGET].astype('int8')
    specs = model_specs(seed, threads)
    selected = list(specs) if models is None else list(models)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(specs):
        raise ValueError('Select unique, supported model names.')
    # Keep previously completed baselines when adding new model families.
    summary_path, scores_path = output_dir / 'summary.csv', output_dir / 'fold_scores.csv'
    summary = (pd.read_csv(summary_path).loc[lambda x: ~x['model'].isin(selected)].to_dict('records')
               if summary_path.exists() else [])
    fold_scores = (pd.read_csv(scores_path).loc[lambda x: ~x['model'].isin(selected)].to_dict('records')
                   if scores_path.exists() else [])
    fold_count = folds['fold'].nunique()
    for name in selected:
        branch, factory = specs[name]
        started = time.perf_counter()
        model_dir = output_dir / name
        model_dir.mkdir(exist_ok=True)
        oof = np.full(len(train), np.nan)
        test_probability = np.zeros(len(test), dtype=np.float64)
        scores = []
        print(f'Training {name}', flush=True)
        for data in iter_preprocessed_folds(train, folds, branch):
            fold_started = time.perf_counter()
            model = factory()
            # Treat convergence failure as an error instead of accepting an unfinished fit.
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(data['X_fit'], data['y_fit'])
            valid_probability = positive_probabilities(model, data['X_valid'])
            oof[data['valid_rows']] = valid_probability
            transformed_test = data['preprocessor'].transform(X_test)
            if branch == 'linear':
                transformed_test = transformed_test.astype('float32')
            test_probability += positive_probabilities(model, transformed_test) / fold_count
            auc = roc_auc_score(data['y_valid'], valid_probability)
            scores.append(auc)
            fold_scores.append({
                'model': name, 'fold': data['fold'], 'roc_auc': auc,
                'log_loss': log_loss(data['y_valid'], valid_probability),
                'fit_rows': len(data['fit_rows']), 'valid_rows': len(data['valid_rows']),
                'seconds': time.perf_counter() - fold_started,
            })
            pipeline = Pipeline([('preprocessor', data['preprocessor']), ('model', model)])
            joblib.dump(pipeline, model_dir / f'fold_{data["fold"]}.joblib', compress=3)
            print(f'  fold {data["fold"]}: ROC-AUC={auc:.8f}', flush=True)
        if not np.isfinite(oof).all():
            raise ValueError('OOF predictions do not cover every sample.')
        pd.DataFrame({'id': train['id'], 'fold': folds['fold'].to_numpy(),
                      TARGET: y, 'probability': oof}).to_csv(model_dir / 'oof.csv', index=False)
        predictions = submission.copy()
        predictions[TARGET] = test_probability
        predictions.to_csv(model_dir / 'test_predictions.csv', index=False)
        summary.append({
            'model': name, 'mean_roc_auc': float(np.mean(scores)),
            'std_roc_auc': float(np.std(scores)), 'oof_roc_auc': roc_auc_score(y, oof),
            'oof_log_loss': log_loss(y, oof), 'seconds': time.perf_counter() - started,
        })
        # Persist completed models' results as training progresses.
        pd.DataFrame(fold_scores).to_csv(output_dir / 'fold_scores.csv', index=False)
        pd.DataFrame(summary).to_csv(output_dir / 'summary.csv', index=False)
    ranking = pd.DataFrame(summary).sort_values('mean_roc_auc', ascending=False)
    ranking.to_csv(output_dir / 'summary.csv', index=False)
    best = ranking.iloc[0]['model']
    best_predictions = pd.read_csv(output_dir / best / 'test_predictions.csv')
    if not best_predictions['id'].equals(submission['id']):
        raise ValueError('Test prediction id order differs from submission.')
    if best_predictions[TARGET].isna().any() or not best_predictions[TARGET].between(0, 1).all():
        raise ValueError('Invalid test probabilities.')
    best_predictions.to_csv(output_dir / 'submission.csv', index=False)
    return ranking, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--models', nargs='+', choices=list(model_specs()),
                        help='Train selected models and retain other completed results.')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    paths = [args.data_dir / name for name in ['train.csv', 'test.csv', 'sample_submission.csv']]
    frames = [pd.read_csv(path) for path in paths]
    folds = pd.read_csv(args.folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for path in paths + [args.folds]:
        with path.open('rb') as source:
            fingerprints[path.name] = hashlib.file_digest(source, 'sha256').hexdigest()
    config = {'seed': args.seed, 'threads': args.threads, 'metric': 'roc_auc',
              'selection': 'highest mean fold ROC-AUC', 'prediction': 'mean of fold probabilities',
              'source_sha256': fingerprints,
              'models': {name: {'branch': branch, 'parameters': factory().get_params()}
                         for name, (branch, factory) in model_specs(args.seed, args.threads).items()}}
    config_path = args.output_dir / 'config.json'
    if args.models:
        config['models'] = {k: config['models'][k] for k in args.models}
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous['source_sha256'] != fingerprints or previous['seed'] != args.seed:
            raise ValueError('Source data, folds or seed changed; use a new --output-dir.')
        if args.models:
            retained = {k: v for k, v in previous['models'].items() if k not in args.models}
            config['models'] = retained | {k: config['models'][k] for k in args.models}
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    ranking, best = train_baselines(*frames, folds, args.output_dir, args.seed, args.threads, args.models)
    print(ranking.to_string(index=False), flush=True)
    print(f'Best baseline: {best}; submission: {args.output_dir / "submission.csv"}', flush=True)
