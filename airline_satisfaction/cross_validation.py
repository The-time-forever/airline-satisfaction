"""Deterministic stratified folds with independently fitted preprocessing."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .preprocessing import ROOT, TARGET, TreePreprocessor, CatBoostPreprocessor, make_linear_preprocessor


def validate_training_data(train):
    if not {'id', TARGET}.issubset(train.columns):
        raise ValueError('Training data requires id and satisfaction.')
    if train['id'].isna().any() or train['id'].duplicated().any():
        raise ValueError('Training ids must be complete and unique.')
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {False, True}:
        raise ValueError('Expected complete binary False/True labels.')


def make_folds(train, n_splits=5, seed=42):
    validate_training_data(train)
    if n_splits < 2 or n_splits > train[TARGET].value_counts().min():
        raise ValueError('Each class must contain at least n_splits samples; n_splits >= 2.')
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = np.full(len(train), -1, dtype=np.int32)
    for fold, (_, valid_rows) in enumerate(splitter.split(np.zeros(len(train)), train[TARGET])):
        assignment[valid_rows] = fold
    return pd.DataFrame({'id': train['id'].to_numpy(), 'fold': assignment})


def validate_folds(train, assignments):
    validate_training_data(train)
    if list(assignments.columns) != ['id', 'fold']:
        raise ValueError('Fold file requires columns id,fold.')
    if not assignments['id'].reset_index(drop=True).equals(train['id'].reset_index(drop=True)):
        raise ValueError('Fold ids must match training ids in original row order.')
    folds = assignments['fold']
    if folds.isna().any() or not pd.api.types.is_integer_dtype(folds):
        raise ValueError('Fold numbers must be complete integers.')
    unique = np.sort(folds.unique())
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Fold numbers must be contiguous starting at zero.')
    labels = train[TARGET].to_numpy()
    for fold in unique:
        mask = folds.to_numpy() == fold
        if len(np.unique(labels[mask])) != 2 or len(np.unique(labels[~mask])) != 2:
            raise ValueError('Every training/validation fold must contain both classes.')


def iter_preprocessed_folds(train, assignments, branch='tree'):
    """Yield one fold at a time; fit all learned transforms on its training rows.

    Row arrays are positional indices for assigning out-of-fold predictions.
    Start each model from a new estimator within the caller's fold loop.
    """
    validate_folds(train, assignments)
    if branch not in {'tree', 'linear', 'catboost'}:
        raise ValueError('branch must be tree, linear or catboost.')
    X = train.drop(columns=['id', TARGET])
    y = train[TARGET].astype('int8')
    fold_numbers = assignments['fold'].to_numpy()
    for fold in sorted(assignments['fold'].unique()):
        fit_rows = np.flatnonzero(fold_numbers != fold)
        valid_rows = np.flatnonzero(fold_numbers == fold)
        preprocessor = (TreePreprocessor() if branch == 'tree'
                        else CatBoostPreprocessor() if branch == 'catboost'
                        else make_linear_preprocessor(X.columns))
        X_fit = preprocessor.fit_transform(X.iloc[fit_rows])
        X_valid = preprocessor.transform(X.iloc[valid_rows])
        if branch == 'linear':
            X_fit = X_fit.astype('float32')
            X_valid = X_valid.astype('float32')
        for value in [X_fit, X_valid]:
            numeric = (value.select_dtypes(include='number').to_numpy()
                       if branch in {'tree', 'catboost'} else value.data)
            if not np.isfinite(numeric).all():
                raise ValueError(f'Non-finite numerical features in fold {fold}.')
        yield {
            'fold': int(fold), 'fit_rows': fit_rows, 'valid_rows': valid_rows,
            'preprocessor': preprocessor, 'X_fit': X_fit, 'X_valid': X_valid,
            'y_fit': y.iloc[fit_rows], 'y_valid': y.iloc[valid_rows],
        }


def cross_validate_model(estimator, train, assignments, branch='tree'):
    """Clone and fit an estimator per fold; return scores and aligned OOF probabilities."""
    oof = np.full(len(train), np.nan, dtype=np.float64)
    scores = []
    for data in iter_preprocessed_folds(train, assignments, branch):
        model = clone(estimator)
        model.fit(data['X_fit'], data['y_fit'])
        positive_column = np.flatnonzero(model.classes_ == 1)
        if len(positive_column) != 1:
            raise ValueError('Estimator must expose positive class 1.')
        probabilities = np.asarray(model.predict_proba(data['X_valid']))[:, positive_column[0]]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError('Estimator returned invalid probabilities.')
        oof[data['valid_rows']] = probabilities
        scores.append({'fold': data['fold'], 'roc_auc': roc_auc_score(data['y_valid'], probabilities)})
    if not np.isfinite(oof).all():
        raise ValueError('OOF predictions do not cover all training samples.')
    fold_scores = pd.DataFrame(scores)
    return {
        'fold_scores': fold_scores,
        'mean_roc_auc': float(fold_scores['roc_auc'].mean()),
        'std_roc_auc': float(fold_scores['roc_auc'].std(ddof=0)),
        'oof_roc_auc': float(roc_auc_score(train[TARGET].astype(int), oof)),
        'oof': pd.DataFrame({'id': train['id'].to_numpy(),
                             'fold': assignments['fold'].to_numpy(),
                             TARGET: oof}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/train.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/cv')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    train = pd.read_csv(args.train)
    assignments = make_folds(train, args.n_splits, args.seed)
    validate_folds(train, assignments)
    summary = []
    preprocessors = {'tree': [], 'linear': []}
    for branch in preprocessors:
        for data in iter_preprocessed_folds(train, assignments, branch):
            preprocessors[branch].append(data['preprocessor'])
            row = {'branch': branch, 'fold': data['fold'],
                   'fit_rows': len(data['fit_rows']), 'valid_rows': len(data['valid_rows']),
                   'fit_positive_rate': float(data['y_fit'].mean()),
                   'valid_positive_rate': float(data['y_valid'].mean()),
                   'features': data['X_fit'].shape[1]}
            summary.append(row)
            print(f'{branch} fold {data["fold"]}: fit={row["fit_rows"]}, '
                  f'valid={row["valid_rows"]}, positive={row["valid_positive_rate"]:.6%}, '
                  f'features={row["features"]}', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output_dir / 'folds.csv', index=False)
    with args.train.open('rb') as source:
        source_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    joblib.dump({
        'metric': 'roc_auc', 'n_splits': args.n_splits, 'seed': args.seed,
        'train_sha256': source_hash,
        'feature_columns': train.columns.drop(['id', TARGET]).tolist(),
        'assignments': assignments, 'summary': pd.DataFrame(summary),
        'preprocessors': preprocessors,
    }, args.output_dir / 'preprocessors.joblib', compress=3)
    print(f'Saved folds.csv and preprocessors.joblib to {args.output_dir}')
