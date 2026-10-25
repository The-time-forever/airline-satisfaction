"""Leakage-safe preprocessing for Kaggle Playground Series S6E10."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

TARGET = 'satisfaction'
CATEGORICAL = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
DELAY = 'Arrival Delay in Minutes'
INDICATOR = 'Arrival Delay Missing'
ROOT = Path(__file__).resolve().parents[1]


class TreePreprocessor(TransformerMixin, BaseEstimator):
    """Training-only medians and category vocabularies; preserve rating zeroes."""

    def fit(self, X, y=None):
        if TARGET in X or 'id' in X:
            raise ValueError('Remove id and target before fitting.')
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.numeric_ = [c for c in X if c not in CATEGORICAL]
        if not set(CATEGORICAL + [DELAY]).issubset(X.columns):
            raise ValueError('Required competition features are missing.')
        self.medians_ = X[self.numeric_].median()
        if self.medians_.isna().any():
            raise ValueError('A numeric feature contains no observed training values.')
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL}
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Feature columns or order differ from fitted schema.')
        result = X.copy()
        result[INDICATOR] = result[DELAY].isna().astype('int8')
        result[self.numeric_] = result[self.numeric_].fillna(self.medians_).astype('float32')
        for c in CATEGORICAL:
            # Unseen values become missing categories, supported by LightGBM.
            values = result[c].where(result[c].isin(self.categories_[c]))
            result[c] = pd.Categorical(values, categories=self.categories_[c])
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return np.append(self.feature_names_in_, INDICATOR)


class CatBoostPreprocessor(TreePreprocessor):
    """Expose categorical values as strings, including unknown/missing levels."""

    def transform(self, X):
        result = super().transform(X)
        for c in CATEGORICAL:
            values = result[c].astype(object)
            result[c] = values.where(values.notna(), '__MISSING__').astype(str)
        return result


def make_linear_preprocessor(features):
    numeric = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True,
                                     keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', dtype=np.float32)),
        ]), CATEGORICAL),
    ], sparse_threshold=1.0)


def validate_inputs(train, test, submission):
    if TARGET not in train or 'id' not in train:
        raise ValueError('Training data requires id and satisfaction.')
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {False, True}:
        raise ValueError('Expected complete binary False/True labels.')
    if not train.drop(columns=TARGET).columns.equals(test.columns):
        raise ValueError('Train/test feature schemas differ.')
    if not set(CATEGORICAL + [DELAY]).issubset(test.columns):
        raise ValueError('Required competition features are missing.')
    if train['id'].duplicated().any() or test['id'].duplicated().any():
        raise ValueError('Duplicate ids found.')
    if set(train['id']) & set(test['id']):
        raise ValueError('Train/test ids overlap.')
    if list(submission.columns) != ['id', TARGET] or not submission['id'].equals(test['id']):
        raise ValueError('Submission schema or id order differs from test.')


def prepare_data(train, test, submission, validation_size=0.2, seed=42):
    validate_inputs(train, test, submission)
    X = train.drop(columns=['id', TARGET])
    X_test = test.drop(columns='id')
    y = train[TARGET].astype('int8')
    fit_rows, valid_rows = train_test_split(
        np.arange(len(train)), test_size=validation_size,
        random_state=seed, stratify=y,
    )
    raw_fit, raw_valid = X.iloc[fit_rows], X.iloc[valid_rows]
    tree = TreePreprocessor().fit(raw_fit)
    linear = make_linear_preprocessor(X.columns)
    linear_fit = linear.fit_transform(raw_fit)
    data = {
        'metric': 'roc_auc', 'seed': seed, 'validation_size': validation_size,
        'feature_columns': list(X.columns), 'categorical_columns': CATEGORICAL.copy(),
        'fit_rows': fit_rows, 'valid_rows': valid_rows,
        'fit_ids': train['id'].iloc[fit_rows].copy(),
        'valid_ids': train['id'].iloc[valid_rows].copy(),
        'test_ids': test['id'].copy(),
        'y_fit': y.iloc[fit_rows].copy(), 'y_valid': y.iloc[valid_rows].copy(),
        'tree': {'preprocessor': tree, 'X_fit': tree.transform(raw_fit),
                 'X_valid': tree.transform(raw_valid), 'X_test': tree.transform(X_test)},
        'linear': {'preprocessor': linear,
                   'feature_names': linear.get_feature_names_out().tolist(),
                   'X_fit': linear_fit.astype('float32'),
                   'X_valid': linear.transform(raw_valid).astype('float32'),
                   'X_test': linear.transform(X_test).astype('float32')},
    }
    validate_prepared(data, len(train), len(test))
    return data


def validate_prepared(data, train_size, test_size):
    fit_rows, valid_rows = data['fit_rows'], data['valid_rows']
    if len(np.intersect1d(fit_rows, valid_rows)):
        raise ValueError('Training and validation rows overlap.')
    if not np.array_equal(np.sort(np.concatenate([fit_rows, valid_rows])), np.arange(train_size)):
        raise ValueError('Split does not cover training rows exactly once.')
    sizes = {'X_fit': len(fit_rows), 'X_valid': len(valid_rows), 'X_test': test_size}
    for branch in ['tree', 'linear']:
        width = data[branch]['X_fit'].shape[1]
        for key, size in sizes.items():
            value = data[branch][key]
            if value.shape != (size, width):
                raise ValueError(f'Unexpected shape for {branch}/{key}.')
            numeric = value.select_dtypes(include='number').to_numpy() if branch == 'tree' else value.data
            if not np.isfinite(numeric).all():
                raise ValueError(f'Non-finite numerical data in {branch}/{key}.')
        if branch == 'tree':
            for c in CATEGORICAL:
                expected = data['tree']['X_fit'][c].dtype
                for key in sizes:
                    if data['tree'][key][c].dtype != expected:
                        raise ValueError('Inconsistent categorical vocabularies.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/preprocessing.joblib')
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.validation_size < 1:
        parser.error('--validation-size must be between 0 and 1')
    frames = [pd.read_csv(args.data_dir / name) for name in
              ['train.csv', 'test.csv', 'sample_submission.csv']]
    prepared = prepare_data(*frames, validation_size=args.validation_size, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared, args.output, compress=3)
    print(f'Saved: {args.output}')
    print(f'Metric: {prepared["metric"]}; seed: {args.seed}')
    print(f'Positive rates: fit={prepared["y_fit"].mean():.6%}, valid={prepared["y_valid"].mean():.6%}')
    for branch in ['tree', 'linear']:
        print(branch, {k: prepared[branch][k].shape for k in ['X_fit', 'X_valid', 'X_test']})
