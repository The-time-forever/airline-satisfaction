"""Cross-fitted target statistics motivated by public S6E10 experiments."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.validation import check_is_fitted
import joblib

from .features import FeaturePreprocessor
from .preprocessing import ROOT, TARGET, validate_inputs
from .cross_validation import validate_folds, make_folds
from .evaluation import load_predictions, markdown_table
from .feature_experiments import run_experiment
from .baseline import positive_probabilities

KEYS = {
    'TE_distance': ['Flight Distance'], 'TE_age': ['Age'],
    'TE_distance_age': ['Flight Distance', 'Age'],
    'TE_distance_class': ['Flight Distance', 'Class'],
    'TE_distance_travel': ['Flight Distance', 'Type of Travel'],
    'TE_segment': ['Age', 'Class', 'Type of Travel', 'Customer Type'],
}


def category_keys(X):
    result = {}
    for name, columns in KEYS.items():
        value = X[columns[0]].astype('string').fillna('__MISSING__')
        for column in columns[1:]:
            value = value + '|' + X[column].astype('string').fillna('__MISSING__')
        result[name] = value
    return pd.DataFrame(result, index=X.index)


class TargetStatisticsPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, groups=('context_interactions',), smooth=20.0, inner_splits=5, seed=42):
        self.groups = groups
        self.smooth = smooth
        self.inner_splits = inner_splits
        self.seed = seed

    def _initialize(self, X):
        self.base_ = FeaturePreprocessor(groups=self.groups).fit(X)
        self.encoder_ = TargetEncoder(target_type='binary', smooth=self.smooth,
            cv=StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.seed))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self.base_.transform(X)

    def _combine(self, base, encoded):
        for i, name in enumerate(KEYS):
            base[name] = encoded[:, i].astype('float32')
        return base

    def fit(self, X, y):
        self._initialize(X)
        self.encoder_.fit(category_keys(X), np.asarray(y, dtype=int))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError('Target statistics require fitting labels.')
        base = self._initialize(X)
        # Internal cross-fitting is essential: no training row sees its own label.
        encoded = self.encoder_.fit_transform(category_keys(X), np.asarray(y, dtype=int))
        return self._combine(base, encoded)

    def transform(self, X):
        check_is_fitted(self, 'encoder_')
        return self._combine(self.base_.transform(X), self.encoder_.transform(category_keys(X)))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'encoder_')
        return np.append(self.base_.get_feature_names_out(), list(KEYS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/target_statistics')
    parser.add_argument('--confirmation-seed', type=int, default=2029)
    args = parser.parse_args(argv)
    if args.confirmation_seed in {42, 2026, 2027, 2028}:
        parser.error('Use a new confirmation seed.')
    train = pd.read_csv(args.data_dir / 'train.csv')
    folds = pd.read_csv(args.folds)
    validate_folds(train, folds)
    tuning = ROOT / 'artifacts/tuning'
    config = json.loads((tuning / 'config.json').read_text())
    for name, path in [('train.csv', args.data_dir / 'train.csv'), ('folds.csv', args.folds)]:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != config['source_sha256'][name]:
            raise ValueError('Source files differ from the single-model reference.')
    selected = json.loads((tuning / 'selection.json').read_text())
    reference_dir = Path(selected['pipeline_directory'])
    probabilities = load_predictions(train, folds, reference_dir.parent, [reference_dir.name])[reference_dir.name]
    from sklearn.metrics import roc_auc_score
    baseline_scores = pd.Series({int(f): roc_auc_score(train[TARGET][folds.fold.eq(f)], probabilities[folds.fold.eq(f)])
                                 for f in sorted(folds.fold.unique())})
    parameters = json.loads((tuning / 'best_params.json').read_text())
    result = run_experiment(train, folds, ('context_interactions',), parameters,
        args.output_dir, baseline_scores, preprocessor_factory=TargetStatisticsPreprocessor)
    test = pd.read_csv(args.data_dir / 'test.csv')
    submission = pd.read_csv(args.data_dir / 'sample_submission.csv')
    validate_inputs(train, test, submission)
    finalize_experiment(train, test, submission, folds, parameters, reference_dir,
                        args.output_dir, result, args.confirmation_seed)
    print(json.dumps(result,indent=2),flush=True)


def finalize_experiment(train, test, submission, folds, parameters, reference_dir, destination, result, seed=2029):
    """Confirm one frozen candidate, then write the single-model production selection."""
    from sklearn.metrics import roc_auc_score
    validate_inputs(train, test, submission)
    reference_prob = load_predictions(train, folds, reference_dir.parent, [reference_dir.name])[reference_dir.name]
    baseline_scores = pd.Series({int(f): roc_auc_score(train[TARGET][folds.fold.eq(f)], reference_prob[folds.fold.eq(f)])
                                 for f in sorted(folds.fold.unique())})
    fresh = make_folds(train, n_splits=folds.fold.nunique(), seed=seed)
    fresh.to_csv(destination / 'confirmation_folds.csv', index=False)
    control = run_experiment(train, fresh, ('context_interactions',), parameters,
                             destination / 'confirmation_reference', None)
    control_scores = pd.read_csv(destination / 'confirmation_reference/fold_scores.csv').set_index('fold').roc_auc
    candidate = run_experiment(train, fresh, ('context_interactions',), parameters,
        destination / 'confirmation_candidate', control_scores,
        preprocessor_factory=TargetStatisticsPreprocessor)
    pd.DataFrame([control,candidate]).to_csv(destination / 'confirmation.csv',index=False)
    adopted = (result['delta_auc'] > 0 and result['win_folds'] >= 3
               and candidate['delta_auc'] > 0 and candidate['win_folds'] >= 3)
    selected_dir = destination if adopted else reference_dir
    probability = np.zeros(len(test))
    for fold in sorted(folds.fold.unique()):
        pipeline = joblib.load(selected_dir / f'fold_{fold}.joblib')
        probability += positive_probabilities(pipeline,test.drop(columns='id')) / folds.fold.nunique()
    output = submission.copy()
    output[TARGET] = probability
    if not output[TARGET].between(0,1).all():
        raise ValueError('Invalid final probabilities.')
    output.to_csv(destination / 'submission.csv',index=False)
    selection = {'model':'lightgbm', 'target_statistics_adopted':bool(adopted),
                 'pipeline_directory':str(selected_dir.resolve()), 'original_cv':result,
                 'confirmation_seed':seed, 'confirmation_reference':control,
                 'confirmation_candidate':candidate, 'fusion':False}
    (destination / 'selection.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(destination / 'submission.csv', ROOT / 'artifacts/submission.csv')
    (ROOT / 'artifacts/selected_model.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8')
    report = ['# 目标统计特征验证',
        '动机来自公开 S6E10 对照实验；本地只增加六个平滑目标统计特征，模型参数、原特征和外层五折保持一致。',
        '每个外层训练折内，TargetEncoder.fit_transform 使用独立的五折交叉拟合。验证样本仅通过该外层训练折的完整映射转换。平滑参数固定为 20。',
        markdown_table(pd.DataFrame([{'feature': k, 'columns': ' × '.join(v)} for k,v in KEYS.items()])),
        markdown_table(pd.DataFrame([{'方案':'调优单模型','mean_auc':baseline_scores.mean()},
                                     {'方案':'增加目标统计','mean_auc':result['mean_roc_auc']} ])),
        f'原划分平均 AUC 变化 {result["delta_auc"]:+.8f}，{result["win_folds"]}/5 折领先；'
        f'整体 OOF AUC 为 {result["oof_roc_auc"]:.8f}。',
        f'## 新划分复核（种子 {seed}）',
        markdown_table(pd.DataFrame([control,candidate])[['experiment','mean_roc_auc','std_roc_auc','delta_auc','win_folds','oof_roc_auc']]),
        '## 当前采用方案',
        ('采用增加目标统计的 LightGBM。' if adopted else '保留原调优 LightGBM。') +
        '两套划分均要求平均 AUC 提升且至少三折领先。所有编码参数在对应训练部分拟合。',
        '最终预测为同一模型配置的原五折概率平均。artifacts/selected_model.json 记录当前方案，artifacts/submission.csv 为当前提交文件。融合实验保持停用。',
        '来源：[公开对照 Notebook](https://www.kaggle.com/code/sachith7/s6e10-what-each-step-was-worth)、'
        '[TargetEncoder 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)。',
        '复现：`python main.py targetstats`。']
    (destination / 'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(f'Target statistics adopted: {adopted}; current submission: {ROOT / "artifacts/submission.csv"}',flush=True)
    return selection
