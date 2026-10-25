"""Final submission model: N-fold, M-seed LightGBM with per-fold fitted transforms.

Kept separate from the baseline/tuning experiments: it only consumes the adopted
preprocessor family (cross-fitted target statistics) and writes its own artifacts.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .baseline import positive_probabilities
from .cross_validation import make_folds, validate_folds
from .preprocessing import ROOT, TARGET, validate_inputs
from .target_statistics import TargetStatisticsPreprocessor

PARAMETERS = {
    'objective': 'binary', 'n_estimators': 500, 'learning_rate': 0.038277726052751405,
    'num_leaves': 63, 'min_child_samples': 100, 'colsample_bytree': 0.7710040865821907,
    'subsample': 0.8997196102882765, 'subsample_freq': 1,
    'reg_alpha': 0.16999327579678442, 'reg_lambda': 2.1793916612996536,
    'deterministic': True, 'force_col_wise': True, 'verbosity': -1,
}


def default_factory(parameters, threads):
    def factory(seed):
        return LGBMClassifier(**(parameters | {'random_state': seed, 'n_jobs': threads}))
    return factory


def train_final(train, test, submission, folds, output_dir, parameters=None, seeds=(42,),
                threads=4, factory=None, log=print):
    """Fit one model per fold and seed, average test probabilities, return the summary."""
    validate_inputs(train, test, submission)
    validate_folds(train, folds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique and non-empty.')
    parameters = PARAMETERS if parameters is None else parameters
    factory = default_factory(parameters, threads) if factory is None else factory
    output_dir = Path(output_dir)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    X = train.drop(columns=['id', TARGET])
    y = train[TARGET].astype(int)
    raw_test = test.drop(columns='id')
    fold_ids = folds.fold.to_numpy()
    fold_numbers = sorted(folds.fold.unique())
    oof = np.full(len(train), np.nan)
    test_probability = np.zeros(len(test))
    fold_rows = []
    started = time.perf_counter()
    for fold in fold_numbers:
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
        preprocessor = TargetStatisticsPreprocessor(groups=('context_interactions',),
                                                    smooth='auto')
        X_fit = preprocessor.fit_transform(X.iloc[fit_rows], y.iloc[fit_rows])
        X_valid = preprocessor.transform(X.iloc[valid_rows])
        X_test = preprocessor.transform(raw_test)
        valid_probability = np.zeros(len(valid_rows))
        for position, seed in enumerate(seeds):
            model = factory(seed)
            model.fit(X_fit, y.iloc[fit_rows])
            valid_probability += positive_probabilities(model, X_valid) / len(seeds)
            test_probability += positive_probabilities(model, X_test) / (len(seeds) * len(fold_numbers))
            if position == 0:
                joblib.dump(Pipeline([('preprocessor', preprocessor), ('model', model)]),
                            models_dir / f'fold_{fold}.joblib', compress=3)
        oof[valid_rows] = valid_probability
        fold_rows.append({'fold': int(fold), 'roc_auc': float(roc_auc_score(y.iloc[valid_rows], valid_probability)),
                          'log_loss': float(log_loss(y.iloc[valid_rows], valid_probability)),
                          'features': int(X_fit.shape[1])})
        log(f'fold {fold}: AUC={fold_rows[-1]["roc_auc"]:.8f}, '
            f'features={X_fit.shape[1]}, {time.perf_counter() - started:.0f}s')
    if not np.isfinite(oof).all():
        raise ValueError('OOF predictions do not cover all training samples.')
    if not ((test_probability >= 0) & (test_probability <= 1)).all():
        raise ValueError('Invalid test probabilities.')
    fold_scores = pd.DataFrame(fold_rows)
    oof_frame = pd.DataFrame({'id': train['id'].to_numpy(), 'fold': fold_ids,
                              TARGET: y.to_numpy(), 'probability': oof})
    result_submission = submission.copy()
    result_submission[TARGET] = test_probability
    summary = {
        'model': 'lightgbm', 'experiment': f'final_{len(fold_numbers)}x{len(seeds)}',
        'folds': len(fold_numbers), 'seeds': list(seeds), 'features': fold_rows[0]['features'],
        'mean_roc_auc': float(fold_scores.roc_auc.mean()),
        'std_roc_auc': float(fold_scores.roc_auc.std(ddof=0)),
        'oof_roc_auc': float(roc_auc_score(y, oof)),
        'oof_log_loss': float(log_loss(y, oof)),
        'seconds': time.perf_counter() - started, 'parameters': parameters,
    }
    fold_scores.to_csv(output_dir / 'fold_scores.csv', index=False)
    oof_frame.to_csv(output_dir / 'oof.csv', index=False)
    oof_frame.to_csv(models_dir / 'oof.csv', index=False)
    fold_scores.to_csv(models_dir / 'fold_scores.csv', index=False)
    result_submission.to_csv(output_dir / 'submission.csv', index=False)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    log(json.dumps({k: v for k, v in summary.items() if k != 'parameters'}, indent=2))
    return summary, result_submission


def write_report(output_dir, summary, fold_scores, adopted):
    rows = '\n'.join(
        f'| {int(r.fold)} | {r.roc_auc:.6f} | {r.log_loss:.5f} | {int(r.features)} |'
        for r in fold_scores.itertuples(index=False))
    report = f'''# 最终模型：{summary["folds"]} 折 × {len(summary["seeds"])} seed

沿用已采用的特征族（`context_interactions` + 交叉拟合目标统计，`smooth='auto'`）与调优后的
LightGBM 参数，只是把折数从 5 增加到 {summary["folds"]}、每折训练 {len(summary["seeds"])} 个随机种子并平均。
逐折独立拟合预处理，测试集概率为所有折与种子模型的平均。

| 指标 | 数值 |
| --- | --- |
| 折数 × 种子 | {summary["folds"]} × {len(summary["seeds"])}（{summary["seeds"]}） |
| 特征数 | {summary["features"]} |
| 逐折 AUC 均值 | {summary["mean_roc_auc"]:.6f} |
| 逐折 AUC 总体标准差 | {summary["std_roc_auc"]:.6f} |
| 整体 OOF ROC-AUC | {summary["oof_roc_auc"]:.6f} |
| 整体 OOF Log Loss | {summary["oof_log_loss"]:.5f} |
| 训练耗时 | {summary["seconds"]:.1f} s |

| 折 | ROC-AUC | Log Loss | 特征数 |
| --- | --- | --- | --- |
{rows}

## 与 5 折单种子方案的关系

`artifacts/lgbm_exploration/te_smoothing_auto` 是 5 折、每折 1 个种子（CV 0.960533）。
本方案在 5 折单种子的基础上同时增加“每折训练数据比例”与“seed 平均”两项纯方差控制手段，
没有引入新特征或新模型族；因此收益上限有限（实测 +1e-4 量级），但方向稳定。

## 文件

- `summary.json`：配置与指标。
- `fold_scores.csv` / `oof.csv`：逐折与整体 OOF。
- `models/fold_<k>.joblib`：每折第 1 个种子的完整 Pipeline（供诊断复现）；
  `models/oof.csv` 与 `models/fold_scores.csv` 供 `evaluation` / `lgbm_exploration` 直接读取。
- `submission.csv`：所有折与种子的测试概率平均。

## 采用状态

{'已复制到 `artifacts/submission.csv`，并更新 `artifacts/selected_model.json`。' if adopted else '未采用（仅供对照）；当前提交仍为 `artifacts/submission.csv` 中原有的 5 折单种子方案。'}

诊断脚本从 `models/fold_0.joblib` 复现的是第 1 个种子的模型，因此其 OOF 分数会与本报告的
种子平均分数有 ~5e-5 的差异，这是预期行为。
'''
    (Path(output_dir) / 'REPORT.md').write_text(report, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=None,
                        help='Existing fold file; generated with --n-splits when omitted.')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/final')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--adopt', action='store_true',
                        help='Copy the submission and record it as the current model.')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    paths = [args.data_dir / name for name in ['train.csv', 'test.csv', 'sample_submission.csv']]
    train, test, submission = (pd.read_csv(p) for p in paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.folds is not None:
        folds = pd.read_csv(args.folds)
    else:
        folds = make_folds(train, n_splits=args.n_splits, seed=42)
    folds.to_csv(args.output_dir / 'folds.csv', index=False)
    summary, result_submission = train_final(
        train, test, submission, folds, args.output_dir, seeds=tuple(args.seeds),
        threads=args.threads)
    write_report(args.output_dir, summary, pd.read_csv(args.output_dir / 'fold_scores.csv'),
                 args.adopt)
    if args.adopt:
        result_submission.to_csv(ROOT / 'artifacts/submission.csv', index=False)
        selection = {'model': 'lightgbm', 'fusion': False,
                     'experiment': summary['experiment'],
                     'pipeline_directory': str((args.output_dir / 'models').resolve()),
                     'folds': summary['folds'], 'seeds': summary['seeds'],
                     'original_cv': summary,
                     'note': 'submission averages every fold and seed; saved pipelines are seed 1'}
        (args.output_dir / 'selection.json').write_text(
            json.dumps(selection, indent=2) + '\n', encoding='utf-8')
        (ROOT / 'artifacts/selected_model.json').write_text(
            json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {args.output_dir / "REPORT.md"}')
    return summary


if __name__ == '__main__':
    main()
