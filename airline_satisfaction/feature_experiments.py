"""Ablation experiments on fixed folds and paired confirmation on a new split."""
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
from .features import FeaturePreprocessor, GROUPS, DESCRIPTIONS
from .preprocessing import ROOT, TARGET, validate_inputs


def run_experiment(train, folds, groups, parameters, destination, baseline_scores, preprocessor_factory=None):
    destination.mkdir(parents=True, exist_ok=True)
    X = train.drop(columns=['id', TARGET])
    y = train[TARGET].astype(int)
    oof = np.full(len(train), np.nan)
    fold_rows = []
    started = time.perf_counter()
    for fold in sorted(folds.fold.unique()):
        fit_rows = np.flatnonzero(folds.fold.to_numpy() != fold)
        valid_rows = np.flatnonzero(folds.fold.to_numpy() == fold)
        preprocessor = (FeaturePreprocessor(groups=tuple(groups)) if preprocessor_factory is None
                        else preprocessor_factory())
        fit = preprocessor.fit_transform(X.iloc[fit_rows], y.iloc[fit_rows])
        valid = preprocessor.transform(X.iloc[valid_rows])
        model = LGBMClassifier(**parameters)
        model.fit(fit, y.iloc[fit_rows])
        probability = positive_probabilities(model, valid)
        oof[valid_rows] = probability
        auc = roc_auc_score(y.iloc[valid_rows], probability)
        row = {'fold': int(fold), 'roc_auc': auc,
               'delta_auc': auc - baseline_scores.loc[fold] if baseline_scores is not None else 0.,
               'log_loss': log_loss(y.iloc[valid_rows], probability), 'features': fit.shape[1]}
        fold_rows.append(row)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        joblib.dump(pipeline, destination / f'fold_{fold}.joblib', compress=3)
        print(f'{destination.name} fold {fold}: AUC={auc:.8f}, delta={row["delta_auc"]:+.8f}, '
              f'features={fit.shape[1]}', flush=True)
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF predictions.')
    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(destination / 'fold_scores.csv', index=False)
    pd.DataFrame({'id': train.id, 'fold': folds.fold.to_numpy(), TARGET: y,
                  'probability': oof}).to_csv(destination / 'oof.csv', index=False)
    result = {'experiment': destination.name, 'groups': '+'.join(groups) or 'original',
              'features': int(per_fold.features.iloc[0]),
              'mean_roc_auc': float(per_fold.roc_auc.mean()),
              'std_roc_auc': float(per_fold.roc_auc.std(ddof=0)),
              'oof_roc_auc': roc_auc_score(y, oof), 'log_loss': log_loss(y, oof),
              'delta_auc': float(per_fold.delta_auc.mean()),
              'win_folds': int(per_fold.delta_auc.gt(0).sum()),
              'seconds': time.perf_counter() - started}
    (destination / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'artifacts/baseline')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/features')
    parser.add_argument('--confirmation-seed', type=int, default=2026)
    args = parser.parse_args(argv)
    train, test, submission = [pd.read_csv(args.data_dir / name) for name in
                               ['train.csv', 'test.csv', 'sample_submission.csv']]
    validate_inputs(train, test, submission)
    folds = pd.read_csv(args.folds)
    validate_folds(train, folds)
    config = json.loads((args.baseline_dir / 'config.json').read_text(encoding='utf-8'))
    for name, path in [('train.csv', args.data_dir / 'train.csv'), ('test.csv', args.data_dir / 'test.csv'),
                       ('folds.csv', args.folds)]:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != config['source_sha256'][name]:
            raise ValueError(f'{name} differs from the original baseline.')
    if args.confirmation_seed == config['seed']:
        parser.error('--confirmation-seed must differ from the original split seed')
    parameters = config['models']['lightgbm']['parameters']
    baseline_folds = pd.read_csv(args.baseline_dir / 'fold_scores.csv')
    baseline_scores = baseline_folds[baseline_folds.model.eq('lightgbm')].set_index('fold').roc_auc
    if set(baseline_scores.index) != set(folds.fold.unique()) or baseline_scores.index.duplicated().any():
        raise ValueError('Baseline folds do not match experiment folds.')
    baseline = pd.read_csv(args.baseline_dir / 'summary.csv').set_index('model').loc['lightgbm']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {'source_sha256': config['source_sha256'], 'parameters': parameters,
                         'groups': DESCRIPTIONS, 'confirmation_seed': args.confirmation_seed,
                         'selection_rule': 'positive mean AUC delta and >=3 winning folds on both splits'}
    (args.output_dir / 'config.json').write_text(json.dumps(experiment_config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    rows = [{'experiment': 'baseline', 'groups': 'original', 'features': 22,
             'mean_roc_auc': baseline.mean_roc_auc, 'std_roc_auc': baseline.std_roc_auc,
             'oof_roc_auc': baseline.oof_roc_auc, 'log_loss': baseline.oof_log_loss,
             'delta_auc': 0., 'win_folds': 0, 'seconds': baseline.seconds}]
    specifications = {}

    def trial(name, groups):
        specifications[name] = tuple(groups)
        rows.append(run_experiment(train, folds, groups, parameters,
                                   args.output_dir / name, baseline_scores))
        pd.DataFrame(rows).to_csv(args.output_dir / 'summary.csv', index=False)

    for group in GROUPS:
        trial(group, (group,))
    positive_groups = tuple(row['experiment'] for row in rows[1:]
                            if row['delta_auc'] > 0 and row['win_folds'] >= 3)
    trial('all_features', GROUPS)
    if len(positive_groups) > 1 and set(positive_groups) != set(GROUPS):
        trial('positive_combination', positive_groups)
    candidates = [row for row in rows[1:] if row['delta_auc'] > 0 and row['win_folds'] >= 3]
    chosen = 'baseline'
    confirmation_rows = []
    if candidates:
        best = max(candidates, key=lambda x: x['mean_roc_auc'])['experiment']
        confirmation = make_folds(train, n_splits=folds.fold.nunique(), seed=args.confirmation_seed)
        confirmation.to_csv(args.output_dir / 'confirmation_folds.csv', index=False)
        original = run_experiment(train, confirmation, (), parameters,
                                  args.output_dir / 'confirmation_baseline', None)
        fresh_scores = pd.read_csv(args.output_dir / 'confirmation_baseline/fold_scores.csv').set_index('fold').roc_auc
        candidate = run_experiment(train, confirmation, specifications[best], parameters,
                                   args.output_dir / 'confirmation_candidate', fresh_scores)
        original['delta_auc'] = 0.
        original['win_folds'] = 0
        confirmation_rows = [original, candidate]
        pd.DataFrame(confirmation_rows).to_csv(args.output_dir / 'confirmation.csv', index=False)
        if candidate['delta_auc'] > 0 and candidate['win_folds'] >= 3:
            chosen = best
    ranking = pd.DataFrame(rows).sort_values('mean_roc_auc', ascending=False)
    ranking.to_csv(args.output_dir / 'summary.csv', index=False)
    selected_dir = args.baseline_dir / 'lightgbm' if chosen == 'baseline' else args.output_dir / chosen
    test_probability = np.zeros(len(test))
    raw_test = test.drop(columns='id')
    for fold in sorted(folds.fold.unique()):
        pipeline = joblib.load(selected_dir / f'fold_{fold}.joblib')
        test_probability += positive_probabilities(pipeline, raw_test) / folds.fold.nunique()
    result_submission = submission.copy()
    result_submission[TARGET] = test_probability
    if not result_submission[TARGET].between(0, 1).all():
        raise ValueError('Invalid selected-model predictions.')
    result_submission.to_csv(args.output_dir / 'submission.csv', index=False)
    selection = {'selected': chosen, 'groups': specifications.get(chosen, ()),
                 'pipeline_directory': str(selected_dir.resolve()),
                 'confirmation': confirmation_rows}
    (args.output_dir / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    write_report(ranking, confirmation_rows, chosen, args.output_dir)
    print(ranking.to_string(index=False), flush=True)
    print(f'Selected: {chosen}; report: {args.output_dir / "REPORT.md"}', flush=True)


def write_report(ranking, confirmation_rows, chosen, destination):
    from .evaluation import markdown_table
    import matplotlib.pyplot as plt
    descriptions = pd.DataFrame([{'特征组': k, '内容': v} for k, v in DESCRIPTIONS.items()])
    sections = ['# 特征工程实验',
        '使用 LightGBM 原始基线参数和相同 5 折分配，逐组新增特征；所有填充值与类别词表仅在每折训练部分拟合。评分 0 保留，新增统计全部按单条样本计算。',
        markdown_table(descriptions),
        '## 原始划分消融实验',
        markdown_table(ranking[['experiment', 'features', 'mean_roc_auc', 'std_roc_auc', 'delta_auc', 'win_folds', 'oof_roc_auc', 'log_loss']]),
        'delta_auc 为与原始特征在同折上的 AUC 差值均值，win_folds 为优于基线的折数。先测试四组特征及全部组合，再测试至少两组单独有效特征的组合（如有）。',
        '## 新划分复核']
    if confirmation_rows:
        sections += [markdown_table(pd.DataFrame(confirmation_rows)[['experiment', 'mean_roc_auc', 'std_roc_auc', 'delta_auc', 'win_folds', 'oof_roc_auc']]),
            '原划分中平均 AUC 提升且至少 3 折领先的最佳候选，使用另一套固定种子的 5 折与原始特征配对复核。该划分仅复核此候选。']
    else:
        sections += ['原划分没有同时满足平均 AUC 提升且至少 3 折领先的候选，保留原始特征。']
    sections += ['## 选择结果', f'按两套划分均平均 AUC 提升且至少 3 折领先的规则，选择 {chosen}。selection.json 记录所用特征组和五折 Pipeline 位置；submission.csv 为原始划分五折模型的测试概率平均。',
                 '实验分数用于本轮候选筛选和复核。后续继续调参时保留本轮结果作为比较依据。',
                 '![各组特征的 AUC 变化](ablation.png)',
                 '## 复现', '```powershell\nuv run python main.py features\n```']
    (destination / 'REPORT.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    plot = ranking[ranking.experiment.ne('baseline')].sort_values('delta_auc')
    ax.barh(plot.experiment, plot.delta_auc, color=['#2563a6' if v > 0 else '#cb7527' for v in plot.delta_auc])
    ax.axvline(0, color='#444444', lw=1)
    ax.set(title='Feature ablation: mean paired ROC-AUC change vs baseline', xlabel='Mean fold ROC-AUC delta')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(destination / 'ablation.png', dpi=160)
    plt.close(fig)
