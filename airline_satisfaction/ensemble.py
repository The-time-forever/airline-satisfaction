"""OOF blend search followed by frozen-weight, retrained-fold confirmation."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from .baseline import positive_probabilities
from .cross_validation import make_folds, validate_folds
from .evaluation import load_predictions, markdown_table
from .preprocessing import ROOT, TARGET, validate_inputs

NAMES = ['lightgbm_tuned', 'xgboost', 'catboost', 'logistic_regression']


def blend(probabilities, weights, method='probability'):
    p, w = np.asarray(probabilities, dtype=float), np.asarray(weights, dtype=float)
    if p.ndim != 2 or w.shape != (p.shape[1],):
        raise ValueError('Probability matrix and weight dimensions differ.')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Inputs must be finite probabilities in [0,1].')
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('Weights must be nonnegative and sum to one.')
    if method == 'probability':
        return p @ w
    if method == 'logit':
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        z = np.log(clipped / (1 - clipped)) @ w
        return 1 / (1 + np.exp(-z))
    if method == 'rank':
        # This is a batch-based ranking score, rather than a calibrated probability.
        return pd.DataFrame(p).rank(method='average', pct=True).to_numpy() @ w
    raise ValueError('Unknown blend method.')


def simplex_grid(units=10):
    if units < 1:
        raise ValueError('Grid units must be positive.')
    return [(i / units, j / units, (units - i - j) / units)
            for i in range(units + 1) for j in range(units - i + 1)]


def candidates():
    result = []
    for i, name in enumerate(NAMES):
        weights = [0.] * len(NAMES)
        weights[i] = 1.
        result.append({'name': 'single_' + name, 'method': 'probability', 'weights': weights})
    result += [
        {'name': 'equal_trees', 'method': 'probability', 'weights': [1/3, 1/3, 1/3, 0.]},
        {'name': 'equal_all', 'method': 'probability', 'weights': [.25] * 4},
        {'name': 'logit_trees', 'method': 'logit', 'weights': [1/3, 1/3, 1/3, 0.]},
        {'name': 'rank_trees', 'method': 'rank', 'weights': [1/3, 1/3, 1/3, 0.]},
    ]
    for weights in simplex_grid():
        if max(weights) == 1:
            continue  # The three single-model controls are already present.
        name = 'weighted_' + '_'.join(str(round(w * 100)) for w in weights)
        result.append({'name': name, 'method': 'probability', 'weights': [*weights, 0.]})
    return result


def fresh_pipeline(prototype):
    estimator = prototype.named_steps['model']
    # CatBoost copies cat_features in its constructor; sklearn's identity check rejects it.
    model = (CatBoostClassifier(**estimator.get_params())
             if isinstance(estimator, CatBoostClassifier) else clone(estimator))
    return Pipeline([('preprocessor', clone(prototype.named_steps['preprocessor'])), ('model', model)])


def evaluate_blend(y, p, folds, spec, reference_scores):
    oof = np.full(len(y), np.nan)
    records = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        probability = blend(p[mask], spec['weights'], spec['method'])
        oof[mask] = probability
        auc = roc_auc_score(y[mask], probability)
        records.append({'candidate': spec['name'], 'fold': int(fold), 'roc_auc': auc,
                        'delta_auc': auc - reference_scores[int(fold)],
                        'log_loss': log_loss(y[mask], probability)})
    frame = pd.DataFrame(records)
    return {
        'candidate': spec['name'], 'method': spec['method'],
        **{f'weight_{name}': weight for name, weight in zip(NAMES, spec['weights'])},
        'mean_roc_auc': frame.roc_auc.mean(), 'std_roc_auc': frame.roc_auc.std(ddof=0),
        'oof_roc_auc': roc_auc_score(y, oof), 'log_loss': log_loss(y, oof),
        'delta_auc': frame.delta_auc.mean(), 'win_folds': int(frame.delta_auc.gt(0).sum()),
    }, frame, oof


def confirm_sources(train, folds, source_dirs, needed, destination):
    X, y = train.drop(columns=['id', TARGET]), train[TARGET].astype(int)
    probabilities = np.zeros((len(train), len(NAMES)))
    for name in needed:
        column = NAMES.index(name)
        prototype = joblib.load(source_dirs[name] / 'fold_0.joblib')
        model_dir = destination / name
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold in sorted(folds.fold.unique()):
            fit_rows = np.flatnonzero(folds.fold.to_numpy() != fold)
            valid_rows = np.flatnonzero(folds.fold.to_numpy() == fold)
            pipeline = fresh_pipeline(prototype)
            pipeline.fit(X.iloc[fit_rows], y.iloc[fit_rows])
            probability = positive_probabilities(pipeline, X.iloc[valid_rows])
            probabilities[valid_rows, column] = probability
            joblib.dump(pipeline, model_dir / f'fold_{fold}.joblib', compress=3)
            print(f'confirmation {name} fold {fold}: AUC={roc_auc_score(y.iloc[valid_rows], probability):.8f}', flush=True)
        pd.DataFrame({'id': train.id, 'fold': folds.fold.to_numpy(), TARGET: y,
                      'probability': probabilities[:, column]}).to_csv(model_dir / 'oof.csv', index=False)
    return probabilities


def test_source_predictions(test, source_dirs, needed):
    result = np.zeros((len(test), len(NAMES)))
    X = test.drop(columns='id')
    for name in needed:
        models = sorted(source_dirs[name].glob('fold_*.joblib'))
        if len(models) != 5:
            raise ValueError('Expected five source fold pipelines.')
        for path in models:
            result[:, NAMES.index(name)] += positive_probabilities(joblib.load(path), X) / len(models)
        print(f'Test probabilities ready: {name}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'artifacts/baseline')
    parser.add_argument('--tuning-dir', type=Path, default=ROOT / 'artifacts/tuning')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/ensemble')
    parser.add_argument('--confirmation-seed', type=int, default=2028)
    parser.add_argument('--adopt-blend', action='store_true',
                        help='Explicitly adopt a blend that passes both CV comparisons.')
    args = parser.parse_args(argv)
    if args.confirmation_seed in {42, 2026, 2027}:
        parser.error('Use a confirmation seed distinct from previous selection splits.')
    train, test, submission = [pd.read_csv(args.data_dir / name) for name in
                               ['train.csv', 'test.csv', 'sample_submission.csv']]
    validate_inputs(train, test, submission)
    folds = pd.read_csv(args.folds)
    validate_folds(train, folds)
    tuning_config = json.loads((args.tuning_dir / 'config.json').read_text(encoding='utf-8'))
    baseline_config = json.loads((args.baseline_dir / 'config.json').read_text(encoding='utf-8'))
    for name, path in [('train.csv', args.data_dir / 'train.csv'), ('test.csv', args.data_dir / 'test.csv'),
                       ('sample_submission.csv', args.data_dir / 'sample_submission.csv'), ('folds.csv', args.folds)]:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if any(digest != cfg['source_sha256'][name] for cfg in [tuning_config, baseline_config]):
            raise ValueError(f'{name} differs from source model data.')
    selected = json.loads((args.tuning_dir / 'selection.json').read_text(encoding='utf-8'))
    source_dirs = {'lightgbm_tuned': Path(selected['pipeline_directory']),
                   **{name: args.baseline_dir / name for name in NAMES[1:]}}
    p = np.column_stack([load_predictions(train, folds, directory.parent, [directory.name])[directory.name]
                         for directory in source_dirs.values()])
    y = train[TARGET].astype(int).to_numpy()
    fold_numbers = folds.fold.to_numpy()
    reference_scores = {int(f): roc_auc_score(y[fold_numbers == f], p[fold_numbers == f, 0])
                        for f in sorted(folds.fold.unique())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual = pd.DataFrame(p - y[:, None], columns=NAMES).corr()
    residual.to_csv(args.output_dir / 'residual_correlation.csv')
    rows, all_folds, specifications = [], [], {}
    best_score, best_spec, best_oof = -np.inf, None, None
    for spec in candidates():
        row, per_fold, oof = evaluate_blend(y, p, fold_numbers, spec, reference_scores)
        rows.append(row)
        all_folds.append(per_fold)
        specifications[spec['name']] = spec
        if row['delta_auc'] > 0 and row['win_folds'] >= 3 and row['mean_roc_auc'] > best_score:
            best_score, best_spec, best_oof = row['mean_roc_auc'], spec, oof
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.output_dir / 'candidates.csv', index=False)
            print(f'Blend search: {len(rows)} candidates evaluated', flush=True)
    ranking = pd.DataFrame(rows).sort_values('mean_roc_auc', ascending=False)
    ranking.to_csv(args.output_dir / 'candidates.csv', index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(args.output_dir / 'fold_scores.csv', index=False)
    reference_spec = specifications['single_lightgbm_tuned']
    chosen = reference_spec
    confirmation_rows = []
    if best_spec is not None:
        frozen = {'candidate': best_spec, 'source_directories': {k: str(v.resolve()) for k, v in source_dirs.items()},
                  'source_sha256': tuning_config['source_sha256'], 'confirmation_seed': args.confirmation_seed}
        (args.output_dir / 'frozen_blend.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
        pd.DataFrame({'id': train.id, 'fold': fold_numbers, TARGET: y,
                      'probability': best_oof}).to_csv(args.output_dir / 'candidate_oof.csv', index=False)
        print(f'Frozen best blend: {best_spec}', flush=True)
        confirmation = make_folds(train, n_splits=folds.fold.nunique(), seed=args.confirmation_seed)
        confirmation.to_csv(args.output_dir / 'confirmation_folds.csv', index=False)
        needed = [name for name, weight in zip(NAMES, best_spec['weights']) if weight > 0]
        if 'lightgbm_tuned' not in needed:
            needed.insert(0, 'lightgbm_tuned')
        fresh = confirm_sources(train, confirmation, source_dirs, needed, args.output_dir / 'confirmation')
        fresh_numbers = confirmation.fold.to_numpy()
        control = {int(f): roc_auc_score(y[fresh_numbers == f], fresh[fresh_numbers == f, 0])
                   for f in sorted(confirmation.fold.unique())}
        control_row, control_folds, _ = evaluate_blend(y, fresh, fresh_numbers, reference_spec, control)
        candidate_row, candidate_folds, fresh_oof = evaluate_blend(y, fresh, fresh_numbers, best_spec, control)
        confirmation_rows = [control_row, candidate_row]
        pd.DataFrame(confirmation_rows).to_csv(args.output_dir / 'confirmation.csv', index=False)
        pd.concat([control_folds, candidate_folds]).to_csv(args.output_dir / 'confirmation_scores.csv', index=False)
        pd.DataFrame({'id': train.id, 'fold': fresh_numbers, TARGET: y,
                      'probability': fresh_oof}).to_csv(args.output_dir / 'confirmation_oof.csv', index=False)
        if args.adopt_blend and candidate_row['delta_auc'] > 0 and candidate_row['win_folds'] >= 3:
            chosen = best_spec
    needed = [name for name, weight in zip(NAMES, chosen['weights']) if weight > 0]
    test_p = test_source_predictions(test, source_dirs, needed)
    submission[TARGET] = blend(test_p, chosen['weights'], chosen['method'])
    if not submission[TARGET].between(0, 1).all():
        raise ValueError('Invalid final blend scores.')
    submission.to_csv(args.output_dir / 'submission.csv', index=False)
    result = {'selected': chosen, 'model_order': NAMES,
              'source_directories': {k: str(v.resolve()) for k, v in source_dirs.items()},
              'confirmation_seed': args.confirmation_seed, 'confirmation': confirmation_rows}
    (args.output_dir / 'selection.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    write_report(ranking, confirmation_rows, chosen, residual, args.output_dir, args.confirmation_seed)
    print(ranking.head(10).to_string(index=False), flush=True)
    print(f'Selected: {chosen}; report: {args.output_dir / "REPORT.md"}', flush=True)


def write_report(ranking, confirmation, chosen, residual, destination, seed):
    import matplotlib.pyplot as plt
    top = ranking.head(12)
    representative_names = ['single_lightgbm_tuned', 'equal_trees', 'equal_all',
                            'logit_trees', 'rank_trees', ranking.iloc[0]['candidate']]
    representatives = ranking[ranking.candidate.isin(representative_names)]
    weights = pd.DataFrame({'model': NAMES, 'weight': chosen['weights']})
    sections = ['# 模型融合实验',
        '基模型为调优 LightGBM、XGBoost、CatBoost 和逻辑回归。校验原始文件摘要、OOF 行序、标签和折分配后，使用相同五折比较融合方案。',
        '## 方法',
        '比较单模型、三树等权平均、四模型等权平均、三树 Logit 等权平均、三树排名平均，以及步长 0.1 的三树非负权重网格。权重和为 1，网格角点由单模型对照覆盖。',
        '概率平均直接加权概率；Logit 平均先将概率裁剪至 [1e-6,1-1e-6] 后取 Logit，平均后应用 Sigmoid；排名平均在同一预测批次内按各模型分数的平均秩百分位加权，输出排序分数。原 OOF 和复核 OOF 均按验证折分别计算秩；测试预测整批计算秩。',
        '## 融合方法比较', markdown_table(representatives[['candidate', 'method', 'mean_roc_auc', 'delta_auc', 'win_folds', 'log_loss']]),
        '## 原划分筛选结果', markdown_table(top[['candidate', 'method', 'mean_roc_auc', 'std_roc_auc', 'delta_auc', 'win_folds', 'oof_roc_auc', 'log_loss']]),
        f'共评估 {len(ranking)} 个方案。完整参数和各折分数见 candidates.csv 与 fold_scores.csv。原划分分数用于选择方案和权重。',
        '## 冻结后新划分复核']
    if confirmation:
        sections += [markdown_table(pd.DataFrame(confirmation)[['candidate', 'mean_roc_auc', 'std_roc_auc', 'delta_auc', 'win_folds', 'oof_roc_auc', 'log_loss']]),
            f'筛选结束后先将方案和权重写入 frozen_blend.json，再在种子 {seed} 的新 5 折中重新训练所需基模型，每折预处理与模型均只拟合该折训练部分。复核仅比较冻结候选与调优 LightGBM。',
            '两套划分均平均 AUC 提升且至少 3 折领先时标记为有效候选。默认保留调优单模型，显式指定 --adopt-blend 后才采用通过复核的融合。']
    else:
        sections += ['原划分没有同时取得平均 AUC 提升和至少 3 折领先的融合方案，保留调优 LightGBM。']
    sections += ['## 最终选择', f'选择 {chosen["name"]}，方法为 {chosen["method"]}。', markdown_table(weights),
        'submission.csv 使用原划分五折模型各自的测试概率平均，再应用所选融合规则。selection.json 记录模型顺序、权重及源模型目录。',
        '## 模型残差相关性', markdown_table(residual.reset_index(names='model')),
        '残差定义为 OOF 预测概率减实际标签；相关性结合融合验证分数评估互补性。',
        '![融合比较](comparison.png)',
        '## 复现', '```powershell\nuv run python main.py ensemble\n```']
    (destination / 'REPORT.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    plot = ranking.head(10).sort_values('delta_auc')
    axes[0].barh(plot.candidate, plot.delta_auc, color='#2563a6')
    axes[0].axvline(0, color='#444444', lw=1)
    axes[0].set(title='Top OOF blend candidates', xlabel='Mean fold ROC-AUC delta vs tuned LightGBM')
    axes[0].grid(axis='x', alpha=.2)
    im = axes[1].imshow(residual, vmin=0, vmax=1, cmap='Blues')
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            axes[1].text(j, i, f'{residual.iloc[i,j]:.3f}', ha='center', va='center',
                         color='white' if residual.iloc[i,j] > .6 else '#222222')
    labels = ['Tuned LGBM', 'XGBoost', 'CatBoost', 'Logistic']
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
                title='OOF residual correlations')
    fig.colorbar(im, ax=axes[1], shrink=.8)
    fig.savefig(destination / 'comparison.png', dpi=160)
    plt.close(fig)
