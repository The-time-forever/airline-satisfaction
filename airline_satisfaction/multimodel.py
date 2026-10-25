"""LightGBM + XGBoost + embedding-MLP blend on the project folds (experimental fusion).

Each member is trained per fold with transforms fitted on that fold only; the neural
member consumes the raw frame, so its errors stay decorrelated from the boosted trees.
The blend is a logit average with fixed weights; adoption requires both the original
and the confirmation split to improve while winning at least four of five folds.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .baseline import positive_probabilities
from .cross_validation import make_folds, validate_folds
from .ensemble import blend
from .evaluation import markdown_table, paired_auc_delta
from .neural import EmbeddingMLPClassifier
from .preprocessing import ROOT, TARGET, validate_inputs
from .target_statistics import TargetStatisticsPreprocessor

MEMBERS = ('lightgbm', 'xgboost', 'neural')
LIGHTGBM = {
    'objective': 'binary', 'n_estimators': 500, 'learning_rate': 0.038277726052751405,
    'num_leaves': 63, 'min_child_samples': 100, 'colsample_bytree': 0.7710040865821907,
    'subsample': 0.8997196102882765, 'subsample_freq': 1,
    'reg_alpha': 0.16999327579678442, 'reg_lambda': 2.1793916612996536,
    'deterministic': True, 'force_col_wise': True, 'verbosity': -1,
}
XGBOOST = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
    'enable_categorical': True, 'device': 'cpu', 'n_estimators': 1500,
    'learning_rate': 0.03, 'max_depth': 8, 'min_child_weight': 10, 'subsample': 0.9,
    'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'random_state': 42, 'verbosity': 0,
}
NEURAL = {'epochs': 12, 'seed': 42}


def train_member(name, train, test, folds, destination, threads=4, seeds=(42,), log=print):
    """Fit one member per fold and seed; return aligned OOF and test probabilities."""
    if name not in MEMBERS:
        raise ValueError(f'Unknown member {name}.')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique and non-empty.')
    X = train.drop(columns=['id', TARGET])
    y = train[TARGET].astype(int)
    raw_test = test.drop(columns='id')
    fold_ids = folds.fold.to_numpy()
    fold_numbers = sorted(folds.fold.unique())
    member_dir = Path(destination) / 'members' / name
    member_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(train), np.nan)
    test_probability = np.zeros(len(test))
    rows = []
    started = time.perf_counter()
    for fold in fold_numbers:
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
        if name == 'neural':
            X_fit, X_valid, X_test = X.iloc[fit_rows], X.iloc[valid_rows], raw_test
        else:
            preprocessor = TargetStatisticsPreprocessor(groups=('context_interactions',),
                                                        smooth='auto')
            X_fit = preprocessor.fit_transform(X.iloc[fit_rows], y.iloc[fit_rows])
            X_valid = preprocessor.transform(X.iloc[valid_rows])
            X_test = preprocessor.transform(raw_test)
        valid_probability = np.zeros(len(valid_rows))
        for position, seed in enumerate(seeds):
            if name == 'neural':
                estimator = EmbeddingMLPClassifier(**(NEURAL | {'seed': seed}))
            elif name == 'lightgbm':
                estimator = LGBMClassifier(**(LIGHTGBM | {'random_state': seed,
                                                          'n_jobs': threads}))
            else:
                estimator = XGBClassifier(**(XGBOOST | {'random_state': seed,
                                                        'n_jobs': threads}))
            estimator.fit(X_fit, y.iloc[fit_rows])
            valid_probability += positive_probabilities(estimator, X_valid) / len(seeds)
            test_probability += (positive_probabilities(estimator, X_test)
                                 / (len(seeds) * len(fold_numbers)))
            if position == 0:
                saved = (estimator if name == 'neural'
                         else Pipeline([('preprocessor', preprocessor), ('model', estimator)]))
                joblib.dump(saved, member_dir / f'fold_{fold}.joblib', compress=3)
        oof[valid_rows] = valid_probability
        rows.append({'fold': int(fold),
                     'roc_auc': float(roc_auc_score(y.iloc[valid_rows], valid_probability)),
                     'log_loss': float(log_loss(y.iloc[valid_rows], valid_probability))})
        log(f'  {name} fold {fold}: AUC={rows[-1]["roc_auc"]:.6f} '
            f'({time.perf_counter() - started:.0f}s)')
    if not np.isfinite(oof).all():
        raise ValueError(f'Incomplete OOF predictions for {name}.')
    fold_scores = pd.DataFrame(rows)
    pd.DataFrame({'id': train['id'].to_numpy(), 'fold': fold_ids,
                  TARGET: y.to_numpy(), 'probability': oof}).to_csv(
        member_dir / 'oof.csv', index=False)
    pd.DataFrame({'id': test['id'].to_numpy(), TARGET: test_probability}).to_csv(
        member_dir / 'test_predictions.csv', index=False)
    fold_scores.to_csv(member_dir / 'fold_scores.csv', index=False)
    return {'name': name, 'oof': oof, 'test': test_probability,
            'mean_roc_auc': float(fold_scores.roc_auc.mean()),
            'fold_scores': fold_scores, 'seconds': time.perf_counter() - started}


def summarise(name, y, fold_ids, reference, candidate):
    """Fold means plus a paired DeLong test of the candidate against the reference."""
    def fold_auc(probability):
        return np.array([roc_auc_score(y[fold_ids == fold], probability[fold_ids == fold])
                         for fold in np.unique(fold_ids)])

    base, promoted = fold_auc(reference), fold_auc(candidate)
    delta, error = paired_auc_delta(y, reference, candidate)
    p_value = 2 * (1 - norm.cdf(abs(delta / error))) if error > 0 else 1.0
    return {'candidate': name, 'mean_roc_auc': float(promoted.mean()),
            'std_roc_auc': float(promoted.std(ddof=0)),
            'delta_auc': float((promoted - base).mean()),
            'win_folds': int((promoted > base).sum()), 'folds': int(len(promoted)),
            'oof_roc_auc': float(roc_auc_score(y, candidate)),
            'log_loss': float(log_loss(y, candidate)),
            'paired_delta': float(delta), 'paired_se': float(error),
            'paired_p': float(p_value)}


def run_split(train, test, folds, destination, weights, threads=4, seeds_by_member=None,
              log=print):
    """Train every member on one split and evaluate the fixed logit blend."""
    y = train[TARGET].astype(int).to_numpy()
    fold_ids = folds.fold.to_numpy()
    seeds_by_member = seeds_by_member or {}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    members = {name: train_member(name, train, test, folds, destination, threads,
                                  tuple(seeds_by_member.get(name, (42,))), log)
               for name in MEMBERS}
    reference = members['lightgbm']['oof']
    candidate = blend(np.column_stack([members[name]['oof'] for name in MEMBERS]),
                      weights, 'logit')
    test_blend = blend(np.column_stack([members[name]['test'] for name in MEMBERS]),
                       weights, 'logit')
    rows = [{'candidate': 'single_lightgbm', 'mean_roc_auc': members['lightgbm']['mean_roc_auc'],
             'delta_auc': 0.0, 'win_folds': 0, 'folds': len(np.unique(fold_ids)),
             'oof_roc_auc': float(roc_auc_score(y, reference)),
             'log_loss': float(log_loss(y, reference))}]
    rows += [{'candidate': name, 'mean_roc_auc': members[name]['mean_roc_auc'],
              'oof_roc_auc': float(roc_auc_score(y, members[name]['oof'])),
              'log_loss': float(log_loss(y, members[name]['oof']))}
             for name in MEMBERS if name != 'lightgbm']
    summary = summarise('blend_' + '_'.join(f'{w:g}' for w in weights), y, fold_ids,
                        reference, candidate)
    rows.append({key: summary[key] for key in
                 ('candidate', 'mean_roc_auc', 'delta_auc', 'win_folds', 'folds',
                  'oof_roc_auc', 'log_loss')})
    table = pd.DataFrame(rows)
    table.to_csv(destination / 'candidates.csv', index=False)
    pd.DataFrame({'id': train['id'].to_numpy(), 'fold': fold_ids,
                  TARGET: y, 'probability': candidate}).to_csv(
        destination / 'blend_oof.csv', index=False)
    pd.DataFrame({'id': test['id'].to_numpy(), TARGET: test_blend}).to_csv(
        destination / 'blend_test.csv', index=False)
    return {'members': {name: {'mean_roc_auc': members[name]['mean_roc_auc'],
                               'seconds': members[name]['seconds']} for name in MEMBERS},
            'table': table, 'blend': summary, 'test': test_blend, 'folds': folds}


def write_report(destination, weights, primary, confirmation, adopted, rule):
    table = primary['table'].copy()
    sections = ['# 三模型融合实验',
                'LightGBM（目标统计特征）、XGBoost（同一预处理）与 embedding MLP（原始特征，'
                '不经目标编码）逐折训练，按固定权重做 logit 平均。对照臂是同折的 LightGBM 单模型。',
                f'权重（{", ".join(MEMBERS)}）= {weights}；logit 空间平均，输出仍为概率。',
                '## 原划分', markdown_table(table),
                f'配对 DeLong：Δ = {primary["blend"]["paired_delta"]:+.6f}，'
                f'SE = {primary["blend"]["paired_se"]:.6f}，p = {primary["blend"]["paired_p"]:.3g}。',
                '## 新划分复核（种子 %d）' % confirmation['seed'] if confirmation else '未做复核。']
    if confirmation:
        sections += [markdown_table(confirmation['result']['table']),
                     f'配对 DeLong：Δ = {confirmation["result"]["blend"]["paired_delta"]:+.6f}，'
                     f'SE = {confirmation["result"]["blend"]["paired_se"]:.6f}，'
                     f'p = {confirmation["result"]["blend"]["paired_p"]:.3g}。']
    sections += ['## 采用规则', rule, '## 结论',
                 ('两套划分均通过，已采用该融合：`artifacts/submission.csv` 与 '
                  '`artifacts/selected_model.json` 已更新。' if adopted
                  else '未通过采用规则，实验保留在 `artifacts/multimodel/`，当前提交不变。'),
                 '## 文件',
                 '- `members/<name>/`：逐折 Pipeline（NN 保存估计器本身）、OOF、测试预测。',
                 '- `candidates.csv`、`summary.json`、`submission.csv`（本划分的融合结果）。']
    (Path(destination) / 'REPORT.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--weights', type=float, nargs='+', default=[0.8, 0.1, 0.1],
                        help=f'Weights for {", ".join(MEMBERS)}; must sum to one.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Seeds for the LightGBM member, averaged within each fold.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--confirm-seed', type=int, default=2026)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/multimodel')
    parser.add_argument('--skip-confirmation', action='store_true')
    parser.add_argument('--adopt', action='store_true')
    args = parser.parse_args(argv)
    weights = [float(w) for w in args.weights]
    if len(weights) != len(MEMBERS):
        parser.error(f'--weights needs exactly {len(MEMBERS)} values.')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.confirm_seed == 42:
        parser.error('Use a fresh confirmation seed.')
    paths = [args.data_dir / name for name in ['train.csv', 'test.csv', 'sample_submission.csv']]
    train, test, submission = (pd.read_csv(path) for path in paths)
    validate_inputs(train, test, submission)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_folds = (pd.read_csv(args.folds) if args.folds.exists()
                     else make_folds(train, n_splits=args.n_splits, seed=42))
    validate_folds(train, primary_folds)
    primary = run_split(train, test, primary_folds, args.output_dir / 'primary',
                        weights, args.threads, {'lightgbm': tuple(args.seeds)})
    confirmation = None
    if not args.skip_confirmation:
        fresh = make_folds(train, n_splits=primary_folds.fold.nunique(),
                           seed=args.confirm_seed)
        result = run_split(train, test, fresh, args.output_dir / f'confirm_{args.confirm_seed}',
                           weights, args.threads, {'lightgbm': tuple(args.seeds)})
        confirmation = {'seed': args.confirm_seed, 'result': result}

    rule = ('两套划分都要求 Δ>0、至少 4/5 折领先，且原划分的配对检验 p<0.10。'
            '融合是方差控制手段，机制明确，因此不再套用单特征那套 Δ≥3e-4 的高门槛。')
    adopted = bool(primary['blend']['delta_auc'] > 0 and primary['blend']['win_folds'] >= 4
                   and primary['blend']['paired_p'] < 0.10)
    if confirmation is not None:
        second = confirmation['result']['blend']
        adopted = bool(adopted and second['delta_auc'] > 0 and second['win_folds'] >= 4)

    result_submission = submission.copy()
    result_submission[TARGET] = primary['test']
    if not result_submission[TARGET].between(0, 1).all():
        raise ValueError('Invalid blended probabilities.')
    result_submission.to_csv(args.output_dir / 'submission.csv', index=False)
    summary = {'experiment': 'multimodel_logit', 'members': list(MEMBERS),
               'weights': weights, 'member_seeds': {'lightgbm': list(args.seeds)},
               'primary': primary['blend'],
               'members_primary': primary['members'],
               'confirmation': None if confirmation is None else {
                   'seed': confirmation['seed'], 'blend': confirmation['result']['blend'],
                   'members': confirmation['result']['members']},
               'adoption_rule': rule, 'adopted': adopted}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                                  encoding='utf-8')
    write_report(args.output_dir, weights, primary, confirmation, adopted, rule)
    if args.adopt and adopted:
        result_submission.to_csv(ROOT / 'artifacts/submission.csv', index=False)
        selection = {'model': 'lightgbm', 'fusion': True, 'experiment': 'multimodel_logit',
                     'pipeline_directory': str((args.output_dir / 'primary' / 'members').resolve()),
                     'members': list(MEMBERS), 'weights': weights,
                     'member_seeds': {'lightgbm': list(args.seeds)},
                     'original_cv': primary['blend'],
                     'confirmation_seed': args.confirm_seed,
                     'confirmation': None if confirmation is None else confirmation['result']['blend']}
        (args.output_dir / 'selection.json').write_text(
            json.dumps(selection, indent=2) + '\n', encoding='utf-8')
        (ROOT / 'artifacts/selected_model.json').write_text(
            json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    print(primary['table'].to_string(index=False))
    print(f'blend delta={primary["blend"]["delta_auc"]:+.6f} '
          f'wins={primary["blend"]["win_folds"]} p={primary["blend"]["paired_p"]:.3g} '
          f'adopted={adopted}')
    print(f'Report: {args.output_dir / "REPORT.md"}')
    return summary


if __name__ == '__main__':
    main()
