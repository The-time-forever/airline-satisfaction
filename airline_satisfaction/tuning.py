"""Persistent Optuna search for LightGBM with verified feature engineering."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.trial import TrialState
from sklearn.metrics import log_loss, roc_auc_score

from .baseline import positive_probabilities
from .cross_validation import make_folds, validate_folds
from .evaluation import load_predictions, markdown_table
from .feature_experiments import run_experiment
from .features import FeaturePreprocessor
from .preprocessing import ROOT, TARGET, validate_inputs

SPACE = {
    'n_estimators': [300, 1200, 100], 'learning_rate': [0.02, 0.1],
    'num_leaves': [15, 63, 16], 'min_child_samples': [20, 150, 10],
    'colsample_bytree': [0.75, 1.0], 'subsample': [0.75, 1.0],
    'reg_alpha': [0.0, 2.0], 'reg_lambda': [0.001, 10.0],
}


def suggest_parameters(trial, reference):
    parameters = reference.copy()
    for name in ['n_estimators', 'num_leaves', 'min_child_samples']:
        low, high, step = SPACE[name]
        parameters[name] = trial.suggest_int(name, low, high, step=step)
    for name in ['learning_rate', 'colsample_bytree', 'subsample', 'reg_alpha', 'reg_lambda']:
        parameters[name] = trial.suggest_float(name, *SPACE[name], log=name in {'learning_rate', 'reg_lambda'})
    return parameters


def export_trials(study, destination):
    trials, folds = [], []
    for trial in study.trials:
        row = {'trial': trial.number, 'state': trial.state.name,
               'mean_cv_auc': trial.value if trial.state == TrialState.COMPLETE else None,
               'partial_mean_auc': trial.user_attrs.get('partial_mean_auc'),
               'fold_std_auc': trial.user_attrs.get('fold_std_auc'),
               'completed_folds': len(trial.user_attrs.get('fold_scores', [])),
               'seconds': trial.duration.total_seconds() if trial.duration else None,
               'cached_reference': trial.user_attrs.get('cached_reference', False),
               'started': trial.datetime_start, 'finished': trial.datetime_complete,
               **{f'param_{key}': value for key, value in trial.params.items()}}
        trials.append(row)
        for score in trial.user_attrs.get('fold_scores', []):
            folds.append({'trial': trial.number, 'state': trial.state.name, **score})
    pd.DataFrame(trials).to_csv(destination / 'trials.csv', index=False)
    pd.DataFrame(folds).to_csv(destination / 'trial_folds.csv', index=False)
    joblib.dump(study.sampler, destination / 'sampler.joblib')


def prepare_fold_cache(train, folds, groups):
    X = train.drop(columns=['id', TARGET])
    y = train[TARGET].astype(int).to_numpy()
    cache = []
    for fold in sorted(folds.fold.unique()):
        fit_rows = np.flatnonzero(folds.fold.to_numpy() != fold)
        valid_rows = np.flatnonzero(folds.fold.to_numpy() == fold)
        prep = FeaturePreprocessor(groups=tuple(groups))
        cache.append({'fold': int(fold), 'X_fit': prep.fit_transform(X.iloc[fit_rows]),
                      'X_valid': prep.transform(X.iloc[valid_rows]),
                      'y_fit': y[fit_rows], 'y_valid': y[valid_rows]})
    return cache


def build_objective(cache, reference, reference_rows, sampler_path):
    def objective(trial):
        parameters = suggest_parameters(trial, reference)
        # Persist sampler RNG state after drawing parameters, as SQLite stores trials only.
        joblib.dump(trial.study.sampler, sampler_path)
        scores = []
        use_reference = all(parameters[k] == reference[k] for k in SPACE)
        trial.set_user_attr('cached_reference', use_reference)
        started = time.perf_counter()
        try:
            for data in cache:
                fold_started = time.perf_counter()
                if use_reference:
                    source = reference_rows.loc[data['fold']]
                    auc, loss = float(source.roc_auc), float(source.log_loss)
                else:
                    model = LGBMClassifier(**parameters)
                    model.fit(data['X_fit'], data['y_fit'])
                    p = positive_probabilities(model, data['X_valid'])
                    auc = roc_auc_score(data['y_valid'], p)
                    loss = log_loss(data['y_valid'], p)
                scores.append({'fold': data['fold'], 'roc_auc': auc, 'log_loss': loss,
                               'seconds': time.perf_counter() - fold_started})
                mean = float(np.mean([row['roc_auc'] for row in scores]))
                trial.set_user_attr('fold_scores', scores)
                trial.set_user_attr('partial_mean_auc', mean)
                trial.set_user_attr('fold_std_auc', float(np.std([row['roc_auc'] for row in scores])))
                trial.report(mean, step=data['fold'])
                print(f'trial {trial.number} fold {data["fold"]}: AUC={auc:.8f}, '
                      f'running mean={mean:.8f}' + (' [cached reference]' if use_reference else ''), flush=True)
                if len(scores) < len(cache) and trial.should_prune():
                    raise optuna.TrialPruned(f'Pruned after {len(scores)} folds')
            return mean
        finally:
            trial.set_user_attr('elapsed_seconds', time.perf_counter() - started)
    return objective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--features-dir', type=Path, default=ROOT / 'artifacts/features')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/tuning')
    parser.add_argument('--trials', type=int, default=16, help='Target total trial count, including reference and pruned trials.')
    parser.add_argument('--confirmation-seed', type=int, default=2027)
    args = parser.parse_args(argv)
    if args.trials < 2:
        parser.error('--trials must be at least 2')
    train, test, submission = [pd.read_csv(args.data_dir / name) for name in
                               ['train.csv', 'test.csv', 'sample_submission.csv']]
    validate_inputs(train, test, submission)
    folds = pd.read_csv(args.folds)
    validate_folds(train, folds)
    feature_config = json.loads((args.features_dir / 'config.json').read_text(encoding='utf-8'))
    selection = json.loads((args.features_dir / 'selection.json').read_text(encoding='utf-8'))
    if args.confirmation_seed in {42, feature_config['confirmation_seed']}:
        parser.error('Use a new confirmation seed, distinct from previous evaluations.')
    groups = selection['groups']
    reference = feature_config['parameters']
    reference_dir = Path(selection['pipeline_directory'])
    fingerprints = {}
    for name, path in [('train.csv', args.data_dir / 'train.csv'), ('test.csv', args.data_dir / 'test.csv'),
                       ('sample_submission.csv', args.data_dir / 'sample_submission.csv'), ('folds.csv', args.folds)]:
        with path.open('rb') as source:
            fingerprints[name] = hashlib.file_digest(source, 'sha256').hexdigest()
        if fingerprints[name] != feature_config['source_sha256'][name]:
            raise ValueError(f'{name} differs from the feature engineering experiment.')
    reference_p = load_predictions(train, folds, reference_dir.parent, [reference_dir.name])[reference_dir.name]
    reference_rows = []
    for fold in sorted(folds.fold.unique()):
        mask = folds.fold.eq(fold).to_numpy()
        reference_rows.append({'fold': int(fold), 'roc_auc': roc_auc_score(train[TARGET][mask], reference_p[mask]),
                               'log_loss': log_loss(train[TARGET][mask], reference_p[mask])})
    reference_rows = pd.DataFrame(reference_rows).set_index('fold')
    source_code = {}
    for name in ['features.py', 'preprocessing.py']:
        with Path(__file__).with_name(name).open('rb') as source:
            source_code[name] = hashlib.file_digest(source, 'sha256').hexdigest()
    identity = {'source_sha256': fingerprints, 'feature_code_sha256': source_code,
                'groups': groups, 'reference_parameters': reference, 'space': SPACE,
                'sampler_seed': 42, 'startup_trials': 5, 'confirmation_seed': args.confirmation_seed,
                'optuna_version': optuna.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.output_dir.resolve() / 'study.db'
    sampler_path = args.output_dir / 'sampler.joblib'
    sampler = (joblib.load(sampler_path) if sampler_path.exists()
               else optuna.samplers.TPESampler(seed=42, n_startup_trials=5))
    study = optuna.create_study(study_name='lightgbm_features_v1',
        storage=f'sqlite:///{database.as_posix()}', direction='maximize', load_if_exists=True,
        sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    previous_identity = study.user_attrs.get('identity')
    if previous_identity is not None and previous_identity != identity:
        raise ValueError('Study inputs/configuration changed; use a new --output-dir.')
    if any(t.state == TrialState.RUNNING for t in study.trials):
        raise ValueError('A trial is RUNNING; finish or resolve it before resuming this study.')
    study.set_user_attr('identity', identity)
    (args.output_dir / 'config.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    if not study.trials:
        study.enqueue_trial({k: reference[k] for k in SPACE}, user_attrs={'role': 'reference'})
    # WAITING reference is part of the target total and will be consumed by optimize.
    finished = sum(t.state.is_finished() for t in study.trials)
    remaining = max(0, args.trials - finished)
    if remaining:
        print('Preparing five independently fitted feature matrices', flush=True)
        cache = prepare_fold_cache(train, folds, groups)
        objective = build_objective(cache, reference, reference_rows, sampler_path)
        try:
            study.optimize(objective, n_trials=remaining, n_jobs=1,
                           callbacks=[lambda s, t: export_trials(s, args.output_dir)])
        finally:
            export_trials(study, args.output_dir)
        del cache
    else:
        export_trials(study, args.output_dir)
    best_parameters = reference | study.best_params
    (args.output_dir / 'best_params.json').write_text(json.dumps(best_parameters, indent=2) + '\n', encoding='utf-8')
    best_dir = args.output_dir / 'best_trial'
    best_result = run_experiment(train, folds, groups, best_parameters, best_dir, reference_rows.roc_auc)
    np.testing.assert_allclose(best_result['mean_roc_auc'], study.best_value, rtol=0, atol=1e-10)
    confirmation = make_folds(train, n_splits=folds.fold.nunique(), seed=args.confirmation_seed)
    confirmation.to_csv(args.output_dir / 'confirmation_folds.csv', index=False)
    reference_check = run_experiment(train, confirmation, groups, reference,
                                     args.output_dir / 'confirmation_reference', None)
    control = pd.read_csv(args.output_dir / 'confirmation_reference/fold_scores.csv').set_index('fold').roc_auc
    candidate_check = run_experiment(train, confirmation, groups, best_parameters,
                                     args.output_dir / 'confirmation_candidate', control)
    pd.DataFrame([reference_check, candidate_check]).to_csv(args.output_dir / 'confirmation.csv', index=False)
    adopted = (best_result['delta_auc'] > 0 and best_result['win_folds'] >= 3
               and candidate_check['delta_auc'] > 0 and candidate_check['win_folds'] >= 3)
    selected_dir = best_dir if adopted else reference_dir
    probability = np.zeros(len(test))
    for fold in sorted(folds.fold.unique()):
        pipeline = joblib.load(selected_dir / f'fold_{fold}.joblib')
        probability += positive_probabilities(pipeline, test.drop(columns='id')) / folds.fold.nunique()
    submission[TARGET] = probability
    if not submission[TARGET].between(0, 1).all():
        raise ValueError('Invalid selected-model probabilities.')
    submission.to_csv(args.output_dir / 'submission.csv', index=False)
    selected = {'adopted_tuned_parameters': bool(adopted), 'best_trial': study.best_trial.number,
                'groups': groups, 'pipeline_directory': str(selected_dir.resolve()),
                'original_cv': best_result, 'confirmation_reference': reference_check,
                'confirmation_candidate': candidate_check}
    (args.output_dir / 'selection.json').write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    write_report(study, selected, reference_rows, args.output_dir, args.confirmation_seed)
    print(f'Best trial={study.best_trial.number}, mean CV AUC={study.best_value:.8f}, adopted={adopted}', flush=True)
    print(f'Report: {args.output_dir / "REPORT.md"}', flush=True)


def write_report(study, selection, reference_rows, destination, confirmation_seed):
    import matplotlib.pyplot as plt
    trials = pd.read_csv(destination / 'trials.csv')
    complete = trials[trials.state.eq('COMPLETE')].sort_values('mean_cv_auc', ascending=False)
    comparison = pd.DataFrame([
        {'划分': '搜索划分', '对照 AUC': reference_rows.roc_auc.mean(),
         '候选 AUC': selection['original_cv']['mean_roc_auc'],
         'AUC 差': selection['original_cv']['delta_auc'], '领先折数': selection['original_cv']['win_folds']},
        {'划分': f'复核种子 {confirmation_seed}', '对照 AUC': selection['confirmation_reference']['mean_roc_auc'],
         '候选 AUC': selection['confirmation_candidate']['mean_roc_auc'],
         'AUC 差': selection['confirmation_candidate']['delta_auc'], '领先折数': selection['confirmation_candidate']['win_folds']},
    ])
    parameters = pd.DataFrame([{'参数': k, '最优值': str(v)} for k, v in study.best_params.items()])
    sections = ['# Optuna 超参数优化',
        '固定已验证的特征组和原 5 折分配，优化 LightGBM 的平均折 ROC-AUC。每折预处理仅使用训练部分，并缓存转换结果供各试验共享；模型在每次试验、每一折重新训练。',
        f'共记录 {len(trials)} 次试验：' + '，'.join(f'{k}={v}' for k, v in trials.state.value_counts().items()) + '。',
        '使用固定种子的 TPE 顺序搜索。参数对照复用已验证的原始 OOF 分数，并在 cached_reference 字段标记。至少完成三折后，MedianPruner 可剪枝较差候选；剪枝试验的部分均值单独记录，不参与最优完整试验排名。',
        '## 最优完整试验', f'试验编号 {study.best_trial.number}，平均 ROC-AUC {study.best_value:.8f}。',
        markdown_table(parameters), '## 配对比较与复核', markdown_table(comparison),
        f'复核使用种子 {confirmation_seed} 的新 5 折，仅比较本轮搜索的最佳候选与同特征组的原参数。两套划分均要求平均 AUC 提升且至少 3 折领先。',
        '最终选择：' + ('采用调优参数。' if selection['adopted_tuned_parameters'] else '保留调优前参数。'),
        '搜索分数用于参数选择；复核结果用于本轮采用决策。后续继续搜索时保留本轮复核记录。',
        '## 完整试验排名', markdown_table(complete[['trial', 'mean_cv_auc', 'fold_std_auc', 'completed_folds', 'seconds']]),
        '![搜索过程](optimization.png)',
        '## 实验记录',
        '- study.db：Optuna 持久化记录，包括参数、状态、中间分数和逐折属性。\n- trials.csv：每次试验的参数、完整/部分 AUC、折数、耗时和状态。\n- trial_folds.csv：每次试验每个已完成折的 AUC、Log Loss、耗时。\n- best_params.json：最佳完整试验的全部模型参数。\n- selection.json：最终采用决策和模型位置。\n- submission.csv：最终采用的原划分五折模型概率平均。',
        '## 复现与继续搜索', '```powershell\nuv run python main.py tune --trials 16\n# Continue the same study to 32 total trials\nuv run python main.py tune --trials 32\n```',
        '输入文件、特征代码、搜索空间及参数身份均会核对；更换配置使用新的 --output-dir。SQLite 保存试验，sampler.joblib 保存采样器状态。']
    (destination / 'REPORT.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    chronological = trials.sort_values('trial')
    finished = chronological[chronological.state.eq('COMPLETE')]
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.scatter(finished.trial, finished.mean_cv_auc, color='#2563a6', label='Complete trial')
    ax.plot(finished.trial, finished.mean_cv_auc.cummax(), color='#cb7527', label='Best complete AUC')
    ax.axhline(reference_rows.roc_auc.mean(), color='#444444', ls='--', label='Reference parameters')
    ax.set(title='Optuna LightGBM search (complete 5-fold evaluations)', xlabel='Trial', ylabel='Mean CV ROC-AUC')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(destination / 'optimization.png', dpi=160)
    plt.close(fig)
