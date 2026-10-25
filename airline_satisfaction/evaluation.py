"""Evaluate saved out-of-fold probabilities without retraining models."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from .cross_validation import validate_folds
from .preprocessing import ROOT, TARGET, CATEGORICAL, DELAY

COLORS = {'lightgbm': '#2563a6', 'xgboost': '#cb7527', 'catboost': '#667f36',
          'logistic_regression': '#ab527c', 'constant': '#767676'}


def calibration_bins(y, p, n_bins=10):
    """Equal-width bins; probability 1 belongs to the last bin."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bins == i
        if mask.any():
            rows.append({'bin': i, 'n': int(mask.sum()), 'mean_probability': p[mask].mean(),
                         'observed_rate': y[mask].mean()})
    return pd.DataFrame(rows)


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected matching labels and finite probabilities in [0, 1].')
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = calibration_bins(y, p)
    return {
        'n': len(y), 'positive_rate': y.mean(),
        'roc_auc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        'average_precision': average_precision_score(y, p) if y.sum() else np.nan,
        'log_loss': log_loss(y, p, labels=[0, 1]), 'brier': brier_score_loss(y, p),
        'ece_10': float((bins['n'] * (bins.mean_probability - bins.observed_rate).abs()).sum() / len(y)),
        'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
        'fpr': fp / (tn + fp) if tn + fp else np.nan,
        'fnr': fn / (tp + fn) if tp + fn else np.nan,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def load_predictions(train, folds, model_dir, models):
    probabilities = {}
    for name in models:
        oof = pd.read_csv(model_dir / name / 'oof.csv')
        if not {'id', 'fold', TARGET, 'probability'}.issubset(oof):
            raise ValueError(f'Missing OOF columns for {name}.')
        if not oof['id'].equals(train['id']) or not oof['fold'].equals(folds['fold']):
            raise ValueError(f'OOF row alignment differs for {name}.')
        if not np.array_equal(oof[TARGET].to_numpy(), train[TARGET].astype(int).to_numpy()):
            raise ValueError(f'OOF labels differ for {name}.')
        p = oof['probability'].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f'Invalid OOF probabilities for {name}.')
        probabilities[name] = p
    return probabilities


def paired_auc_delta(y_true, reference, candidate):
    """DeLong paired standard error for AUC(candidate) - AUC(reference).

    Returns (delta, standard_error); the two score vectors must be aligned row by row.
    """
    def components(scores):
        labels = np.asarray(y_true, dtype=bool)
        scores = np.asarray(scores, dtype=float)
        positive, negative = scores[labels], scores[~labels]
        m, n = len(positive), len(negative)
        if m < 2 or n < 2:
            raise ValueError('Paired AUC test needs at least two rows per class.')
        ranks = rankdata(np.concatenate([positive, negative]))
        return (ranks[:m] - (m + 1) / 2) / n, (ranks[m:].mean() - ranks[m:]) / m, m, n

    a10, a01, m, n = components(reference)
    b10, b01, _, _ = components(candidate)
    variance = ((a10.var(ddof=1) + b10.var(ddof=1) - 2 * np.cov(a10, b10)[0, 1]) / m
                + (a01.var(ddof=1) + b01.var(ddof=1) - 2 * np.cov(a01, b01)[0, 1]) / n)
    return float(b10.mean() - a10.mean()), float(np.sqrt(max(variance, 0.0)))


def segment_definitions(train):
    segments = {c: train[c] for c in CATEGORICAL}
    segments['Age band'] = pd.cut(train['Age'], [0, 17, 29, 44, 59, np.inf],
                                  labels=['<=17', '18-29', '30-44', '45-59', '60+'])
    segments['Arrival delay status'] = pd.Series(np.select(
        [train[DELAY].isna(), train[DELAY].eq(0)], ['missing', 'zero'], default='positive'))
    segments['WiFi zero'] = train['Inflight wifi service'].eq(0).map({True: 'zero', False: 'nonzero'})
    return segments


def markdown_table(frame):
    def fmt(x):
        if pd.isna(x):
            return '—'
        if isinstance(x, (float, np.floating)):
            return f'{x:.6f}'
        if isinstance(x, (int, np.integer)):
            return f'{x:,}'
        return str(x).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(fmt, row)) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def plot_evaluation(y, predictions, overview, fold_metrics, best, output_dir):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout='constrained')
    for name, p in predictions.items():
        color = COLORS.get(name, '#333333')
        fpr, tpr, _ = roc_curve(y, p)
        axes[0, 0].plot(fpr, tpr, color=color, lw=1.6,
                        label=f'{name}: {overview.loc[name, "roc_auc"]:.6f}')
        precision, recall, _ = precision_recall_curve(y, p, drop_intermediate=True)
        axes[0, 1].step(recall, precision, where='post', color=color, lw=1.6,
                        label=f'{name}: {overview.loc[name, "average_precision"]:.6f}')
        bins = calibration_bins(y, p)
        axes[1, 0].plot(bins.mean_probability, bins.observed_rate, 'o-',
                        color=color, markersize=4, lw=1.5, label=name)
    axes[0, 0].plot([0, 1], [0, 1], '--', color='#444444', lw=1)
    axes[0, 0].set(title='OOF ROC curves (legend: ROC-AUC)', xlabel='False positive rate',
                   ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].axhline(np.mean(y), ls='--', color='#444444', lw=1)
    axes[0, 1].set(title='OOF precision-recall curves (legend: AP)', xlabel='Recall',
                   ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    axes[1, 0].plot([0, 1], [0, 1], '--', color='#444444', lw=1)
    axes[1, 0].set(title='OOF calibration (10 equal-width probability bins)',
                   xlabel='Mean predicted probability', ylabel='Observed positive rate',
                   xlim=(0, 1), ylim=(0, 1))
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=.15)
    matrix = confusion_matrix(y, predictions[best] >= .5, labels=[0, 1])
    axes[1, 1].imshow(matrix, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, f'{matrix[i,j]:,}', ha='center', va='center',
                            fontsize=18, color='white' if matrix[i,j] > matrix.max()/2 else '#222222')
    axes[1, 1].set(title=f'{best}: OOF confusion matrix (threshold=0.5)',
                   xlabel='Predicted label', ylabel='Actual label',
                   xticks=[0, 1], yticks=[0, 1], xticklabels=['0', '1'], yticklabels=['0', '1'])
    fig.suptitle(f'Model evaluation | {len(y):,} OOF samples | positive rate {np.mean(y):.2%}', fontsize=15)
    fig.savefig(output_dir / 'evaluation.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for name in ['lightgbm', 'xgboost', 'catboost']:
        if name in predictions:
            rows = fold_metrics[fold_metrics.model.eq(name)].sort_values('fold')
            ax.plot(rows.fold, rows.roc_auc, 'o-', color=COLORS[name], label=name)
    ax.set(title='Tree baseline ROC-AUC on identical validation folds', xlabel='Fold',
           ylabel='ROC-AUC', xticks=sorted(fold_metrics.fold.unique()))
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output_dir / 'fold_auc.png', dpi=160)
    plt.close(fig)


def evaluate(train, folds, model_dir, output_dir):
    validate_folds(train, folds)
    saved_summary = pd.read_csv(model_dir / 'summary.csv')
    if saved_summary.model.duplicated().any():
        raise ValueError('Duplicate models in saved summary.')
    models = saved_summary.sort_values('mean_roc_auc', ascending=False).model.tolist()
    predictions = load_predictions(train, folds, model_dir, models)
    y = train[TARGET].astype(int).to_numpy()
    fold_numbers = folds.fold.to_numpy()
    all_rows, fold_rows, segment_rows = [], [], []
    segments = segment_definitions(train)
    for name, p in predictions.items():
        print(f'Evaluating {name}', flush=True)
        local_folds = []
        for fold in sorted(folds.fold.unique()):
            mask = fold_numbers == fold
            row = {'model': name, 'fold': fold, **metrics(y[mask], p[mask])}
            fold_rows.append(row)
            local_folds.append(row['roc_auc'])
        row = {'model': name, 'mean_fold_auc': np.mean(local_folds),
               'std_fold_auc': np.std(local_folds), **metrics(y, p)}
        previous = saved_summary.set_index('model').loc[name]
        np.testing.assert_allclose([row['mean_fold_auc'], row['std_fold_auc'], row['roc_auc']],
            previous[['mean_roc_auc', 'std_roc_auc', 'oof_roc_auc']].to_numpy(dtype=float), atol=1e-10, rtol=0)
        all_rows.append(row)
        for feature, values in segments.items():
            groups = (values.cat.categories if isinstance(values.dtype, pd.CategoricalDtype)
                      else sorted(values.dropna().unique()))
            for group in groups:
                mask = np.asarray(values == group)
                if not mask.any():
                    continue
                segment_rows.append({'model': name, 'feature': feature, 'group': str(group),
                                     **metrics(y[mask], p[mask])})
    overview = pd.DataFrame(all_rows).set_index('model')
    fold_metrics = pd.DataFrame(fold_rows)
    segment_metrics = pd.DataFrame(segment_rows)
    best = overview.mean_fold_auc.idxmax()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(output_dir / 'metrics.csv')
    fold_metrics.to_csv(output_dir / 'fold_metrics.csv', index=False)
    segment_metrics.to_csv(output_dir / 'segment_metrics.csv', index=False)
    plot_evaluation(y, predictions, overview, fold_metrics, best, output_dir)
    ranking = overview.reset_index()
    best_row = overview.loc[best]
    paired = fold_metrics.pivot(index='fold', columns='model', values='roc_auc')
    differences = []
    for name in models:
        if name == best:
            continue
        delta = paired[best] - paired[name]
        differences.append({'比较': f'{best} - {name}', '平均折 AUC 差': delta.mean(),
                            '领先折数': int((delta > 0).sum()), '总折数': len(delta),
                            '最小差': delta.min(), '最大差': delta.max()})
    segment_best = segment_metrics[segment_metrics.model.eq(best)].copy()
    group_findings = []
    for feature, group in [('Type of Travel', 'Personal Travel'), ('Class', 'Eco'),
                           ('Customer Type', 'disloyal Customer'), ('WiFi zero', 'zero')]:
        group_row = segment_best.loc[segment_best.feature.eq(feature) & segment_best.group.eq(group)].iloc[0]
        group_findings.append(
            f'- {feature}={group}：{int(group_row["n"]):,} 条，满意率 {group_row.positive_rate:.2%}，'
            f'组内 AUC {group_row.roc_auc:.6f}，误报率 {group_row.fpr:.2%}，'
            f'漏报率 {group_row.fnr:.2%}。')
    report = ['# 基线模型评估',
        f'使用 data/train.csv 的 {len(train):,} 条样本和固定 5 折 OOF 预测。每条概率来自未使用该样本训练的折模型；每折预处理独立拟合。正类为 satisfaction=True，分类阈值固定为 0.5。',
        '## 概率预测与排序能力',
        markdown_table(ranking[['model', 'mean_fold_auc', 'std_fold_auc', 'roc_auc', 'average_precision', 'log_loss', 'brier', 'ece_10']]),
        f'按平均折 ROC-AUC 排名，当前最佳模型为 {best}，均值 {best_row.mean_fold_auc:.6f}，折间标准差 {best_row.std_fold_auc:.6f}，整体 OOF ROC-AUC {best_row.roc_auc:.6f}。',
        'mean_fold_auc 是各折 AUC 的平均值，roc_auc 是合并 OOF 的 AUC；std_fold_auc 使用 ddof=0，描述本次划分的折间波动。average_precision 为 AP，使用召回增量加权精确率；Log Loss、Brier 和 ECE 越低越好。ECE 使用 10 个等宽概率区间，按各区间样本数加权计算预测均值与实际满意率的绝对差。',
        '常数模型的 ECE 为 0、ROC-AUC 约为 0.5；校准程度和排序能力分别通过校准指标与 AUC/AP 评估。',
        '## 固定阈值分类结果',
        markdown_table(ranking[['model', 'accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr']]),
        f'{best} 的混淆矩阵：TN={int(best_row.tn):,}，FP={int(best_row.fp):,}，FN={int(best_row.fn):,}，TP={int(best_row.tp):,}。FP 为不满意样本被判为满意，FN 为满意样本被判为不满意。',
        'Precision、Recall、F1 及误报率/漏报率均对应阈值 0.5；没有预测正类时 Precision 记为 0。阈值调整服务于具体分类成本，比赛提交仍使用连续概率。',
        '![OOF ROC、PR、校准和混淆矩阵](evaluation.png)',
        '## 相同折上的模型差异', markdown_table(pd.DataFrame(differences)),
        '差值按同一验证折配对计算。后续模型改进沿用该折分配，同时保留新划分复核小幅增益。',
        '![三个树模型的逐折 AUC](fold_auc.png)',
        '## 最佳模型分组表现',
        markdown_table(segment_best[['feature', 'group', 'n', 'positive_rate', 'roc_auc', 'log_loss', 'fpr', 'fnr']]),
        'n 为该组样本数，positive_rate 为实际满意率；各分组的 AUC 衡量组内排序能力。FPR 的分母为实际负类数，FNR 的分母为实际正类数。全部模型的分组结果见 segment_metrics.csv。',
        '重点分组：', '\n'.join(group_findings),
        '## 下一步',
        f'保留 {best} 作为比较基准。个人出行、经济舱和非忠诚客户的满意样本漏报率较高，优先验证出行类型与舱位、客户类型与服务评分的交互。WiFi 评分 0 组的负类误报率较高，可验证单独的零值指示及其与出行类型的交互。若尝试融合，使用 OOF 预测选择方案，并在独立划分复核增益。',
        '## 复现', '```powershell\nuv run python main.py evaluate\n```',
        '本脚本读取已有 OOF 预测并重新计算指标；绘图使用完整预测。产物位于 artifacts/evaluation/。']
    (output_dir / 'REPORT.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
    print(ranking[['model', 'mean_fold_auc', 'roc_auc', 'average_precision', 'log_loss', 'accuracy', 'f1']].to_string(index=False))
    print(f'Report: {output_dir / "REPORT.md"}')
    return overview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/train.csv')
    parser.add_argument('--folds', type=Path, default=ROOT / 'artifacts/cv/folds.csv')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts/baseline')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/evaluation')
    args = parser.parse_args(argv)
    config = json.loads((args.model_dir / 'config.json').read_text(encoding='utf-8'))
    for name, path in [('train.csv', args.train), ('folds.csv', args.folds)]:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != config['source_sha256'][name]:
            raise ValueError(f'{name} differs from model training source.')
    evaluate(pd.read_csv(args.train), pd.read_csv(args.folds), args.model_dir, args.output_dir)
