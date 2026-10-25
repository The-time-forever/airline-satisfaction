"""Read-only dataset audit: duplicate structure, label noise, signal and drift.

Run from the repository root: python analysis/audit.py

Writes analysis/AUDIT.md. The optional --original flag adds a comparison with the
public 129,880-row airline survey the synthetic competition data derives from.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATINGS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness',
]
CATEGORICAL = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
SIGNATURE = RATINGS + CATEGORICAL
ORIGINAL_RATINGS = [
    'In-flight Wifi Service', 'Departure and Arrival Time Convenience',
    'Ease of Online Booking', 'Gate Location', 'Food and Drink', 'Online Boarding',
    'Seat Comfort', 'In-flight Entertainment', 'On-board Service', 'Leg Room Service',
    'Baggage Handling', 'Check-in Service', 'Cleanliness',
]
ORIGINAL_COLUMNS = (set(ORIGINAL_RATINGS)
                    | {'Flight Distance', 'Satisfaction', 'Type of Travel', 'Class',
                       'Customer Type', 'Gender'})


def table(frame):
    def fmt(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:,.0f}' if value.is_integer() else f'{value:,.4f}'
        return str(value).replace('|', '/')

    return '\n'.join(
        ['| ' + ' | '.join(frame.columns) + ' |',
         '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
        + ['| ' + ' | '.join(map(fmt, row)) + ' |'
           for row in frame.itertuples(index=False, name=None)]
    )


def signature_keys(frame):
    """Stable 64-bit key per row; identical discrete profile -> identical key."""
    values = pd.concat([frame[SIGNATURE[k]].rename(k).astype('string')
                        for k in range(len(SIGNATURE))], axis=1)
    return pd.util.hash_pandas_object(values, index=False)


def grouping(frame, y, columns, label):
    keys = (signature_keys(frame) if columns == 'signature'
            else pd.util.hash_pandas_object(frame[columns], index=False))
    grouped = pd.DataFrame({'key': keys, 'y': y}).groupby('key')['y'].agg(['size', 'sum'])
    grouped['mean'] = grouped['sum'] / grouped['size']
    multi = grouped[grouped['size'] >= 2]
    pure = multi[(multi['mean'] == 0) | (multi['mean'] == 1)]
    impure = grouped[(grouped['mean'] > 0) & (grouped['mean'] < 1)]
    p = impure['mean'].to_numpy()
    rows = impure['size'].to_numpy()
    entropy = -(rows * (p * np.log(p) + (1 - p) * np.log(1 - p))).sum() / len(frame)
    majority_flip = 1 - (multi[['mean']].assign(m=lambda v: np.maximum(v['mean'], 1 - v['mean']))
                         .assign(n=multi['size']).eval('m * n').sum() / multi['size'].sum())
    return {
        '分组': label,
        '组数': len(grouped),
        '重复组数': len(multi),
        '重复组覆盖行占比 (%)': 100 * multi['size'].sum() / len(frame),
        '重复组中纯净组占比 (%)': 100 * len(pure) / max(len(multi), 1),
        '同组标签冲突行占比 (%)': 100 * impure['size'].sum() / len(frame),
        '组内多数标签翻转率 (%)': 100 * majority_flip,
        '条件熵下界 (nats)': entropy,
    }


def drift_checks(train, test):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    features = [c for c in train.columns if c not in ('id', 'satisfaction')]
    frame = pd.concat([train[features], test[features]], ignore_index=True)
    for c in CATEGORICAL:
        frame[c] = frame[c].astype('category')
    labels = np.r_[np.zeros(len(train)), np.ones(len(test))]
    scores, importances = [], None
    splitter = StratifiedKFold(3, shuffle=True, random_state=0)
    for fit_rows, valid_rows in splitter.split(frame, labels):
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               random_state=0, n_jobs=4, verbosity=-1)
        model.fit(frame.iloc[fit_rows], labels[fit_rows])
        scores.append(roc_auc_score(labels[valid_rows],
                                    model.predict_proba(frame.iloc[valid_rows])[:, 1]))
        gain = pd.Series(model.booster_.feature_importance('gain'), index=model.booster_.feature_name())
        importances = gain if importances is None else importances + gain
    top = (importances / len(scores)).sort_values(ascending=False).head(5)
    return scores, top


def rating_table(train, y):
    rows = []
    for c in RATINGS:
        rate = train.groupby(c)['satisfaction'].mean()
        steps = rate.diff().abs()
        rows.append({
            '评分字段': c,
            '取值数': int(train[c].nunique()),
            '零值占比 (%)': 100 * (train[c] == 0).mean(),
            '零值满意率': float(rate.get(0, np.nan)),
            '一值满意率': float(rate.get(1, np.nan)),
            '最大值满意率': float(rate.get(5, np.nan)),
            '单调性缺口': float(steps.dropna().max()),
            'Pearson r': float(np.corrcoef(train[c], y)[0, 1]),
        })
    return pd.DataFrame(rows)


def load_original(path):
    frame = pd.read_csv(path)
    missing = ORIGINAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f'原始调查文件缺少列：{sorted(missing)}')
    return frame


def original_comparison(train, test, original):
    """Compare the synthetic data with the public survey it was generated from."""
    mapping = dict(zip(ORIGINAL_RATINGS, RATINGS))
    mapping.update({'Satisfaction': 'satisfaction', 'Flight Distance': 'Flight Distance',
                    'Departure Delay': 'Departure Delay in Minutes',
                    'Arrival Delay': 'Arrival Delay in Minutes', 'Age': 'Age'})
    original = original.rename(columns=mapping)
    original['Type of Travel'] = original['Type of Travel'].map(
        {'Business travel': 'Business travel', 'Business': 'Business travel',
         'Personal': 'Personal Travel', 'Personal Travel': 'Personal Travel'})
    original['Class'] = original['Class'].map(
        {'Business': 'Business', 'Economy': 'Eco', 'Eco': 'Eco',
         'Economy Plus': 'Eco Plus', 'Eco Plus': 'Eco Plus'})
    original['Customer Type'] = original['Customer Type'].map(
        {'Loyal Customer': 'Loyal Customer', 'Returning': 'Loyal Customer',
         'disloyal Customer': 'disloyal Customer', 'First-time': 'disloyal Customer'})
    original['satisfaction'] = original['satisfaction'].map(
        {'Satisfied': 1, 'satisfied': 1, 'Neutral or Dissatisfied': 0,
         'neutral or dissatisfied': 0})
    if not original['satisfaction'].notna().any():
        raise ValueError('原始调查的 Satisfaction 列无法解析为 0/1。')
    original = original.dropna(subset=['Flight Distance', 'satisfaction'])

    unseen = set(train['Flight Distance']) - set(original['Flight Distance'])
    distances = train.groupby('Flight Distance')['satisfaction'].agg(['size', 'mean'])
    reference = original.groupby('Flight Distance')['satisfaction'].mean().rename('mean_original')
    joined = distances.join(reference, how='inner')
    joined = joined[joined['size'] >= 30]
    correlation = joined['mean'].corr(joined['mean_original'])

    original_keys = set(signature_keys(original))
    keys = signature_keys(train)
    matched = keys.isin(original_keys)
    keyed = pd.DataFrame({'key': keys, 'y': train['satisfaction'].astype(int),
                          'oy': keys.map(original.groupby(signature_keys(original))['satisfaction'].mean())})
    agreement = (keyed.loc[matched, 'y'] == (keyed.loc[matched, 'oy'] >= 0.5)).mean()

    delay = []
    for column in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        delay.append({'字段': column, '合成均值': train[column].mean(),
                      '原始均值': original[column].mean(),
                      '合成 99 分位': train[column].quantile(0.99),
                      '原始 99 分位': original[column].quantile(0.99)})
    summary = pd.DataFrame([
        {'检查项': '原始数据行数', '结果': len(original)},
        {'检查项': '训练集未出现在原始数据的航程数', '结果': len(unseen)},
        {'检查项': '对应训练行占比 (%)', '结果': 100 * train['Flight Distance'].isin(unseen).mean()},
        {'检查项': '航程满意率与原始数据相关系数', '结果': correlation},
        {'检查项': '17 列签名命中原始数据行占比 (%)', '结果': 100 * matched.mean()},
        {'检查项': '命中行与原标签一致率 (%)', '结果': 100 * agreement},
        {'检查项': '测试集签名命中原始数据行占比 (%)', '结果': 100 * signature_keys(test).isin(original_keys).mean()},
    ])
    return summary, pd.DataFrame(delay)


def adversarial_section(train, test):
    scores, top = drift_checks(train, test)
    frame = pd.DataFrame([{'折': str(i), 'AUC': float(s)} for i, s in enumerate(scores)]
                         + [{'折': '平均', 'AUC': float(np.mean(scores))}])
    gains = top.rename('平均增益').rename_axis('字段').reset_index()
    gains['字段'] = gains['字段'].str.replace('_', ' ')
    return frame, gains


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/AUDIT.md')
    parser.add_argument('--original', type=Path, default=None,
                        help='Optional public survey CSV used to check what the generator copied.')
    parser.add_argument('--skip-adversarial', action='store_true')
    args = parser.parse_args(argv)
    train = pd.read_csv(args.data_dir / 'train.csv')
    test = pd.read_csv(args.data_dir / 'test.csv')
    original = load_original(args.original) if args.original is not None else None
    y = train['satisfaction'].astype(int)

    structure = [
        grouping(train, y, 'signature', '13 评分 + 4 类别（17 列签名）'),
        grouping(train, y, RATINGS, '仅 13 评分'),
        grouping(train, y, RATINGS + ['Age'], '13 评分 + Age'),
        grouping(train, y, ['Flight Distance'], '航程'),
    ]
    ids_contiguous = set(test['id']) == set(range(train['id'].max() + 1, test['id'].max() + 1))
    keys = signature_keys(train)
    coverage = signature_keys(test).isin(set(keys)).mean()
    ratings = rating_table(train, y)

    sections = [
        '# 数据集审计：重复结构、标签噪声与可用信号',
        '本报告补充 analysis/EDA.md，聚焦“还能从数据里挤出什么”。所有统计只读 data/，'
        '签名指 13 个服务评分加 4 个类别字段组成的离散指纹。',
        '## 划分与一致性',
        table(pd.DataFrame([
            {'检查项': '训练 / 测试行数', '结果': f'{len(train)} / {len(test)}'},
            {'检查项': '训练 + 测试 id 覆盖连续区间', '结果': bool(ids_contiguous)},
            {'检查项': '满意率', '结果': float(y.mean())},
            {'检查项': '测试行签名命中训练集占比 (%)', '结果': 100 * coverage},
        ])),
        '训练与测试 id 连续且无交集，说明是一次随机切分，不存在按时间或按批次的划分差异。',
        '## 离散签名与标签噪声',
        '把特征当作分组键，可以直接观察“同特征不同标签”的比例，这是标签噪声的直接证据。',
        table(pd.DataFrame(structure)),
        '条件熵下界只统计存在标签冲突的组，因此是 H(y|x) 的乐观下界；组内多数标签翻转率给出重复样本中不可约噪声的比例。'
        '加入 Age 后冲突大幅减少，说明连续特征仍在承担真实分辨力，离散指纹不是全部信息。',
        '## 评分 0 的特殊含义',
        '评分 0 不是“最差”，而是“未使用 / 不适用”。单调性缺口衡量相邻取值的满意率最大跳变，缺口越大越不适合线性模型。',
        table(ratings),
        'Wi-Fi、在线登机、餐饮等字段在 0 处出现满意率反转，因此逻辑回归等线性模型必须独热编码评分，'
        '树模型可以直接利用分裂。Departure/Arrival time convenient 与 Gate location 的满意率几乎不随取值变化，属于近噪声字段。',
    ]
    if not args.skip_adversarial:
        fold_frame, gains = adversarial_section(train, test)
        sections += ['## 对抗验证',
                     '用 LightGBM 区分训练与测试样本。AUC 接近 0.5 表示两者同分布，交叉验证分数应能外推到排行榜。',
                     table(fold_frame), table(gains)]
    if original is not None:
        summary, delay = original_comparison(train, test, original)
        sections += ['## 与公开原始调查数据的对照',
                     '本赛题数据由公开的 129,880 行航空服务调查合成而来。用户可以下载原始文件后用 --original 复现本节。',
                     table(summary),
                     '航程取值几乎没有新增，且每个航程的满意率与原始调查高度相关，这说明生成器保留了“按航程分组的标签条件分布”，'
                     '也是航程目标编码有效的根本原因。',
                     table(delay),
                     '延误分布在合成过程中被严重压缩，导致延误字段几乎不含信号；针对延误做特征工程难有回报。']
    destination = args.output
    destination.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print(f'Report: {destination}')
    print(table(pd.DataFrame(structure)))
    return structure


if __name__ == '__main__':
    main()
