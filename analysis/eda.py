"""Read-only EDA. Run from the repository root: python analysis/eda.py."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(frame):
    def fmt(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:,.4f}'
        return str(value).replace('|', '/')

    return '\n'.join(
        ['| ' + ' | '.join(frame.columns) + ' |',
         '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
        + ['| ' + ' | '.join(map(fmt, row)) + ' |'
           for row in frame.itertuples(index=False, name=None)]
    )


def main():
    train = pd.read_csv(ROOT / 'data/train.csv')
    test = pd.read_csv(ROOT / 'data/test.csv')
    submission = pd.read_csv(ROOT / 'data/sample_submission.csv')
    assert train['satisfaction'].notna().all()
    assert set(train['satisfaction'].unique()) == {False, True}
    y = train['satisfaction'].astype(int)
    features = train.columns.drop(['id', 'satisfaction']).tolist()
    numeric = train[features].select_dtypes(include='number').columns.tolist()
    categorical = [c for c in features if c not in numeric]
    ratings = [c for c in numeric if train[c].nunique() <= 6]
    continuous = [c for c in numeric if c not in ratings]
    sections = ['# 航空旅客满意度 EDA',
                '来源：data/train.csv、data/test.csv、data/sample_submission.csv。以下统计使用完整文件，每行作为一条样本。满意率 = satisfaction=True 样本数 / 该组样本数。',
                '## 数据规模', table(pd.DataFrame([
                    {'文件': name, '样本数': len(df), '列数': df.shape[1]}
                    for name, df in [('train.csv', train), ('test.csv', test),
                                     ('sample_submission.csv', submission)]])),
                f'训练集包含 {len(features)} 个候选特征、1 个 id 和 1 个标签。',
                '## 特征类型',
                '存储类型为 pandas.read_csv 的推断结果。评分字段按离散评分解释。']
    schema = []
    for c in train:
        role = ('二元标签' if c == 'satisfaction' else '标识符，基线移除' if c == 'id'
                else '无序类别' if c in categorical else '离散评分' if c in ratings else '数值测量')
        schema.append({'字段': c, 'pandas 类型': str(train[c].dtype), '语义': role,
                       '训练非空取值数': train[c].nunique(),
                       '测试非空取值数': test[c].nunique() if c in test else None})
    sections += [table(pd.DataFrame(schema)),
                 f'特征组成：{len(continuous)} 个数值测量、{len(ratings)} 个离散评分、{len(categorical)} 个无序类别。',
                 '；'.join(c + ': ' + ', '.join(sorted(train[c].unique())) for c in categorical),
                 '## 缺失值']
    nulls = []
    for name, df in [('train', train), ('test', test), ('sample_submission', submission)]:
        for c in df:
            count = int(df[c].isna().sum())
            if count:
                nulls.append({'文件': name, '字段': c, '缺失数': count,
                              '缺失率 (%)': 100 * count / len(df)})
    sections += [table(pd.DataFrame(nulls)),
                 '其余字段缺失数为 0，训练标签完整。到达延误可由 LightGBM 原生处理，或在训练折内拟合中位数填充，并比较增加缺失指示的效果。',
                 '## 标签分布']
    counts = y.value_counts().sort_index()
    sections += [table(pd.DataFrame([
        {'标签': f'{i} ({bool(i)})', '样本数': int(n), '占比 (%)': 100 * n / len(train)}
        for i, n in counts.items()])),
        f'满意率 {y.mean():.4%}；多数类 Accuracy 基线 {counts.max()/len(train):.4%}。优先分层划分，保留原始类别比例。',
        '\n'.join(['```mermaid', 'xychart-beta', '    title "训练集标签分布"',
                   '    x-axis ["False", "True"]', '    y-axis "样本数" 0 --> 400000',
                   f'    bar [{counts[0]}, {counts[1]}]', '```']),
        '## 数值分布与评分取值']
    desc = train[continuous].describe(percentiles=[.01, .25, .5, .75, .99]).T.reset_index(names='字段')
    sections += [table(desc[['字段', 'min', '1%', '25%', '50%', '75%', '99%', 'max', 'mean']]),
                 table(pd.DataFrame([
                     {'字段': c, '训练取值': ', '.join(map(str, sorted(train[c].unique()))),
                      '零值数': int(train[c].eq(0).sum()), '零值占比 (%)': 100 * train[c].eq(0).mean()}
                     for c in ratings])),
                 f'起飞延误零值占 {train["Departure Delay in Minutes"].eq(0).mean():.2%}；到达延误零值占全部训练样本的 {train["Arrival Delay in Minutes"].eq(0).mean():.2%}。延误分布集中于零并有右长尾。',
                 '年龄、航程和延误均未发现负值。先保留长尾样本。评分 0 的业务定义需查比赛字段说明，当前保留原值。',
                 '## 特征与标签的关系',
                 '以下为训练集单变量描述性关联；模型效果使用独立验证评估。']
    group_rows = []
    for c in categorical + ['Inflight wifi service', 'Online boarding']:
        grouped = train.assign(_target=y).groupby(c)['_target'].agg(['size', 'mean'])
        for value, row in grouped.iterrows():
            group_rows.append({'字段': c, '取值': str(value), '样本数': int(row['size']),
                               '满意率 (%)': 100 * row['mean']})
    sections += [table(pd.DataFrame(group_rows)),
                 'Wi-Fi 评分 0 的满意率为 88.73%，在线登机评分 0 的满意率为 62.04%，评分与满意率存在非单调关系。先保留 0 值，避免全区间单调约束。',
                 '数值特征与标签的 Pearson 相关系数（到达延误使用非缺失样本）：']
    corr = train[numeric].corrwith(y).sort_values(key=abs, ascending=False)
    sections += [table(corr.rename('Pearson r').rename_axis('字段').reset_index()),
                 'Online boarding 的线性关联最强。舱位、出行类型、客户类型也呈现明显的分组满意率差异。',
                 '数值特征两两相关性最高的组合：']
    matrix = train[numeric].corr()
    pairs = sorted([(a, b, matrix.loc[a, b]) for i, a in enumerate(numeric)
                    for b in numeric[i+1:]], key=lambda x: abs(x[2]), reverse=True)[:8]
    sections += [table(pd.DataFrame(pairs, columns=['字段 A', '字段 B', 'Pearson r'])),
                 '## 数据质量与训练/测试一致性']
    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    candidates = test.loc[test_hash.isin(set(train_hash)), features]
    overlap = len(candidates.merge(train[features].drop_duplicates(), on=features, how='inner'))
    checks = [
        ('特征列名和顺序一致', train.drop(columns='satisfaction').columns.equals(test.columns)),
        ('训练重复 id', train['id'].duplicated().sum()),
        ('测试重复 id', test['id'].duplicated().sum()),
        ('训练/测试 id 交集', len(set(train['id']) & set(test['id']))),
        ('训练重复特征行（排除 id、标签）', train.duplicated(features).sum()),
        ('测试重复特征行（排除 id）', test.duplicated(features).sum()),
        ('测试样本与训练样本特征完全相同行数', overlap),
        ('提交样例行数与测试一致', len(submission) == len(test)),
        ('提交样例 id 顺序与测试一致', submission['id'].equals(test['id'])),
    ]
    sections.append(table(pd.DataFrame(checks, columns=['检查项', '结果'])))
    drift = []
    for c in features:
        a, b = train[c], test[c]
        discrete = c in categorical or c in ratings
        max_diff = None
        if discrete:
            pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
            pa, pb = pa.align(pb, fill_value=0)
            max_diff = 100 * (pa - pb).abs().max()
        drift.append({'字段': c, '最大取值占比差 (百分点)': max_diff,
                      '训练均值': a.mean() if c in numeric else None,
                      '测试均值': b.mean() if c in numeric else None,
                      '测试新增类别/评分数': len(set(b.dropna()) - set(a.dropna())) if discrete else None})
    sections += [table(pd.DataFrame(drift)),
                 '本次均值、类别频率和评分频率检查显示训练/测试分布接近，测试无新增类别或评分取值。后续可使用分位数比较和对抗验证检查联合分布差异。',
                 f'提交样例 satisfaction 只有 {submission["satisfaction"].nunique()} 个取值：{submission["satisfaction"].iloc[0]:.17g}，与训练满意率一致。提交文件采用 id 和概率格式，官方评估指标需查比赛页面。',
                 '## 下一步计划',
                 '1. 确认比赛链接、官方评估指标、评分 0 的定义及数据来源，确定主要验证指标。',
                 '2. 固定 random_state=42，按标签做 80%/20% 分层划分。基线移除 id；所有填充、编码与特征选择在训练折内拟合。',
                 '3. 建立常数概率和逻辑回归基线。逻辑回归采用数值填充及标准化、类别独热编码。再训练 LightGBM，显式指定四个类别字段，保留评分原值并使用早停。暂记 ROC-AUC、Log Loss、Accuracy、F1，确认官方指标后以其为主。',
                 '4. 在同一划分逐项比较缺失指示、延误 log1p/是否延误、服务评分汇总、舱位与出行类型交互。结合分组误差与置换重要性，保留验证有效的特征。',
                 '5. 对有效方案做 5 折分层交叉验证，记录均值和标准差。按验证需求决定概率校准或阈值优化。最终全训练集重训，按提交样例 id 顺序导出正类概率，检查行数、空值和 [0,1] 范围。',
                 '## 复现', '在项目根目录执行：',
                 '```powershell\nuv run python analysis/eda.py\n```',
                 '依赖为项目已有 pandas。脚本读取 data/，生成 analysis/EDA.md。此次完成 EDA，模型训练留待下一阶段。']
    destination = ROOT / 'analysis/EDA.md'
    destination.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print(f'Report: {destination}')
    print(f'Train {train.shape}; test {test.shape}; positive rate {y.mean():.10f}')
    print(pd.DataFrame(checks, columns=['check', 'result']).to_string(index=False))


if __name__ == '__main__':
    main()
