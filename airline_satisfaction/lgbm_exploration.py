"""Controlled capacity and data-association experiments for the current LightGBM."""
import argparse
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.validation import check_is_fitted

from .baseline import positive_probabilities
from .cross_validation import make_folds, validate_folds
from .evaluation import load_predictions, markdown_table, metrics
from .feature_experiments import run_experiment
from .features import RATINGS
from .preprocessing import ROOT, TARGET, validate_inputs
from .target_statistics import TargetStatisticsPreprocessor, category_keys


def rating_keys(X):
    result = {}
    for i, rating in enumerate(RATINGS):
        values = X[rating].astype('string').fillna('__MISSING__')
        for j, context in enumerate(['Class', 'Type of Travel', 'Customer Type']):
            result[f'RT_{i}_{j}'] = values + '|' + X[context].astype('string').fillna('__MISSING__')
    return pd.DataFrame(result, index=X.index)


class AssociationPreprocessor(TransformerMixin, BaseEstimator):
    """Add fold-fitted route profiles or cross-fitted rating/context target means."""

    def __init__(self, base=None, block='route_profile'):
        self.base = base
        self.block = block

    def _setup(self, X):
        if self.block not in {'route_profile', 'rating_context'}:
            raise ValueError('Unknown association block.')
        self.base_ = clone(self.base) if self.base is not None else TargetStatisticsPreprocessor()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        if self.block == 'route_profile':
            self.profile_columns_ = RATINGS + ['Age']
            self.global_means_ = X[self.profile_columns_].mean()
            grouped = X.groupby('Flight Distance')[self.profile_columns_].agg(['sum', 'count'])
            self.route_means_ = pd.DataFrame({c: (grouped[c]['sum'] + 20 * self.global_means_[c]) /
                (grouped[c]['count'] + 20) for c in self.profile_columns_})
            keys = category_keys(X)
            self.support_ = {c: keys[c].value_counts() for c in keys}
        else:
            self.rating_encoder_ = TargetEncoder(target_type='binary', smooth=20.,
                cv=StratifiedKFold(5, shuffle=True, random_state=42))

    def _append(self, result, X, encoded=None):
        if self.block == 'route_profile':
            for c in self.profile_columns_:
                mean = X['Flight Distance'].map(self.route_means_[c]).fillna(self.global_means_[c])
                result['RP_mean_' + c] = mean.astype('float32')
                if c in RATINGS:
                    result['RP_deviation_' + c] = (X[c] - mean).astype('float32')
            keys = category_keys(X)
            for c in keys:
                result['RP_support_' + c] = keys[c].map(self.support_[c]).fillna(0).astype('float32')
        else:
            if encoded is None:
                encoded = self.rating_encoder_.transform(rating_keys(X))
            for i, c in enumerate(self.rating_encoder_.get_feature_names_out()):
                result[c] = encoded[:, i].astype('float32')
        if not np.isfinite(result.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Non-finite association features.')
        self.output_names_ = np.asarray(result.columns, dtype=object)
        return result

    def fit(self, X, y):
        self._setup(X)
        self.base_.fit(X, y)
        if self.block == 'rating_context':
            self.rating_encoder_.fit(rating_keys(X), np.asarray(y, dtype=int))
        self._append(self.base_.transform(X.iloc[:1]), X.iloc[:1])
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self._setup(X)
        base = self.base_.fit_transform(X, y)
        encoded = (self.rating_encoder_.fit_transform(rating_keys(X), np.asarray(y, dtype=int))
                   if self.block == 'rating_context' else None)
        return self._append(base, X, encoded)

    def transform(self, X):
        check_is_fitted(self, 'output_names_')
        return self._append(self.base_.transform(X), X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_names_')
        return self.output_names_.copy()


def diagnose(train, folds, reference_dir, probabilities, destination):
    X, y = train.drop(columns=['id', TARGET]), train[TARGET].astype(int).to_numpy()
    curves, gains = [], []
    for fold in sorted(folds.fold.unique()):
        pipeline = joblib.load(reference_dir / f'fold_{fold}.joblib')
        model = pipeline.named_steps['model']
        mask = folds.fold.eq(fold).to_numpy()
        transformed = pipeline.named_steps['preprocessor'].transform(X.loc[mask])
        maximum = model.n_estimators_
        for trees in sorted({min(n, maximum) for n in [100, 250, 400, maximum]}):
            p = model.predict_proba(transformed, num_iteration=trees)[:, 1]
            curves.append({'fold':int(fold), 'trees':trees, 'auc':roc_auc_score(y[mask], p)})
        gains.append(pd.Series(model.booster_.feature_importance(importance_type='gain'),
                               index=model.booster_.feature_name()))
    pd.DataFrame(curves).to_csv(destination/'iteration_curve.csv',index=False)
    pd.concat(gains,axis=1).mean(axis=1).sort_values(ascending=False).rename('mean_gain').to_csv(destination/'feature_gain.csv')
    support = train['Flight Distance'].map(train['Flight Distance'].value_counts())
    groups = {'Class':train['Class'], 'Travel':train['Type of Travel'], 'Customer':train['Customer Type'],
              'WiFi':train['Inflight wifi service'], 'Boarding':train['Online boarding'],
              'Distance_support':pd.cut(support,[0,25,100,300,np.inf],labels=['1-25','26-100','101-300','301+'])}
    rows=[]
    for name, values in groups.items():
        for group in values.dropna().unique():
            mask=values.eq(group).to_numpy()
            rows.append({'feature':name,'group':str(group),
                         'mean_residual':float((y[mask]-probabilities[mask]).mean()),
                         **metrics(y[mask],probabilities[mask])})
    pd.DataFrame(rows).to_csv(destination/'residual_groups.csv',index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/lgbm_exploration')
    parser.add_argument('--confirmation-seed', type=int, default=2030)
    parser.add_argument('--min-gain', type=float, default=0.00005)
    args = parser.parse_args(argv)
    if args.confirmation_seed in {42, 2026, 2027, 2028, 2029}:
        parser.error('Use a new confirmation seed.')
    if args.min_gain < 0:
        parser.error('--min-gain must be nonnegative')
    train, test, submission = [pd.read_csv(ROOT / 'data' / name) for name in
                               ['train.csv', 'test.csv', 'sample_submission.csv']]
    folds = pd.read_csv(ROOT / 'artifacts/cv/folds.csv')
    validate_inputs(train, test, submission)
    validate_folds(train, folds)
    current = json.loads((ROOT / 'artifacts/selected_model.json').read_text())
    reference_dir = Path(current['pipeline_directory'])
    prototype = joblib.load(reference_dir / 'fold_0.joblib')
    base = clone(prototype.named_steps['preprocessor'])
    parameters = prototype.named_steps['model'].get_params()
    if not isinstance(base, TargetStatisticsPreprocessor):
        raise ValueError('This experiment requires the current target-statistics baseline; use its saved selection.')
    reference_p = load_predictions(train, folds, reference_dir.parent, [reference_dir.name])[reference_dir.name]
    reference_scores = pd.Series({int(f): roc_auc_score(train[TARGET][folds.fold.eq(f)], reference_p[folds.fold.eq(f)])
                                  for f in sorted(folds.fold.unique())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnose(train, folds, reference_dir, reference_p, args.output_dir)
    source_hash = {}
    for path in [ROOT / 'data/train.csv', ROOT / 'data/test.csv', ROOT / 'artifacts/cv/folds.csv']:
        with path.open('rb') as source:
            source_hash[path.name] = hashlib.file_digest(source, 'sha256').hexdigest()
    config = {'reference_directory':str(reference_dir), 'parameters':parameters,
              'base_preprocessor':base.get_params(), 'confirmation_seed':args.confirmation_seed,
              'minimum_adoption_gain':args.min_gain, 'source_sha256':source_hash}
    (args.output_dir / 'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    specs = {
        'trees_1000': (parameters | {'n_estimators':1000}, partial(clone, base)),
        'leaves_31': (parameters | {'num_leaves':31}, partial(clone, base)),
        'bins_511': (parameters | {'max_bin':511}, partial(clone, base)),
        'te_smoothing_auto': (parameters, partial(TargetStatisticsPreprocessor,
                                                 **(base.get_params() | {'smooth':'auto'}))),
        'route_profiles': (parameters, partial(AssociationPreprocessor, base=base, block='route_profile')),
        'rating_context_te': (parameters, partial(AssociationPreprocessor, base=base, block='rating_context')),
    }
    rows = []
    for name, (params, factory) in specs.items():
        result = run_experiment(train,folds,(),params,args.output_dir/name,reference_scores,
                                preprocessor_factory=factory)
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir/'summary.csv',index=False)
    ranking = pd.DataFrame(rows).sort_values('mean_roc_auc',ascending=False)
    ranking.to_csv(args.output_dir/'summary.csv',index=False)
    eligible = [r for r in rows if r['delta_auc']>0 and r['win_folds']>=3]
    confirmation_rows = []
    adopted = False
    chosen = 'reference'
    if eligible:
        candidate_name = max(eligible,key=lambda r:r['mean_roc_auc'])['experiment']
        params,factory = specs[candidate_name]
        fresh = make_folds(train,n_splits=folds.fold.nunique(),seed=args.confirmation_seed)
        fresh.to_csv(args.output_dir/'confirmation_folds.csv',index=False)
        control = run_experiment(train,fresh,(),parameters,args.output_dir/'confirmation_reference',None,
                                 preprocessor_factory=partial(clone,base))
        control_scores = pd.read_csv(args.output_dir/'confirmation_reference/fold_scores.csv').set_index('fold').roc_auc
        candidate = run_experiment(train,fresh,(),params,args.output_dir/'confirmation_candidate',control_scores,
                                   preprocessor_factory=factory)
        confirmation_rows=[control,candidate]
        pd.DataFrame(confirmation_rows).to_csv(args.output_dir/'confirmation.csv',index=False)
        original = next(r for r in rows if r['experiment']==candidate_name)
        adopted=(original['delta_auc']>=args.min_gain and candidate['delta_auc']>=args.min_gain
                 and candidate['win_folds']>=3)
        if adopted:
            chosen=candidate_name
            selected_dir=args.output_dir/chosen
            probability=np.zeros(len(test))
            for fold in sorted(folds.fold.unique()):
                pipeline=joblib.load(selected_dir/f'fold_{fold}.joblib')
                probability+=positive_probabilities(pipeline,test.drop(columns='id'))/folds.fold.nunique()
            output=submission.copy()
            output[TARGET]=probability
            if not output[TARGET].between(0,1).all():
                raise ValueError('Invalid selected predictions.')
            output.to_csv(args.output_dir/'submission.csv',index=False)
            shutil.copy2(args.output_dir/'submission.csv',ROOT/'artifacts/submission.csv')
            selection={'model':'lightgbm','fusion':False,'experiment':chosen,
                       'pipeline_directory':str(selected_dir.resolve()),'original_cv':original,
                       'confirmation_seed':args.confirmation_seed,'confirmation_candidate':candidate,
                       'confirmation_reference':control}
            (ROOT/'artifacts/selected_model.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8')
    result={'adopted':bool(adopted),'chosen':chosen,'reference_directory':str(reference_dir),
            'minimum_adoption_gain':args.min_gain,'confirmation':confirmation_rows}
    (args.output_dir/'selection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_report(ranking,reference_scores,confirmation_rows,result,args.output_dir)
    print(ranking.to_string(index=False),flush=True)
    print(f'Adopted: {adopted}; chosen: {chosen}',flush=True)


def write_report(ranking,reference_scores,confirmation,result,destination):
    import matplotlib.pyplot as plt
    diagnostic_sections=[]
    if (destination/'iteration_curve.csv').exists():
        curve=pd.read_csv(destination/'iteration_curve.csv').groupby('trees').auc.mean().reset_index(name='mean_validation_auc')
        diagnostic_sections += [markdown_table(curve)]
        lookup=curve.set_index('trees').mean_validation_auc
        if 400 in lookup and 500 in lookup:
            diagnostic_sections += [f'在已训练模型上截取前 400/500 棵树，平均验证 AUC 的差为 {lookup[500]-lookup[400]:+.8f}。']
    if (destination/'residual_groups.csv').exists():
        residual=pd.read_csv(destination/'residual_groups.csv')
        support=residual[residual.feature.eq('Distance_support')]
        diagnostic_sections += ['按航程重复次数分组的 OOF 残差：',
                                markdown_table(support[['group','n','mean_residual','roc_auc']]),
                                'mean_residual = 实际标签减预测概率；正值表示该组平均概率偏低。频次分组用于诊断；实验特征的频次和画像均在每折训练部分重新计算。']
    sections=['# LightGBM 与数据关联探索',
        f'基准为目标统计单模型，原五折平均 AUC {reference_scores.mean():.8f}。所有实验沿用原折分配，预处理仅在训练部分拟合。',
        '## 初始诊断',
        '当前模型的验证轮数曲线、特征增益和分组残差分别保存在 iteration_curve.csv、feature_gain.csv、residual_groups.csv。组内漏报率结合满意率和概率残差解释。',
        *diagnostic_sections,
        '## 控制实验',
        '- trees_1000：仅将树数从 500 增至 1000。\n- leaves_31：仅将叶子数从 63 降至 31。\n- bins_511：将数值分箱上限从默认 255 增至 511，检查连续目标统计的切分分辨率。\n- te_smoothing_auto：将六个目标统计的固定平滑改为经验贝叶斯平滑。\n- route_profiles：增加按航程分组的 14 项平滑画像均值、13 项评分相对画像偏差、六个目标统计键的训练频次。\n- rating_context_te：增加 13 项评分分别与舱位、出行类型、客户类型组合的 39 个内层交叉拟合目标编码。',
        markdown_table(ranking[['experiment','features','mean_roc_auc','std_roc_auc','delta_auc','win_folds','log_loss']]),
        '## 复核与采用']
    if confirmation:
        sections += [markdown_table(pd.DataFrame(confirmation)[['experiment','mean_roc_auc','delta_auc','win_folds','oof_roc_auc']])]
    sections += [f'原划分中有平均增益且至少三折领先的最佳候选接受新五折复核。考虑额外特征和复杂度，采用要求两套划分的平均 AUC 增益均至少为 {result["minimum_adoption_gain"]:.8f}，且至少三折领先。',
        f'当前采用：{result["chosen"]}。' + ('已更新 artifacts/selected_model.json 和 artifacts/submission.csv。' if result['adopted'] else '当前模型和提交文件保持原方案。'),
        '![消融比较](comparison.png)',
        '参考：[LightGBM 参数说明](https://lightgbm.readthedocs.io/en/latest/Parameters.html)。',
        '复现：`python main.py explore`。']
    (destination/'REPORT.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    data=ranking.sort_values('delta_auc')
    ax.barh(data.experiment,data.delta_auc,color=['#2563a6' if x>0 else '#cb7527' for x in data.delta_auc])
    ax.axvline(0,color='#444444',lw=1)
    ax.set(title='LightGBM exploration vs target-statistics baseline',xlabel='Mean paired fold ROC-AUC delta')
    ax.grid(axis='x',alpha=.2)
    fig.savefig(destination/'comparison.png',dpi=160)
    plt.close(fig)
