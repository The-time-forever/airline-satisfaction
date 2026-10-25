# Airline Satisfaction

当前方案及提交文件由 `artifacts/selected_model.json` 和 `artifacts/submission.csv` 记录。融合默认停用。

Kaggle [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10/overview) 使用 ROC-AUC 评价满意概率预测。

将 `train.csv`、`test.csv`、`sample_submission.csv` 放入 `data/`，运行预处理：

```bat
conda activate ml
cd /d D:\Document\airline-satisfaction
python -m pip install -r requirements.txt
python main.py
```

输出为 `artifacts/preprocessing.joblib`，包含固定种子 42 的 80%/20% 分层划分、标签、行位置、ID、已拟合预处理器和两套特征：

- `tree`：训练部分中位数填充、到达延误缺失指示、统一类别词表，供 LightGBM 使用。
- `linear`：训练部分中位数填充、缺失指示、标准化和类别独热编码，输出稀疏矩阵。

两套输入均移除 `id`，保留评分 0 和长尾样本。所有学习参数仅使用训练部分。测试顺序与提交样例一致。

```python
import joblib

data = joblib.load('artifacts/preprocessing.joblib')
X_train = data['tree']['X_fit']
X_valid = data['tree']['X_valid']
y_train = data['y_fit']
y_valid = data['y_valid']
```

交叉验证时，每折重新拟合预处理器；最终全量训练时重新使用完整训练集拟合。现有产物用于当前训练/验证划分。

EDA：`python analysis/eda.py`，报告见 [analysis/EDA.md](analysis/EDA.md)。

数据审计：`python analysis/audit.py`，报告见 [analysis/AUDIT.md](analysis/AUDIT.md)。该脚本额外统计离散签名的标签噪声、评分 0 的语义、对抗验证，并可用 `--original <公开原始调查 csv>` 对照生成过程保留/抹掉了哪些结构。优化空间与路线图见 [analysis/OPTIMIZATION_REVIEW.md](analysis/OPTIMIZATION_REVIEW.md)。

运行 5 折分层交叉验证的数据准备（shuffle=True，seed=42）：

```powershell
python main.py cv
```

使用完整训练集建立固定折分配，每条样本恰好作为一次验证样本。逐折重新拟合并检查两套预处理器，输出 `artifacts/cv/folds.csv`（原始行序的 id、fold，折编号 0–4）和 `artifacts/cv/preprocessors.joblib`（折分配、源文件摘要、每折规模及预处理器）。此步骤完成交叉验证数据准备；ROC-AUC 在模型训练阶段计算。

后续训练使用以下接口，每折新建模型；将验证概率写回 `valid_rows` 对应位置，即可得到全训练集的 OOF 概率：

```python
import numpy as np
import pandas as pd
from airline_satisfaction.cross_validation import iter_preprocessed_folds

train = pd.read_csv('data/train.csv')
folds = pd.read_csv('artifacts/cv/folds.csv')
oof = np.full(len(train), np.nan)
for fold in iter_preprocessed_folds(train, folds, branch='tree'):
    # Create and fit a new model with fold['X_fit'] and fold['y_fit'].
    # Store positive probabilities at oof[fold['valid_rows']].
    print(fold['fold'], fold['X_fit'].shape, fold['X_valid'].shape)
```

`branch='linear'` 使用线性模型预处理。两条路线共用折分配，便于比较 ROC-AUC；每折训练完成后汇总各折分数均值、标准差及整体 OOF ROC-AUC。

模型确定后，可直接调用 `cross_validate_model(estimator, train, folds, branch='tree')`。它逐折克隆并训练模型，返回 `fold_scores`、`mean_roc_auc`、`std_roc_auc`（5 折总体标准差，ddof=0）、`oof_roc_auc` 和按原始 ID 排列的 `oof` 概率表。模型需支持 `predict_proba`；随机种子在模型参数中设置。

训练五个基线（常数概率、逻辑回归、LightGBM、XGBoost、CatBoost）：

```powershell
python main.py baseline
```

使用已保存的 `artifacts/cv/folds.csv`，每折独立拟合预处理和模型。LightGBM 固定 500 棵树，逻辑回归最多 1000 次迭代。三个提升树模型默认使用 4 个 CPU 线程，可通过 `--threads` 调整。

增量加入 XGBoost 和 CatBoost，保留已有模型结果：

```powershell
python main.py baseline --models xgboost catboost
```

两者均固定 500 轮、学习率 0.05、深度 6、CPU 4 线程（`--threads` 可调整）。XGBoost 使用 hist 和原生类别支持；CatBoost 显式声明四个类别列，并将缺失/未知类别转为字符串占位符。预处理仍在每折训练部分拟合。已有产物的源文件摘要及种子需一致，新数据或新划分使用新的 `--output-dir`。

输出位于 `artifacts/baseline/`：`summary.csv` 包含交叉验证均值、总体标准差、OOF ROC-AUC 和 Log Loss；`fold_scores.csv` 包含逐折分数；`config.json` 记录参数和输入文件摘要。各模型目录保存 5 个完整预测 Pipeline、`oof.csv` 和 `test_predictions.csv`。按平均折 ROC-AUC 选择最佳基线，其 5 折概率平均输出为 `submission.csv`，ID 顺序与提交样例一致。

评估已有模型的完整 OOF 预测：

```powershell
python main.py evaluate
```

输出 `artifacts/evaluation/REPORT.md`、整体/逐折/分组指标 CSV，以及 ROC、PR、校准、混淆矩阵和逐折 AUC 图。指标包含 ROC-AUC、AP、Log Loss、Brier、ECE，以及阈值 0.5 下的 Accuracy、Precision、Recall、F1、误报率和漏报率。评估会校验源文件摘要、OOF 行序、标签及原有 AUC 分数。

运行特征工程消融实验：

```powershell
python main.py features
```

保持 LightGBM 基线参数不变，在相同 5 折上分别测试服务评分汇总、场景类别交互、零评分及其交互、延误/航程变换，并测试特征组合。每折独立拟合预处理。原划分平均 AUC 提升且至少 3 折领先的最佳候选，再与原始特征在种子 2026 的新 5 折上配对复核；两套划分均满足该规则时采用候选，否则保留基线。

输出位于 `artifacts/features/`：`REPORT.md`、`summary.csv`、消融图、各实验的折分数/OOF/模型，以及最终 `selection.json` 和 `submission.csv`。实验结果与原有 `artifacts/baseline/` 分开保存。

使用 Optuna 优化已选特征下的 LightGBM：

```powershell
python main.py tune --trials 16
```

固定原 5 折，顺序执行 TPE 搜索，至少完成三折后可剪枝。16 是目标试验总数，包含对照和剪枝试验；对照复用既有 OOF 分数并显式标记。模型使用原有 4 个 CPU 线程。每折预处理独立拟合并在试验间缓存，搜索树数、学习率、叶子数、最小叶节点样本数、行/列采样与正则化。

实验保存在 `artifacts/tuning/study.db`，每次试验结束更新 `trials.csv` 和 `trial_folds.csv`。最佳完整试验会重新训练保存模型，并在种子 2027 的新 5 折上与原参数配对复核。两套划分都提升平均 AUC 且至少 3 折领先时采用调优参数。最终报告、参数、OOF、模型和 `submission.csv` 位于同一目录。

继续同一搜索至 32 次总试验：`python main.py tune --trials 32`。输入/特征代码/搜索空间变化时使用新的 `--output-dir`。

比较并复核模型融合：

```powershell
python main.py ensemble
```

读取调优 LightGBM、XGBoost、CatBoost、逻辑回归的 OOF 概率，比较单模型、等权概率平均、三树非负权重网格、Logit 平均和排名平均。原五折用于选方案和权重；最佳候选冻结后，在种子 2028 的新五折上重新训练所需基模型，并与调优 LightGBM 比较。两套划分均平均 AUC 提升且至少三折领先时标记为有效候选。当前采用调优后的单模型；融合默认仅作实验，显式增加 `--adopt-blend` 后才采用通过复核的融合。

输出 `artifacts/ensemble/REPORT.md`、71 个方案的 `candidates.csv`、逐折分数、冻结权重、复核模型及预测、残差相关性、`selection.json` 和 `submission.csv`。排名法按每折/整批测试集计算百分位，输出排序分数；概率法和 Logit 法输出概率。

验证目标统计方向：`python main.py targetstats`。在当前调优单模型上增加航程、年龄及其场景组合的六个平滑目标编码，外层沿用原五折，训练行编码使用内层五折交叉拟合，验证行只使用外层训练折映射。结果保存在 `artifacts/target_statistics/`，并在种子 2029 新五折上配对复核；两套划分均提升平均 AUC 且至少三折领先时采用新特征，更新当前方案及提交文件。

进一步探索当前目标统计模型：`python main.py explore`。固定原五折比较树数、叶子数、目标编码平滑、航程画像/频次和评分×场景目标统计，最佳有增益候选在种子 2030 五折配对复核。考虑复杂度，默认两套划分平均增益均至少 0.00005 且至少三折领先才更新当前模型。产物位于 `artifacts/lgbm_exploration/`。

最终训练协议（折数 × 随机种子）：`python main.py final --seeds 42 1042 2042`。沿用已采用的特征族与调优参数，逐折独立拟合预处理，每折训练多个种子并把所有折、所有种子的测试概率平均。写入 `artifacts/final/`（`REPORT.md`、`summary.json`、`oof.csv`、`models/fold_<k>.joblib`、`submission.csv`）。只有显式加 `--adopt` 才会覆盖 `artifacts/submission.csv` 与 `artifacts/selected_model.json`；保存的每折 Pipeline 是第 1 个种子的模型，供诊断复现。同一探针上的对照显示 5 折 × 3 种子比 5 折 × 1 种子 +5.1e-5，而 10 折 × 3 种子与 5 折 × 3 种子持平（0.961355 vs 0.961356），因此默认保持 5 折。
