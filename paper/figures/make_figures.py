"""Generate the data figures for the paper (vector PDF, serif, colour-blind safe)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY = (
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9')


def save(fig, name):
    fig.savefig(OUT / name)
    fig.savefig((OUT / name).with_suffix('.png'), dpi=170)
    plt.close(fig)
    print('wrote', name, flush=True)


def fig_audit():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    labels = ['17 discrete', '+ Age', '13 ratings', 'distance']
    multi = [48.96, 22.72, 63.66, 99.98]
    conflict = [19.76, 4.06, 30.43, 99.54]
    flip = [5.66, 5.46, 6.73, 29.13]
    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, multi, width, label='in repeated groups', color=SKY,
           edgecolor='white', linewidth=0.4)
    ax.bar(x, conflict, width, label='conflicting labels', color=ORANGE,
           edgecolor='white', linewidth=0.4)
    ax.bar(x + width, flip, width, label='majority-flip noise', color=VERMILLION,
           edgecolor='white', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('share of training rows (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=1.0)

    ax = axes[1]
    curves = {
        'Inflight wifi': ([0.887, 0.391, 0.270, 0.276, 0.622, 0.950], '-', 'o', BLUE),
        'Online boarding': ([0.620, 0.105, 0.085, 0.128, 0.642, 0.875], '--', 's', GREEN),
        'Ease of booking': ([0.708, 0.415, 0.314, 0.320, 0.556, 0.754], '-.', '^', ORANGE),
        'Gate location': ([0.543, 0.514, 0.465, 0.337, 0.411, 0.625], ':', 'D', PURPLE),
    }
    for name, (values, style, marker, colour) in curves.items():
        ax.plot(range(6), values, style, marker=marker, markersize=2.6, color=colour,
                label=name, linewidth=1.0)
    ax.set_xlabel('stored rating value')
    ax.set_ylabel('satisfaction rate')
    ax.set_xticks(range(6))
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              handlelength=1.8, columnspacing=1.0)
    save(fig, 'fig_audit.pdf')


def fig_progress():
    stages = ['LGB', '+inter.', 'tune', '+TE', 'auto', '5$\\times$3', '3 fam', '7 fam',
              '+FT', 'FT$\\times$3']
    cv = [0.958618, 0.958770, 0.959022, 0.960369, 0.960533, 0.960589, 0.960696,
          0.960800, 0.960901, 0.960942]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    x = np.arange(len(stages))
    ax.plot(x, cv, marker='o', markersize=3.2, color=BLUE)
    ax.annotate(f'{cv[0]:.4f}', (0, cv[0]), textcoords='offset points', xytext=(6, -9),
                fontsize=6, color='#333333')
    ax.annotate(f'{cv[3]:.4f}', (3, cv[3]), textcoords='offset points', xytext=(-16, 7),
                fontsize=6, color='#333333')
    ax.annotate(f'{cv[-1]:.4f}', (x[-1], cv[-1]), textcoords='offset points', xytext=(-24, 7),
                fontsize=6, color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=0, fontsize=6.2)
    ax.set_ylabel('5-fold mean ROC-AUC')
    ax.set_ylim(0.9580, 0.9614)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    ax = axes[1]
    cv_points = [0.960533, 0.960589, 0.960696, 0.960800, 0.960901]
    lb_points = [0.96027, 0.96028, 0.96034, 0.96047, 0.96062]
    names = ['5$\\times$1', '5$\\times$3', '3 fam.', '7 fam.', '+ FT']
    ax.plot(cv_points, lb_points, color=ORANGE, linewidth=0.8, alpha=0.5, zorder=2)
    ax.scatter(cv_points, lb_points, s=24, color=ORANGE, edgecolor='white', linewidth=0.5,
               zorder=3)
    offsets = [(5, -10), (5, 4), (5, 4), (5, 4), (-6, -12)]
    for point, label, offset in zip(zip(cv_points, lb_points), names, offsets):
        ax.annotate(label, point, textcoords='offset points', xytext=offset, fontsize=6.2)
    ax.set_xlabel('cross-validated ROC-AUC')
    ax.set_ylabel('public leaderboard ROC-AUC')
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_xlim(0.96047, 0.96098)
    ax.set_ylim(0.96021, 0.96070)
    ax.text(0.96072, 0.960245, 'CV$\\to$LB transfer per step:\n0.18, 0.56, 1.25, 1.50',
            fontsize=6.0, color='#555555', ha='left', va='bottom')
    save(fig, 'fig_progress.pdf')



def fig_members():
    members = [
        ("linear-tree", 0.960474, 0.9983, 0.179, "tree"),
        ("LightGBM", 0.960580, 1.0000, 0.012, "tree"),
        ("DART", 0.960326, 0.9978, 0.104, "tree"),
        ("CatBoost", 0.960219, 0.9937, 0.207, "tree"),
        ("XGBoost", 0.960028, 0.9954, 0.155, "tree"),
        ("MLP (encoded)", 0.959126, 0.9841, 0.076, "neural"),
        ("FT-Transformer", 0.958548, 0.9816, 0.249, "neural"),
        ("MLP (raw)", 0.958703, 0.9807, 0.018, "neural"),
        ("teacher", 0.954326, 0.9468, 0.003, "external"),
    ]
    colours = {"tree": BLUE, "neural": ORANGE, "external": PURPLE}
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.0), sharey=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.55})
    y = np.arange(len(members))[::-1]
    ax = axes[0]
    for position, (name, auc, correlation, weight, family) in zip(y, members):
        ax.plot([0.9530, auc], [position, position], color="#CCCCCC", linewidth=0.7, zorder=1)
        ax.scatter([auc], [position], s=22, color=colours[family], zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.text(auc + 0.00035, position, f"{correlation:.3f}", fontsize=5.6, va="center",
                color="#666666")
    ax.set_xlim(0.9530, 0.9622)
    ax.set_ylabel("member")
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in members], fontsize=6.2)
    ax.set_xlabel("member ROC-AUC (5-fold mean)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax = axes[1]
    for position, (name, auc, correlation, weight, family) in zip(y, members):
        ax.barh(position, weight, height=0.55, color=colours[family], alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.text(weight + 0.006, position, f"{weight:.3f}", fontsize=5.6, va="center",
                color="#444444")
    ax.set_xlim(0, 0.30)
    ax.tick_params(labelleft=False)
    ax.set_xlabel("weight in the 8-member logit blend")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=4,
                          label=family) for family, colour in colours.items()]
    axes[0].legend(handles=handles, frameon=False, loc="lower right", handletextpad=0.2,
                   borderpad=0.2, labelspacing=0.25, fontsize=6.0)
    save(fig, "fig_members.pdf")


def fig_ablation():
    entries = [
        ('cross-fitted target statistics', 13.5, True),
        ('3-family blend', 1.07, True),
        ('5 new families (batch)', 1.00, True),
        ('FT-Transformer member', 1.00, True),
        ('multi-view auxiliary features', 0.57, False),
        ('seeds per fold', 0.55, False),
        ('best stacker (LightGBM meta)', 0.26, False),
        ('ten folds instead of five', 0.14, False),
        ('external teacher member', 0.00, False),
        ('max\\_bin $=4095$', -0.22, False),
        ('pseudo-label member', -0.41, False),
        ('delay / distance transforms', -0.68, False),
        ('extra distance encodings', -1.04, False),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    y = np.arange(len(entries))
    values = [value for _, value, _ in entries]
    colours = [BLUE if accepted else '#AEADAD' for _, _, accepted in entries]
    ax.barh(y, values, color=colours, height=0.62, edgecolor='white', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _ in entries], fontsize=6.0)
    ax.invert_yaxis()
    ax.axvline(0, color='#444444', linewidth=0.7)
    ax.axvline(2.2, color='#666666', linewidth=0.6, linestyle=':')
    ax.set_xlabel('$\\Delta$ ROC-AUC ($\\times 10^{-4}$)')
    ax.set_xlim(-2.2, 16.5)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.text(2.5, 8.4, 'paired noise floor\n$2.2\\times10^{-4}$', fontsize=5.6,
            color='#666666', ha='left', va='center')
    save(fig, 'fig_ablation.pdf')


def fig_rank(leaderboard_csv):
    frame = pd.read_csv(leaderboard_csv)
    scores = frame["Score"].dropna().sort_values(ascending=False).to_numpy()
    ranks = np.arange(1, len(scores) + 1)
    keep = (scores > 0.9595) & (scores < 0.9625)
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.plot(scores[keep], ranks[keep], color=BLUE, linewidth=1.1)
    ours = [(0.96027, "5x1", 8, -2), (0.96034, "3 fam.", 8, -2),
            (0.96047, "7 fam.", 8, -2), (0.96062, "8 fam.", -34, -10)]
    for score, label, dx, dy in ours:
        rank = int((scores > score).sum()) + 1
        ax.scatter([score], [rank], s=22, color=ORANGE, zorder=3, edgecolor="white",
                   linewidth=0.4)
        ax.annotate(label, (score, rank), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.0)

    ax.annotate("", xy=(0.96100, 1105), xytext=(0.96080, 1105),
                arrowprops=dict(arrowstyle="<->", color="#666666", linewidth=0.6))
    ax.text(0.96090, 1115, "2e-4 gives ~100 places", fontsize=5.8,
            color="#666666", ha="center")
    ax.set_xlabel("public leaderboard ROC-AUC")
    ax.set_ylabel("rank (of %d teams)" % len(scores))
    ax.set_xlim(0.9596, 0.9625)
    ax.set_ylim(1050, 1450)
    ax.grid(alpha=0.25, linewidth=0.5)
    save(fig, "fig_rank.pdf")


if __name__ == '__main__':
    import sys
    fig_audit()
    fig_progress()
    fig_members()
    fig_ablation()
    fig_rank(sys.argv[1])
