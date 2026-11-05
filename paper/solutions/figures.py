"""Figures for ``paper/solutions`` (vector PDF + PNG preview, Okabe--Ito palette).

Every number below is taken from a published solution write-up or from the two
leaderboard snapshots in ``artifacts/solutions``; the comment above each block names
the source. The TikZ figures (generation/scoring directions, protocol) live in the
LaTeX sources, because they are diagrams rather than plots.

Run from the repository root:
    D:\\SoftWare\\Anaconda\\envs\\ml\\python.exe paper/solutions/figures.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
DATA = HERE.parents[1] / 'artifacts' / 'solutions'

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
GREY = '#8C8C8C'


def save(fig, name):
    fig.savefig(OUT / name)
    fig.savefig((OUT / name).with_suffix('.png'), dpi=170)
    plt.close(fig)
    print('wrote', name, flush=True)


def read_private_top(limit=200):
    """Parse ``kaggle competitions leaderboard --show -v`` (teamId,name,date,score)."""
    rows = []
    for line in (DATA / 'leaderboard_top200.csv').read_text(encoding='utf-8').splitlines():
        match = re.match(r'^(\d+),("?)(.+?)\2,([\d\-: .]+),([\d.]+)$', line.strip())
        if match:
            rows.append({'team_id': int(match.group(1)), 'name': match.group(3),
                         'private': float(match.group(5))})
    frame = pd.DataFrame(rows).head(limit)
    frame.insert(0, 'private_rank', range(1, len(frame) + 1))
    return frame

def fig_ladder():
    """Single-model OOF/CV AUC, assembled from the published write-ups.

    Sources: 1st place #754683 (GPT-2 family, TFM table, XGBoost, 245-pair stack),
    4th #754814 (AO-ARM, RealMLP), 7th #754646 (GenPure, own stack),
    11th #754666 (Q-FT), our own artefacts (LightGBM ladder, 8-member blend).
    """
    entries = [
        ('LightGBM baseline (ours)', 0.9586, 'gbdt'),
        ('tuned LightGBM (ours)', 0.960589, 'gbdt'),
        ('tuned XGBoost (1st)', 0.96149, 'gbdt'),
        ('RealMLP (4th)', 0.96140, 'gbdt'),
        ('TabPFN-3.5 (1st)', 0.96115, 'tfm'),
        ('Q-FT, 17 members (11th)', 0.96201, 'transformer'),
        ('AO-ARM (4th)', 0.96201, 'transformer'),
        ('GenPure, one seed (7th)', 0.96119, 'transformer'),
        ('GPT-2, target last (1st)', 0.9610, 'generative'),
        ('GPT-2, target first (1st)', 0.96242, 'generative'),
        ('GPT-2, target anywhere (1st)', 0.96245, 'generative'),
        ('245-pair public stack (1st)', 0.96218, 'stack'),
        ('56-stream own stack (7th)', 0.962287, 'stack'),
        ('8-family blend (ours)', 0.960901, 'stack'),
    ]
    colours = {'gbdt': BLUE, 'tfm': GREEN, 'transformer': PURPLE,
               'generative': ORANGE, 'stack': GREY}
    labels = {'gbdt': 'GBDT / MLP', 'tfm': 'tabular foundation model',
              'transformer': 'tabular transformer', 'generative': 'generative scorer',
              'stack': 'ensemble'}
    entries.sort(key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    y = np.arange(len(entries))
    for yi, (name, value, family) in zip(y, entries):
        ax.plot([0.9582, value], [yi, yi], color=colours[family], linewidth=0.8, alpha=0.55)
        ax.scatter([value], [yi], s=26, color=colours[family], zorder=3,
                   edgecolor='white', linewidth=0.5)
        ax.annotate(f'{value:.5f}', (value, yi), textcoords='offset points', xytext=(5, -2.2),
                    fontsize=5.8, color='#333333')
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _ in entries], fontsize=6.4)
    ax.set_xlim(0.9582, 0.9631)
    ax.set_xlabel('out-of-fold ROC-AUC (five folds, seed 42 unless the team states otherwise)')
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=colour, markersize=4,
                          label=labels[family]) for family, colour in colours.items()]
    ax.legend(handles=handles, frameon=False, loc='lower right', ncol=2,
              handletextpad=0.2, columnspacing=0.8, borderpad=0.2)
    save(fig, 'fig_ladder.pdf')


def fig_pools():
    """Stream counts in the final ensembles, split into own runs and public OOF pairs.

    Sources: 1st #754683 (245 public + 14 own = 259), 4th #754814 (189 own, no public),
    7th #754646 (56 own; slot B adds the public 245-column stack),
    16th #754658 (251 public + 58 own = 312), 25th #754782 (293 public + 6 own),
    29th #754691 (284 public + 80 own = 364).
    """
    rows = [
        ('1st  Chris Deotte', 14, 245, 0.96318),
        ('4th  Ole-Jakob', 189, 0, 0.96304),
        ('7th  UIT-Loc & An', 56, 245, 0.96263),
        ('16th Tuong Trinh', 58, 251, 0.96244),
        ('25th Vedant', 6, 293, 0.96241),
        ('29th Alexdruso', 80, 284, 0.96240),
    ]
    rows.sort(key=lambda row: row[3])
    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    y = np.arange(len(rows))
    own = np.array([row[1] for row in rows])
    public = np.array([row[2] for row in rows])
    ax.barh(y, own, height=0.6, color=BLUE, edgecolor='white', linewidth=0.5,
            label='streams trained by the team')
    ax.barh(y, public, left=own, height=0.6, color=SKY, edgecolor='white', linewidth=0.5,
            label='public OOF/test pairs')
    for yi, (name, n_own, n_public, score) in zip(y, rows):
        ax.text(n_own + n_public + 6, yi, f'{score:.5f}', va='center', fontsize=6.2)
        ax.text(n_own / 2 if n_own > 25 else n_own + 8, yi, str(n_own), va='center',
                ha='center', fontsize=5.8, color='white' if n_own > 25 else '#222222')
        if n_public:
            ax.text(n_own + n_public / 2, yi, str(n_public), va='center', ha='center',
                    fontsize=5.8, color='#12384f')
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=6.6)
    ax.set_xlim(0, 560)
    ax.set_xlabel('columns in the final stacker   (value at the right: private ROC-AUC)')
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc='lower right', handlelength=1.2)
    save(fig, 'fig_pools.pdf')



def fig_lb():
    """Public versus private leaderboard, from the two CLI snapshots."""
    public = pd.read_csv(DATA / 'leaderboard_full.csv')
    private = read_private_top()
    merged = private.merge(public[['TeamId', 'TeamName', 'Rank', 'Score']],
                           left_on='team_id', right_on='TeamId', how='left')
    merged['name'] = merged['TeamName'].fillna(merged['name'])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75),
                             gridspec_kw={'width_ratios': [1.5, 1.0]})
    ax = axes[0]
    ax.scatter(merged['Score'], merged['private'], s=7, color=GREY, alpha=0.55,
               linewidth=0, label='private top 200')
    top = merged.head(10)
    ax.scatter(top['Score'], top['private'], s=22, color=BLUE, zorder=3,
               edgecolor='white', linewidth=0.4, label='private top 10')
    ax.axhline(0.96142, color=GREY, linewidth=0.6, linestyle=':')
    ax.text(0.96355, 0.96145, 'top-200 cut', fontsize=5.4, color='#555555', ha='right')
    label_at = {'Chris Deotte': (0.96288, 0.96321), 'Prior': (0.96288, 0.96309),
                'Christoffer Thimsen': (0.96288, 0.96297)}
    for _, row in top.head(3).iterrows():
        ax.annotate(row['name'], (row['Score'], row['private']),
                    xytext=label_at[row['name']], textcoords='data', fontsize=5.6,
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color='#999999', linewidth=0.4,
                                    shrinkA=1, shrinkB=1))
    ax.scatter([0.96062], [0.96133], marker='*', s=110, color=ORANGE, zorder=4,
               edgecolor='white', linewidth=0.4)
    ax.annotate('our submission (0.96062 / 0.96133)', (0.96062, 0.96133),
                xytext=(10, -5), textcoords='offset points', fontsize=5.8,
                color=VERMILLION, va='top')
    ax.annotate('public #1--#5: 0.9632--0.9647\nfour of them miss the top 200',
                (0.96145, 0.96156), fontsize=5.6, color=VERMILLION, ha='left')
    ax.set_xlim(0.9604, 0.96365)
    ax.set_ylim(0.96116, 0.96335)
    ax.set_xlabel('public ROC-AUC')
    ax.set_ylabel('private ROC-AUC')
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc='upper left', handletextpad=0.3, borderpad=0.2)

    ax = axes[1]
    gap = (merged['private'] - merged['Score']).to_numpy() * 1e4
    ax.hist(gap, bins=np.arange(-2.5, 11.5, 1.0), color=BLUE, edgecolor='white',
            linewidth=0.5)
    ax.axvline(0, color='#444444', linewidth=0.7)
    ax.axvline(np.mean(gap), color=VERMILLION, linewidth=0.9, linestyle='--')
    ax.text(np.mean(gap) + 0.55, 138, f'mean\n+{np.mean(gap):.1f}', fontsize=5.8,
            color=VERMILLION, va='top')
    ax.set_ylim(0, 145)
    ax.set_xlabel('private $-$ public ($\\times10^{-4}$)')
    ax.set_ylabel('teams (private top 200)')
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    save(fig, 'fig_lb.pdf')


if __name__ == '__main__':
    fig_ladder()
    fig_pools()
    fig_lb()

