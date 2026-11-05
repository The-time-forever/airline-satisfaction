# Paper: *How the Top Ten Solved Playground S6E10*

Companion report to `paper/` (*Where the Gains Come From*). It surveys the public solutions
of the ten highest-placed teams in Kaggle Playground S6E10, explains the two solution
families (generative scoring of `p(x|y)` and large logistic stacks over shared OOF), and
quantifies what the final public and private leaderboards rewarded. Every number is taken
from a published write-up or from the leaderboard snapshots in `artifacts/solutions/`; the
sources are cited as blog-style references with their Kaggle discussion URLs.

## Build

```bat
cd paper\solutions
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`latexmk -pdf main.tex` also works once `main.bbl` exists. TeX Live 2025 with `newtx`,
`tikz`, `booktabs` and `natbib` is enough.

## Layout

| Path | Contents |
| --- | --- |
| `main.tex` | preamble, abstract, scope section, section includes |
| `sections/02_board.tex` | final leaderboard (Table 1) and public-vs-private analysis (Fig. 1) |
| `sections/03_generator.tex` | GReaT evidence, the two scoring directions (Fig. 2), single-model table (Table 2) |
| `sections/04_systems.tex` | generative members and stack mechanics (Figs. 3-4, Table 3) |
| `sections/05_evidence.tex` | reported effects of each ingredient (Table 4) |
| `sections/06_discipline.tex` | the shared validation protocol (Fig. 5) |
| `sections/07_discussion.tex` | discussion, limitations, reproduction, appendix A, bibliography |
| `figures.py` | regenerates `figures/fig_ladder`, `fig_pools`, `fig_lb` (PDF + PNG preview) |
| `references.bib` | Kaggle write-ups (URL + discussion id) and the method papers |

## Regenerate the figures

```bat
D:\SoftWare\Anaconda\envs\ml\python.exe paper\solutions\figures.py
```

The script reads `artifacts/solutions/leaderboard_full.csv` (public board) and
`artifacts/solutions/leaderboard_top200.csv` (private top 200). Both are produced by the
Kaggle CLI; see section 9 of the report for the exact commands.
