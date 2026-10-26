# Paper: *Where the Gains Come From*

LaTeX source of the study report for Kaggle Playground S6E10, plus the scripts that generate
every figure. The paper is self-contained: all numbers come from the experiment artefacts in
`artifacts/` and from `analysis/OPTIMIZATION_REVIEW.md`; no value is estimated by hand.

## Build

```bat
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or, in one step, `latexmk -pdf main.tex`. TeX Live 2025 (pdflatex + bibtex) is sufficient; no
extra fonts or packages are needed beyond `newtx`, `tikz`, `booktabs` and `natbib`.

## Layout

| Path | Contents |
| --- | --- |
| `main.tex` | preamble, abstract, introduction, related work, TikZ pipeline figure (Fig. 1), section includes |
| `sections/02_data.tex` | data and audit (Fig. 2) |
| `sections/03_protocol.tex` | protocol, paired inference, single-model ceiling (Fig. 3, Table 1) |
| `sections/05_families.tex` | member table, blends (Fig. 4, Table 2), rejected interventions (Fig. 5, Tables 3–4) |
| `sections/07_lb.tex` | leaderboard analysis (Fig. 6) |
| `sections/08_discussion.tex` | discussion, limitations, reproducibility, references |
| `references.bib` | every entry verified against arXiv, Crossref or the DOI resolver --- no citation is written from memory |
| `figures/make_figures.py` | generates `fig_audit`, `fig_progress`, `fig_members`, `fig_ablation`, `fig_rank` (PDF + PNG preview) |

Fig. 1 is drawn in TikZ so that it uses the paper's fonts and stays vector-crisp; the five data
figures are matplotlib output exported as PDF with the Okabe--Ito colour-blind safe palette.
Each figure's PDF is what LaTeX includes; the PNG next to it is a preview for reading on GitHub.

The bibliography was checked with throwaway scripts that queried the arXiv API, Crossref and the
DOI resolver, and then deleted; every entry in `references.bib` still carries its arXiv id, DOI or
URL so the check can be repeated by hand in a browser.

## Regenerate the figures

```bat
D:\SoftWare\Anaconda\envs\ml\python.exe figures\make_figures.py ^
  ..\artifacts\audit\lb5\playground-series-s6e10-publicleaderboard-*.csv
```

`fig_rank` needs a leaderboard snapshot; the rest use constants copied from the experiment
reports (each figure documents its source in the comment above the data).
