# TurboPress papers

Two self-contained LaTeX write-ups of the TurboPress experiments, plus the
figures and bibliography they share. Every figure and table value is taken
directly from `../results/*.json` (regenerate figures with
`python make_figures.py`), so the papers cannot drift from the measurements.

| file | what it is |
|------|-----------|
| `turbopress_preprint.tex` | Full arXiv/preprint version (single column, `article`). |
| `turbopress_openreview.tex` | ICLR/OpenReview double-blind conference version, with Reproducibility Statement, Ethics Statement, and a paper checklist. |
| `references.bib` | Bibliography. Every entry's title, authors, year, and arXiv id / venue was verified against the source; the four recent preprints (BASE-Q 2506.15689, PiSO 2606.10890, QAM-W 2605.26339, OrbitQuant 2607.02461) had their authors checked on arXiv. |
| `make_figures.py` | Regenerates all eight figures (PDF + PNG) strictly from the result JSONs — no hand-typed values. Fig 7 reads `results/ldlq_micro.json` (regenerate with `python -m turbopress.ldlq_micro`); Fig 8 is the equilibration-exponent ablation (Proposition 1) on Qwen3-0.6B/1.7B. |
| `figures/` | Vector PDFs used by both papers (and PNG previews). |

## Build

Requires a LaTeX distribution (TeX Live / MiKTeX) with `pdflatex` + `bibtex`,
or just upload the `paper/` folder to Overleaf. The single-binary
[Tectonic](https://tectonic-typesetting.github.io) engine also builds both
papers with one command each (`tectonic turbopress_preprint.tex`) — it runs
BibTeX and the reruns automatically, which is how the checked-in PDFs were
last built.

```bash
python make_figures.py          # (re)build figures from results/
make                            # builds both PDFs, or:
pdflatex turbopress_preprint && bibtex turbopress_preprint && \
  pdflatex turbopress_preprint && pdflatex turbopress_preprint
```

## Notes for the authors (read before submitting)

- **Fill in the author block** in `turbopress_preprint.tex` (the OpenReview
  version stays anonymous for double-blind review).
- **For the venue style**: `turbopress_openreview.tex` uses plain `article` so it
  compiles anywhere. Swap in the official `iclr20xx_conference.sty` (or the
  venue's class) and keep the section content, statements, and checklist.
- **Scope honesty is deliberate.** Both papers explicitly *decline* a
  state-of-the-art claim and list the scale (≤0.6B) and missing-baseline
  (WikiText-2 / zero-shot vs released GPTQ/AWQ/QuIP#/QTIP) gaps. Do not remove
  the limitations — the equilibration exponent and the QJL negative result are
  the load-bearing contributions, and the isotropic-error assumptions behind
  Proposition 1 must stay stated. Verify BASE-Q §3.4/Eq. 12 in the PDF before
  finalizing the positioning against it.
