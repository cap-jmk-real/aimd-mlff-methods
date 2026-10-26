# Manuscript and research notes

This directory holds the methods paper sources and **local-only planning documents** for the `aimd-mlff-methods` project (MLFF-based AIMD for electrochemical systems).

## LaTeX manuscript

Build with `latexmk` (recommended) or `pdflatex` + `bibtex`:

```bash
cd paper
latexmk -pdf main.tex
```

Author: **Julian M. Kleber** (sole author).

## Planning documents (draft, not for publication as-is)

| File | Purpose |
|------|---------|
| [`OUTLINE.md`](OUTLINE.md) | Paper outline: research questions (RQ1–RQ5), hypotheses, section structure, and how experiments map to claims. |
| [`inventory.md`](inventory.md) | Audit of prior AIMD/MLFF work (June 2025 repos), diagnosis of the Verlet unit bug and validation gaps, and engine/package inventory. |
| [`plan.md`](plan.md) | Recommended MLFF workflow, engine constraints, staged experiment schedule (S0–S6), and cloud-agent PR specs for code scaffolding. |

These three files were added as a snapshot on 26 Oct 2026. They inform implementation in `src/` and future experiment configs under `experiments/`; they are **not** wired into the LaTeX build.
