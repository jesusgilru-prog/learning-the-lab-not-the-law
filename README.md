# Learning the Lab, Not the Law — reproducibility repository

Code and data for the paper *"Learning the Lab, Not the Law: A Verdict-Based
Audit Protocol for Structure-Level Domain Confounding in Symbolic Scientific
Discovery"* (submitted to *Machine Learning and Knowledge Extraction*, MDPI).

This repository contains the dataset, the audit implementation (Algorithms 1
and 2 in the paper), and the scripts that regenerate every number and figure
reported in the paper and its supplement.

## Contents

- `data/` — the 114-point cross-facility windage dataset
  (`cross_rotor_dataset.csv`, one row per measurement, with per-row
  `data_origin` and `error_pct` provenance fields — see Table A1 in the
  paper's appendix for source-level licensing) plus two auxiliary
  hand-digitized CSVs used in a spin-down-friction cross-check
  (`liu_fig6_spindown.csv`, `liu_fig7a_redigitized.csv`).
- `sr_engine/` — the audit engine: Buckingham-Pi dimensional analysis
  (`buckingham_pi.py`), the class-SR fitting procedure (`class_sr.py`),
  the conformal-prediction implementations (split/Mondrian/normalized,
  `conformal.py`; the adaptive-clustering ablation, `adaptive_conformal.py`),
  Bayesian structure selection (`bayesian_structural_sr.py`), and the
  honesty-suite (SRSD-Feynman recovery check, noise-robustness sweep) under
  `honesty_tests/`. This is the implementation of Algorithms 1–2.
- `scripts/` — top-level scripts referenced by name in the paper:
  - `remediation_experiments.py` — the S1–S6 structured-form discriminant
    battery (Tests 1–4) and the specificity/power analysis.
  - `run_conformal_prediction.py` — the original (leaky) LOGO conformal
    run; kept for transparency about the leak described in the paper's
    §5.9, superseded by the nested cross-fit scripts below.
  - `analyze_prefactors.py` — the single-group Π-regression analysis
    (Figure 5) and the multi-group regression (Eq. 16).
  - `cp_s5_discriminant.py` — the S5 permutation-null test (Figure 7) and
    the leave-one-facility-out structure-stability check (Figure 10).
  - `compute_pi_groups.py` — derives the four candidate dimensionless
    groups (Π_gap, Π_blockage, Π_aspect,axial, Π_confinement) from the
    raw geometry columns in `data/cross_rotor_dataset.csv`.
- `analysis/` — the verification scripts written during the paper's
  external-review cycle, each independently re-deriving one specific
  reported number, plus their raw output (`*_results.json`,
  `RESULTS_*.md`):
  - `conformal_logo_crossfit.py`, `conformal_logo_nested_crossfit.py`,
    `conformal_logo_nested_crossfit_split.py` — the honest, fully nested
    (outer geometry-fold + inner calibration-split) cross-fit conformal
    re-evaluation (Table 11).
  - `s5_null_calibration_real_support.py` — the support-matched null
    calibration for the S5 finding (§5.7(iv)).
  - `stage0_validation.py`, `design_identifiability_v2.py`,
    `pooled_rank_check.py`, `ddp_rank_analysis.py` — the Stage-0
    design-identifiability screen and its synthetic validation
    (Proposition 1, §5.1–5.2).
  - `threshold_sensitivity_30pct.py` — the Test-2 physical-threshold
    sensitivity sweep (Table 7).
  - `build_corrected_dataset.py`, `extract_liu_fig6_spindown.py`,
    `redigitize_liu_fig7a.py`, `regenerate_orphan_numbers.py`,
    `spindown_friction_analysis.py`, `verify_fable.py`, `recompute_logo_conformal.py` —
    supporting data-provenance and cross-checking scripts.
- `supplement.pdf` — the paper's supplementary material (Sections S1–S7
  referenced throughout the main text).

## Reproducing a specific number or figure

Every script is self-contained (`python <script>.py`) and writes a
`run_log_*.txt` and/or `*_results.json` alongside itself; the paper cites
these exact filenames next to the number they produced. Start from
`sr_engine/` for the core audit protocol (Algorithms 1–2), or from
`analysis/` for any specific verification re-run mentioned in a table or
figure caption.

## Requirements

Python 3.10+, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.

## Data licensing

The dataset in `data/cross_rotor_dataset.csv` redistributes only derived
numerical measurements (source, geometry, operating conditions, $C_p$,
$\mathrm{Re}_\Omega$), not copyrighted figures or text, under the
redistribution terms documented per source in Table A1 of the paper
(NASA public domain; CC-BY 4.0; publisher-derived-values-only). Code in
this repository is released under the MIT License (see `LICENSE`).

## Citation

If you use this code or data, please cite the paper (citation details to
be added on acceptance).
