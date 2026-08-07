# Learning the Lab, Not the Law — reproducibility repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21776087.svg)](https://doi.org/10.5281/zenodo.21776087)

Code and data for the paper *"Learning the Lab, Not the Law: A Verdict-Based
Audit Protocol for Structure-Level Domain Confounding in Symbolic Scientific
Discovery"* (submitted to *Algorithms*, MDPI).

This repository contains the dataset, the audit implementation (Algorithms 1
and 2 in the paper), and the scripts that regenerate essentially every number
and figure reported in the paper and its supplement. Run `./run_all.sh` to
regenerate the three Supplement S4/S6 outputs in one shot; see "Reproducing a
specific number or figure" below for everything else, and the one disclosed
exception (a single archived Table 6 value) in that same section.

## Contents

- `data/` — the 114-point cross-facility windage dataset
  (`cross_rotor_dataset.csv`, one row per measurement, with per-row
  `data_origin` and `error_pct` provenance fields — see Table A1 in the
  paper's appendix for source-level licensing) plus two auxiliary
  hand-digitized CSVs used in a spin-down-friction cross-check
  (`liu_fig6_spindown.csv`, `liu_fig7a_redigitized.csv`), and
  `data/processed_checkpoints/` — intermediate results consumed by the
  `analysis/` scripts:
  - `model_comparison_table.csv` — the source of Table 6 (the
    leave-one-facility-out baseline comparison: Class-SR, the global
    power law, the classical correlations, IRM, GroupDRO) and of the
    IRM/GroupDRO rows referenced in Table 7's discussion. Produced by
    `scripts/remediation_experiments.py`'s Experiment 5. **Note on a
    metric that looks similar but is not the same:** `class_sr_results.json`
    in this same folder also has an `r2_loso_cv` field (0.021) that is
    *not* the Table 6 number (−1.001). It is a different, internal
    diagnostic computed by `sr_engine/class_sr.py`'s own
    `class_sr_fit()`: a **leave-one-geometry-out** (12 folds) check that
    substitutes the mean of the training geometries' intercepts for the
    held-out geometry. Table 6 instead evaluates
    **leave-one-facility-out** (4 folds): since every geometry belongs
    to exactly one facility, holding out a facility means every
    held-out row's geometry is entirely unseen, so no intercept proxy
    is available — see `analysis/verify_table6_class_sr_loso_facility.py`
    below.
  - `threshold_sweep_bf.csv` — the full 37-point Bayes-factor threshold
    sweep behind Table 7 (the manuscript table shows selected values).
    Also produced by `scripts/remediation_experiments.py`.
  - `class_sr_results.json`, `prefactor_analysis.json`,
    `conformal_prediction_results.json`, `cross_rotor_dataset_v3.csv`/
    `.parquet` — other intermediate checkpoints read by specific
    `analysis/` scripts (see their own headers for which one uses
    which).
- `sr_engine/` — the audit engine: Buckingham-Pi dimensional analysis
  (`buckingham_pi.py`), the class-SR fitting procedure (`class_sr.py`),
  the conformal-prediction implementations (split/Mondrian/normalized,
  `conformal.py`; the adaptive-clustering ablation, `adaptive_conformal.py`),
  Bayesian structure selection (`bayesian_structural_sr.py`), and the
  honesty-suite (SRSD-Feynman recovery check, noise-robustness sweep) under
  `honesty_tests/`. This is the implementation of Algorithms 1–2.
- `scripts/` — top-level scripts referenced by name in the paper:
  - `remediation_experiments.py` — the S1–S6 structured-form discriminant
    battery (Tests 1–4), the specificity/power analysis, the IRM/GroupDRO
    domain-generalization baselines, and the writer of
    `model_comparison_table.csv` / `threshold_sweep_bf.csv` above. Note:
    this script still has the path constants from the original research
    repository (`data/processed/cross_rotor_dataset_v3.parquet`,
    `results/`) rather than the relative paths used elsewhere in this
    package, because it runs several long, seeded (`SEED=42`) experiments
    together and re-running it fresh risks silently diverging numbers
    already cited elsewhere in the paper if the environment differs even
    slightly; the CSVs it produced are archived above instead of being
    regenerated on clone. If you need to re-run it, point `DATASET` at
    `../data/cross_rotor_dataset.csv` and adjust `RESULTS_DIR`.
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
  - `facility_mape_rmse.py` — per-facility MAPE/RMSE in original $C_p$
    units for the baseline models (Supplement S4). Run via `./run_all.sh`
    or directly; verified to reproduce the supplement's cited figures
    exactly (e.g. Class-SR: 4.8% MAPE on Zheng2024, 22.5% on Vrancik1968).
  - `naive_sr_replication.py` — off-the-shelf, unconstrained symbolic
    regression (via `gplearn`, no hand-picked candidate family) checking
    whether a generic SR search independently rediscovers the S5
    Mach-conditioned structure (Supplement S6). Requires
    `pip install gplearn`. Verified: converges to the simple one-term fit
    with no Mach dependence, exactly as the supplement reports.
  - `structural_confound_power.py` — synthetic detection-power sweep for
    an injected *structural* (Reynolds-exponent) confound, as opposed to
    the prefactor-offset confound already in the main text (Supplement
    S6). Requires `remediation_experiments.py` in the same folder
    (imported for `run_framework_verdict`/`SEED`, not executed). Verified
    to reproduce every row of the supplement's power table exactly.
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
  - `verify_table6_class_sr_loso_facility.py` — an independent
    from-scratch re-implementation of the Table 6 leave-one-facility-out
    evaluation, written to check `model_comparison_table.csv`'s Class-SR
    row against the same profiled-MLE fitting machinery
    (`sr_engine/class_sr.py`) used everywhere else in this repository.
    Honest result (`table6_class_sr_loso_facility_result.json`): it
    reproduces the same qualitative finding (a strongly negative
    held-out R², around −4.4 with a zero-intercept assumption for the
    unseen facility, versus the published −1.001) but does not hit the
    published figure to the digit — the exact historical fitting
    procedure behind `model_comparison_table.csv`'s Class-SR row was not
    preserved as a standalone script (see
    `results/REMEDIATION_RESULTS.md`'s original note, "from prior
    computation," in the source research repository). This is flagged
    here rather than hidden.
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
- `run_all.sh` — regenerates the three Supplement S4/S6 outputs
  (`facility_mape_rmse.py`, `structural_confound_power.py`,
  `naive_sr_replication.py`) in one command.

## Reproducing a specific number or figure

Every script is self-contained (`python <script>.py`) and writes a
`run_log_*.txt` and/or `*_results.json` alongside itself; the paper cites
these exact filenames next to the number they produced. Start from
`sr_engine/` for the core audit protocol (Algorithms 1–2), or from
`analysis/` for any specific verification re-run mentioned in a table or
figure caption. `./run_all.sh` runs the three scripts behind Supplement
S4 and S6 (`facility_mape_rmse.py`, `naive_sr_replication.py`,
`structural_confound_power.py`) in one command.

**The one disclosed exception:** Table 6's Class-SR leave-one-facility-out
value ($-1.001$) is archived in `data/processed_checkpoints/
model_comparison_table.csv` from an earlier computation whose exact
script was not preserved (see that file's entry above, and
`analysis/verify_table6_class_sr_loso_facility.py` for an independent
re-implementation that corroborates the finding qualitatively — a
strongly negative held-out R² — without hitting the published digit).
Every other number in the paper and supplement is reproducible from a
script in this repository.

## Requirements

Python 3.10+, `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`,
`scikit-learn`. `naive_sr_replication.py` additionally needs `gplearn`
(`pip install gplearn`).

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
