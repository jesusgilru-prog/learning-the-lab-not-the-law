import os
#!/usr/bin/env python
"""
Leakage-free LOGO (leave-one-geometry-out) normalized conformal.

Round-3 external review (ChatGPT, item A1) correctly identified that the manuscript's
normalized-conformal LOGO evaluation reuses sigma_per_geometry estimated ONCE on the
FULL dataset (via class_sr_fit on all 114 points) for every fold, including the fold
where that very geometry is held out. Since sigma_g = std(residuals of geometry g) in
the global fit, the held-out geometry's own residuals informed the sigma used to
normalize its own test scores -- direct leakage.

This script recomputes LOGO honestly with strict cross-fitting:
  - For each held-out geometry g, refit class_sr_fit on D \\ D_g (exponent + per-geometry
    log_C + per-geometry sigma all re-estimated from the 11 remaining geometries only).
  - Calibration scores use residuals/sigmas from D \\ D_g (no leakage: each calibration
    geometry's sigma still comes only from its own points, none of which are g's).
  - For the held-out geometry g (unseen at fit time), there is no valid geometry-specific
    sigma. We use the SAME fallback convention already established in Table 6 for unseen
    geometries: prefactor = mean(log_C over the 11 training geometries), sigma = mean
    (sigma over the 11 training geometries). This is an explicit, documented, leakage-free
    fallback rule, not an ad hoc patch.

Output: per-geometry coverage, LOGO-min, comparison against the original (leaky) numbers
already in the manuscript.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sr_engine.class_sr import class_sr_fit  # noqa: E402

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
SR_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "class_sr_results.json")
ALPHA = 0.10


def main():
    df = pd.read_csv(DATASET)
    with open(SR_RESULTS) as f:
        sr_full = json.load(f)
    feature_names = sr_full["feature_names"]

    geom_ids = df["geometry_id"].values
    log_y = np.log(df["Cp"].values)
    log_X = np.column_stack([np.log(df[f].values) for f in feature_names])
    unique_geoms = np.unique(geom_ids)

    print(f"n={len(df)}  geometries={len(unique_geoms)}  features={feature_names}")
    print(f"Original (leaky) global exponent: {sr_full['global_exponents']}")

    per_geom_coverage = {}
    per_geom_n = {}
    q_hats = {}

    for g in unique_geoms:
        test_mask = geom_ids == g
        cal_mask = ~test_mask

        res = class_sr_fit(
            log_y=log_y[cal_mask],
            log_X=log_X[cal_mask],
            group_ids=geom_ids[cal_mask],
            feature_names=feature_names,
            n_bootstrap=0,
            seed=42,
        )
        exponent_cf = np.array([res.global_exponents[f] for f in feature_names])
        log_C_cf = {k: v["log_C"] for k, v in res.prefactors.items()}
        sigma_cf = res.sigma_per_geometry

        # Calibration residuals (leakage-free: g excluded entirely from this fit)
        cal_geoms = geom_ids[cal_mask]
        cal_logy = log_y[cal_mask]
        cal_logX = log_X[cal_mask]
        cal_pred = np.array([
            log_C_cf[str(gg)] + exponent_cf @ cal_logX[i]
            for i, gg in enumerate(cal_geoms)
        ])
        cal_resid = cal_logy - cal_pred
        cal_sigma = np.array([max(sigma_cf[str(gg)], 0.01) for gg in cal_geoms])
        cal_norm_scores = np.abs(cal_resid) / cal_sigma

        n_cal = len(cal_norm_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
        q_hat = np.quantile(cal_norm_scores, q_level)
        q_hats[g] = float(q_hat)

        # Held-out geometry: fallback prefactor/sigma from training geometries only
        fallback_log_C = float(np.mean(list(log_C_cf.values())))
        fallback_sigma = max(float(np.mean(list(sigma_cf.values()))), 0.01)

        test_logy = log_y[test_mask]
        test_logX = log_X[test_mask]
        test_pred = fallback_log_C + test_logX @ exponent_cf
        test_resid = test_logy - test_pred
        test_norm_scores = np.abs(test_resid) / fallback_sigma

        coverage_g = float(np.mean(test_norm_scores <= q_hat))
        per_geom_coverage[g] = coverage_g
        per_geom_n[g] = int(test_mask.sum())

    print("\nPer-geometry LOGO coverage (leakage-free cross-fit, target 0.90):")
    for g in sorted(unique_geoms):
        print(f"  {g:<32} n={per_geom_n[g]:<4} coverage={per_geom_coverage[g]:.4f}  q_hat={q_hats[g]:.4f}")

    logo_min = min(per_geom_coverage.values())
    logo_min_geom = min(per_geom_coverage, key=per_geom_coverage.get)
    print(f"\nLOGO-min (cross-fit, honest) = {logo_min:.4f}  (geometry: {logo_min_geom})")

    out = {
        "per_geom_coverage": per_geom_coverage,
        "per_geom_n": per_geom_n,
        "q_hats": q_hats,
        "logo_min": logo_min,
        "logo_min_geometry": logo_min_geom,
        "fallback_rule": "mean(log_C) and mean(sigma) over training geometries, same convention as Table 6",
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformal_logo_crossfit_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: conformal_logo_crossfit_results.json")


if __name__ == "__main__":
    sys.exit(main())
