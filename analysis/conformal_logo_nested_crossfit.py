import os
#!/usr/bin/env python
"""
Leakage-free LOGO conformal, with a genuine internal train/calibration split.

Round-4 external review (ChatGPT, item P0-1) correctly pointed out that
conformal_logo_crossfit.py, while honestly excluding the held-out geometry from the
model fit (outer fold), still used the SAME 11 training geometries' points both to fit
class_sr AND to generate the calibration scores for the quantile -- i.e. resubstitution
residuals, not out-of-fold ones, at the calibration step.

This script adds the missing inner split, mirroring the repeated-random-split
convention ALREADY used elsewhere in this codebase's split/normalized conformal
(sr_engine/conformal.py: n_splits=200 random 50/50 splits, pooled). For each outer fold
(held-out geometry g):
  1. Fit class_sr on all 11 training geometries once -> exponent_final, per-geometry
     log_C_final, sigma_final. This FINAL fit is used ONLY to score g's held-out test
     points (via the same across-geometry mean fallback as before) -- never to generate
     calibration scores.
  2. For n_splits=200 repeated random 50/50 splits of the 106ish training points:
     refit class_sr on the inner-train half only, compute normalized residuals of the
     inner-calibration half using THAT split's own fit (genuinely out-of-fold for those
     points). Pool all inner-calibration normalized scores across the 200 splits.
  3. q_hat = quantile of the pooled out-of-fold calibration scores.
  4. Coverage of g's test points = fraction with |resid|/sigma_fallback <= q_hat.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sr_engine.class_sr import class_sr_fit  # noqa: E402

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
SR_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "class_sr_results.json")
ALPHA = 0.10
N_SPLITS = 200
SEED = 42


def fit_no_boot(log_y, log_X, group_ids, feature_names):
    return class_sr_fit(log_y=log_y, log_X=log_X, group_ids=group_ids,
                         feature_names=feature_names, n_bootstrap=0, seed=42)


def main():
    t0 = time.time()
    df = pd.read_csv(DATASET)
    with open(SR_RESULTS) as f:
        sr_full = json.load(f)
    feature_names = sr_full["feature_names"]

    geom_ids = df["geometry_id"].values
    log_y = np.log(df["Cp"].values)
    log_X = np.column_stack([np.log(df[f].values) for f in feature_names])
    unique_geoms = np.unique(geom_ids)
    rng = np.random.default_rng(SEED)

    print(f"n={len(df)}  geometries={len(unique_geoms)}  n_splits={N_SPLITS}")

    per_geom_coverage = {}
    per_geom_n = {}
    q_hats = {}

    for g in unique_geoms:
        test_mask = geom_ids == g
        cal_mask = ~test_mask
        train_idx = np.where(cal_mask)[0]
        n_train = len(train_idx)

        # Step 1: final fit on all 11 training geometries, for scoring g only
        res_final = fit_no_boot(log_y[cal_mask], log_X[cal_mask], geom_ids[cal_mask], feature_names)
        exponent_final = np.array([res_final.global_exponents[f] for f in feature_names])
        log_C_final = {k: v["log_C"] for k, v in res_final.prefactors.items()}
        sigma_final = res_final.sigma_per_geometry
        fallback_log_C = float(np.mean(list(log_C_final.values())))
        fallback_sigma = max(float(np.mean(list(sigma_final.values()))), 0.01)

        test_pred = fallback_log_C + log_X[test_mask] @ exponent_final
        test_resid = log_y[test_mask] - test_pred
        test_norm_scores = np.abs(test_resid) / fallback_sigma

        # Step 2: repeated inner splits for out-of-fold calibration scores
        pooled_cal_scores = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n_train)
            half = n_train // 2
            inner_train_idx = train_idx[perm[:half]]
            inner_cal_idx = train_idx[perm[half:]]

            # need every geometry represented in inner_train for class_sr_fit to work;
            # if a geometry has 0 points in inner_train, drop those inner_cal points
            # (can't score them without a fitted C_i/sigma_i for that geometry)
            itr_geoms = set(geom_ids[inner_train_idx])
            keep = np.array([geom_ids[i] in itr_geoms for i in inner_cal_idx])
            inner_cal_idx = inner_cal_idx[keep]
            if len(inner_cal_idx) == 0:
                continue

            res_inner = fit_no_boot(log_y[inner_train_idx], log_X[inner_train_idx],
                                     geom_ids[inner_train_idx], feature_names)
            exp_inner = np.array([res_inner.global_exponents[f] for f in feature_names])
            logC_inner = {k: v["log_C"] for k, v in res_inner.prefactors.items()}
            sigma_inner = res_inner.sigma_per_geometry

            cal_geoms_i = geom_ids[inner_cal_idx]
            cal_pred_i = np.array([logC_inner[str(gg)] + exp_inner @ log_X[idx]
                                    for gg, idx in zip(cal_geoms_i, inner_cal_idx)])
            cal_resid_i = log_y[inner_cal_idx] - cal_pred_i
            cal_sigma_i = np.array([max(sigma_inner[str(gg)], 0.01) for gg in cal_geoms_i])
            pooled_cal_scores.extend(np.abs(cal_resid_i) / cal_sigma_i)

        pooled_cal_scores = np.array(pooled_cal_scores)
        n_cal = len(pooled_cal_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
        q_hat = np.quantile(pooled_cal_scores, q_level)
        q_hats[g] = float(q_hat)

        coverage_g = float(np.mean(test_norm_scores <= q_hat))
        per_geom_coverage[g] = coverage_g
        per_geom_n[g] = int(test_mask.sum())
        print(f"  {g:<32} n_test={per_geom_n[g]:<4} n_pooled_cal={n_cal:<6} "
              f"q_hat={q_hat:.4f}  coverage={coverage_g:.4f}  [{time.time()-t0:.0f}s]")

    logo_min = min(per_geom_coverage.values())
    logo_min_geom = min(per_geom_coverage, key=per_geom_coverage.get)
    n_total = sum(per_geom_n.values())
    weighted_mean = sum(per_geom_coverage[g] * per_geom_n[g] for g in per_geom_coverage) / n_total

    print(f"\nLOGO-min (nested, out-of-fold calibration) = {logo_min:.4f} ({logo_min_geom})")
    print(f"Weighted mean coverage = {weighted_mean:.4f}")

    out = {
        "per_geom_coverage": per_geom_coverage,
        "per_geom_n": per_geom_n,
        "q_hats": q_hats,
        "logo_min": logo_min,
        "logo_min_geometry": logo_min_geom,
        "weighted_mean_coverage": weighted_mean,
        "n_splits_inner": N_SPLITS,
        "method": "outer LOGO fold excludes test geometry entirely; inner calibration "
                  "scores come from repeated 50/50 random splits within the 11 training "
                  "geometries, pooled, genuinely out-of-fold; final predictor for "
                  "scoring the test geometry is fit on all 11 training geometries",
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformal_logo_nested_crossfit_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved. Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
