import os
#!/usr/bin/env python
"""
Split-conformal counterpart to conformal_logo_nested_crossfit.py (round-5 review,
Kimi B2): re-run the SAME outer-LOGO + inner-nested-split construction used for
normalized conformal, but for vanilla split conformal (raw |residual| quantile, no
per-geometry sigma normalization), so Table 11's honest-cross-fit column is not
asymmetric across variants.
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

    print(f"n={len(df)}  geometries={len(unique_geoms)}  n_splits={N_SPLITS}  variant=split (raw residuals)")

    per_geom_coverage = {}
    per_geom_n = {}
    q_hats = {}

    for g in unique_geoms:
        test_mask = geom_ids == g
        cal_mask = ~test_mask
        train_idx = np.where(cal_mask)[0]
        n_train = len(train_idx)

        res_final = fit_no_boot(log_y[cal_mask], log_X[cal_mask], geom_ids[cal_mask], feature_names)
        exponent_final = np.array([res_final.global_exponents[f] for f in feature_names])
        log_C_final = {k: v["log_C"] for k, v in res_final.prefactors.items()}
        fallback_log_C = float(np.mean(list(log_C_final.values())))

        test_pred = fallback_log_C + log_X[test_mask] @ exponent_final
        test_resid = log_y[test_mask] - test_pred
        test_abs_scores = np.abs(test_resid)  # no sigma normalization: split conformal

        pooled_cal_scores = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n_train)
            half = n_train // 2
            inner_train_idx = train_idx[perm[:half]]
            inner_cal_idx = train_idx[perm[half:]]

            itr_geoms = set(geom_ids[inner_train_idx])
            keep = np.array([geom_ids[i] in itr_geoms for i in inner_cal_idx])
            inner_cal_idx = inner_cal_idx[keep]
            if len(inner_cal_idx) == 0:
                continue

            res_inner = fit_no_boot(log_y[inner_train_idx], log_X[inner_train_idx],
                                     geom_ids[inner_train_idx], feature_names)
            exp_inner = np.array([res_inner.global_exponents[f] for f in feature_names])
            logC_inner = {k: v["log_C"] for k, v in res_inner.prefactors.items()}

            cal_geoms_i = geom_ids[inner_cal_idx]
            cal_pred_i = np.array([logC_inner[str(gg)] + exp_inner @ log_X[idx]
                                    for gg, idx in zip(cal_geoms_i, inner_cal_idx)])
            cal_resid_i = log_y[inner_cal_idx] - cal_pred_i
            pooled_cal_scores.extend(np.abs(cal_resid_i))

        pooled_cal_scores = np.array(pooled_cal_scores)
        n_cal = len(pooled_cal_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
        q_hat = np.quantile(pooled_cal_scores, q_level)
        q_hats[g] = float(q_hat)

        coverage_g = float(np.mean(test_abs_scores <= q_hat))
        per_geom_coverage[g] = coverage_g
        per_geom_n[g] = int(test_mask.sum())
        print(f"  {g:<32} n_test={per_geom_n[g]:<4} n_pooled_cal={n_cal:<6} "
              f"q_hat={q_hat:.4f}  coverage={coverage_g:.4f}  [{time.time()-t0:.0f}s]")

    logo_min = min(per_geom_coverage.values())
    logo_min_geom = min(per_geom_coverage, key=per_geom_coverage.get)
    n_total = sum(per_geom_n.values())
    weighted_mean = sum(per_geom_coverage[g] * per_geom_n[g] for g in per_geom_coverage) / n_total

    print(f"\nLOGO-min (split, nested) = {logo_min:.4f} ({logo_min_geom})")
    print(f"Weighted mean coverage = {weighted_mean:.4f}")

    out = {
        "variant": "split",
        "per_geom_coverage": per_geom_coverage,
        "per_geom_n": per_geom_n,
        "q_hats": q_hats,
        "logo_min": logo_min,
        "logo_min_geometry": logo_min_geom,
        "weighted_mean_coverage": weighted_mean,
        "n_splits_inner": N_SPLITS,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformal_logo_nested_crossfit_split_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved. Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
