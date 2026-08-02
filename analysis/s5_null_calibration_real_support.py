import os
#!/usr/bin/env python
"""
S5 calibration under the null, preserving the REAL 8/106 pooled support imbalance.

Round-3 external review (ChatGPT, item A3) correctly pointed out that the existing
specificity/power experiment (structural_confound_power.py) injects a confound via
one facility occupying its OWN, DISJOINT Mach range -- a different mechanism from
what Table 10/tab:contingency actually shows for S5, which is a POOLED support
imbalance (8/114 low-Mach points scattered across facilities, Liu2024's own Mach
range nested inside Vrancik1968's, not disjoint). The existing 13%-FPR number is
therefore a real but partial specificity result: it does not calibrate the specific
mechanism the S5 case exhibits.

This script closes that gap directly: simulate y under the TRUE null S1 (single
shared exponent, no regime), using the REAL x's (Re, Mach, geometry, facility) of
the actual 114-point corpus unchanged -- so the real 8/106 pooled support imbalance
is baked into every replicate exactly as observed -- and real per-geometry
prefactors/sigmas as the generating parameters. Then run the SAME threshold-search
+ Bayes-factor procedure (run_framework_verdict, reused unmodified from
remediation_experiments.py) that produced the real S5 finding, and record how often
pure noise manufactures an apparent two-regime detection this strong.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from remediation_experiments import run_framework_verdict  # noqa: E402

import json

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
SR_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "class_sr_results.json")
K = 200
SEED = 20260802

REAL_BF_FITTED = 5.8e5
REAL_BF_PEAK = 6.2e6


def main():
    df = pd.read_csv(DATASET)
    with open(SR_RESULTS) as f:
        sr = json.load(f)

    q_true = sr["global_exponents"]["Re_Omega"]
    log_C = {k: v["log_C"] for k, v in sr["prefactors"].items()}
    sigma_g = sr["sigma_per_geometry"]

    log_Re = np.log(df["Re_Omega"].values)
    mach = df["M_tip"].values
    group_ids = df["geometry_id"].values
    facility_ids = df["source"].values
    n_pts = len(df)

    mu = np.array([log_C[g] + q_true * lr for g, lr in zip(group_ids, log_Re)])
    sigmas = np.array([max(sigma_g[g], 1e-3) for g in group_ids])

    print(f"n={n_pts}  q_true={q_true:.4f}  K={K} replicates")
    print(f"Real corpus support: 8/114 below Ma=0.127 (preserved exactly, same x's every replicate)")

    rng = np.random.default_rng(SEED)
    bfs = []
    verdicts = []

    for k in range(K):
        noise = rng.normal(0, 1, n_pts) * sigmas
        log_y_sim = mu + noise
        verdict, info = run_framework_verdict(log_y_sim, log_Re, mach, group_ids, facility_ids, n_pts)
        bfs.append(info.get("bf_s5_s1", np.nan))
        verdicts.append(verdict)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{K}  last BF={info.get('bf_s5_s1', float('nan')):.3g}  verdict={verdict}")

    bfs = np.array(bfs)
    from collections import Counter
    vc = Counter(verdicts)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Verdict distribution under null (real support structure): {dict(vc)}")
    fp = sum(v not in ("facility-artifact", "no-regime-detected") for v in verdicts)
    print(f"False-positive rate (verdict is NOT facility-artifact/no-regime): {fp}/{K} = {fp/K:.4f}")
    print(f"\nBF_S5/S1 under null: mean={np.nanmean(bfs):.3g}  median={np.nanmedian(bfs):.3g}"
          f"  max={np.nanmax(bfs):.3g}")
    exceed_fitted = np.mean(bfs >= REAL_BF_FITTED)
    exceed_peak = np.mean(bfs >= REAL_BF_PEAK)
    print(f"Fraction of null replicates with BF >= real fitted value ({REAL_BF_FITTED:.1e}): {exceed_fitted:.4f}")
    print(f"Fraction of null replicates with BF >= real sweep peak ({REAL_BF_PEAK:.1e}): {exceed_peak:.4f}")

    out = {
        "K": K, "q_true": float(q_true),
        "verdict_counts": dict(vc),
        "false_positive_rate": fp / K,
        "bf_null_mean": float(np.nanmean(bfs)),
        "bf_null_median": float(np.nanmedian(bfs)),
        "bf_null_max": float(np.nanmax(bfs)),
        "frac_exceed_real_fitted_bf": float(exceed_fitted),
        "frac_exceed_real_peak_bf": float(exceed_peak),
        "real_bf_fitted": REAL_BF_FITTED,
        "real_bf_peak": REAL_BF_PEAK,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_null_calibration_real_support_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: s5_null_calibration_real_support_results.json")


if __name__ == "__main__":
    sys.exit(main())
