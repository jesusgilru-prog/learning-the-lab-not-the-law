"""Synthetic power curve with INJECTED STRUCTURAL confounding
(Kimi review, item 1.4) -- distinct from the existing prefactor-offset
power curve (experiment_1c in remediation_experiments.py, which
injects a constant additive log-Cp offset on one facility and, by
design, has low detection power since class-SR absorbs it into the
per-geometry prefactor).

Here the injected confound changes one facility's REYNOLDS EXPONENT
itself (a genuine structure-level effect, of the same kind as the S5
case study), not just its prefactor. Critically -- matching the real
mechanism behind S5, where Liu2024's distinct behaviour coincides
with it occupying a distinct Mach range -- the confounded facility's
Mach values are also shifted into their own range, so the exponent
change is correlated with Mach exactly as a Bayesian regime-in-Mach
search (S1-S6) could pick up. (An earlier version of this script
gave the confounded facility the SAME random Mach range as everyone
else; since the S1-S6 family only searches for regimes defined ON
MACH, an exponent shift uncorrelated with Mach gave that family
nothing to find, and power stayed at the false-positive floor
regardless of delta_q -- a real bug in the experiment design, not a
statement about the protocol, fixed here.)  We sweep the exponent
gap delta_q and measure the fraction of K simulated datasets the
protocol correctly flags as facility-artifact / statistically-real-
spurious, i.e. the protocol's power against the type of confound it
is actually designed to catch.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson CI for a binomial proportion k/n."""
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remediation_experiments import run_framework_verdict, SEED  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    q_true = -0.10
    K = 100
    delta_qs = [0.0, 0.02, 0.05, 0.10, 0.20]
    sigma_noise = 0.05  # matches the scale used for the existing power curve
    rng = np.random.default_rng(SEED + 2000)

    facility_configs = [
        {"name": "fac_A", "n": 45, "re_range": (1e5, 1e7), "n_geom": 5, "confounded": False},
        {"name": "fac_B", "n": 41, "re_range": (5e5, 5e7), "n_geom": 4, "confounded": False},
        {"name": "fac_C", "n": 20, "re_range": (1e6, 8e7), "n_geom": 3, "confounded": False},
        {"name": "fac_D", "n": 8, "re_range": (3e5, 3e6), "n_geom": 2, "confounded": True},
    ]

    power_rows = []
    for delta_q in delta_qs:
        true_positives = 0
        verdict_counts = {}
        for sim_id in range(K):
            dfs = []
            for fc in facility_configs:
                re_vals = np.exp(rng.uniform(np.log(fc["re_range"][0]), np.log(fc["re_range"][1]), fc["n"]))
                geom_ids = np.array([f"{fc['name']}_g{i%fc['n_geom']}" for i in range(fc["n"])])
                log_C = {f"{fc['name']}_g{j}": -2.0 + 0.3 * j / fc["n_geom"] for j in range(fc["n_geom"])}
                q_local = q_true + (delta_q if fc["confounded"] else 0.0)
                log_cp = np.array([log_C[g] + q_local * np.log(re) + rng.normal(0, sigma_noise)
                                   for g, re in zip(geom_ids, re_vals)])
                # Confounded facility occupies its own Mach range, exactly
                # as Liu2024 does in the real data (100% above Ma=0.127) --
                # the mechanism that makes the exponent shift show up as
                # a Mach-conditioned "regime" to the S1-S6 search.
                if fc["confounded"]:
                    mach_vals = rng.uniform(0.35, 0.6, fc["n"])
                else:
                    mach_vals = rng.uniform(0.05, 0.3, fc["n"])
                sub = pd.DataFrame({
                    "source": fc["name"], "geometry_id": geom_ids,
                    "Re_Omega": re_vals, "Cp": np.exp(log_cp), "M_tip": mach_vals,
                })
                dfs.append(sub)
            sim_df = pd.concat(dfs, ignore_index=True)

            log_y = np.log(sim_df["Cp"].values)
            log_Re = np.log(sim_df["Re_Omega"].values)
            mach = sim_df["M_tip"].values
            group_ids = sim_df["geometry_id"].values
            fac_ids = sim_df["source"].values

            verdict, _ = run_framework_verdict(log_y, log_Re, mach, group_ids, fac_ids, len(sim_df))
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            if verdict in ("facility-artifact", "ambiguous", "report-with-disclaimer",
                           "statistically-real-spurious"):
                true_positives += 1

        power = true_positives / K
        ci_lo, ci_hi = clopper_pearson(true_positives, K)
        print(f"delta_q={delta_q}: power={power:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  verdicts={verdict_counts}")
        power_rows.append({
            "delta_q": delta_q, "K": K, "power": power,
            "power_CI95_lo": ci_lo, "power_CI95_hi": ci_hi,
            "confound_type": "structural_exponent_shift",
            "verdict_counts": str(verdict_counts),
        })

    out = pd.DataFrame(power_rows)
    out.to_csv(os.path.join(RESULTS_DIR, "structural_confound_power.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
