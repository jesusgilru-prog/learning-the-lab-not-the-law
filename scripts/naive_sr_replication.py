"""Naive/blind SR replication (Kimi review, item 1.3): does an
off-the-shelf, unconstrained symbolic regression search (gplearn;
no hand-picked S1-S6 candidate family) independently rediscover a
Mach-conditioned residual structure, without ever being told about
the Bayesian S1-S6 family or Liu2024/facility identity?

Setup: residualize log(Cp) against the per-geometry prefactor
log(C_i) already fit by class-SR (data/processed/prefactor_analysis.json)
-- the same target the S1-S6 family operates on -- and let gplearn
search freely over {log(Re_Omega), Mach} with generic arithmetic
primitives (no piecewise/threshold primitive is handed to it).
We then check whether Mach appears in the fittest program, and if
so, run it through Test 1 (facility-threshold contingency) exactly
as in the discriminant battery.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.parquet")
PREFACTORS = os.path.join(HERE, "..", "data", "processed_checkpoints", "prefactor_analysis.json")
RESULTS_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


def main():
    df = pd.read_parquet(DATASET)
    with open(PREFACTORS) as f:
        pf = json.load(f)["prefactors"]

    log_Cp = np.log(df["Cp"].values)
    log_C = np.array([pf[g]["log_C"] for g in df["geometry_id"].values])
    residual = log_Cp - log_C
    log_Re = np.log(df["Re_Omega"].values)
    mach = df["M_tip"].values
    facility = df["source"].values

    X = np.column_stack([log_Re, mach])
    y = residual

    est = SymbolicRegressor(
        population_size=3000,
        generations=40,
        function_set=("add", "sub", "mul", "div"),
        metric="mse",
        parsimony_coefficient=0.001,
        stopping_criteria=0.0,
        random_state=SEED,
        n_jobs=2,
        verbose=1,
        feature_names=["logRe", "Mach"],
    )
    est.fit(X, y)

    program = str(est._program)
    r2 = est.score(X, y)
    uses_mach = "Mach" in program

    print("\n=== NAIVE SR RESULT ===")
    print("Best program:", program)
    print("In-sample R^2:", r2)
    print("Uses Mach:", uses_mach)

    out = {
        "best_program": program,
        "r2_in_sample": float(r2),
        "uses_mach": bool(uses_mach),
    }

    if uses_mach:
        # Fit a threshold on Mach the same way S5's threshold was fit:
        # find the split point that most reduces residual variance
        # (simple grid search, not the differential-evolution MLE used
        # for S5, since gplearn's program is not itself piecewise).
        order = np.argsort(mach)
        m_sorted = mach[order]
        best_var, best_m = np.inf, None
        for i in range(5, len(m_sorted) - 5):
            m_try = m_sorted[i]
            lo = residual[mach < m_try]
            hi = residual[mach >= m_try]
            if len(lo) < 3 or len(hi) < 3:
                continue
            v = np.var(lo) * len(lo) + np.var(hi) * len(hi)
            if v < best_var:
                best_var, best_m = v, m_try

        low = mach < best_m
        unique_fac = np.unique(facility)
        ct = np.array([[np.sum((facility == f) & low),
                         np.sum((facility == f) & ~low)] for f in unique_fac])
        from scipy.stats import chi2_contingency
        try:
            chi2, p_test1, _, _ = chi2_contingency(ct)
        except Exception:
            p_test1 = float("nan")

        out["implied_threshold_mach"] = float(best_m)
        out["contingency_table"] = {f: row.tolist() for f, row in zip(unique_fac.tolist(), ct)}
        out["test1_chi2_p"] = float(p_test1)
        print(f"Implied best-split Mach threshold: {best_m:.4f}")
        print("Facility x threshold contingency:")
        for f, row in zip(unique_fac, ct):
            print(f"  {f}: below={row[0]}, above={row[1]}")
        print(f"Test 1 chi2 p-value: {p_test1:.4f}")

    with open(os.path.join(RESULTS_DIR, "naive_sr_replication.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
