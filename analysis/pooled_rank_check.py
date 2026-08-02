import os
#!/usr/bin/env python
"""
Pooled, nuisance-adjusted design rank across facilities.

Round-2 external review (ChatGPT) correctly pointed out two gaps in the original
Stage-0 write-up:

  A1.3 -- Stage 0's Z never residualizes against the per-geometry intercepts C_i that
          class-SR actually fits, so it can count between-geometry variation as
          identifying information that the fitted model itself absorbs.
  A1.4 -- per-facility rank deficiency does not imply the POOLED, multi-facility design
          is also deficient; stacked designs can recover rank that no single block has.

This script checks both at once: residualize (log Ng, log Re, log M) against facility
dummies, then against the actual geometry dummies used by class-SR, and report the
rank and condition number of what is left in each case.
"""
import sys

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
KNOBS = ["omega_rad_s", "R_m", "p_Pa", "T_K"]
E = np.array([
    [2.0, 1.0, 0.0, 0.0],     # Ng  = Omega^2 R / g
    [1.0, 2.0, 1.0, -1.7],    # Re  = p Omega R^2 / (R_s T mu(T))
    [1.0, 1.0, 0.0, -0.5],    # M   = Omega R / a(T)
])
GROUPS = ["Ng", "Re", "M"]


def residual_rank(df, group_col, Z):
    levels = sorted(df[group_col].unique())
    D = np.column_stack([(df[group_col] == v).astype(float) for v in levels])
    n = len(df)
    P = D @ np.linalg.pinv(D.T @ D) @ D.T
    resid = (np.eye(n) - P) @ Z
    r = np.linalg.matrix_rank(resid, tol=1e-8)
    Zc = resid - resid.mean(0)
    norms = np.linalg.norm(Zc, axis=0)
    keep = norms > 1e-8
    Zn = Zc[:, keep] / norms[keep]
    sv = np.linalg.svd(Zn, compute_uv=False)
    rr = np.linalg.matrix_rank(Zn, tol=1e-8)
    kappa = sv[0] / sv[rr - 1] if rr else np.inf
    return r, sv, kappa, len(levels)


def main():
    df = pd.read_csv(DATA)
    Z = np.log(df[KNOBS].values) @ E.T
    print(f"n={len(df)}  groups={GROUPS}  knobs={KNOBS}\n")
    for col, label in [("source", "facility (F=4)"),
                       ("geometry_id", "geometry (G=12, what class-SR's C_i actually use)")]:
        r, sv, kappa, n_levels = residual_rank(df, col, Z)
        print(f"nuisance = {label}  (n_levels={n_levels})")
        print(f"  residual rank of (Ng,Re,M) after removing {col}: {r}/3")
        print(f"  singular values (unit-norm columns): {np.round(sv, 4)}")
        print(f"  kappa = {kappa:.2f}"
              f"  {'-- MAL CONDICIONADO (>30)' if kappa > 30 else '-- bien condicionado'}\n")


if __name__ == "__main__":
    sys.exit(main())
