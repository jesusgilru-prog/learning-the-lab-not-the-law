import os
#!/usr/bin/env python
"""
Design-rank identifiability analysis + within-facility Reynolds-similarity test
for the cross-facility windage benchmark of "Learning the Lab, Not the Law".

Reproduces every number quoted in RESULTS_identifiability.md.

Usage:
    python ddp_rank_analysis.py            # prints the full report
"""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
G_EARTH = 9.80665
SEED = 20260730

# Physical controls an experimenter can set, and the exponent of each control
# in each candidate dimensionless group.  Air is treated as an ideal gas, so
# the speed of sound is a function of temperature alone.
CONTROLS = ["omega_rad_s", "R_m", "rho_kgm3", "mu_Pas", "a_sound"]
EXPONENTS = {
    "g_level":  {"omega_rad_s": 2, "R_m": 1},                                  # Omega^2 R / g
    "Re_Omega": {"omega_rad_s": 1, "R_m": 2, "rho_kgm3": 1, "mu_Pas": -1},     # rho Omega R^2 / mu
    "M_tip":    {"omega_rad_s": 1, "R_m": 1, "a_sound": -1},                   # Omega R / a
}
GROUPS = list(EXPONENTS)


def load():
    df = pd.read_csv(DATA)
    df["a_sound"] = np.sqrt(1.4 * 287.05 * df["T_K"])
    return df


def ols(A, y):
    """Least squares with classical covariance. A already includes the intercept."""
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    n, k = A.shape
    cov = (resid @ resid / (n - k)) * np.linalg.pinv(A.T @ A)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return b, cov, r2, n, k


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_monomials(df):
    section("0. The tabulated Pi groups are exactly the theoretical monomials")
    pred = {
        "g_level": df.omega_rad_s ** 2 * df.R_m / G_EARTH,
        "Re_Omega": df.rho_kgm3 * df.omega_rad_s * df.R_m ** 2 / df.mu_Pas,
        "M_tip": df.omega_rad_s * df.R_m / df.a_sound,
    }
    for name, values in pred.items():
        if name == "g_level":
            # Only the two genuine centrifuges carry a rotor g-level; the 1-g
            # benches are tabulated as g_level = 1 by convention.
            for src in ("Liu2024", "Xia2024"):
                m = df.source == src
                err = np.abs(values[m] / df.loc[m, name] - 1).max()
                print(f"  g_level  ({src:11s}) max relative error = {err:.2e}")
        else:
            err = np.abs(values / df[name] - 1).max()
            print(f"  {name:9s}              max relative error = {err:.2e}")


def design_rank(df):
    section("1. Structural identifiability = rank of the exponent matrix "
            "restricted to the controls actually varied")
    for src, g in df.groupby("source"):
        varied = [c for c in CONTROLS if g[c].nunique() > 1]
        M = np.array([[EXPONENTS[p].get(c, 0) for c in varied] for p in GROUPS], float)
        rank = np.linalg.matrix_rank(M) if M.size else 0
        print(f"\n{src}  (n={len(g)})")
        print(f"  controls varied : {varied}")
        print(f"  exponent matrix (rows = {', '.join(GROUPS)}):\n{M}")
        print(f"  rank = {rank}  -> at most {rank} of the {len(GROUPS)} groups "
              f"are separately identifiable")
        for p1, p2 in itertools.combinations(GROUPS, 2):
            v1 = np.array([EXPONENTS[p1].get(c, 0) for c in varied], float)
            v2 = np.array([EXPONENTS[p2].get(c, 0) for c in varied], float)
            exact = np.linalg.matrix_rank(np.vstack([v1, v2])) < 2
            if g[p1].nunique() > 1 and g[p2].nunique() > 1:
                c = np.corrcoef(np.log(g[p1]), np.log(g[p2]))[0, 1]
                print(f"    {p1:9s} vs {p2:9s}: exactly degenerate={str(exact):5s} "
                      f" corr(log,log)={c:+.6f}")


def practical_identifiability(df):
    section("2. Practical identifiability = conditioning of the realised "
            "log-design matrix")
    for src, g in df.groupby("source"):
        cols = [c for c in GROUPS if g[c].nunique() > 1]
        if len(cols) < 2:
            print(f"{src:12s} only {cols} varies -> nothing to condition")
            continue
        X = np.column_stack([np.log(g[c]) for c in cols])
        X = X - X.mean(0)
        sv = np.linalg.svd(X, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        print(f"{src:12s} groups={cols}")
        print(f"{'':12s} singular values={np.array2string(sv, precision=4)}  "
              f"cond={cond:.3g}")


def reynolds_similarity(df):
    section("3. Reynolds-similarity test inside a single facility (Liu2024, "
            "balanced 5x4 factorial in Omega and chamber pressure)")
    print("If Cp = f(Re) with Re ~ rho*Omega, then in")
    print("    log Cp = c + alpha*log(Omega) + beta*log(rho)")
    print("Reynolds similarity requires alpha = beta.  H0: alpha - beta = 0.")
    print("Because R and T are fixed, M_tip ~ Omega, so alpha - beta is exactly")
    print("the exponent of the second group (Mach -- or, indistinguishably, g-level).")

    liu = df[df.source == "Liu2024"]
    subsets = [
        ("all 20 points", liu),
        ("digitized only (drops the 10 kPa column,\n"
         "                 which is a fitted constant)",
         liu[liu.data_origin == "digitized_from_figure"]),
    ]
    for label, sub in subsets:
        y = np.log(sub.Cp.values)
        A = np.column_stack([np.ones(len(sub)),
                             np.log(sub.omega_rad_s.values),
                             np.log(sub.rho_kgm3.values)])
        b, cov, r2, n, k = ols(A, y)
        d = b[1] - b[2]
        sd = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
        t = d / sd
        p = 2 * (1 - stats.t.cdf(abs(t), n - k))
        print(f"\n  {label}   n={n}  R2={r2:.3f}")
        print(f"    alpha(Omega) = {b[1]:+.4f} +- {np.sqrt(cov[1,1]):.4f}")
        print(f"    beta (rho)   = {b[2]:+.4f} +- {np.sqrt(cov[2,2]):.4f}")
        print(f"    alpha - beta = {d:+.4f} +- {sd:.4f}   t={t:+.2f}  p={p:.4g}"
              f"   CI95=[{d-1.96*sd:+.4f}, {d+1.96*sd:+.4f}]")
        print(f"    verdict: {'REJECTS' if p < 0.05 else 'compatible with'} "
              f"Reynolds similarity")

    # Robustness on the full factorial.
    y = np.log(liu.Cp.values)
    A = np.column_stack([np.ones(len(liu)),
                         np.log(liu.omega_rad_s.values),
                         np.log(liu.rho_kgm3.values)])
    n = len(liu)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(20000):
        idx = rng.integers(0, n, n)
        bb, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
        boot.append(bb[1] - bb[2])
    boot = np.array(boot)
    print(f"\n  bootstrap (20k, seed {SEED}): mean={boot.mean():+.4f}  "
          f"CI95=[{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  "
          f"P(>=0)={np.mean(boot >= 0):.4f}")

    jack = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        bb, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        jack.append(bb[1] - bb[2])
    jack = np.array(jack)
    print(f"  jackknife: min={jack.min():+.4f} max={jack.max():+.4f}  "
          f"all negative = {bool(np.all(jack < 0))}")

    for factor, name in (("omega_rad_s", "Omega"), ("p_Pa", "pressure")):
        outs = []
        for v in sorted(liu[factor].unique()):
            m = (liu[factor] != v).values
            bb, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            outs.append(f"{v:.6g}:{bb[1]-bb[2]:+.3f}")
        print(f"  leave-one-{name}-out: " + "  ".join(outs))

    # Same fit expressed directly in (Re, M): identical by construction.
    A2 = np.column_stack([np.ones(n),
                          np.log(liu.Re_Omega.values),
                          np.log(liu.M_tip.values)])
    b2, cov2, r22, n2, k2 = ols(A2, y)
    se2 = np.sqrt(np.diag(cov2))
    print(f"\n  same fit in (Re, M):  log Cp = {b2[0]:+.3f} "
          f"+ ({b2[1]:+.4f}+-{se2[1]:.4f})*log Re "
          f"+ ({b2[2]:+.4f}+-{se2[2]:.4f})*log M    R2={r22:.3f}")
    for j, nm in ((1, "Re"), (2, "M ")):
        t = b2[j] / se2[j]
        print(f"    {nm}: t={t:+.2f}  p={2*(1-stats.t.cdf(abs(t), n2-k2)):.4g}")


def per_geometry_exponents(df):
    section("4. Cp ~ Re^a fitted inside each geometry (12 geometries, not 4 facilities)")
    rows = []
    for (src, gid), g in df.groupby(["source", "geometry_id"]):
        if len(g) < 4 or g.Re_Omega.nunique() < 4:
            continue
        y = np.log(g.Cp.values)
        A = np.column_stack([np.ones(len(g)), np.log(g.Re_Omega.values)])
        b, cov, r2, n, k = ols(A, y)
        span = g.Re_Omega.max() / g.Re_Omega.min()
        rows.append(dict(source=src, geometry=gid, n=n, a=b[1], r2=r2, re_span=span))
        print(f"  {src:12s} {str(gid):30s} n={n:3d}  a={b[1]:+.3f}  "
              f"R2={r2:.3f}  Re span x{span:.0f}")
    R = pd.DataFrame(rows)
    print(f"\n  across {len(R)} geometries: mean={R.a.mean():+.3f}  "
          f"median={R.a.median():+.3f}  sd={R.a.std():.3f}  "
          f"range=[{R.a.min():+.3f}, {R.a.max():+.3f}]")
    print(R.groupby("source").a.agg(["count", "mean", "std"]).to_string())


def pooled_vs_logo(df):
    section("5. Pooled fit vs leave-one-facility-out (why the pooled law is a "
            "facility artifact)")
    pis = ["Pi_confinement", "Pi_gap", "Pi_aspect_axial", "Pi_blockage"]
    for feats in (["Re_Omega"], ["Re_Omega", "M_tip"],
                  ["Re_Omega"] + pis, ["Re_Omega", "M_tip"] + pis):
        y = np.log(df.Cp.values)
        A = np.column_stack([np.ones(len(df))] + [np.log(df[f].values) for f in feats])
        _, _, r2, _, _ = ols(A, y)
        errs = {}
        for src in df.source.unique():
            tr, te = df[df.source != src], df[df.source == src]
            Atr = np.column_stack([np.ones(len(tr))] + [np.log(tr[f].values) for f in feats])
            btr, *_ = np.linalg.lstsq(Atr, np.log(tr.Cp.values), rcond=None)
            Ate = np.column_stack([np.ones(len(te))] + [np.log(te[f].values) for f in feats])
            errs[src] = np.sqrt(((np.log(te.Cp.values) - Ate @ btr) ** 2).mean())
        print(f"  {str(feats):68s}")
        print(f"     in-sample R2={r2:.3f}   LOGO rmse(log): "
              + "  ".join(f"{k}={v:.3f}" for k, v in errs.items())
              + f"   mean={np.mean(list(errs.values())):.3f}")


def main():
    df = load()
    print(f"dataset: {DATA}")
    print(f"n={len(df)}  sources={sorted(df.source.unique())}  "
          f"geometries={df.geometry_id.nunique()}")
    check_monomials(df)
    design_rank(df)
    practical_identifiability(df)
    reynolds_similarity(df)
    per_geometry_exponents(df)
    pooled_vs_logo(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
