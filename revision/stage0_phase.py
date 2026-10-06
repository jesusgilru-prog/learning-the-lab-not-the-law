"""Stage 0 phase study (reviewer 1: when does Stage 0 matter; reviewer 2 items 6 and 8:
kappa threshold and recovery-error threshold sensitivity).

Extends experiment D of stage0_validation.py (three candidate groups Re, Ma, N_g over
the controls Omega, R, p, T) with a factorial over:
  n_speeds            {3, 4, 6, 8}           (sample size)
  controls varied     {3: Omega,R,p ; 4: Omega,R,p,T}
  replication imbalance {balanced, half of the runs piled on one design point}
  facilities          {1, 2}   (2 = a second facility with its own spans)
  facility shift      {0, 0.3} additive offset of the second facility (F=2 only)
and random spans of R, p, T (collinearity), random noise and random true effects.

The response fit always includes facility fixed effects. Stage 0 is computed on the
nuisance-adjusted design (columns centred within facility, as Algorithm 1 states);
the pairwise heuristic is computed on the raw pooled log-groups, as a practitioner
would. Recovery := |theta_M_hat - theta_M| < c |theta_M| for c in {0.2,0.3,0.5,0.7}.
"""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "analysis"))
from stage0_validation import E, GROUPS, KNOBS, make_design  # noqa: E402

SEED = 20261002
TRIALS = 300
C_LIST = (0.2, 0.3, 0.5, 0.7)
KAPPA_LIST = (10, 20, 30, 50, 100)
CORR_LIST = (0.95, 0.99, 0.999)


def logZ(design):
    return np.log(design[KNOBS].values) @ E.T  # raw log-groups (n x 3)


def facility_design(rng, n_speeds, n_controls, imbalance):
    spanR = 10 ** rng.uniform(-3.5, 0.5)
    spanP = 10 ** rng.uniform(-3.5, 1.0)
    spanT = 10 ** rng.uniform(-3.5, -0.3)
    lev = {"Omega": np.linspace(40, 100, n_speeds) * rng.uniform(0.7, 1.3),
           "R": np.array([1.0, 1.0 + spanR]) * rng.uniform(0.5, 2.0),
           "p": np.array([101e3, 101e3 / (1 + spanP)]),
           "T": (np.array([293.15, 293.15 * (1 + spanT)]) if n_controls == 4
                 else np.array([293.15]))}
    d = make_design(lev)
    if imbalance:
        d = pd.concat([d, pd.DataFrame([d.iloc[0]] * len(d))], ignore_index=True)
    return d


def screens(Zadj, Zraw):
    Zc = Zadj - Zadj.mean(0)
    r = np.linalg.matrix_rank(Zc, tol=1e-9)
    nrm = np.linalg.norm(Zc, axis=0)
    if (nrm < 1e-12).any():
        kappa = np.inf
    else:
        sv = np.linalg.svd(Zc / nrm, compute_uv=False)
        kappa = sv[0] / sv[-1] if sv[-1] > 1e-14 else np.inf
    cmax = 0.0
    for i, j in itertools.combinations(range(3), 2):
        a, b = Zraw[:, i], Zraw[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            cmax = 1.0
            break
        cmax = max(cmax, abs(np.corrcoef(a, b)[0, 1]))
    return r, kappa, cmax


def trial(rng, n_speeds, n_controls, imbalance, F, shift):
    designs = [facility_design(rng, n_speeds, n_controls, imbalance) for _ in range(F)]
    Zraw = np.vstack([logZ(d) for d in designs])
    Zadj = np.vstack([logZ(d) - logZ(d).mean(0) for d in designs])
    fac = np.concatenate([np.full(len(d), k) for k, d in enumerate(designs)])
    sigma = 10 ** rng.uniform(-2.5, -1.0)
    theta = np.array([rng.uniform(-0.6, 0.0), rng.uniform(-0.3, -0.05), rng.uniform(-0.15, 0.0)])
    y = Zraw @ theta + shift * (fac == 1) + rng.normal(0, sigma, len(fac))
    r, kappa, cmax = screens(Zadj, Zraw)
    Fd = (fac[:, None] == np.arange(F)[None, :]).astype(float)
    A = np.hstack([Fd, Zraw])
    if np.linalg.matrix_rank(A, tol=1e-9) < A.shape[1]:
        err = np.inf
    else:
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        err = abs(b[F + 1] - theta[1]) / abs(theta[1])
    return dict(rank=r, kappa=kappa, cmax=cmax, rel_err=err, sigma=sigma, n=len(fac))


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n_sp, nc, imb, F, sh in itertools.product((3, 4, 6, 8), (3, 4), (0, 1), (1, 2), (0.0, 0.3)):
        if F == 1 and sh > 0:
            continue
        for _ in range(TRIALS):
            t = trial(rng, n_sp, nc, imb, F, sh)
            t.update(n_speeds=n_sp, controls=nc, imbalance=imb, F=F, shift=sh)
            rows.append(t)
    R = pd.DataFrame(rows)
    R.to_csv("stage0_phase_trials.csv", index=False)
    print(f"{len(R)} trials")

    def conf(v, rec):
        tp = (v & rec).sum(); fp = (v & ~rec).sum(); fn = (~v & rec).sum(); tn = (~v & ~rec).sum()
        return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                    accuracy=(tp + tn) / len(v), false_promise=fp / max(tp + fp, 1))

    # threshold sensitivity
    sens = []
    for c in C_LIST:
        rec = R.rel_err < c
        for k in KAPPA_LIST:
            v = (R["rank"] == 3) & (R.kappa <= k)
            sens.append(dict(rule=f"Stage0 kappa<={k}", c=c, **conf(v, rec)))
        for cc in CORR_LIST:
            v = R.cmax < cc
            sens.append(dict(rule=f"pairwise |r|<{cc}", c=c, **conf(v, rec)))
    S = pd.DataFrame(sens)
    S.to_csv("stage0_threshold_sensitivity.csv", index=False)
    print("\n=== threshold sensitivity (recovery error threshold c) ===")
    print(S.pivot_table(index="rule", columns="c", values="accuracy").round(3).to_string())
    print(S.pivot_table(index="rule", columns="c", values="false_promise").round(3).to_string())

    # by factor, frozen rules (kappa<=30, |r|<0.99, c=0.5)
    R["rec"] = R.rel_err < 0.5
    R["s0"] = (R["rank"] == 3) & (R.kappa <= 30)
    R["pw"] = R.cmax < 0.99
    R["blind"] = R.pw & ~R.s0
    fac_rows = []
    for col in ("n_speeds", "controls", "imbalance", "F", "shift"):
        for val, g in R.groupby(col):
            fac_rows.append(dict(factor=col, level=val, n=len(g), recoverable=g.rec.mean(),
                                 blind_zone=g.blind.mean(),
                                 recoverable_in_blind=g[g.blind].rec.mean() if g.blind.any() else np.nan,
                                 acc_stage0=((g.s0 == g.rec)).mean(), acc_pairwise=((g.pw == g.rec)).mean(),
                                 fp_stage0=(g.s0 & ~g.rec).sum() / max(g.s0.sum(), 1),
                                 fp_pairwise=(g.pw & ~g.rec).sum() / max(g.pw.sum(), 1)))
    FR = pd.DataFrame(fac_rows)
    FR.to_csv("stage0_by_factor.csv", index=False)
    print("\n=== by factor (frozen: kappa<=30, |r|<0.99, c=0.5) ===")
    print(FR.round(3).to_string(index=False))

    # phase map
    R["lk"] = pd.cut(np.log10(R.kappa.replace(np.inf, 1e12)), [-np.inf, 1, np.log10(30), 2, 3, np.inf],
                     labels=["<10", "10-30", "30-100", "100-1e3", ">1e3"])
    R["lc"] = pd.cut(R.cmax, [0, 0.9, 0.99, 0.999, 1.0001], labels=["<0.9", "0.9-0.99", "0.99-0.999", ">0.999"])
    P = R.pivot_table(index="lk", columns="lc", values="rec", aggfunc="mean", observed=False)
    N = R.pivot_table(index="lk", columns="lc", values="rec", aggfunc="size", observed=False)
    P.to_csv("stage0_phase_map_recovery.csv"); N.to_csv("stage0_phase_map_counts.csv")
    print("\n=== phase map: recovery rate (rows kappa, cols max pairwise |r|) ===")
    print(P.round(3).to_string()); print(N.to_string())
    print(f"\noverall: blind zone {R.blind.mean():.3f}, recoverable inside it {R[R.blind].rec.mean():.3f}; "
          f"acc Stage0 {(R.s0 == R.rec).mean():.3f} vs pairwise {(R.pw == R.rec).mean():.3f}")


if __name__ == "__main__":
    main()
