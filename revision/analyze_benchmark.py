"""Analysis of the factorial benchmark exactly as pre-registered in FROZEN_PROTOCOL.md."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def cp(k, n, a=0.05):
    lo = beta_dist.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


path = sys.argv[1] if len(sys.argv) > 1 else "bench_K30.jsonl"
rows = [json.loads(l) for l in open(path)]
D = pd.DataFrame(rows)
for c in ("t1", "t2", "t3", "t4"):
    D[c] = D[c].fillna(False).astype(bool)
D["boot_sel"] = D["boot_sel"].fillna(0.0)
print(f"datasets: {len(D)}")
REG = ["R1", "R2", "R3", "R4a", "R4b"]

# ---------------------------------------------------------------- Track B verdicts
vt = pd.crosstab(D.regime, D.verdict_B, normalize="index").reindex(REG)
vc = pd.crosstab(D.regime, D.verdict_B).reindex(REG)
print("\n=== Track B verdict distribution (fraction) ===")
print(vt.round(3).to_string())
print(vc.to_string())

claim = D.verdict_B == "STRUCTURE-SUPPORTED"
D["claim_full"] = claim
correct = {
    "R1": D.verdict_B.isin(["NO-STRUCTURE", "INCONCLUSIVE"]),
    "R2": D.verdict_B.isin(["NO-STRUCTURE", "INCONCLUSIVE"]),
    "R3": D.verdict_B == "STRUCTURE-SUPPORTED",
    "R4a": D.verdict_B == "SUPPORTED-UNATTRIBUTED",
    "R4b": D.verdict_B == "SUPPORTED-UNATTRIBUTED",
}
D["exact_ok"] = False
for r, m in correct.items():
    D.loc[D.regime == r, "exact_ok"] = m[D.regime == r]

# ---------------------------------------------------------------- ablation
sel = D.selected
abl = {
    "1 model selection only": sel,
    "2 Test 3 only": sel & D.t3,
    "3 Test 4 only": sel & D.t4,
    "4 cluster bootstrap only": sel & (D.boot_sel >= 0.5),
    "5 Test 1 only": sel & D.t1,
    "6 Test 2 only": sel & D.t2,
    "7 Tests 1+2": sel & D.t1 & D.t2,
    "8 Tests 3+4": sel & D.t3 & D.t4,
    "9 full protocol": sel & D.t1 & D.t2 & D.t3 & D.t4,
    "Stage 0 only (all designs identifiable)": sel & True,
}
out = []
for name, c in abl.items():
    row = {"rule": name}
    for r in REG:
        m = D.regime == r
        row[r] = c[m].mean()
    pos = D.regime == "R3"
    row["sensitivity"] = c[pos].mean()
    row["specificity"] = 1 - c[~pos].mean()
    row["balanced_acc"] = (row["sensitivity"] + row["specificity"]) / 2
    out.append(row)
A = pd.DataFrame(out)
print("\n=== Track B ablation: P(claims physical structure) by regime ===")
print(A.round(3).to_string(index=False))
A.to_csv("ablation_trackB.csv", index=False)

# ---------------------------------------------------------------- primary metrics
print("\n=== Full protocol, primary metric with Clopper-Pearson 95% CI ===")
for r in REG:
    m = D.regime == r
    k, n = int(claim[m].sum()), int(m.sum())
    lo, hi = cp(k, n)
    print(f"{r:4s} P(STRUCTURE-SUPPORTED)={k}/{n}={k/n:.3f} [{lo:.3f},{hi:.3f}]  exact-correct={D.exact_ok[m].mean():.3f}")

# ---------------------------------------------------------------- Track A
print("\n=== Track A ===")
ta = pd.crosstab(D.regime, D.verdict_A, normalize="index").reindex(REG)
print(ta.round(3).to_string())
nt = D.verdict_A == "NON-TRANSPORTABLE"
for r in ("R1", "R2"):
    m = D.regime == r
    k, n = int(nt[m].sum()), int(m.sum()); lo, hi = cp(k, n)
    print(f"{r} P(NON-TRANSPORTABLE)={k/n:.3f} [{lo:.3f},{hi:.3f}]")
trA = []
flags = {
    "(a) LOSO only": D.loso_r2 <= 0,
    "(b) IM only": (~D.im_excl0) | (D.im_n_canonical > 1),
    "(c) conformal only": D.logo_min < 0.80,
    "(d) full Track A (not report-as-physics)": D.verdict_A != "REPORT-AS-PHYSICS",
}
for name, f in flags.items():
    trA.append({"rule": name, "flag_R1(FPR)": f[D.regime == "R1"].mean(),
                "flag_R2(sens)": f[D.regime == "R2"].mean()})
TA = pd.DataFrame(trA)
print(TA.round(3).to_string(index=False))
TA.to_csv("ablation_trackA.csv", index=False)

# ---------------------------------------------------------------- by factor
print("\n=== Full protocol by factor: P(STRUCTURE-SUPPORTED) ===")
for fct in ("level", "sigma", "scale", "F"):
    t = D.groupby(["regime", fct]).claim_full.mean().unstack(fct).reindex(REG)
    print(f"-- {fct}"); print(t.round(3).to_string())
print("\n=== Full protocol by factor: exact-verdict correct ===")
for fct in ("level", "sigma", "scale", "F"):
    t = D.groupby(["regime", fct]).exact_ok.mean().unstack(fct).reindex(REG)
    print(f"-- {fct}"); print(t.round(3).to_string())

# ---------------------------------------------------------------- Test 1 threshold sensitivity
print("\n=== Test 1 threshold sensitivity: P(STRUCTURE-SUPPORTED) with alternative (single/minority) ===")
keys = sorted({k for g in D.t1_grid.dropna() for k in g})
sens = []
for k in keys:
    t1k = D.t1_grid.apply(lambda g: bool(g.get(k)) if isinstance(g, dict) else False)
    c = sel & t1k & D.t2 & D.t3 & D.t4
    ua = sel & (~t1k) & (~D.t2) & D.t3 & D.t4
    row = {"single/minority": k}
    for r in REG:
        m = D.regime == r
        row[f"claim_{r}"] = c[m].mean()
    row["unattr_R4a"] = ua[D.regime == "R4a"].mean(); row["unattr_R4b"] = ua[D.regime == "R4b"].mean()
    sens.append(row)
T1 = pd.DataFrame(sens)
print(T1.round(3).to_string(index=False))
T1.to_csv("test1_threshold_sensitivity.csv", index=False)
D.drop(columns=["t1_grid", "t4_sel"]).to_csv("bench_flat.csv", index=False)
