"""Analysis of the v2 benchmark (CORRECTIONS_v2.md). Two Track-B endpoints:
(a) discriminant label (verdict_B_disc) and (b) final capped verdict (verdict_B)."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def cp(k, n, a=0.05):
    lo = beta_dist.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


path = sys.argv[1] if len(sys.argv) > 1 else "bench_v2_K20.jsonl"
D = pd.DataFrame([json.loads(l) for l in open(path)])
for c in ("t1", "t2", "t3", "t4"):
    D[c] = D[c].fillna(False).astype(bool)
D["boot_sel"] = D["boot_sel"].fillna(0.0)
REG = ["R1", "R2", "R4a", "R4b", "R4c", "R3", "R3w", "R3m", "R3s"]
GEN = ["R3", "R3w", "R3m", "R3s"]
NUL = ["R1", "R2", "R4a", "R4b", "R4c"]
D = D[D.regime.isin(REG)]
print(f"datasets: {len(D)}  per regime: {D.regime.value_counts().to_dict()}")

print("\n=== (a) discriminant label ===")
print(pd.crosstab(D.regime, D.verdict_B_disc, normalize="index").reindex(REG).round(3).to_string())
print(pd.crosstab(D.regime, D.verdict_B_disc).reindex(REG).to_string())
print("\n=== (b) final capped verdict of S ===")
print(pd.crosstab(D.regime, D.verdict_B, normalize="index").reindex(REG).round(3).to_string())
print(pd.crosstab(D.regime, D.verdict_B).reindex(REG).to_string())

sel = D.selected
disc = D.verdict_B_disc == "STRUCTURE-SUPPORTED"
final_phys = D.verdict_B == "STRUCTURE-AS-PHYSICS"
cf = D.conformal_fail.astype(bool)
print("\n=== primary metrics with Clopper-Pearson 95% CI ===")
for name, c in [("discriminant STRUCTURE-SUPPORTED", disc), ("final STRUCTURE-AS-PHYSICS", final_phys)]:
    print(f"-- {name}")
    for r in REG:
        m = D.regime == r
        k, n = int(c[m].sum()), int(m.sum()); lo, hi = cp(k, n)
        print(f"   {r:4s} {k:4d}/{n} = {k/n:.3f} [{lo:.3f},{hi:.3f}]")
    k = int(c[D.regime.isin(NUL)].sum()); n = int(D.regime.isin(NUL).sum()); lo, hi = cp(k, n)
    print(f"   null regimes pooled {k}/{n} = {k/n:.4f} [{lo:.4f},{hi:.4f}]")

rules = {
    "model selection only": sel,
    "+ cluster bootstrap": sel & (D.boot_sel >= 0.5),
    "+ Test 3": sel & D.t3,
    "+ Test 4": sel & D.t4,
    "+ Tests 3,4": sel & D.t3 & D.t4,
    "+ Test 1": sel & D.t1,
    "+ Test 2": sel & D.t2,
    "+ Tests 1,2": sel & D.t1 & D.t2,
    "full discriminant (T1-4)": disc,
    "full minus Test 1": sel & D.t2 & D.t3 & D.t4,
    "full minus Test 2": sel & D.t1 & D.t3 & D.t4,
    "full minus Test 3": sel & D.t1 & D.t2 & D.t4,
    "full minus Test 4": sel & D.t1 & D.t2 & D.t3,
    "full with conformal cap (as physics)": final_phys,
}
rows = []
for name, c in rules.items():
    row = {"rule": name}
    for r in REG:
        row[r] = c[D.regime == r].mean()
    row["null_pooled"] = c[D.regime.isin(NUL)].mean()
    rows.append(row)
A = pd.DataFrame(rows)
print("\n=== ablation: fraction presenting a structure as physical ===")
print(A.round(3).to_string(index=False))
A.to_csv("ablation_v2.csv", index=False)

print("\n=== Track A ===")
print(pd.crosstab(D.regime, D.verdict_A, normalize="index").reindex(REG).round(3).to_string())
nt = D.verdict_A == "NON-TRANSPORTABLE"
for r in ("R1", "R2"):
    m = D.regime == r; k, n = int(nt[m].sum()), int(m.sum()); lo, hi = cp(k, n)
    print(f"{r} NON-TRANSPORTABLE {k}/{n}={k/n:.3f} [{lo:.3f},{hi:.3f}]")
TA = pd.DataFrame([{"rule": nm, "R1": f[D.regime == "R1"].mean(), "R2": f[D.regime == "R2"].mean()} for nm, f in {
    "LOSO only": D.loso_r2 <= 0,
    "IM only (CI incl. 0 or not exactly one regime)": (~D.im_excl0) | (D.im_n_canonical != 1),
    "conformal only": cf,
    "full Track A (not report-as-physics)": D.verdict_A != "REPORT-AS-PHYSICS"}.items()])
print(TA.round(3).to_string(index=False)); TA.to_csv("ablation_trackA_v2.csv", index=False)
print("conformal fail rate by regime:", D.groupby("regime").conformal_fail.mean().reindex(REG).round(3).to_dict())
print("components of conformal failure (R1): logo<0.8", (D[D.regime=="R1"].logo_min < 0.8).mean().round(3),
      " n_eff<30", (D[D.regime=="R1"].n_eff < 30).mean().round(3), " maxw>=0.5", (D[D.regime=="R1"].max_w_share >= 0.5).mean().round(3))

print("\n=== by factor: discriminant STRUCTURE-SUPPORTED ===")
D["disc"] = disc
for f in ("level", "sigma", "scale", "F"):
    print(f"-- {f}"); print(D.groupby(["regime", f]).disc.mean().unstack(f).reindex(REG).round(3).to_string())

print("\n=== Test 1 threshold sensitivity (discriminant label) ===")
keys = sorted({k for g in D.t1_grid.dropna() for k in g})
out = []
for k in keys:
    t1k = D.t1_grid.apply(lambda g: bool(g.get(k)) if isinstance(g, dict) else False)
    c = sel & t1k & D.t2 & D.t3 & D.t4
    ua = sel & (~t1k) & (~D.t2) & D.t3 & D.t4
    row = {"thr": k}
    for r in REG:
        row[r] = c[D.regime == r].mean()
    row["UA_R4b"] = ua[D.regime == "R4b"].mean()
    out.append(row)
T = pd.DataFrame(out); print(T.round(3).to_string(index=False)); T.to_csv("test1_sens_v2.csv", index=False)
D.drop(columns=["t1_grid", "t4_sel"]).to_csv("bench_v2_flat.csv", index=False)
