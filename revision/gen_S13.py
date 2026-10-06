"""Generates supp_S13.tex (full tables of the version-2 benchmark) from bench_v2_K20.jsonl."""
import json
import pandas as pd

D = pd.DataFrame([json.loads(l) for l in open("bench_v2_K20.jsonl")])
for c in ("t1", "t2", "t3", "t4"):
    D[c] = D[c].fillna(False).astype(bool)
REG = ["R1", "R2", "R4a", "R4b", "R4c", "R3", "R3w", "R3m", "R3s"]
EXPECT = {r: "NO" for r in ("R1", "R2", "R4a", "R4c")}
EXPECT.update(R4b="SUPPORTED-UNATTRIBUTED", R3="STRUCTURE-SUPPORTED", R3w="STRUCTURE-SUPPORTED",
              R3m="STRUCTURE-SUPPORTED", R3s="STRUCTURE-SUPPORTED")
D["P"] = D.verdict_B_disc == "STRUCTURE-SUPPORTED"
D["E"] = [(not p) if EXPECT[r] == "NO" else (v == EXPECT[r])
          for r, p, v in zip(D.regime, D.P, D.verdict_B_disc)]
sel = D.selected.astype(bool)
from decimal import Decimal, ROUND_HALF_UP
f = lambda x: str(Decimal(repr(round(float(x), 10))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
L = []
L.append(r"""\section{S13: Simulation benchmark (version 2), full tables}
Generator: \texttt{revision/benchmark\_v2.py}; implementation:
\texttt{revision/audit\_v2.py}; corrections to the first (frozen) implementation and
expected outcomes recorded before the run: \texttt{revision/CORRECTIONS\_v2.md};
per-run outputs: \texttt{revision/bench\_v2\_K20.jsonl}; analysis:
\texttt{revision/analyze\_v2.py}; these tables:
\texttt{revision/gen\_S13.py}. The first implementation
(\texttt{audit\_frozen.py}, \texttt{FROZEN\_PROTOCOL.md}, K=30, five regimes) is kept
unchanged and is not reported. Nine regimes $\times$ $16$ cells $\times$ $20$
replicates $=2{,}880$ data sets; master seed $20261002$. Effect levels
(low/high): R2 facility-offset SD $0.15/0.40$; R3, R3m, R3s exponent jump
$0.05/0.15$; R3w $0.005/0.015$; R4a exponent shift of the banded facility
$0.05/0.15$; R4b, R4c additive artefact $0.15/0.40$. Geometry prefactors
$\log C_g=-2+\mathcal N(0,0.15^2)$; Mach $U(0.05,0.35)$ except the R4a banded
facility, $U(0.40,0.65)$, and R4b, where all points are $U(0.12,0.35)$ except
the artefact subset, $U(0.05,0.10)$; R3s uses the same Mach distribution as R4b (about $8\%$ of the points of every facility moved to $U(0.05,0.10)$, below the genuine threshold $0.11$). Mach is drawn from distributions that span both sides of the threshold in every facility, but a given corpus need not contain points on both sides in every facility. Proportions are rounded half up to three decimals. Four regimes (R3w, R3s, R3m, R4c) and
the corrections were added after the results of the first run had been seen.
""")
# by factor
cols = [("level", 0, "effect low"), ("level", 1, "effect high"), ("sigma", 0.03, r"$\sigma=0.03$"),
        ("sigma", 0.08, r"$\sigma=0.08$"), ("scale", 1, r"$n\times1$"), ("scale", 2, r"$n\times2$"),
        ("F", 4, r"$F=4$"), ("F", 6, r"$F=6$")]
hdr = " & ".join(rf"\multicolumn{{2}}{{c}}{{{c[2]}}}" for c in cols)
L.append(r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}\caption{Track~B (discriminant endpoint): fraction labelled \textsc{structure-supported} (P) and fraction with the expected label (E), by factor level. Expected: no \textsc{structure-supported} for R1, R2, R4a, R4c; \textsc{supported-unattributed} for R4b; \textsc{structure-supported} for R3, R3w, R3m, R3s (R3s is anticipated to fail by design).}")
L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "rr" * 8 + r"}\toprule")
L.append("Regime & " + hdr + r"\\")
L.append(" & " + " & ".join("P & E" for _ in cols) + r"\\\midrule")
for r in REG:
    d = D[D.regime == r]
    cells = []
    for k, v, _ in cols:
        s = d[d[k] == v]
        cells += [f(s.P.mean()), f(s.E.mean())]
    L.append(r + " & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule\end{tabular}}\end{table}")
# T1 sensitivity
keys = sorted({k for g in D.t1_grid.dropna() for k in g})
L.append(r"\begin{table}[H]\centering\scriptsize\caption{Sensitivity to the Test~1 thresholds (single-facility share / minority share). Fraction labelled \textsc{structure-supported} (discriminant endpoint) in each regime, and fraction \textsc{supported-unattributed} in R4a and R4b. The rule used elsewhere is 0.90/0.10. Changing the limit changes which datasets are \textsc{supported-unattributed}, so verdict distributions differ slightly between 0.10 and 0.15 (R4b: see last column).}")
L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrrrrrr}\toprule")
L.append(r"Thresholds & R1 & R2 & R4a & R4b & R4c & R3 & R3w & R3m & R3s & S-U R4a & S-U R4b\\\midrule")
for k in keys:
    t1k = D.t1_grid.apply(lambda g: bool(g.get(k)) if isinstance(g, dict) else False)
    c = sel & t1k & D.t2 & D.t3 & D.t4
    ua = sel & (~t1k) & (~D.t2) & D.t3 & D.t4
    row = [f(c[D.regime == r].mean()) for r in REG]
    row += [f(ua[D.regime == "R4a"].mean()), f(ua[D.regime == "R4b"].mean())]
    L.append(k + " & " + " & ".join(row) + r" \\")
L.append(r"\bottomrule\end{tabular}}\end{table}")
# Track A
TA = pd.crosstab(D.regime, D.verdict_A, normalize="index").reindex(REG).fillna(0)
L.append(r"\begin{table}[H]\centering\small\caption{Track~A verdicts by regime. Only R1 (transportable) and R2 (not transportable) have a truth defined in advance; the other regimes are descriptive.}\begin{tabular}{lrrr}\toprule")
L.append(r"Regime & \textsc{non-transportable} & \textsc{report-with-disclaimer} & \textsc{report-as-physics}\\\midrule")
for r in REG:
    L.append(f"{r} & {f(TA.loc[r].get('NON-TRANSPORTABLE', 0))} & {f(TA.loc[r].get('REPORT-WITH-DISCLAIMER', 0))} & {f(TA.loc[r].get('REPORT-AS-PHYSICS', 0))} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
# Track A ablation
IMEX = lambda s: (~s.im_excl0.astype(bool)) | (s.im_n_canonical != 1)
rows = []
for name, fn in [("(a) LOSO only", lambda s: s.loso_r2 <= 0),
                 ("(b) IM only (CI includes 0 or not exactly one canonical regime)", IMEX),
                 ("(c) conformal only", lambda s: s.conformal_fail.astype(bool)),
                 ("(d) full Track A (not report-as-physics)", lambda s: s.verdict_A != "REPORT-AS-PHYSICS")]:
    rows.append((name, fn(D[D.regime == "R1"]).mean(), fn(D[D.regime == "R2"]).mean()))
L.append(r"\begin{table}[H]\centering\small\caption{Track~A component ablation: flag rate in R1 (false positives) and in R2 (detection of a pure facility shift).}\begin{tabular}{lrr}\toprule")
L.append(r"Rule & R1 & R2\\\midrule")
for n, a, b in rows:
    L.append(f"{n} & {f(a)} & {f(b)} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
open("../manuscript_R1/supp_S13.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L[-14:]))
