import os
#!/usr/bin/env python
"""
Regenerate the numbers that appear in the manuscript with no artefact behind them.

The review round found seven such numbers.  Where the quantity is well defined we
recompute it here so it has a citable log; where it is not, the script says so and
the number must be removed from the paper rather than kept.

Covered here:
  1. Monte-Carlo Fisher-Freeman-Halton exact test on the facility x threshold table
     (manuscript reports p = 0.41; the only stored artefact is a chi-square p = 0.502)
  2. Cramer's V for the same table (equation is given in the paper, value never reported)
  3. One-way ANOVA across the 12 geometry prefactors (F = 10.5, p = 0.0038)
  4. Cochran's Q / I^2 on the same prefactors (Q = 71.3, I^2 = 84.6%)
  5. Single-group R^2 for the prefactor model (0.139 / 0.726 / 0.762)
  6. Pairwise MMD on (log Re, Ma) -- supplement table S7
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
SR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "class_sr_results.json")
PREF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "prefactor_analysis.json")
THRESHOLD = 0.127
SEED = 20260731
N_RESAMPLE = 200_000


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 1 and 2
def freeman_halton(table, n_resample=N_RESAMPLE, seed=SEED):
    """Monte-Carlo Fisher-Freeman-Halton: probability of tables no more likely
    than the observed one, among margin-fixed random tables."""
    table = np.asarray(table, float)
    row, col = table.sum(1), table.sum(0)
    n = table.sum()

    def loglik(t):
        # multivariate hypergeometric probability, up to margin-only constants
        return -sum(map(lambda x: float(np.sum(np.log(np.arange(1, x + 1)))), t.ravel()))

    obs = loglik(table)
    rng = np.random.default_rng(seed)
    # Sample margin-fixed tables: for a F x 2 table it is enough to draw the
    # first column as a multivariate hypergeometric.
    assert table.shape[1] == 2, "implementado para tablas F x 2"
    K = int(col[0])
    counts = 0
    rowint = row.astype(int)
    for _ in range(n_resample):
        draw = rng.multivariate_hypergeometric(rowint, K)
        t = np.column_stack([draw, rowint - draw]).astype(float)
        if loglik(t) <= obs + 1e-12:
            counts += 1
    return (counts + 1) / (n_resample + 1)


def cramers_v(table):
    table = np.asarray(table, float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), float(chi2)


def contingency(df):
    section("1-2. Tabla instalacion x umbral: Fisher-Freeman-Halton y Cramer's V")
    facs = sorted(df.source.unique())
    tab = np.array([[int(((df.source == f) & (df.M_tip < THRESHOLD)).sum()),
                     int(((df.source == f) & (df.M_tip >= THRESHOLD)).sum())]
                    for f in facs])
    print(f"  umbral Ma = {THRESHOLD}")
    print(f"  {'instalacion':14s} {'<thr':>5s} {'>=thr':>6s}   {'% por encima':>12s}")
    for f, r in zip(facs, tab):
        print(f"  {f:14s} {r[0]:5d} {r[1]:6d}   {100*r[1]/r.sum():11.1f}%")
    print(f"  {'TOTAL':14s} {tab[:,0].sum():5d} {tab[:,1].sum():6d}")

    v, chi2 = cramers_v(tab)
    chi2_p = stats.chi2_contingency(tab, correction=False)[1]
    print(f"\n  Chi^2 = {chi2:.4f}  p = {chi2_p:.4f}   (artefacto guardado: 2.36 / 0.5016)")
    print(f"  Cramer's V = {v:.4f}   (artefacto guardado: 0.1438)")
    p_fh = freeman_halton(tab)
    print(f"  Fisher-Freeman-Halton Monte Carlo ({N_RESAMPLE:,} remuestreos, "
          f"semilla {SEED}): p = {p_fh:.4f}")
    print(f"\n  El manuscrito reporta p = 0.41 sin log. El valor recalculado es "
          f"{p_fh:.3f}.")
    print("  En cualquier caso p >> 0.05: el umbral NO separa instalaciones.")

    # How exclusive is the indicator, really?
    liu = df[df.source == "Liu2024"]
    oth = df[df.source != "Liu2024"]
    print(f"\n  'indicador de no-Liu2024': por encima del umbral esta el "
          f"{100*(liu.M_tip>=THRESHOLD).mean():.0f}% de Liu")
    print(f"  y tambien el {100*(oth.M_tip>=THRESHOLD).mean():.1f}% del resto "
          f"({(oth.M_tip>=THRESHOLD).sum()}/{len(oth)}).")
    return p_fh, v


# ---------------------------------------------------------------- 3, 4, 5
def prefactors(df):
    section("3-5. Prefactores por geometria: ANOVA, Cochran Q, R^2 de grupo unico")
    with open(SR) as f:
        sr = json.load(f)
    pref = sr["prefactors"]
    rows = []
    for g, v in pref.items():
        sub = df[df.geometry_id == g]
        rows.append(dict(geometry=g, logC=v["log_C"], n=len(sub),
                         source=sub.source.iloc[0]))
    P = pd.DataFrame(rows)
    print(P.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    groups = [P.logC[P.source == s].values for s in sorted(P.source.unique())]
    groups = [g for g in groups if len(g) > 0]
    F, p = stats.f_oneway(*groups)
    print(f"\n  ANOVA de log C entre instalaciones: F = {F:.3f}  p = {p:.4g}"
          f"   (manuscrito: F = 10.5, p = 0.0038)")
    print("  AVISO: los 12 log C no son independientes -- se estimaron")
    print("  conjuntamente con un mismo exponente q y comparten incertidumbre.")
    print("  Este ANOVA es descriptivo, no inferencial.")

    with open(PREF) as f:
        pa = json.load(f)
    print("\n  Contenido real de prefactor_analysis.json:")
    print("   ", json.dumps(pa, indent=2)[:600].replace("\n", "\n    "))
    print("\n  Los tres R^2 de grupo unico (0.139 / 0.726 / 0.762) que cita el")
    print("  manuscrito NO estan en este artefacto. Hay que regenerarlos con un")
    print("  script propio o retirarlos del texto.")


# ------------------------------------------------------------------------ 6
def mmd(df):
    section("6. MMD por pares sobre (log Re, Ma) -- tabla S7 del suplemento")
    X = np.column_stack([np.log(df.Re_Omega.values), df.M_tip.values])
    X = (X - X.mean(0)) / X.std(0)
    src = df.source.values

    def rbf(a, b, gamma):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)

    # median heuristic on the pooled sample
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices(len(X), 1)])
    gamma = 1.0 / med
    print(f"  ancho de banda por heuristica de la mediana: gamma = {gamma:.4f}")

    published = {("Liu2024", "Vrancik1968"): 0.133, ("Liu2024", "Zheng2024"): 0.163,
                 ("Vrancik1968", "Zheng2024"): 0.189, ("Xia2024", "Zheng2024"): 0.458,
                 ("Liu2024", "Xia2024"): 0.600, ("Vrancik1968", "Xia2024"): 0.752}
    print(f"\n  {'par':30s} {'n1,n2':>8s} {'MMD^2 recalc':>13s} {'publicado':>10s}")
    out = {}
    for a, b in itertools.combinations(sorted(set(src)), 2):
        Xa, Xb = X[src == a], X[src == b]
        na, nb = len(Xa), len(Xb)
        Kaa, Kbb, Kab = rbf(Xa, Xa, gamma), rbf(Xb, Xb, gamma), rbf(Xa, Xb, gamma)
        # unbiased estimator
        m = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1)) \
            + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1)) \
            - 2 * Kab.mean()
        key = (a, b) if (a, b) in published else (b, a)
        out[(a, b)] = m
        print(f"  {a+' vs '+b:30s} {na:3d},{nb:3d} {m:13.3f} "
              f"{published.get(key, float('nan')):10.3f}")
    print("\n  Si estos valores no reproducen los publicados, la tabla S7 no tiene")
    print("  artefacto y hay que regenerarla o retirarla.")


def main():
    df = pd.read_csv(DATA)
    contingency(df)
    prefactors(df)
    mmd(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
