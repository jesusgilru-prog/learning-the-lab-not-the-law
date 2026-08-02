#!/usr/bin/env python
"""
Synthetic validation of Stage 0 (design-identifiability audit).

The case study shows the screen catching a real problem in a real corpus, but a
single corpus cannot show that the screen *generalises*, nor that it beats the
obvious heuristic (flag pairs whose log-log correlation is near one).  This
script supplies that evidence on data where the ground truth is known.

Three experiments:

  A. Attribution is the algorithm's prior, not the data.  On a design that is
     rank-deficient in (M, N_g), two different "discovered laws" fit the data
     identically, so the structure search decides between them on tie-breaking
     alone.  Adding a third design direction repairs it.

  B. Does the condition number predict recovery?  Over random designs, relate
     kappa to the actual error in the recovered exponent.

  C. Stage 0 as a gate, against the correlation heuristic.  Over many random
     (design, truth, noise) triples, compare the rank+kappa verdict and the
     |corr| > 1-eps verdict against whether the exponent is actually
     recoverable.  Reports the full confusion matrix for both.

Ground-truth model throughout:
    log Cp = c + theta_Re log Re + theta_M log M + theta_g log N_g + eps
with Re, M and N_g monomials in the controls (Omega, R, p, T).
"""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260730
G_EARTH = 9.80665
R_SPECIFIC = 287.05

GROUPS = ["Re", "M", "Ng"]
KNOBS = ["Omega", "R", "p", "T"]
# rows = groups, cols = knobs;  mu ~ T^0.7, a ~ T^0.5
E = np.array([
    [1.0, 2.0, 1.0, -1.7],      # Re = p Omega R^2 / (R_s T mu(T))
    [1.0, 1.0, 0.0, -0.5],      # M  = Omega R / a(T)
    [2.0, 1.0, 0.0,  0.0],      # Ng = Omega^2 R / g_earth
])


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------- design tools
def make_design(levels):
    """levels: dict knob -> array of values. Returns the full factorial."""
    grids = np.meshgrid(*[levels[k] for k in KNOBS], indexing="ij")
    return pd.DataFrame({k: g.ravel() for k, g in zip(KNOBS, grids)})


def group_matrix(design):
    """Realised log-design in group space, Z = D_c E^T (n x 3)."""
    D = np.log(design[KNOBS].values)
    return (D - D.mean(0)) @ E.T


def screen(design, subset=None):
    """Stage 0 verdict for a subset of groups: rank, kappa, degenerate pairs."""
    Z = group_matrix(design)
    idx = [GROUPS.index(g) for g in (subset or GROUPS)]
    Zs = Z[:, idx]
    Zc = Zs - Zs.mean(0)
    r = np.linalg.matrix_rank(Zc, tol=1e-9)
    norms = np.linalg.norm(Zc, axis=0)
    keep = norms > 1e-12
    if keep.sum() == 0:
        return 0, np.inf
    Zn = Zc[:, keep] / norms[keep]
    sv = np.linalg.svd(Zn, compute_uv=False)
    rn = np.linalg.matrix_rank(Zn, tol=1e-9)
    kappa = sv[0] / sv[rn - 1] if rn else np.inf
    return r, kappa


def corr_screen(design, subset):
    """The naive heuristic: max |corr(log Pi_i, log Pi_j)| over the subset."""
    Z = group_matrix(design)
    idx = [GROUPS.index(g) for g in subset]
    best = 0.0
    for i, j in itertools.combinations(idx, 2):
        a, b = Z[:, i], Z[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 1.0
        best = max(best, abs(np.corrcoef(a, b)[0, 1]))
    return best


def simulate(design, theta, sigma, rng):
    Z = group_matrix(design)
    return Z @ theta + rng.normal(0, sigma, len(design))


def fit(design, subset, y):
    idx = [GROUPS.index(g) for g in subset]
    Z = group_matrix(design)[:, idx]
    A = np.column_stack([np.ones(len(Z)), Z])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    return b, r @ r


# ------------------------------------------------------------- experiment A
def experiment_a():
    section("A. La atribucion la decide el prior del algoritmo, no los datos")
    rng = np.random.default_rng(SEED)

    # Liu-like design: 5 speeds x 4 pressures, R and T fixed.
    deg = make_design({"Omega": np.array([44.27, 62.61, 76.68, 88.54, 98.99]),
                       "R": np.array([1.0]),
                       "p": np.array([10e3, 30e3, 50e3, 101e3]),
                       "T": np.array([293.15])})
    # Same, plus two gas temperatures: adds the third design direction.
    rep = make_design({"Omega": np.array([44.27, 62.61, 76.68, 88.54, 98.99]),
                       "R": np.array([1.0]),
                       "p": np.array([10e3, 30e3, 50e3, 101e3]),
                       "T": np.array([273.15, 333.15])})

    theta = np.array([-0.03, -0.12, 0.0])       # only Re and Mach are real
    print(f"verdad: theta_Re={theta[0]:+.3f}  theta_M={theta[1]:+.3f}  "
          f"theta_Ng={theta[2]:+.3f}   (la gravedad NO interviene)\n")

    for name, design in [("diseno degenerado (R y T fijos)", deg),
                         ("diseno + 2 temperaturas de gas", rep)]:
        r, k = screen(design, ["Re", "M", "Ng"])
        print(f"{name}: n={len(design)}  Stage 0 -> rango={r}/3  kappa={k:.3g}  "
              f"corr max={corr_screen(design, GROUPS):.6f}")
        y = simulate(design, theta, 0.02, rng)
        rows = []
        for k_ in range(1, 4):
            for sub in itertools.combinations(GROUPS, k_):
                b, sse = fit(design, list(sub), y)
                n, npar = len(y), len(sub) + 1
                bic = n * np.log(sse / n) + npar * np.log(n)
                rows.append(dict(structure="+".join(sub), sse=sse, bic=bic,
                                 coefs=np.array2string(b[1:], precision=4)))
        R = pd.DataFrame(rows).sort_values("bic")
        print(R.to_string(index=False, float_format=lambda v: f"{v:12.6g}"))
        best = R.iloc[0]
        ties = R[np.abs(R.bic - best.bic) < 1e-6]
        print(f"  -> mejor BIC: {best.structure};  estructuras empatadas a "
              f"1e-6: {list(ties.structure)}")
        print()

    print("En el diseno degenerado, log N_g = 2 log M + cte, asi que la ley")
    print("'C_p depende de la gravedad con exponente theta_M/2' ajusta EXACTAMENTE")
    print("igual de bien que la verdadera. Ningun criterio basado en bondad de")
    print("ajuste puede separarlas: la eleccion la hace el desempate del buscador.")


# ------------------------------------------------------------- experiment B
def experiment_b(n_designs=400, sigma=0.02):
    section("B. El condicionamiento predice el error de recuperacion")
    rng = np.random.default_rng(SEED + 1)
    theta = np.array([-0.03, -0.12, 0.0])
    rows = []
    for _ in range(n_designs):
        # Random facility: speed always varies; radius and pressure vary by a
        # random amount, which is what moves the conditioning.
        spanR = 10 ** rng.uniform(-3, 0.3)          # 0.1% to 2x in radius
        spanP = 10 ** rng.uniform(-3, 1.0)          # negligible to 10x in pressure
        design = make_design({
            "Omega": np.linspace(40, 100, 5),
            "R": np.array([1.0, 1.0 * (1 + spanR)]),
            "p": np.array([101e3, 101e3 / (1 + spanP)]),
            "T": np.array([293.15]),
        })
        r, k = screen(design, ["Re", "M"])
        if r < 2:
            continue
        y = simulate(design, theta, sigma, rng)
        b, _ = fit(design, ["Re", "M"], y)
        rows.append(dict(kappa=k, err_M=abs(b[2] - theta[1]),
                         err_Re=abs(b[1] - theta[0])))
    R = pd.DataFrame(rows)
    print(f"{len(R)} disenos de rango completo, ruido sigma={sigma}\n")
    R["bin"] = pd.cut(np.log10(R.kappa), [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
                      labels=["k<3", "3-10", "10-32", "32-100", ">100"])
    g = R.groupby("bin", observed=True).agg(
        n=("kappa", "size"), kappa_med=("kappa", "median"),
        err_M_med=("err_M", "median"), err_M_p90=("err_M", lambda s: s.quantile(0.9)))
    print(g.to_string(float_format=lambda v: f"{v:10.4g}"))
    rho = stats.spearmanr(R.kappa, R.err_M)
    print(f"\n  Spearman(kappa, |error en theta_M|) = {rho.statistic:+.3f}  "
          f"p={rho.pvalue:.3g}")
    print(f"  error mediano con kappa<30: {R[R.kappa<30].err_M.median():.4f}   "
          f"con kappa>30: {R[R.kappa>30].err_M.median():.4f}   "
          f"(la verdad es |theta_M|={abs(theta[1]):.3f})")


# ------------------------------------------------------------- experiment C
def experiment_c(n_trials=3000):
    section("C. Stage 0 como filtro, frente a la heuristica de correlacion")
    print("Recuperable := |theta_M estimado - theta_M real| < |theta_M|/2,")
    print("es decir, el exponente se recupera con menos de un 50% de error.")
    print("Stage 0 dice SI si rango completo y kappa <= 30.")
    print("Heuristica dice SI si max |corr(log Pi_i, log Pi_j)| < 0.99.\n")
    rng = np.random.default_rng(SEED + 2)
    rec, v_stage0, v_corr, kap, cor = [], [], [], [], []
    for _ in range(n_trials):
        spanR = 10 ** rng.uniform(-3.5, 0.3)
        spanP = 10 ** rng.uniform(-3.5, 1.0)
        sigma = 10 ** rng.uniform(-2.5, -1.0)
        theta = np.array([rng.uniform(-0.6, 0.0), rng.uniform(-0.3, -0.05), 0.0])
        design = make_design({
            "Omega": np.linspace(40, 100, 5),
            "R": np.array([1.0, 1.0 * (1 + spanR)]),
            "p": np.array([101e3, 101e3 / (1 + spanP)]),
            "T": np.array([293.15]),
        })
        r, k = screen(design, ["Re", "M"])
        c = corr_screen(design, ["Re", "M"])
        y = simulate(design, theta, sigma, rng)
        if r < 2:
            est = np.nan
        else:
            b, _ = fit(design, ["Re", "M"], y)
            est = b[2]
        ok = np.isfinite(est) and abs(est - theta[1]) < abs(theta[1]) * 0.3
        rec.append(ok)
        v_stage0.append(r == 2 and k <= 30)
        v_corr.append(c < 0.99)
        kap.append(k)
        cor.append(c)
    R = pd.DataFrame(dict(recoverable=rec, stage0=v_stage0, corr=v_corr,
                          kappa=kap, cmax=cor))

    def confusion(col, label):
        tp = ((R[col]) & (R.recoverable)).sum()
        fp = ((R[col]) & (~R.recoverable)).sum()
        fn = ((~R[col]) & (R.recoverable)).sum()
        tn = ((~R[col]) & (~R.recoverable)).sum()
        prec = tp / max(tp + fp, 1)
        rec_ = tp / max(tp + fn, 1)
        print(f"  {label:34s} TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}   "
              f"precision={prec:.3f}  sensibilidad={rec_:.3f}  "
              f"acierto={(tp+tn)/len(R):.3f}")
        print(f"  {'':34s} tasa de falsa promesa (dice SI y no se recupera) "
              f"= {fp/max(tp+fp,1):.3f}")
        return prec, rec_

    print(f"{n_trials} ensayos.  Recuperables: {R.recoverable.mean():.3f}\n")
    confusion("stage0", "Stage 0 (rango + kappa<=30)")
    confusion("corr", "heuristica (|corr| < 0.99)")

    print("\n  Casos donde discrepan:")
    d1 = R[(R.stage0) & (~R["corr"])]
    d2 = R[(~R.stage0) & (R["corr"])]
    print(f"    Stage 0 dice SI y la heuristica NO: n={len(d1)}  "
          f"de ellos recuperables={d1.recoverable.mean() if len(d1) else float('nan'):.3f}")
    print(f"    Stage 0 dice NO y la heuristica SI: n={len(d2)}  "
          f"de ellos recuperables={d2.recoverable.mean() if len(d2) else float('nan'):.3f}")
    print("\n  RESULTADO HONESTO: con DOS grupos las dos pantallas son el MISMO")
    print("  estadistico. Para dos columnas centradas y normalizadas,")
    print("      kappa = sqrt((1+|r|)/(1-|r|)),")
    print("  de modo que |corr|<0.99 equivale exactamente a kappa<14.1. La unica")
    print("  diferencia observada es el umbral, no el criterio. La comparacion")
    print("  relevante esta en el experimento D, con tres grupos.")
    r_eq = np.sqrt((1 + R.cmax) / (1 - R.cmax).clip(lower=1e-15))
    print(f"  comprobacion numerica: max |kappa - sqrt((1+r)/(1-r))| = "
          f"{np.nanmax(np.abs(R.kappa - r_eq)):.3e}")

    # Threshold sweep, so the paper can justify kappa = 30 rather than assert it.
    print("\n  Barrido del umbral de kappa:")
    for thr in [3, 10, 20, 30, 50, 100, 300]:
        v = (R.kappa <= thr)
        tp = (v & R.recoverable).sum(); fp = (v & ~R.recoverable).sum()
        fn = (~v & R.recoverable).sum()
        print(f"    kappa<={thr:4d}: precision={tp/max(tp+fp,1):.3f}  "
              f"sensibilidad={tp/max(tp+fn,1):.3f}  "
              f"acierto={((v & R.recoverable)|(~v & ~R.recoverable)).mean():.3f}")


def experiment_d(n_trials=4000):
    section("D. Tres grupos: donde la correlacion por pares se queda ciega")
    print("Con m>=3 un grupo puede ser casi combinacion lineal de los otros dos")
    print("sin que NINGUNA correlacion por pares sea alta. La heuristica por pares")
    print("no puede verlo; el condicionamiento conjunto si.\n")
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for _ in range(n_trials):
        spanR = 10 ** rng.uniform(-3.5, 0.5)
        spanP = 10 ** rng.uniform(-3.5, 1.0)
        spanT = 10 ** rng.uniform(-3.5, -0.3)
        sigma = 10 ** rng.uniform(-2.5, -1.0)
        theta = np.array([rng.uniform(-0.6, 0.0), rng.uniform(-0.3, -0.05),
                          rng.uniform(-0.15, 0.0)])
        design = make_design({
            "Omega": np.linspace(40, 100, 4),
            "R": np.array([1.0, 1.0 * (1 + spanR)]),
            "p": np.array([101e3, 101e3 / (1 + spanP)]),
            "T": np.array([293.15, 293.15 * (1 + spanT)]),
        })
        r, k = screen(design, GROUPS)
        cmax = corr_screen(design, GROUPS)
        y = simulate(design, theta, sigma, rng)
        if r < 3:
            est = np.nan
        else:
            b, _ = fit(design, GROUPS, y)
            est = b[2]
        ok = np.isfinite(est) and abs(est - theta[1]) < abs(theta[1]) * 0.3
        rows.append(dict(recoverable=ok, kappa=k, cmax=cmax, rank=r,
                         stage0=(r == 3 and k <= 30), corr=(cmax < 0.99)))
    R = pd.DataFrame(rows)
    print(f"{n_trials} ensayos con los tres grupos. Recuperables: "
          f"{R.recoverable.mean():.3f}\n")

    def confusion(col, label):
        tp = ((R[col]) & (R.recoverable)).sum()
        fp = ((R[col]) & (~R.recoverable)).sum()
        fn = ((~R[col]) & (R.recoverable)).sum()
        tn = ((~R[col]) & (~R.recoverable)).sum()
        print(f"  {label:34s} TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}   "
              f"precision={tp/max(tp+fp,1):.3f}  sensibilidad={tp/max(tp+fn,1):.3f}  "
              f"acierto={(tp+tn)/len(R):.3f}")

    confusion("stage0", "Stage 0 (rango + kappa<=30)")
    confusion("corr", "heuristica por pares (|corr|<0.99)")

    blind = R[(R.cmax < 0.99) & (R.kappa > 30)]
    print(f"\n  Zona ciega de la heuristica (todas las correlaciones por pares")
    print(f"  por debajo de 0.99 pero kappa conjunto > 30): n={len(blind)} "
          f"({100*len(blind)/len(R):.1f}% de los ensayos)")
    if len(blind):
        print(f"    de esos, recuperables = {blind.recoverable.mean():.3f}  "
              f"(la heuristica los da todos por buenos)")
        print(f"    correlacion maxima por pares en esa zona: "
              f"mediana={blind.cmax.median():.4f}  max={blind.cmax.max():.4f}")
        print(f"    kappa conjunto en esa zona: mediana={blind.kappa.median():.1f}  "
              f"max={blind.kappa.max():.3g}")


def main():
    experiment_a()
    experiment_b()
    experiment_c()
    experiment_d()
    return 0


if __name__ == "__main__":
    sys.exit(main())
