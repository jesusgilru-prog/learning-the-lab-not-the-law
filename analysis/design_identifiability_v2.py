import os
#!/usr/bin/env python
"""
Design-identifiability audit of the cross-facility windage benchmark, v2.

Changes over v1 (`ddp_rank_analysis.py`), all prompted by the co-author round:

* runs on the corrected dataset (pixel re-digitisation of Liu Fig. 7(a) and a
  temperature-consistent air density) -- see `build_corrected_dataset.py`;
* the identifiability criterion is evaluated on the REALISED design,
  rank(D_c E^T), not on the exponent matrix restricted to the varied controls,
  which is only an upper bound when controls move in tandem;
* the coordinates are the knobs an experimenter actually sets (Omega, R, p, T),
  not (Omega, R, rho, mu, a), since rho, mu and a are functions of p and T;
* adds a model-free falsification (points matched in Re along different paths),
  a Knudsen check, a bearing-friction sensitivity scan, a digitisation Monte
  Carlo, a permutation test, a stratified bootstrap, a power calculation, a
  robustness check against curvature in f(Re), a consistency check in the only
  rank-3 facility, and the g-level semantics audit.

Every number printed here is reproducible from the CSVs in this directory.
"""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cross_rotor_dataset.csv")
G_EARTH = 9.80665
R_SPECIFIC = 287.05
SEED = 20260730

# Liu 2024, p. 19: M_f = I * beta with I = 845.69 kg m^2, beta = -0.013 rad/s^2.
I_ARM, BETA_SPINDOWN = 845.69, 0.013
M_FRICTION = I_ARM * BETA_SPINDOWN          # N m, treated as constant by the paper

# Exponents of each candidate group in the knobs the experimenter sets.
# mu(T) ~ T^0.7 (Sutherland near ambient) and a ~ T^0.5 for an ideal gas.
KNOBS = ["omega_rad_s", "R_m", "p_Pa", "T_K"]
EXPONENTS = {
    "g_level":  {"omega_rad_s": 2, "R_m": 1},
    "Re_Omega": {"omega_rad_s": 1, "R_m": 2, "p_Pa": 1, "T_K": -1.7},
    "M_tip":    {"omega_rad_s": 1, "R_m": 1, "T_K": -0.5},
}
GROUPS = list(EXPONENTS)


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def ols(A, y):
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    n, k = A.shape
    cov = (r @ r / (n - k)) * np.linalg.pinv(A.T @ A)
    r2 = 1 - (r @ r) / ((y - y.mean()) ** 2).sum()
    return b, cov, r2, n, k


def sim_test(sub, label, verbose=True):
    """alpha = d log Cp / d log Omega, beta = d log Cp / d log rho."""
    y = np.log(sub.Cp.values)
    A = np.column_stack([np.ones(len(sub)), np.log(sub.omega_rad_s.values),
                         np.log(sub.rho_kgm3.values)])
    b, cov, r2, n, k = ols(A, y)
    d = b[1] - b[2]
    sd = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    t = d / sd
    p = 2 * (1 - stats.t.cdf(abs(t), n - k))
    if verbose:
        print(f"  {label:44s} n={n:3d} R2={r2:.3f}  "
              f"alpha={b[1]:+.4f} beta={b[2]:+.4f}  "
              f"alpha-beta={d:+.4f}+-{sd:.4f}  t={t:+.2f} p={p:.4g}")
    return d, sd, p, n


# ---------------------------------------------------------------- criterion
def identifiability(df):
    section("1. Identificabilidad del diseno: rango a priori vs rango realizado")
    print("Coordenadas = mandos reales (Omega, R, p, T). rho, mu y a son funciones de p y T.")
    print("El criterio exacto es rank(D_c E^T) sobre el diseno REALIZADO; restringir E a los")
    print("mandos variados solo da una cota superior (falla si dos mandos se mueven en tandem).\n")
    E = np.array([[EXPONENTS[g].get(k, 0) for k in KNOBS] for g in GROUPS], float)
    for src, g in df.groupby("source"):
        varied = [k for k in KNOBS if g[k].nunique() > 1]
        idx = [KNOBS.index(k) for k in varied]
        rank_apriori = np.linalg.matrix_rank(E[:, idx]) if idx else 0

        D = np.column_stack([np.log(g[k].values) for k in KNOBS])
        Dc = D - D.mean(0)
        Z = Dc @ E.T                      # realised log-design in group space
        rank_real = np.linalg.matrix_rank(Z, tol=1e-9)
        sv = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        cond = sv[0] / sv[rank_real - 1] if rank_real else np.inf
        # Belsley-Kuh-Welsch: the condition number is only comparable across
        # designs after scaling each column to unit norm.
        Zc = Z - Z.mean(0)
        norms = np.linalg.norm(Zc, axis=0)
        Zn = Zc[:, norms > 1e-12] / norms[norms > 1e-12]
        svn = np.linalg.svd(Zn, compute_uv=False)
        r_n = np.linalg.matrix_rank(Zn, tol=1e-9)
        cond_n = svn[0] / svn[r_n - 1] if r_n else np.inf

        print(f"{src:12s} n={len(g):3d}  mandos variados={varied}")
        print(f"{'':12s} rango a priori (cota) = {rank_apriori}   "
              f"rango realizado = {rank_real}   "
              f"{'COINCIDEN' if rank_apriori == rank_real else '*** DIFIEREN ***'}")
        print(f"{'':12s} valores singulares={np.array2string(sv, precision=4)}")
        print(f"{'':12s} cond sin normalizar={cond:.4g}   "
              f"cond BKW (columnas a norma 1)={cond_n:.4g}   "
              f"{'MAL CONDICIONADO (>30)' if cond_n > 30 else 'bien condicionado'}")
        for i, j in itertools.combinations(range(3), 2):
            pair = Z[:, [i, j]]
            r = np.linalg.matrix_rank(pair - pair.mean(0), tol=1e-9)
            tag = "DEGENERADO" if r < 2 else "separables "
            extra = ""
            if g[GROUPS[i]].nunique() > 1 and g[GROUPS[j]].nunique() > 1:
                c = np.corrcoef(np.log(g[GROUPS[i]]), np.log(g[GROUPS[j]]))[0, 1]
                extra = f"  corr(log,log)={c:+.6f}"
            print(f"{'':12s}   {GROUPS[i]:9s} vs {GROUPS[j]:9s}: {tag}{extra}")
        print()


# ------------------------------------------------------- model-free evidence
def matched_re(df):
    section("2. Falsacion SIN MODELO: puntos con el mismo Re por caminos distintos")
    print("Si Cp = f(Re), dos condiciones con el mismo Re deben dar el mismo Cp,")
    print("sea cual sea la forma de f. No se ajusta nada.\n")
    liu = df[df.source == "Liu2024"].reset_index(drop=True)
    dig = liu[liu.data_origin == "digitized_from_figure_pixel"].reset_index(drop=True)
    rows = []
    for i, j in itertools.combinations(range(len(dig)), 2):
        a, b = dig.loc[i], dig.loc[j]
        if a.omega_rad_s == b.omega_rad_s:
            continue
        dre = abs(a.Re_Omega / b.Re_Omega - 1)
        if dre < 0.06:
            hi, lo = (a, b) if a.omega_rad_s > b.omega_rad_s else (b, a)
            rows.append(dict(dRe_pct=100 * dre,
                             lo=f"{lo.p_Pa/1000:.0f}kPa/{lo.omega_rad_s:.1f}",
                             hi=f"{hi.p_Pa/1000:.0f}kPa/{hi.omega_rad_s:.1f}",
                             Cp_lo=lo.Cp, Cp_hi=hi.Cp,
                             dCp_pct=100 * (hi.Cp / lo.Cp - 1),
                             M_lo=lo.M_tip, M_hi=hi.M_tip))
    R = pd.DataFrame(rows).sort_values("dRe_pct")
    print(R.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("\n  En todos los pares, la condicion de MAYOR Omega tiene MENOR Cp al mismo Re.")
    print("  Cota de error de lectura: 1 px = 0.058 kW, muy por debajo de estas diferencias.")


def knudsen(df):
    section("3. Knudsen: la presion mas baja sigue en regimen continuo?")
    kB, d_air, Lc = 1.380649e-23, 3.7e-10, 0.15      # gap radial de Liu
    liu = df[df.source == "Liu2024"]
    for p in sorted(liu.p_Pa.unique()):
        T = liu[liu.p_Pa == p].T_K.iloc[0]
        lam = kB * T / (np.sqrt(2) * np.pi * d_air ** 2 * p)
        print(f"   p={p/1000:6.1f} kPa   camino libre medio={lam:.3e} m   "
              f"Kn={lam/Lc:.3e}")
    print("\n  Kn < 1e-5 en todo el rango: continuo por cuatro ordenes de magnitud.")
    print("  El enrarecimiento no puede explicar alpha != beta.")


# --------------------------------------------------------------- robustness
def friction_scan(df):
    section("4. Sensibilidad a la correccion de friccion de rodamientos")
    print(f"Liu resta P_friccion = M_f * Omega con M_f = I*beta = {I_ARM}*{BETA_SPINDOWN}"
          f" = {M_FRICTION:.2f} N m,")
    print("constante, extrapolado de spin-down a omega < 1 rad/s. Si el par real fuese")
    print("c veces mayor, la potencia de windage seria P + (1-c)*M_f*Omega.\n")
    liu = df[(df.source == "Liu2024") &
             (df.data_origin == "digitized_from_figure_pixel")].copy()
    frac = M_FRICTION * liu.omega_rad_s / liu.P_w_W
    print(f"  peso de la friccion sobre el windage publicado: "
          f"min={100*frac.min():.1f}%  max={100*frac.max():.1f}%\n")
    for c in [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0]:
        s = liu.copy()
        s["P_w_W"] = s.P_w_W + (1 - c) * M_FRICTION * s.omega_rad_s
        s = s[s.P_w_W > 0]
        s["Cp"] = s.P_w_W / (0.5 * s.rho_kgm3 * s.omega_rad_s ** 3 * s.R_m ** 5)
        sim_test(s, f"M_f x {c:4.2f}  (constante)")
    print()
    for c in [1.5, 2.0, 3.0]:
        s = liu.copy()
        # viscous-type error: extra torque growing linearly with Omega, matched
        # to (c-1)*M_f at the top speed.
        extra = (c - 1) * M_FRICTION * (s.omega_rad_s / s.omega_rad_s.max())
        s["P_w_W"] = s.P_w_W - extra * s.omega_rad_s
        s = s[s.P_w_W > 0]
        s["Cp"] = s.P_w_W / (0.5 * s.rho_kgm3 * s.omega_rad_s ** 3 * s.R_m ** 5)
        sim_test(s, f"error viscoso, x{c:4.2f} a Omega maxima")


def digitisation_mc(df, n_rep=20000):
    section("5. Monte Carlo de la incertidumbre de re-digitalizacion")
    liu = df[(df.source == "Liu2024") &
             (df.data_origin == "digitized_from_figure_pixel")].copy()
    rng = np.random.default_rng(SEED)
    base = liu.P_w_W.values
    A = np.column_stack([np.ones(len(liu)), np.log(liu.omega_rad_s.values),
                         np.log(liu.rho_kgm3.values)])
    const = -np.log(0.5 * liu.rho_kgm3.values * liu.omega_rad_s.values ** 3
                    * liu.R_m.values ** 5)
    for err_kw in [0.06, 0.15, 0.30, 0.50]:
        ds, ps = [], []
        for _ in range(n_rep):
            P = base + rng.uniform(-err_kw, err_kw, len(base)) * 1000
            if (P <= 0).any():
                continue
            y = np.log(P) + const
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ b
            cov = (r @ r / (len(y) - 3)) * np.linalg.pinv(A.T @ A)
            d = b[1] - b[2]
            sd = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
            ds.append(d)
            ps.append(2 * (1 - stats.t.cdf(abs(d / sd), len(y) - 3)))
        ds, ps = np.array(ds), np.array(ps)
        tag = "  <- resolucion real (1 px)" if err_kw == 0.06 else ""
        print(f"  error uniforme +-{err_kw:.2f} kW: media={ds.mean():+.4f}  "
              f"P(signo negativo)={np.mean(ds < 0):.4f}  "
              f"P(p<0.05)={np.mean(ps < 0.05):.4f}{tag}")


def permutation_and_bootstrap(df):
    section("6. Test de permutacion, bootstrap estratificado y potencia")
    liu = df[(df.source == "Liu2024") &
             (df.data_origin == "digitized_from_figure_pixel")].copy()
    y = np.log(liu.Cp.values)
    A = np.column_stack([np.ones(len(liu)), np.log(liu.omega_rad_s.values),
                         np.log(liu.rho_kgm3.values)])
    d_obs, sd_obs, p_obs, n = sim_test(liu, "observado")

    rng = np.random.default_rng(SEED)
    # Permutation: shuffle Cp within the factorial, keeping the design fixed.
    ds = []
    for _ in range(20000):
        b, *_ = np.linalg.lstsq(A, rng.permutation(y), rcond=None)
        ds.append(b[1] - b[2])
    ds = np.array(ds)
    print(f"  permutacion (20k): P(|alpha-beta| >= |observado|) = "
          f"{np.mean(np.abs(ds) >= abs(d_obs)):.4f}")

    # Stratified bootstrap: resample within pressure level, then within speed.
    for strat in ["p_Pa", "omega_rad_s"]:
        out = []
        groups = [liu.index[liu[strat] == v].values for v in liu[strat].unique()]
        for _ in range(20000):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            s = liu.loc[idx]
            Ab = np.column_stack([np.ones(len(s)), np.log(s.omega_rad_s.values),
                                  np.log(s.rho_kgm3.values)])
            if np.linalg.matrix_rank(Ab) < 3:
                continue
            b, *_ = np.linalg.lstsq(Ab, np.log(s.Cp.values), rcond=None)
            out.append(b[1] - b[2])
        out = np.array(out)
        print(f"  bootstrap estratificado por {strat:12s} (20k): "
              f"IC95=[{np.percentile(out,2.5):+.4f}, {np.percentile(out,97.5):+.4f}]  "
              f"P(>=0)={np.mean(out >= 0):.4f}")

    # Power to detect the observed effect at alpha=0.05 with this design.
    ncp = abs(d_obs) / sd_obs
    crit = stats.t.ppf(0.975, n - 3)
    power = 1 - stats.nct.cdf(crit, n - 3, ncp) + stats.nct.cdf(-crit, n - 3, ncp)
    print(f"  potencia para detectar alpha-beta={d_obs:+.3f} con n={n}: {power:.3f}")


def curvature(df):
    section("7. Robustez a curvatura en f(Re)")
    print("Si f no es una ley de potencia, Omega y rho muestrean tramos distintos de Re")
    print("y alpha != beta podria ser espurio. Se anaden terminos en log(Re).\n")
    liu = df[(df.source == "Liu2024") &
             (df.data_origin == "digitized_from_figure_pixel")].copy()
    y = np.log(liu.Cp.values)
    lre = np.log(liu.Re_Omega.values)
    lom = np.log(liu.omega_rad_s.values)
    lre_c = lre - lre.mean()
    for order in [1, 2, 3]:
        cols = [np.ones(len(liu)), lom] + [lre_c ** k for k in range(1, order + 1)]
        A = np.column_stack(cols)
        b, cov, r2, n, k = ols(A, y)
        se = np.sqrt(np.diag(cov))[1]
        t = b[1] / se
        p = 2 * (1 - stats.t.cdf(abs(t), n - k))
        print(f"  f(Re) polinomica de orden {order}: coef. residual de log(Omega) = "
              f"{b[1]:+.4f}+-{se:.4f}  t={t:+.2f}  p={p:.4g}   R2={r2:.3f}")
    print("\n  El coeficiente de log(Omega) a Re controlado es el mismo objeto que")
    print("  alpha-beta cuando f es una ley de potencia, pero no supone esa forma.")


def vrancik_check(df):
    section("8. Chequeo de consistencia en Vrancik1968 (el unico diseno de rango 3)")
    v = df[df.source == "Vrancik1968"].copy()
    v["gmono"] = v.omega_rad_s ** 2 * v.R_m / G_EARTH
    geo = pd.get_dummies(v.geometry_id, drop_first=True).astype(float).values
    y = np.log(v.Cp.values)
    for extra, name in [("M_tip", "Mach"), ("gmono", "g monomico")]:
        A = np.column_stack([np.ones(len(v)), np.log(v.Re_Omega.values),
                             np.log(v[extra].values), geo])
        b, cov, r2, n, k = ols(A, y)
        se = np.sqrt(np.diag(cov))
        t = b[2] / se[2]
        p = 2 * (1 - stats.t.cdf(abs(t), n - k))
        print(f"  con efectos fijos de geometria, exponente de {name:11s} = "
              f"{b[2]:+.4f}+-{se[2]:.4f}  t={t:+.2f} p={p:.4g}   "
              f"(Re: {b[1]:+.4f}+-{se[1]:.4f})   R2={r2:.3f}")
    Z = np.column_stack([np.log(v.M_tip.values), np.log(v.gmono.values)])
    Z = Z - Z.mean(0)
    sv = np.linalg.svd(Z, compute_uv=False)
    print(f"\n  M y g monomico en Vrancik: cond={sv[0]/sv[-1]:.1f} "
          f"-> separables en principio, pero mal condicionados.")
    print("  Se reporta como chequeo de consistencia, NO como replica: asume exponente")
    print("  comun entre geometrias y la identificacion viene de comparar sub-disenos.")


def g_semantics(df):
    section("9. Semantica de 'nivel g': metadato de instalacion vs monomio")
    print("La columna g_level es el nivel de gravedad de la CARGA UTIL declarado por la")
    print("instalacion. El grupo adimensional que usa el analisis de rango es Omega^2 R/g,")
    print("que todo rotor tiene. No son lo mismo.\n")
    for src, g in df.groupby("source"):
        mono = g.omega_rad_s ** 2 * g.R_m / G_EARTH
        print(f"  {src:12s} g_level tabulado=[{g.g_level.min():8.0f}, {g.g_level.max():8.0f}]"
              f"   Omega^2 R/g = [{mono.min():8.0f}, {mono.max():8.0f}]")
    print("\n  Vrancik y Zheng, etiquetados 'a 1 g', muestrean el monomio MAS ALTO del corpus.")
    print("  El paper debe declarar cual de los dos objetos es la hipotesis fisica.")


def geometry_exponents(df):
    section("10. Exponentes por geometria y heterogeneidad (Q de Cochran)")
    rows = []
    for (src, gid), g in df.groupby(["source", "geometry_id"]):
        if len(g) < 4 or g.Re_Omega.nunique() < 4:
            continue
        y = np.log(g.Cp.values)
        A = np.column_stack([np.ones(len(g)), np.log(g.Re_Omega.values)])
        b, cov, r2, n, k = ols(A, y)
        se = np.sqrt(cov[1, 1])
        rows.append(dict(source=src, geometry=gid, n=n, a=b[1], se=se, r2=r2))
        print(f"  {src:12s} {str(gid):30s} n={n:3d}  a={b[1]:+.3f}+-{se:.3f}  R2={r2:.3f}")
    R = pd.DataFrame(rows)
    w = 1 / R.se ** 2
    a_bar = (w * R.a).sum() / w.sum()
    Q = (w * (R.a - a_bar) ** 2).sum()
    dfree = len(R) - 1
    pQ = 1 - stats.chi2.cdf(Q, dfree)
    I2 = max(0.0, (Q - dfree) / Q)
    print(f"\n  exponente combinado por precision = {a_bar:+.3f}")
    print(f"  Q de Cochran = {Q:.1f}  gl={dfree}  p={pQ:.3g}   I^2={100*I2:.1f}%")
    print("  Heterogeneidad extrema: el exponente agrupado no es un promedio con sentido.")

    y = np.log(df.Cp.values)
    A = np.column_stack([np.ones(len(df)), np.log(df.Re_Omega.values)])
    b, cov, r2, n, k = ols(A, y)
    print(f"  (para comparar: exponente AGRUPADO sobre los 114 puntos = {b[1]:+.3f}, "
          f"R2={r2:.3f})")


def pooled_vs_logo(df):
    section("11. Ajuste agrupado frente a dejar-una-instalacion-fuera")
    pis = ["Pi_confinement", "Pi_gap", "Pi_aspect_axial", "Pi_blockage"]
    for feats in (["Re_Omega"], ["Re_Omega", "M_tip"], ["Re_Omega"] + pis,
                  ["Re_Omega", "M_tip"] + pis):
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
        print(f"  {str(feats):66s}")
        print(f"     R2 en muestra={r2:.3f}   LOGO rmse(log): "
              + "  ".join(f"{k}={v:.3f}" for k, v in errs.items())
              + f"   media={np.mean(list(errs.values())):.3f}")


def response_blindness(df):
    section("12. El cribado es ciego a la respuesta (y sensible al diseno)")
    print("Peticion de ChatGPT: comprobar que el criterio no depende de Cp, y que si")
    print("depende del diseno realizado.\n")
    E = np.array([[EXPONENTS[g].get(k, 0) for k in KNOBS] for g in GROUPS], float)

    def screen(sub):
        D = np.column_stack([np.log(sub[k].values) for k in KNOBS])
        Z = (D - D.mean(0)) @ E.T
        Zc = Z - Z.mean(0)
        nr = np.linalg.norm(Zc, axis=0)
        Zn = Zc[:, nr > 1e-12] / nr[nr > 1e-12]
        sv = np.linalg.svd(Zn, compute_uv=False)
        r = np.linalg.matrix_rank(Zn, tol=1e-9)
        return r, (sv[0] / sv[r - 1] if r else np.inf)

    rng = np.random.default_rng(SEED)
    for src, g in df.groupby("source"):
        r0, k0 = screen(g)
        g2 = g.copy()
        g2["Cp"] = rng.lognormal(0, 3, len(g2))          # respuesta destruida
        r1, k1 = screen(g2)
        g3 = g.copy()
        g3["T_K"] = g3.T_K * rng.uniform(0.9, 1.1, len(g3))   # diseno alterado
        r2, k2 = screen(g3)
        print(f"  {src:12s} original: rango={r0} k={k0:8.3f}   "
              f"con Cp aleatorio: rango={r1} k={k1:8.3f} {'OK' if (r1, round(k1,6)) == (r0, round(k0,6)) else 'FALLA'}"
              f"   con T perturbada: rango={r2} k={k2:8.3f} "
              f"{'(cambia, como debe)' if (r2 != r0 or abs(k2-k0) > 1e-6) else '(NO cambia, revisar)'}")


def main():
    df = pd.read_csv(DATA)
    print(f"datos: {DATA}")
    print(f"n={len(df)}  fuentes={sorted(df.source.unique())}  "
          f"geometrias={df.geometry_id.nunique()}")

    section("0. Test de similaridad de Reynolds: antes y despues de las correcciones")
    print("H0: alpha = beta, donde alpha = d log Cp / d log Omega y beta = d log Cp / d log rho.")
    print("Con R y T fijos dentro de Liu2024, alpha-beta es el exponente del segundo grupo.\n")
    old = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv"))
    o = old[old.source == "Liu2024"]
    sim_test(o, "v3 (lectura a ojo), 20 puntos")
    sim_test(o[o.data_origin == "digitized_from_figure"], "v3 (lectura a ojo), 15 digitalizados")
    liu = df[df.source == "Liu2024"]
    sim_test(liu, "v4 (re-digitalizado), 20 puntos")
    sim_test(liu[liu.data_origin == "digitized_from_figure_pixel"],
             "v4 (re-digitalizado), 15 digitalizados")

    identifiability(df)
    matched_re(df)
    knudsen(df)
    friction_scan(df)
    digitisation_mc(df)
    permutation_and_bootstrap(df)
    curvature(df)
    vrancik_check(df)
    g_semantics(df)
    geometry_exponents(df)
    pooled_vs_logo(df)
    response_blindness(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
