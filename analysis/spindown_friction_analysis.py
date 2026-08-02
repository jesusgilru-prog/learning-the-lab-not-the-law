import os
#!/usr/bin/env python
"""Bearing-friction torque M_f(omega) from Liu 2024 Fig. 6, and what it does
to the alpha != beta finding.

Physics of the spin-down:  I * domega/dt = -(M_f(omega) + C rho omega^2).
At 3-10 kPa and omega < 1.05 rad/s the windage term is < 0.011 N m (0.1% of
M_f), so the spin-down is a direct measurement of M_f(omega) on [0.1, 1.05]
rad/s, replicated at two pressures.

Outputs:
  (a) validation: the constant-M_f fit must reproduce Liu's I*beta = 10.99 N m;
  (b) the measured speed dependence of M_f (parametric ODE fits + local slopes);
  (c) the friction-error SHAPE scan on the corrected Liu dataset: constant
      (q=0), EHL-like (q=0.6, 0.8) and viscous-like (q=1) errors, all matched
      to (c-1)*M_f0 at omega_max.
"""
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

DIR = os.path.dirname(os.path.abspath(__file__))
SPIN = f"{DIR}/liu_fig6_spindown.csv"
DATA = f"{DIR}/cross_rotor_dataset_v4.csv"

I_ARM = 845.69
MF0 = I_ARM * 0.013                    # 10.99 N m, Liu's constant correction
C_W = 0.054                            # P = C rho omega^3 (W), paper's Ces
OM_MAX = 98.99
RHO = lambda p_kpa: p_kpa * 1e3 * 0.029 / (8.314 * 288.0)


def integrate(mf, w0, t, rho):
    w = np.empty(len(t)); w[0] = w0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]; x = w[i - 1]
        f = lambda u: -(mf(max(u, 0.0)) + C_W * rho * max(u, 0.0) ** 2) / I_ARM
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        w[i] = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return w


def fit_model(curves, model, p0, names):
    def resid(q):
        pars = q[:len(p0)]
        return np.concatenate([
            d[:, 1] - integrate(lambda u: model(pars, u), q[len(p0) + j], d[:, 0], rho)
            for j, (d, rho) in enumerate(curves)])
    q0 = np.array(list(p0) + [d[0, 1] + 0.013 for d, _ in curves])
    sol = optimize.least_squares(resid, q0, method="lm", xtol=1e-14)
    res = resid(sol.x)
    n, k = len(res), len(sol.x)
    cov = (res @ res / (n - k)) * np.linalg.pinv(sol.jac.T @ sol.jac)
    se = np.sqrt(np.diag(cov))
    print(f"\n  modelo M_f = {names}: rmse={np.sqrt(np.mean(res**2))*1e3:.2f} mrad/s"
          f"   SSE={res@res:.6f}")
    for nm, v, s in zip(names.split(" + ") + ["w0_3kPa", "w0_10kPa"], sol.x, se):
        print(f"    {nm:10s} = {v:+.5f} +- {s:.5f}")
    return sol.x, se, res


def sim_test(s, label):
    y = np.log(s.Cp.values)
    A = np.column_stack([np.ones(len(s)), np.log(s.omega_rad_s.values),
                         np.log(s.rho_kgm3.values)])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    n, k = A.shape
    cov = (r @ r / (n - k)) * np.linalg.pinv(A.T @ A)
    d = b[1] - b[2]
    sd = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    p = 2 * (1 - stats.t.cdf(abs(d / sd), n - k))
    print(f"  {label:46s} alpha-beta={d:+.4f}+-{sd:.4f}  t={d/sd:+.2f}  p={p:.4g}")


def main():
    sp = pd.read_csv(SPIN)
    curves = []
    for p_kpa in [3, 10]:
        d = sp[sp.pressure_kpa == p_kpa][["t_s", "omega_rad_s"]].values
        curves.append((d[np.argsort(d[:, 0])], RHO(p_kpa)))
        print(f"{p_kpa:2d} kPa: n={len(d)}   windage max durante spin-down = "
              f"{C_W*RHO(p_kpa)*1.05**2:.4f} N m")

    print("\n(a) VALIDACION: ajuste con M_f constante (debe dar ~10.99 N m)")
    fit_model(curves, lambda p, u: p[0], [11.0], "M0")

    print("\n(b) DEPENDENCIA CON LA VELOCIDAD")
    rB, seB, _ = fit_model(curves, lambda p, u: p[0] + p[1] * u, [11.0, 0.0],
                           "M0 + c1*omega")
    fit_model(curves, lambda p, u: p[0] + p[1] * u * u, [11.0, 0.0],
              "M0 + c2*omega^2")

    print("\n  M_f local (ventanas de 12 puntos, pendiente OLS, windage restado):")
    rows = []
    for (d, rho), tag in zip(curves, ["3 kPa", "10 kPa"]):
        t, w = d[:, 0], d[:, 1]
        for i in range(0, len(t) - 12, 6):
            s = slice(i, i + 12)
            b, cv = np.polyfit(t[s], w[s], 1, cov=True)
            rows.append((np.mean(w[s]), -I_ARM * b[0] - C_W * rho * np.mean(w[s]) ** 2,
                         I_ARM * np.sqrt(cv[0, 0]), tag))
    for om, mf, se, tag in rows:
        print(f"    {tag:7s} omega={om:5.3f}  M_f={mf:6.3f} +- {se:.3f} N m")
    R = np.array([(a, b, c) for a, b, c, _ in rows])
    W = np.diag(1 / R[:, 2] ** 2)
    X = np.column_stack([np.ones(len(R)), R[:, 0]])
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ R[:, 1])
    cvb = np.linalg.inv(X.T @ W @ X)
    print(f"\n  recta ponderada sobre ventanas: M_f = ({beta[0]:.3f}+-{np.sqrt(cvb[0,0]):.3f})"
          f" + ({beta[1]:.3f}+-{np.sqrt(cvb[1,1]):.3f})*omega  [N m; omega en 0.1-1.05]")
    print("  -> la friccion BAJA con omega en el rango medido (rama descendente de")
    print("     Stribeck: regimen limite/mixto). La extrapolacion lineal a 44-99 rad/s")
    print("     es negativa, o sea, fisicamente sin sentido: el spin-down acota el")
    print("     termino de contorno (carga), no el termino hidrodinamico EHL.")

    print("\n(c) ESCANEO DE FORMAS DEL ERROR DE FRICCION sobre los 15 puntos v4")
    print("    P_pub = P_true + DM(omega)*omega;  DM = (c-1)*MF0*(omega/omega_max)^q")
    df = pd.read_csv(DATA)
    liu = df[(df.source == "Liu2024") &
             (df.data_origin == "digitized_from_figure_pixel")].copy()
    for q in [0.0, 0.6, 0.8, 1.0]:
        print(f"\n    q = {q}  ({'constante' if q == 0 else 'crece con omega'}):")
        for c in [1.25, 1.5, 2.0]:
            s = liu.copy()
            dm = (c - 1) * MF0 * (s.omega_rad_s / OM_MAX) ** q
            s["P_w_W"] = s.P_w_W - dm * s.omega_rad_s
            s = s[s.P_w_W > 0]
            s["Cp"] = s.P_w_W / (0.5 * s.rho_kgm3 * s.omega_rad_s ** 3 * s.R_m ** 5)
            sim_test(s, f"c={c:.2f}  (M_f(omega_max)={c*MF0:5.1f} N m)")
    print("\n    sobre-sustraccion del termino constante (lo que implica Stribeck):")
    for c in [0.95, 0.90]:
        s = liu.copy()
        s["P_w_W"] = s.P_w_W + (1 - c) * MF0 * s.omega_rad_s
        s["Cp"] = s.P_w_W / (0.5 * s.rho_kgm3 * s.omega_rad_s ** 3 * s.R_m ** 5)
        sim_test(s, f"M_f x {c:.2f} (constante)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
