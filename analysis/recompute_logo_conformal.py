import os
#!/usr/bin/env python
"""
Recompute the conformal LOGO evaluation holding out FACILITIES, not geometries.

`sr_engine/conformal.py::logo_conformal` is documented as "Leave-One-Geometry-Out"
and is called with `geometry_id`, so every published LOGO number is a
leave-one-GEOMETRY-out figure.  The manuscript describes it as leave-one-facility-out
("when an unseen facility is the test set..."), and that number reaches the abstract.

This script recomputes both, side by side, and exposes a second problem the geometry
version hides: two of the three variants are conditioned on geometry, so when a whole
facility is held out its geometries are unseen and the variant is not even defined
without a fallback rule.  That has to be stated, not patched silently.
"""
import json
import sys

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.parquet")
SR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "class_sr_results.json")
PUBLISHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "conformal_prediction_results.json")
ALPHA = 0.10


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def residuals(df, sr):
    """Same construction as the published pipeline."""
    exponent = sr["global_exponents"]["Re_Omega"]
    pref = sr["prefactors"]
    log_y = np.log(df["Cp"].values)
    log_re = np.log(df["Re_Omega"].values)
    geo = df["geometry_id"].values
    log_pred = np.array([pref[g]["log_C"] + exponent * lr for g, lr in zip(geo, log_re)])
    return np.abs(log_y - log_pred), geo, log_pred


def conformal_quantile(cal_scores, alpha=ALPHA):
    n = len(cal_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(cal_scores, q_level))


def logo(abs_res, groups, holdout_by, sigma_map=None, sigma_fallback=None):
    """Leave-one-`holdout_by`-out coverage.

    `groups` is the vector used for sigma lookup (geometry); `holdout_by` is the
    vector defining the folds (geometry or facility).
    """
    out = {}
    for g in np.unique(holdout_by):
        te = holdout_by == g
        ca = ~te
        if sigma_map is None:
            cal, test = abs_res[ca], abs_res[te]
        else:
            s_ca = np.array([sigma_map.get(x, sigma_fallback) for x in groups[ca]])
            s_te = np.array([sigma_map.get(x, sigma_fallback) for x in groups[te]])
            cal, test = abs_res[ca] / np.maximum(s_ca, 0.01), abs_res[te] / np.maximum(s_te, 0.01)
        q = conformal_quantile(cal)
        out[str(g)] = dict(coverage=float(np.mean(test <= q)), n_test=int(te.sum()), q_hat=q)
    return out


def main():
    df = pd.read_parquet(DATA)
    with open(SR) as f:
        sr = json.load(f)
    abs_res, geo, _ = residuals(df, sr)
    fac = df["source"].values

    # sigma per geometry: taken from the SR results, exactly as the published
    # pipeline does (`sigma_per_geometry = sr["sigma_per_geometry"]`).
    sigma_geo = {k: float(v) for k, v in sr["sigma_per_geometry"].items()}
    sigma_global = float(np.std(abs_res))

    section("0. Que se publico realmente")
    pub = json.load(open(PUBLISHED))
    for v, res in pub["logo_results"].items():
        keys = list(res.keys())
        mn = min(r["coverage"] for r in res.values())
        arg = min(res, key=lambda k: res[k]["coverage"])
        print(f"  {v:11s}: {len(keys)} grupos -> {'GEOMETRIAS' if len(keys)==12 else '?'};  "
              f"minimo={mn:.4f} en '{arg}'")
    print("\n  Las 12 claves son geometrias. El manuscrito lo describe como")
    print("  'una instalacion no vista', que no es lo que se calculo.")

    section("1. LOGO por GEOMETRIA (lo publicado, reproducido)")
    g_split = logo(abs_res, geo, geo)
    g_norm = logo(abs_res, geo, geo, sigma_geo, sigma_global)
    print(f"  split      minimo={min(v['coverage'] for v in g_split.values()):.4f}")
    print(f"  normalized minimo={min(v['coverage'] for v in g_norm.values()):.4f}")

    section("2. LOGO por INSTALACION (lo que dice el manuscrito)")
    f_split = logo(abs_res, geo, fac)
    print("  --- split (sin condicionar a geometria: bien definido) ---")
    for k, v in sorted(f_split.items()):
        print(f"     {k:12s} n={v['n_test']:3d}  cobertura={v['coverage']:.4f}")
    print(f"     MINIMO = {min(v['coverage'] for v in f_split.values()):.4f}")

    print("\n  --- normalized (sigma por geometria) ---")
    print("  Problema: al dejar fuera una instalacion, SUS geometrias no aparecen en")
    print("  calibracion, asi que sigma no esta definido para el test. Se necesita una")
    print("  regla de respaldo, y el numero depende de cual se elija:")
    for name, fb in [("sigma global", sigma_global),
                     ("mediana de sigmas de calibracion", float(np.median(list(sigma_geo.values()))))]:
        # The held-out facility's geometries have no sigma: replace them by the
        # fallback, fold by fold, which is the honest evaluation.
        honest = {}
        for g in np.unique(fac):
            te = fac == g
            unseen = {x: fb for x in np.unique(geo[te])}
            sm = {k: v for k, v in sigma_geo.items() if k not in unseen}
            sm.update(unseen)
            honest[str(g)] = logo(abs_res, geo, fac, sm, fb)[str(g)]
        detail = ", ".join(f"{k}={v['coverage']:.3f}" for k, v in sorted(honest.items()))
        worst = min(v["coverage"] for v in honest.values())
        print(f"     con {name:34s} minimo={worst:.4f}   ({detail})")

    print("\n  --- mondrian (bins por geometria) ---")
    print("  NO ESTA DEFINIDO dejando fuera una instalacion: sus geometrias no tienen")
    print("  ningun punto de calibracion en su propio bin. Cualquier numero que se")
    print("  reporte aqui exige declarar la regla de respaldo.")

    section("3. Resumen para el manuscrito")
    print(f"  LOGO-min por geometria, split      : {min(v['coverage'] for v in g_split.values()):.3f}"
          f"   <- es el 0.167 que cita el paper")
    print(f"  LOGO-min por geometria, normalized : {min(v['coverage'] for v in g_norm.values()):.3f}")
    print(f"  LOGO-min por INSTALACION, split    : {min(v['coverage'] for v in f_split.values()):.3f}")
    print("\n  Recomendacion: reportar los dos, decir explicitamente que el criterio de")
    print("  aceptacion se evalua por geometria, y que las variantes condicionadas a")
    print("  geometria no son evaluables dejando fuera una instalacion entera --- que es")
    print("  en si mismo un resultado: la normalizacion por geometria no puede proteger")
    print("  contra una instalacion nueva, porque sus geometrias tambien son nuevas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
