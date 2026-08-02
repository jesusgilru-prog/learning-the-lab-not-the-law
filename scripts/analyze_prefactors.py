"""Post-hoc analysis of Class-SR prefactors C_i vs geometric Pi groups."""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS_JSON = "data/processed/class_sr_results.json"
DATASET = "data/processed/cross_rotor_dataset_v3.parquet"
OUTPUT_JSON = "data/processed/prefactor_analysis.json"
FIGURES_DIR = "data/processed/figures"

PI_GEOM_COLS = ["Pi_gap", "Pi_blockage", "Pi_aspect_axial", "Pi_confinement"]
EXTRA_COLS = ["M_tip"]


def bootstrap_ols(X, y, n_boot=2000, seed=42):
    """OLS with bootstrap CI for coefficients."""
    rng = np.random.default_rng(seed)
    A = np.column_stack([np.ones(len(y)), X])
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    n = len(y)
    boot_coeffs = np.zeros((n_boot, A.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Ab, yb = A[idx], y[idx]
        try:
            boot_coeffs[b] = np.linalg.lstsq(Ab, yb, rcond=None)[0]
        except np.linalg.LinAlgError:
            boot_coeffs[b] = np.nan

    valid = ~np.any(np.isnan(boot_coeffs), axis=1)
    boot_coeffs = boot_coeffs[valid]
    ci_lo = np.percentile(boot_coeffs, 2.5, axis=0) if len(boot_coeffs) > 10 else np.full(A.shape[1], np.nan)
    ci_hi = np.percentile(boot_coeffs, 97.5, axis=0) if len(boot_coeffs) > 10 else np.full(A.shape[1], np.nan)

    return coeffs, r2, ci_lo, ci_hi


def full_model_no_prefactors(log_y, log_X, feature_names):
    """Fit pooled model log(Cp) = log(K) + sum a_i*log(X_i) without per-geometry intercepts."""
    A = np.column_stack([np.ones(len(log_y)), log_X])
    coeffs = np.linalg.lstsq(A, log_y, rcond=None)[0]
    log_y_pred = A @ coeffs
    ss_res = np.sum((log_y - log_y_pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    n = len(log_y)
    p = log_X.shape[1] + 1  # intercept + exponents
    ll = -0.5 * n * np.log(2 * np.pi * ss_res / n) - 0.5 * n
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll

    return {
        "intercept": float(coeffs[0]),
        "K": float(np.exp(coeffs[0])),
        "exponents": {name: float(coeffs[1 + i]) for i, name in enumerate(feature_names)},
        "r2_logspace": float(r2),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": p,
        "n_points": n,
    }


def main():
    print("=" * 60)
    print("Análisis post-hoc: Prefactores C_i vs Pi geométricos")
    print("=" * 60)

    # --- Load Class-SR results ---
    with open(RESULTS_JSON) as f:
        sr_results = json.load(f)

    prefactors = sr_results["prefactors"]
    geom_ids = sorted(prefactors.keys())
    log_C = np.array([prefactors[g]["log_C"] for g in geom_ids])
    print(f"\n{len(geom_ids)} geometrías con prefactores C_i")

    # --- Load dataset and compute per-geometry means ---
    df = pd.read_parquet(DATASET)
    all_cols = PI_GEOM_COLS + EXTRA_COLS + ["source"]
    avail_cols = [c for c in all_cols if c in df.columns]

    geom_means = {}
    for gid in geom_ids:
        mask = df["geometry_id"] == gid
        sub = df.loc[mask, avail_cols]
        means = {}
        for col in avail_cols:
            if col == "source":
                means[col] = sub[col].mode().iloc[0] if len(sub) > 0 else "unknown"
            else:
                means[col] = float(sub[col].mean()) if col in sub.columns else np.nan
        means["n_points"] = int(mask.sum())
        geom_means[gid] = means

    print("\nMedias por geometría:")
    print(f"{'Geometry':<35} {'log(Ci)':>8} {'Pi_gap':>10} {'Pi_block':>10} {'Pi_asp':>10} {'Pi_conf':>10} {'M_tip':>8} {'src'}")
    for gid in geom_ids:
        m = geom_means[gid]
        print(f"  {gid:<33} {prefactors[gid]['log_C']:8.3f} "
              f"{m.get('Pi_gap', float('nan')):10.4f} "
              f"{m.get('Pi_blockage', float('nan')):10.4f} "
              f"{m.get('Pi_aspect_axial', float('nan')):10.4f} "
              f"{m.get('Pi_confinement', float('nan')):10.4f} "
              f"{m.get('M_tip', float('nan')):8.4f} "
              f"{m.get('source', '?')}")

    # --- Secondary regressions: log(C_i) vs log(Pi_geom means) ---
    results = {"regressions": {}, "geom_means": geom_means, "prefactors": prefactors}

    # Build matrix of log(Pi_geom means)
    pi_available = [c for c in PI_GEOM_COLS if all(
        not np.isnan(geom_means[g].get(c, np.nan)) and geom_means[g].get(c, 0) > 0
        for g in geom_ids
    )]
    print(f"\nPi groups disponibles para regresión: {pi_available}")

    log_pi_matrix = np.column_stack([
        np.log(np.array([geom_means[g][c] for g in geom_ids]))
        for c in pi_available
    ])

    # Model 1: log(C_i) ~ log(Pi_gap)
    if "Pi_gap" in pi_available:
        idx_gap = pi_available.index("Pi_gap")
        X1 = log_pi_matrix[:, idx_gap:idx_gap+1]
        coeffs, r2, ci_lo, ci_hi = bootstrap_ols(X1, log_C)
        results["regressions"]["log_Ci_vs_log_Pi_gap"] = {
            "r2": r2,
            "intercept": float(coeffs[0]),
            "slope_Pi_gap": float(coeffs[1]),
            "ci_intercept": [float(ci_lo[0]), float(ci_hi[0])],
            "ci_slope_Pi_gap": [float(ci_lo[1]), float(ci_hi[1])],
        }
        print(f"\nModelo 1: log(C_i) ~ log(Pi_gap)")
        print(f"  R² = {r2:.4f}")
        print(f"  slope = {coeffs[1]:.4f} CI [{ci_lo[1]:.4f}, {ci_hi[1]:.4f}]")

    # Model 2: log(C_i) ~ log(Pi_gap) + log(Pi_blockage)
    if "Pi_gap" in pi_available and "Pi_blockage" in pi_available:
        idx_g = pi_available.index("Pi_gap")
        idx_b = pi_available.index("Pi_blockage")
        X2 = log_pi_matrix[:, [idx_g, idx_b]]
        coeffs, r2, ci_lo, ci_hi = bootstrap_ols(X2, log_C)
        results["regressions"]["log_Ci_vs_log_Pi_gap_blockage"] = {
            "r2": r2,
            "intercept": float(coeffs[0]),
            "slope_Pi_gap": float(coeffs[1]),
            "slope_Pi_blockage": float(coeffs[2]),
            "ci_slope_Pi_gap": [float(ci_lo[1]), float(ci_hi[1])],
            "ci_slope_Pi_blockage": [float(ci_lo[2]), float(ci_hi[2])],
        }
        print(f"\nModelo 2: log(C_i) ~ log(Pi_gap) + log(Pi_blockage)")
        print(f"  R² = {r2:.4f}")
        print(f"  slope_gap = {coeffs[1]:.4f}, slope_block = {coeffs[2]:.4f}")

    # Model 3: log(C_i) ~ all Pi_geom
    if len(pi_available) >= 2:
        coeffs, r2, ci_lo, ci_hi = bootstrap_ols(log_pi_matrix, log_C)
        reg_all = {"r2": r2, "intercept": float(coeffs[0])}
        for i, name in enumerate(pi_available):
            reg_all[f"slope_{name}"] = float(coeffs[1 + i])
            reg_all[f"ci_{name}"] = [float(ci_lo[1 + i]), float(ci_hi[1 + i])]
        results["regressions"]["log_Ci_vs_all_Pi_geom"] = reg_all
        print(f"\nModelo 3: log(C_i) ~ todos los Pi geom")
        print(f"  R² = {r2:.4f}")
        for i, name in enumerate(pi_available):
            print(f"  {name}: {coeffs[1+i]:.4f} CI [{ci_lo[1+i]:.4f}, {ci_hi[1+i]:.4f}]")

    # --- Model 4: If Pi_gap R² > 0.5, fit full pooled model ---
    r2_gap = results["regressions"].get("log_Ci_vs_log_Pi_gap", {}).get("r2", 0)
    if r2_gap > 0.5:
        print(f"\n--- R²(C_i ~ Pi_gap) = {r2_gap:.3f} > 0.5 → ajustando modelo full pooled ---")
        df_clean = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap"])
        df_clean = df_clean[(df_clean["Cp"] > 0) & (df_clean["Re_Omega"] > 0) & (df_clean["Pi_gap"] > 0)]
        log_y_full = np.log(df_clean["Cp"].values)
        log_X_full = np.log(df_clean[["Re_Omega", "Pi_gap"]].values)
        full_res = full_model_no_prefactors(log_y_full, log_X_full, ["Re_Omega", "Pi_gap"])
        results["full_pooled_model_Re_Pi_gap"] = full_res
        print(f"  Full model: Cp = {full_res['K']:.4f} × Re_Ω^{full_res['exponents']['Re_Omega']:.4f} × Pi_gap^{full_res['exponents']['Pi_gap']:.4f}")
        print(f"  R² logspace = {full_res['r2_logspace']:.4f}")
        print(f"  AIC = {full_res['aic']:.2f}, BIC = {full_res['bic']:.2f}")
        print(f"  Class-SR AIC = {sr_results['aic']:.2f}, BIC = {sr_results['bic']:.2f}")

        if "Pi_blockage" in pi_available:
            r2_gap_block = results["regressions"].get("log_Ci_vs_log_Pi_gap_blockage", {}).get("r2", 0)
            if r2_gap_block > r2_gap + 0.05:
                df_clean2 = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_blockage"])
                df_clean2 = df_clean2[(df_clean2["Cp"] > 0) & (df_clean2["Re_Omega"] > 0) &
                                      (df_clean2["Pi_gap"] > 0) & (df_clean2["Pi_blockage"] > 0)]
                log_y_f2 = np.log(df_clean2["Cp"].values)
                log_X_f2 = np.log(df_clean2[["Re_Omega", "Pi_gap", "Pi_blockage"]].values)
                full_res2 = full_model_no_prefactors(log_y_f2, log_X_f2, ["Re_Omega", "Pi_gap", "Pi_blockage"])
                results["full_pooled_model_Re_Pi_gap_blockage"] = full_res2
                print(f"\n  Full model 3-var: R² = {full_res2['r2_logspace']:.4f}, BIC = {full_res2['bic']:.2f}")
    else:
        print(f"\n--- R²(C_i ~ Pi_gap) = {r2_gap:.3f} <= 0.5 → no se ajusta modelo full ---")

    # --- Interpretation ---
    print("\n" + "=" * 60)
    print("INTERPRETACIÓN")
    print("=" * 60)
    if r2_gap > 0.5:
        interpretation = (
            f"Pi_gap es el driver principal de C_i (R²={r2_gap:.3f}), confirmando "
            f"que el grupo dimensional gap/R captura gran parte de la variabilidad "
            f"geométrica inter-rotor. El modelo full Cp = K × Re_Ω^a × Pi_gap^b "
            f"es una alternativa más interpretable al Class-SR con prefactores libres."
        )
    else:
        interpretation = (
            f"Los prefactores son facility-specific sin descomposición paramétrica "
            f"simple identificable (R²(C_i ~ Pi_gap) = {r2_gap:.3f}). Esto refleja "
            f"la diversidad geométrica de las 12 configuraciones y sugiere que las "
            f"variables Pi_geom medias por geometría no capturan suficiente "
            f"variabilidad para reemplazar los prefactores libres."
        )
    results["interpretation"] = interpretation
    print(interpretation)

    # --- Save JSON ---
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados: {OUTPUT_JSON}")

    # --- Generate figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIGURES_DIR, exist_ok=True)
        n_pi = len(pi_available)
        ncols = min(n_pi, 2)
        nrows = (n_pi + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

        for idx, pi_name in enumerate(pi_available):
            ax = axes[idx // ncols, idx % ncols]
            pi_vals = np.array([geom_means[g][pi_name] for g in geom_ids])
            log_pi = np.log(pi_vals)

            # Color by source
            sources = [geom_means[g].get("source", "?") for g in geom_ids]
            unique_sources = sorted(set(sources))
            cmap = plt.cm.Set2
            colors = {s: cmap(i / max(len(unique_sources) - 1, 1)) for i, s in enumerate(unique_sources)}

            for g_idx, gid in enumerate(geom_ids):
                ci = prefactors[gid]["ci_log_C"]
                ax.errorbar(
                    log_pi[g_idx], log_C[g_idx],
                    yerr=[[log_C[g_idx] - ci[0]], [ci[1] - log_C[g_idx]]],
                    fmt="o", color=colors[sources[g_idx]], markersize=8,
                    capsize=3, alpha=0.8,
                )

            # OLS fit line
            coeffs_line = np.polyfit(log_pi, log_C, 1)
            x_range = np.linspace(log_pi.min() - 0.2, log_pi.max() + 0.2, 50)
            ax.plot(x_range, np.polyval(coeffs_line, x_range), "k--", alpha=0.5)

            # R²
            y_pred_line = np.polyval(coeffs_line, log_pi)
            ss_res = np.sum((log_C - y_pred_line)**2)
            ss_tot = np.sum((log_C - np.mean(log_C))**2)
            r2_line = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax.set_xlabel(f"log({pi_name})")
            ax.set_ylabel("log(C_i)")
            ax.set_title(f"log(C_i) vs log({pi_name})  R²={r2_line:.3f}")
            ax.grid(True, alpha=0.3)

        # Legend
        for s in unique_sources:
            axes[0, 0].plot([], [], "o", color=colors[s], label=s, markersize=8)
        axes[0, 0].legend(fontsize=8, loc="best")

        # Hide unused axes
        for idx in range(n_pi, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "fig5_prefactors_vs_pi_geom.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Figura: {fig_path}")
    except ImportError:
        print("matplotlib no disponible, figura omitida")

    print("\nDone.")


if __name__ == "__main__":
    main()
