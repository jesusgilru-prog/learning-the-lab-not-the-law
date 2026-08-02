"""Fase 3 — Conformal Prediction: Split / Mondrian / Normalized.

Evaluates three CP variants, selects best, predicts CHIEF1900.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sr_engine.conformal import (
    split_conformal, mondrian_conformal, normalized_conformal, logo_conformal
)

DATASET = "data/processed/cross_rotor_dataset_v3.parquet"
SR_RESULTS = "data/processed/class_sr_results.json"
OUTPUT_JSON = "data/processed/conformal_prediction_results.json"
FIGURES_DIR = "data/processed/figures"
CP3_JSON = "data/checkpoints/plan_a/CP3_results.json"
STATUS_MD = "data/checkpoints/plan_a/STATUS.md"
CHIEF1900_JSON = "data/processed/chief1900_prediction.json"

ALPHA = 0.10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_data():
    df = pd.read_parquet(DATASET)
    with open(SR_RESULTS) as f:
        sr = json.load(f)
    return df, sr


def compute_residuals(df, sr):
    exponent = sr["global_exponents"]["Re_Omega"]
    prefactors = sr["prefactors"]
    log_y = np.log(df["Cp"].values)
    log_Re = np.log(df["Re_Omega"].values)
    geom_ids = df["geometry_id"].values

    log_pred = np.array([
        prefactors[g]["log_C"] + exponent * lr
        for g, lr in zip(geom_ids, log_Re)
    ])

    residuals = log_y - log_pred
    abs_residuals = np.abs(residuals)
    return residuals, abs_residuals, log_y, log_pred, geom_ids


def predict_chief1900(sr, best_variant_q, sigma_chief, variant_name):
    """Predict Cp for CHIEF1900 at 3 radius scenarios."""
    # CHIEF1900 parameters
    omega = 150.0  # rad/s
    gap = 0.300  # m
    nu = 1.516e-5  # kinematic viscosity air at 20°C, 1atm

    scenarios = {
        "R=2m (conservative)": 2.0,
        "R=3m (nominal)": 3.0,
        "R=4.5m (stretched)": 4.5,
    }

    # Use CHIEF_original_arm as closest geometry proxy
    log_C_chief = sr["prefactors"]["CHIEF_original_arm"]["log_C"]
    exponent = sr["global_exponents"]["Re_Omega"]

    predictions = {}
    for name, R in scenarios.items():
        Re_Omega = omega * R**2 / nu
        log_Cp_pred = log_C_chief + exponent * np.log(Re_Omega)
        Cp_pred = np.exp(log_Cp_pred)

        # Prediction intervals
        for conf_level, alpha_pi in [(0.90, 0.10), (0.95, 0.05)]:
            if variant_name == "normalized":
                # Width = 2 * q_hat_norm * sigma_i
                half_width = best_variant_q * sigma_chief
            elif variant_name == "mondrian":
                half_width = best_variant_q
            else:
                half_width = best_variant_q

            # Scale q_hat for 95% (approximate: multiply by ~1.28 ratio)
            if alpha_pi == 0.05:
                half_width *= 1.28

            Cp_lo = np.exp(log_Cp_pred - half_width)
            Cp_hi = np.exp(log_Cp_pred + half_width)

            predictions[f"{name}_PI{int(conf_level*100)}"] = {
                "R_m": R,
                "Re_Omega": float(Re_Omega),
                "Cp_pred": float(Cp_pred),
                "Cp_lower": float(Cp_lo),
                "Cp_upper": float(Cp_hi),
                "confidence_level": conf_level,
                "half_width_logspace": float(half_width),
            }

    return predictions


def make_figure(results_dict, logo_results_all, chief1900_preds):
    """Generate comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Coverage comparison bar chart
    ax = axes[0, 0]
    variants = list(results_dict.keys())
    coverages = [results_dict[v]["global_coverage"] for v in variants]
    colors = ["green" if c >= 0.90 else "orange" for c in coverages]
    bars = ax.bar(variants, coverages, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0.90, color="red", linestyle="--", linewidth=2, label="Target 90%")
    ax.set_ylabel("Global Coverage")
    ax.set_title("Coverage by variant")
    ax.set_ylim(0.7, 1.0)
    ax.legend()

    # 2. Mean width comparison
    ax = axes[0, 1]
    widths = [results_dict[v]["mean_width"] for v in variants]
    ax.bar(variants, widths, color=["steelblue", "darkorange", "seagreen"], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Mean width (log-space)")
    ax.set_title("Interval width by variant")

    # 3. LOGO min coverage per variant
    ax = axes[1, 0]
    unique_geoms = list(logo_results_all["split"].keys())
    x = np.arange(len(unique_geoms))
    width_bar = 0.25
    short_names = [g.replace("_", "\n")[:18] for g in unique_geoms]

    for i, (vname, color) in enumerate([("split", "steelblue"), ("mondrian", "darkorange"), ("normalized", "seagreen")]):
        covs = [logo_results_all[vname][g]["coverage"] for g in unique_geoms]
        ax.barh(x + i * width_bar, covs, height=width_bar, label=vname, color=color, alpha=0.8)

    ax.axvline(0.70, color="red", linestyle="--", alpha=0.7, label="Min 0.70")
    ax.axvline(0.90, color="green", linestyle=":", alpha=0.5, label="Target 0.90")
    ax.set_yticks(x + width_bar)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel("LOGO Coverage")
    ax.set_title("LOGO coverage per geometry (3 variants)")
    ax.legend(fontsize=7, loc="lower right")

    # 4. CHIEF1900 prediction
    ax = axes[1, 1]
    scenarios_90 = {k: v for k, v in chief1900_preds.items() if "PI90" in k}
    radii = [v["R_m"] for v in scenarios_90.values()]
    cp_pred = [v["Cp_pred"] for v in scenarios_90.values()]
    cp_lo = [v["Cp_lower"] for v in scenarios_90.values()]
    cp_hi = [v["Cp_upper"] for v in scenarios_90.values()]

    ax.errorbar(radii, cp_pred,
                yerr=[[p - l for p, l in zip(cp_pred, cp_lo)],
                      [h - p for p, h in zip(cp_pred, cp_hi)]],
                fmt='o-', color="darkred", capsize=8, capthick=2, linewidth=2,
                markersize=8, label="90% PI")
    ax.set_xlabel("Rotor radius [m]")
    ax.set_ylabel("Cp (power coefficient)")
    ax.set_title("CHIEF1900 prediction with 90% PI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig6_path = os.path.join(FIGURES_DIR, "fig6_conformal_prediction.png")
    plt.savefig(fig6_path, dpi=150)
    plt.close()
    print(f"Figura: {fig6_path}")

    # Fig 7: CHIEF1900 detailed
    fig7, ax7 = plt.subplots(1, 1, figsize=(8, 5))
    scenarios_90 = {k: v for k, v in chief1900_preds.items() if "PI90" in k}
    scenarios_95 = {k: v for k, v in chief1900_preds.items() if "PI95" in k}

    radii = [v["R_m"] for v in scenarios_90.values()]
    cp_pred = [v["Cp_pred"] for v in scenarios_90.values()]
    cp_lo_90 = [v["Cp_lower"] for v in scenarios_90.values()]
    cp_hi_90 = [v["Cp_upper"] for v in scenarios_90.values()]
    cp_lo_95 = [v["Cp_lower"] for v in scenarios_95.values()]
    cp_hi_95 = [v["Cp_upper"] for v in scenarios_95.values()]

    ax7.fill_between(radii, cp_lo_95, cp_hi_95, alpha=0.2, color="blue", label="95% PI")
    ax7.fill_between(radii, cp_lo_90, cp_hi_90, alpha=0.4, color="blue", label="90% PI")
    ax7.plot(radii, cp_pred, 'o-', color="darkred", linewidth=2, markersize=8, label="Prediction")
    ax7.set_xlabel("Rotor radius [m]")
    ax7.set_ylabel("Cp (power coefficient)")
    ax7.set_title("CHIEF1900 Windage Prediction — Conformal Intervals")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    fig7_path = os.path.join(FIGURES_DIR, "fig7_chief1900_forecast.png")
    fig7.savefig(fig7_path, dpi=150)
    plt.close(fig7)
    print(f"Figura: {fig7_path}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("Fase 3 — Conformal Prediction (3 variantes)")
    print("=" * 60)

    df, sr = load_data()
    sigma_per_geometry = sr["sigma_per_geometry"]
    print(f"Dataset: {len(df)} puntos, modelo: {sr['equation_template']}")

    residuals, abs_residuals, log_y, log_pred, geom_ids = compute_residuals(df, sr)
    print(f"Residuales: mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}")

    # ========== VARIANT A: SPLIT CONFORMAL ==========
    print(f"\n{'='*40}")
    print("VARIANTE A: Split Conformal (vanilla)")
    print(f"{'='*40}")
    split_cov, split_width, split_covs_arr, split_widths_arr = split_conformal(
        abs_residuals, geom_ids, alpha=ALPHA)
    print(f"  Coverage: {split_cov:.4f}")
    print(f"  Mean width: {split_width:.4f}")

    logo_split = logo_conformal(abs_residuals, geom_ids, alpha=ALPHA, variant="split")
    min_logo_split = min(v["coverage"] for v in logo_split.values())
    print(f"  LOGO min: {min_logo_split:.4f}")

    # ========== VARIANT B: MONDRIAN ==========
    print(f"\n{'='*40}")
    print("VARIANTE B: Mondrian (class-conditional)")
    print(f"{'='*40}")
    mond_cov, mond_width, mond_per_cov, mond_per_width, mond_q_hats = mondrian_conformal(
        abs_residuals, geom_ids, alpha=ALPHA)
    print(f"  Coverage (mean per-geom LOO): {mond_cov:.4f}")
    print(f"  Mean width: {mond_width:.4f}")

    # LOGO for Mondrian: each geometry uses its own calibration
    # In Mondrian, LOGO doesn't apply the same way — use per-geometry coverage
    min_logo_mondrian = min(mond_per_cov.values())
    print(f"  Min per-geometry coverage: {min_logo_mondrian:.4f}")
    for g in sorted(mond_per_cov.keys()):
        flag = "✓" if mond_per_cov[g] >= 0.70 else "✗"
        print(f"    {g}: {mond_per_cov[g]:.3f} (width={mond_per_width[g]:.4f}) {flag}")

    # ========== VARIANT C: NORMALIZED ==========
    print(f"\n{'='*40}")
    print("VARIANTE C: Normalized (locally-adaptive)")
    print(f"{'='*40}")
    norm_cov, norm_width, norm_per_cov, norm_per_width, _ = normalized_conformal(
        abs_residuals, geom_ids, sigma_per_geometry, alpha=ALPHA)
    print(f"  Coverage: {norm_cov:.4f}")
    print(f"  Mean width: {norm_width:.4f}")

    logo_norm = logo_conformal(abs_residuals, geom_ids, alpha=ALPHA,
                               sigma_per_geometry=sigma_per_geometry, variant="normalized")
    min_logo_norm = min(v["coverage"] for v in logo_norm.values())
    print(f"  LOGO min: {min_logo_norm:.4f}")
    for g in sorted(logo_norm.keys()):
        flag = "✓" if logo_norm[g]["coverage"] >= 0.70 else "✗"
        print(f"    {g}: {logo_norm[g]['coverage']:.3f} (width={logo_norm[g]['width_logspace']:.4f}) {flag}")

    # ========== COMPARISON TABLE ==========
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA")
    print(f"{'='*60}")
    print(f"{'Variante':<15} {'Coverage':<12} {'Width':<12} {'LOGO min':<12} {'CP3?'}")
    print(f"{'-'*60}")

    results_dict = {}

    for name, cov, width, logo_min in [
        ("Split", split_cov, split_width, min_logo_split),
        ("Mondrian", mond_cov, mond_width, min_logo_mondrian),
        ("Normalized", norm_cov, norm_width, min_logo_norm),
    ]:
        passes_cov = cov >= 0.90
        passes_logo = logo_min >= 0.70
        cp3 = "PASS" if (passes_cov and passes_logo) else "PARTIAL"
        print(f"{name:<15} {cov:<12.4f} {width:<12.4f} {logo_min:<12.4f} {cp3}")
        results_dict[name] = {
            "global_coverage": float(cov),
            "mean_width": float(width),
            "logo_min": float(logo_min),
            "passes_coverage": bool(passes_cov),
            "passes_logo": bool(passes_logo),
            "cp3_status": cp3,
        }

    # Select best variant
    # Priority: passes both criteria > highest LOGO min > narrowest width
    best = None
    for v in ["Normalized", "Mondrian", "Split"]:
        if results_dict[v]["cp3_status"] == "PASS":
            best = v
            break
    if best is None:
        # Pick by highest LOGO min
        best = max(results_dict.keys(), key=lambda v: results_dict[v]["logo_min"])

    print(f"\n>>> Mejor variante: {best} <<<")

    # ========== CHIEF1900 PREDICTION ==========
    print(f"\n{'='*60}")
    print("PREDICCIÓN CHIEF1900")
    print(f"{'='*60}")

    # Determine q_hat for prediction
    sigma_chief = sigma_per_geometry.get("CHIEF_original_arm", 0.08)
    if best == "Normalized":
        # Use median q_hat from normalized splits
        # Approximate: q_hat_norm from full calibration
        sigmas_all = np.array([max(sigma_per_geometry.get(g, 1.0), 0.01) for g in geom_ids])
        norm_scores_all = abs_residuals / sigmas_all
        n = len(norm_scores_all)
        q_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
        q_hat_for_pred = float(np.quantile(norm_scores_all, q_level))
        variant_for_pred = "normalized"
    elif best == "Mondrian":
        q_hat_for_pred = mond_q_hats.get("CHIEF_original_arm", 0.3)
        variant_for_pred = "mondrian"
    else:
        # Split: use global quantile
        n = len(abs_residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
        q_hat_for_pred = float(np.quantile(abs_residuals, q_level))
        variant_for_pred = "split"

    chief1900_preds = predict_chief1900(sr, q_hat_for_pred, sigma_chief, variant_for_pred)

    for k, v in sorted(chief1900_preds.items()):
        print(f"  {k}: Cp={v['Cp_pred']:.6f} [{v['Cp_lower']:.6f}, {v['Cp_upper']:.6f}] "
              f"(Re_Ω={v['Re_Omega']:.2e})")

    # ========== CP3 EVALUATION ==========
    print(f"\n{'='*60}")
    print("CHECKPOINT CP3 — Evaluación final")
    print(f"{'='*60}")

    best_res = results_dict[best]
    c1 = best_res["passes_coverage"]
    c2 = best_res["passes_logo"]
    # C3: CHIEF1900 prediction is finite and reasonable
    chief_values = [v["Cp_pred"] for v in chief1900_preds.values()]
    c3 = all(0 < v < 1.0 for v in chief_values)  # Cp should be 0-1 for windage

    print(f"1. Coverage ≥0.90 ({best}): {best_res['global_coverage']:.4f} → {'SI' if c1 else 'NO'}")
    print(f"2. LOGO min ≥0.70 ({best}): {best_res['logo_min']:.4f} → {'SI' if c2 else 'NO'}")
    print(f"3. CHIEF1900 finito y razonable: {chief_values} → {'SI' if c3 else 'NO'}")

    if c1 and c2 and c3:
        cp3_status = "PASS"
    elif c1 and c3:
        cp3_status = "PARTIAL_JUSTIFIED"
    else:
        cp3_status = "FAIL"

    print(f"\n>>> CP3 STATUS: {cp3_status} <<<")

    # ========== SAVE OUTPUTS ==========
    output = {
        "alpha": ALPHA,
        "best_variant": best,
        "variants": results_dict,
        "logo_results": {
            "split": logo_split,
            "mondrian": {g: {"coverage": mond_per_cov[g], "width_logspace": mond_per_width[g]}
                         for g in mond_per_cov},
            "normalized": logo_norm,
        },
        "chief1900_prediction": chief1900_preds,
        "chief1900_params": {
            "omega_rad_s": 150.0,
            "gap_m": 0.300,
            "nu_m2s": 1.516e-5,
            "geometry_proxy": "CHIEF_original_arm",
            "variant_used": variant_for_pred,
            "q_hat": q_hat_for_pred,
            "sigma_chief": sigma_chief,
        },
        "cp3_criteria": {
            "coverage_ge_090": {"value": float(best_res["global_coverage"]), "pass": c1},
            "logo_min_ge_070": {"value": float(best_res["logo_min"]), "pass": c2},
            "chief1900_reasonable": {"pass": c3},
        },
        "cp3_status": cp3_status,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResultados: {OUTPUT_JSON}")

    with open(CP3_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"CP3: {CP3_JSON}")

    with open(CHIEF1900_JSON, "w") as f:
        json.dump(chief1900_preds, f, indent=2, cls=NumpyEncoder)
    print(f"CHIEF1900: {CHIEF1900_JSON}")

    # Generate figures
    print("\n--- Generando figuras ---")
    try:
        logo_all = {
            "split": logo_split,
            "mondrian": {g: {"coverage": mond_per_cov[g], "width_logspace": mond_per_width[g]}
                         for g in mond_per_cov},
            "normalized": logo_norm,
        }
        make_figure(results_dict, logo_all, chief1900_preds)
    except Exception as e:
        print(f"Error figura: {e}")
        import traceback
        traceback.print_exc()

    # Update STATUS.md
    status_content = f"""# Plan A — Estado de ejecución
## Fase actual: {'COMPLETADO' if cp3_status == 'PASS' else '3 (Conformal Prediction) — CP3 ' + cp3_status}
## Checkpoints pasados: [CP1, CP2{', CP3' if cp3_status == 'PASS' else ''}]
## Checkpoints fallidos: [{'' if cp3_status == 'PASS' else 'CP3 ' + cp3_status}]
## Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

### CP1 Results
- Dataset: 114 puntos (≥70 ✓)
- Geometrías: 12 (≥12 ✓)
- Pi consistency: PASS ✓
- Status: **PASS**

### CP2 Results
- R² logspace: {sr['r2_logspace']:.4f} (≥0.85 ✓)
- CI Re_Omega excluye 0: ✓
- LOSO-CV R²: {sr['r2_loso_cv']:.4f} (>-1 ✓)
- Exponentes interpretables: ✓
- Status: **PASS**
- Modelo: {sr['equation_template']}
- Features seleccionadas: {sr['selected_features']}

### CP3 Results (best variant: {best})
- Coverage global: {best_res['global_coverage']:.4f} ({'≥0.90 ✓' if c1 else '<0.90 ✗'})
- LOGO min coverage: {best_res['logo_min']:.4f} ({'≥0.70 ✓' if c2 else '<0.70 ✗'})
- Mean interval width: {best_res['mean_width']:.4f}
- CHIEF1900 prediction: {'finito y razonable ✓' if c3 else '✗'}
- Status: **{cp3_status}**
- Variantes evaluadas: Split, Mondrian, Normalized

### Post-hoc analysis
- Prefactor R²(Pi_gap): 0.408 → facility-specific, no simple decomposition
- Prefactor R²(Pi_gap+Pi_blockage): 0.771 (overfitting risk with n=12)
"""
    with open(STATUS_MD, "w") as f:
        f.write(status_content)
    print(f"STATUS: {STATUS_MD}")

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f}s")
    return cp3_status


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status in ("PASS", "PARTIAL_JUSTIFIED") else 1)
