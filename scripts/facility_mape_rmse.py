"""Per-facility MAPE/RMSE in original Cp units for the baseline models
of Table "baselines", computed from the same in-sample fits used for
the R^2_orig column already reported (not LOGO-CV; that is a separate,
already-reported column)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.parquet")
SR_RESULTS = os.path.join(HERE, "..", "data", "processed_checkpoints", "class_sr_results.json")
RESULTS_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def fit_class_sr_q(log_y, log_Re, group_ids):
    from scipy.optimize import minimize

    unique_groups = np.unique(group_ids)

    def neg_ll(q):
        nll = 0.0
        for g in unique_groups:
            mask = group_ids == g
            r = log_y[mask] - q[0] * log_Re[mask]
            n_i = len(r)
            mu_i = np.mean(r)
            ss = np.sum((r - mu_i) ** 2)
            sigma_sq = max(ss / n_i, 1e-12)
            nll += 0.5 * n_i * np.log(2 * np.pi * sigma_sq) + 0.5 * n_i
        return nll

    res = minimize(neg_ll, x0=[-0.2], method="Nelder-Mead")
    return res.x[0]


def main():
    df = pd.read_parquet(DATASET)
    with open(SR_RESULTS) as f:
        sr = json.load(f)

    y_true = df["Cp"].values
    log_y = np.log(y_true)
    log_Re = np.log(df["Re_Omega"].values)
    geom_ids = df["geometry_id"].values
    facility = df["source"].values

    preds = {}

    # Cp_const_global
    preds["Cp_const_global"] = np.full(len(df), np.mean(log_y))

    # Cp_const_facility
    preds["Cp_const_facility"] = np.array(
        [np.mean(log_y[facility == f]) for f in facility]
    )

    # Cp_const_geometry (== Windage_classic_Cp_const)
    preds["Cp_const_geometry"] = np.array(
        [np.mean(log_y[geom_ids == g]) for g in geom_ids]
    )

    # Cp_global_power
    X = np.column_stack([np.ones(len(log_Re)), log_Re])
    beta = np.linalg.lstsq(X, log_y, rcond=None)[0]
    preds["Cp_global_power"] = beta[0] + beta[1] * log_Re

    # Class_SR
    exponent = sr["global_exponents"]["Re_Omega"]
    prefactors = sr["prefactors"]
    preds["Class_SR"] = np.array(
        [prefactors[g]["log_C"] + exponent * lr for g, lr in zip(geom_ids, log_Re)]
    )

    # Vrancik 1968 correlation: C_M = 0.065 * Re^-0.2
    Cm_vr = 0.065 * df["Re_Omega"].values ** (-0.2)
    preds["Vrancik_1968_corr"] = np.log(Cm_vr)

    # Daily-Nece 1960: C_M = 0.0622 * (s/R)^0.1 * Re^-0.2
    if df["gap_radial_m"].notna().all() and (df["gap_radial_m"] > 0).all():
        s = df["gap_radial_m"].values
        R = df["R_m"].values
        Re = df["Re_Omega"].values
        Cm_dn = 0.0622 * (s / R) ** 0.1 * Re ** (-0.2)
        preds["Daily_Nece_1960"] = np.log(Cm_dn)

    facilities = sorted(np.unique(facility).tolist())
    rows = []
    for model, log_pred in preds.items():
        y_pred = np.exp(log_pred)
        for fac in facilities:
            m = facility == fac
            yt, yp = y_true[m], y_pred[m]
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)
            rows.append(
                {"model": model, "facility": fac, "n": int(m.sum()),
                 "rmse_orig": rmse, "mape_pct": mape}
            )
        # overall row too
        rmse_all = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape_all = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        rows.append({"model": model, "facility": "ALL", "n": len(df),
                     "rmse_orig": rmse_all, "mape_pct": mape_all})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "facility_mape_rmse.csv"), index=False)
    print(out.pivot(index="model", columns="facility", values="mape_pct").round(1).to_string())
    print()
    print(out.pivot(index="model", columns="facility", values="rmse_orig").round(4).to_string())


if __name__ == "__main__":
    main()
