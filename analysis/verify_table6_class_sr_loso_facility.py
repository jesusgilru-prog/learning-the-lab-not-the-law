"""Regenerates the Class-SR row of Table 6 (leave-one-FACILITY-out LOSO),
which is the number that Copyleaks-review round 9 (Codex) flagged as
unverifiable: the manuscript reports -1.001, but the release repo shipped
class_sr_results.json's r2_loso_cv=0.021, which is a DIFFERENT metric
(leave-one-GEOMETRY-out with mean-of-training-intercepts as the proxy for
the held-out geometry, computed inside sr_engine/class_sr.py's own
class_sr_fit()).

Table 6 instead evaluates leave-one-SOURCE(facility)-out: each of the 4
facilities is held out in turn, and because every geometry belongs to
exactly one facility, holding out a facility means every held-out row's
geometry is completely unseen in training, so its intercept genuinely
cannot be estimated from data. Table 6's caption says explicitly that this
uses a zero intercept (log_C=0, i.e. C_i=1) for the unseen geometry -- not
the mean-of-training-intercepts proxy that class_sr.py's built-in
r2_loso_cv uses for its own (different) leave-one-geometry-out metric.

This script fits the same profiled-MLE Class-SR model
(sr_engine/class_sr.py's _neg_log_likelihood/_neg_log_likelihood_grad) on
3-of-4 facilities at a time, predicts the held-out facility's rows with
log_C=0, and reports the pooled LOSO R^2, to check it against the
model_comparison_table.csv row (source: hyperscale-chief/results/) that
Table 6 was built from.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sr_engine"))
from class_sr import _neg_log_likelihood, _neg_log_likelihood_grad

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cross_rotor_dataset.csv")

df = pd.read_csv(DATA_PATH)
log_y_all = np.log(df["Cp"].to_numpy())
log_X_all = np.log(df["Re_Omega"].to_numpy()).reshape(-1, 1)
source = df["source"].to_numpy()
geometry = df["geometry_id"].to_numpy()

facilities = np.unique(source)
assert len(facilities) == 4, facilities

loso_preds = np.full(len(df), np.nan)

for held_out in facilities:
    train_mask = source != held_out
    test_mask = source == held_out

    log_y_tr = log_y_all[train_mask]
    log_X_tr = log_X_all[train_mask]
    gids_tr = geometry[train_mask]
    unique_tr = np.unique(gids_tr)

    # sanity check: the held-out facility's geometries must be disjoint
    # from the training geometries (this is what makes log_C=0 the only
    # honest choice for the held-out predictions)
    test_geoms = set(geometry[test_mask])
    assert test_geoms.isdisjoint(set(unique_tr)), (
        f"{held_out}: expected disjoint geometries, got overlap {test_geoms & set(unique_tr)}"
    )

    x0 = np.zeros(1)
    res_cv = minimize(
        _neg_log_likelihood, x0,
        args=(log_y_tr, log_X_tr, gids_tr, unique_tr),
        jac=_neg_log_likelihood_grad,
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-7},
    )
    exp_cv = res_cv.x

    # log_C = 0 for the held-out facility's (entirely unseen) geometries
    loso_preds[test_mask] = log_X_all[test_mask] @ exp_cv

valid = ~np.isnan(loso_preds)
assert valid.all()
ss_res = np.sum((log_y_all[valid] - loso_preds[valid]) ** 2)
ss_tot = np.sum((log_y_all[valid] - np.mean(log_y_all[valid])) ** 2)
r2_loso_facility = 1.0 - ss_res / ss_tot

print(f"Leave-one-facility-out Class-SR R^2 (log space, zero intercept for held-out): {r2_loso_facility:.10f}")
print(f"Manuscript Table 6 value:                                                     -1.0012986247646127")
print(f"model_comparison_table.csv value:                                             -1.0012986247646127")
print(f"Match (within 1e-6): {abs(r2_loso_facility - (-1.0012986247646127)) < 1e-6}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table6_class_sr_loso_facility_result.json"), "w") as f:
    import json
    json.dump({
        "r2_loso_facility_zero_intercept": r2_loso_facility,
        "manuscript_table6_value": -1.0012986247646127,
        "matches_manuscript": bool(abs(r2_loso_facility - (-1.0012986247646127)) < 1e-6),
    }, f, indent=2)
