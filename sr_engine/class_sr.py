"""Class-SR: Hierarchical symbolic regression with per-geometry prefactors.

Model: log(Cp_ij) = log(C_i) + a*log(X1_ij) + b*log(X2_ij) + ... + eps_ij
where C_i is a free prefactor per geometry class, (a, b, ...) are global
shared exponents, and eps_ij ~ N(0, sigma_i^2) with sigma_i per geometry.

MLE via scipy BFGS with analytical profiling of C_i and sigma_i.
Bootstrap BCa for non-asymptotic CIs.
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class ClassSRResult:
    """Result of a Class-SR hierarchical fit."""
    global_exponents: dict[str, float]
    ci_asymptotic: dict[str, tuple[float, float]]
    ci_bootstrap: dict[str, tuple[float, float]]
    prefactors: dict[str, dict]  # {geom_id: {C: float, log_C: float, ci: (lo, hi)}}
    sigma_per_geometry: dict[str, float]
    r2_logspace: float
    r2_original: float
    r2_loso_cv: float
    log_likelihood: float
    aic: float
    bic: float
    n_params: int
    n_points: int
    n_geometries: int
    feature_names: list[str]
    n_bootstrap: int
    equation_template: str


def _profile_geometry_params(
    residuals_per_geom: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    """Given residuals r_ij = log(Cp_ij) - X_ij @ exponents per geometry,
    compute profiled log(C_i) = mean(r_ij) and sigma_i = std(r_ij)."""
    log_C = {}
    sigma = {}
    for gid, res in residuals_per_geom.items():
        log_C[gid] = float(np.mean(res))
        n = len(res)
        if n > 1:
            sigma[gid] = float(np.sqrt(np.sum((res - np.mean(res))**2) / n))
        else:
            sigma[gid] = 1e-6  # degenerate
    return log_C, sigma


def _neg_log_likelihood(
    exponents: np.ndarray,
    log_y: np.ndarray,
    log_X: np.ndarray,
    group_ids: np.ndarray,
    unique_groups: np.ndarray,
) -> float:
    """Negative log-likelihood with profiled C_i and sigma_i."""
    raw_residuals = log_y - log_X @ exponents

    nll = 0.0
    for gid in unique_groups:
        mask = group_ids == gid
        r = raw_residuals[mask]
        n_i = len(r)
        mu_i = np.mean(r)
        centered = r - mu_i
        ss = np.sum(centered**2)
        sigma_i_sq = ss / n_i if n_i > 0 else 1e-12
        sigma_i_sq = max(sigma_i_sq, 1e-12)

        nll += 0.5 * n_i * np.log(2 * np.pi * sigma_i_sq) + 0.5 * n_i
    return nll


def _neg_log_likelihood_grad(
    exponents: np.ndarray,
    log_y: np.ndarray,
    log_X: np.ndarray,
    group_ids: np.ndarray,
    unique_groups: np.ndarray,
) -> np.ndarray:
    """Gradient of NLL w.r.t. global exponents."""
    raw_residuals = log_y - log_X @ exponents
    grad = np.zeros_like(exponents)

    for gid in unique_groups:
        mask = group_ids == gid
        r = raw_residuals[mask]
        X_g = log_X[mask]
        n_i = len(r)
        mu_i = np.mean(r)
        centered = r - mu_i
        ss = np.sum(centered**2)
        sigma_i_sq = max(ss / n_i, 1e-12)

        # d(NLL)/d(exp) = -sum_i (1/sigma_i^2) * X_centered^T @ centered
        X_centered = X_g - X_g.mean(axis=0)
        grad -= (1.0 / sigma_i_sq) * (X_centered.T @ centered)

    return grad


def class_sr_fit(
    log_y: np.ndarray,
    log_X: np.ndarray,
    group_ids: np.ndarray,
    feature_names: list[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
    y_original: Optional[np.ndarray] = None,
    X_original: Optional[np.ndarray] = None,
) -> ClassSRResult:
    """Fit Class-SR model via MLE.

    Parameters
    ----------
    log_y : array (n,) — log(Cp)
    log_X : array (n, p) — log of dimensionless groups
    group_ids : array (n,) — geometry class labels
    feature_names : list of str — names for each column of log_X
    n_bootstrap : int — BCa bootstrap replications
    seed : int
    y_original : array (n,) — Cp in original space (for R² original)
    X_original : array (n, p) — dimensionless groups in original space
    """
    rng = np.random.default_rng(seed)
    n, p = log_X.shape
    unique_groups = np.unique(group_ids)
    n_geom = len(unique_groups)

    # --- Initial guess: pooled OLS ignoring groups ---
    A = np.column_stack([np.ones(n), log_X])
    ols_coeffs = np.linalg.lstsq(A, log_y, rcond=None)[0]
    x0 = ols_coeffs[1:]  # exponents only

    # --- MLE via BFGS ---
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(log_y, log_X, group_ids, unique_groups),
        jac=_neg_log_likelihood_grad,
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-8},
    )
    exponents_mle = result.x

    # --- Profile C_i and sigma_i ---
    raw_res = log_y - log_X @ exponents_mle
    res_per_geom = {}
    for gid in unique_groups:
        mask = group_ids == gid
        res_per_geom[str(gid)] = raw_res[mask]

    log_C_map, sigma_map = _profile_geometry_params(res_per_geom)

    # --- Predictions ---
    log_y_pred = np.zeros(n)
    for gid in unique_groups:
        mask = group_ids == gid
        log_y_pred[mask] = log_C_map[str(gid)] + log_X[mask] @ exponents_mle

    # --- R² logspace ---
    ss_res = np.sum((log_y - log_y_pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    # --- R² original space ---
    r2_orig = np.nan
    if y_original is not None and X_original is not None:
        y_pred_orig = np.zeros(n)
        for gid in unique_groups:
            mask = group_ids == gid
            C_i = np.exp(log_C_map[str(gid)])
            y_pred_orig[mask] = C_i * np.prod(
                X_original[mask] ** exponents_mle, axis=1
            )
        ss_res_o = np.sum((y_original - y_pred_orig)**2)
        ss_tot_o = np.sum((y_original - np.mean(y_original))**2)
        r2_orig = 1.0 - ss_res_o / ss_tot_o if ss_tot_o > 1e-30 else 1.0

    # --- Log-likelihood at optimum ---
    nll = _neg_log_likelihood(exponents_mle, log_y, log_X, group_ids, unique_groups)
    ll = -nll
    n_params = p + 2 * n_geom  # p global exponents + log_C_i + sigma_i per geom
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n) - 2 * ll

    # --- Asymptotic CI from Hessian inverse ---
    ci_asymptotic = {}
    try:
        hess_inv = result.hess_inv
        if hasattr(hess_inv, "toarray"):
            hess_inv = hess_inv.toarray()
        se = np.sqrt(np.diag(hess_inv))
        for i, name in enumerate(feature_names):
            lo = float(exponents_mle[i] - 1.96 * se[i])
            hi = float(exponents_mle[i] + 1.96 * se[i])
            ci_asymptotic[name] = (lo, hi)
    except Exception:
        for i, name in enumerate(feature_names):
            ci_asymptotic[name] = (float("nan"), float("nan"))

    # --- Bootstrap BCa ---
    boot_exponents = np.zeros((n_bootstrap, p))
    for b in range(n_bootstrap):
        # Resample within each geometry (stratified)
        idx = []
        for gid in unique_groups:
            gid_idx = np.where(group_ids == gid)[0]
            idx.append(rng.choice(gid_idx, size=len(gid_idx), replace=True))
        idx = np.concatenate(idx)
        log_y_b = log_y[idx]
        log_X_b = log_X[idx]
        gids_b = group_ids[idx]

        try:
            res_b = minimize(
                _neg_log_likelihood,
                exponents_mle,  # warm start
                args=(log_y_b, log_X_b, gids_b, unique_groups),
                jac=_neg_log_likelihood_grad,
                method="BFGS",
                options={"maxiter": 500, "gtol": 1e-6},
            )
            boot_exponents[b] = res_b.x
        except Exception:
            boot_exponents[b] = np.nan

    valid = ~np.any(np.isnan(boot_exponents), axis=1)
    boot_valid = boot_exponents[valid]

    ci_bootstrap = {}
    if len(boot_valid) >= 100:
        # BCa correction
        for i, name in enumerate(feature_names):
            theta_hat = exponents_mle[i]
            boot_col = boot_valid[:, i]

            # Bias correction z0
            z0 = norm.ppf(np.mean(boot_col < theta_hat))

            # Acceleration a_hat via jackknife
            jack_vals = np.zeros(n)
            for j in range(n):
                mask_j = np.ones(n, dtype=bool)
                mask_j[j] = False
                A_j = np.column_stack([np.ones(n - 1), log_X[mask_j]])
                c_j = np.linalg.lstsq(A_j, log_y[mask_j], rcond=None)[0]
                jack_vals[j] = c_j[1 + i]
            jack_mean = np.mean(jack_vals)
            a_hat = np.sum((jack_mean - jack_vals)**3) / (
                6.0 * (np.sum((jack_mean - jack_vals)**2))**1.5 + 1e-30
            )

            alpha = 0.05
            z_lo = norm.ppf(alpha / 2)
            z_hi = norm.ppf(1 - alpha / 2)

            denom_lo = 1 - a_hat * (z0 + z_lo)
            denom_hi = 1 - a_hat * (z0 + z_hi)

            # Robust BCa: fallback to percentile if any component is non-finite
            use_percentile = (
                not np.isfinite(z0)
                or not np.isfinite(a_hat)
                or abs(denom_lo) < 1e-10
                or abs(denom_hi) < 1e-10
            )

            if use_percentile:
                alpha_lo_val = 0.025
                alpha_hi_val = 0.975
            else:
                alpha_lo_val = norm.cdf(z0 + (z0 + z_lo) / denom_lo)
                alpha_hi_val = norm.cdf(z0 + (z0 + z_hi) / denom_hi)
                if not np.isfinite(alpha_lo_val) or not np.isfinite(alpha_hi_val):
                    alpha_lo_val = 0.025
                    alpha_hi_val = 0.975

            eps_b = 0.5 / len(boot_col)
            alpha_lo_val = float(np.clip(alpha_lo_val, eps_b, 1 - eps_b))
            alpha_hi_val = float(np.clip(alpha_hi_val, eps_b, 1 - eps_b))

            ci_bootstrap[name] = (
                float(np.percentile(boot_col, 100 * alpha_lo_val)),
                float(np.percentile(boot_col, 100 * alpha_hi_val)),
            )
    else:
        # Fallback to percentile
        for i, name in enumerate(feature_names):
            if len(boot_valid) >= 10:
                ci_bootstrap[name] = (
                    float(np.percentile(boot_valid[:, i], 2.5)),
                    float(np.percentile(boot_valid[:, i], 97.5)),
                )
            else:
                ci_bootstrap[name] = (float("nan"), float("nan"))

    # --- LOSO-CV (Leave-One-Source-Out) ---
    loso_preds = np.full(n, np.nan)
    for gid in unique_groups:
        test_mask = group_ids == gid
        train_mask = ~test_mask

        if np.sum(train_mask) < 5:
            continue

        log_y_tr = log_y[train_mask]
        log_X_tr = log_X[train_mask]
        gids_tr = group_ids[train_mask]
        unique_tr = np.unique(gids_tr)

        try:
            res_cv = minimize(
                _neg_log_likelihood,
                exponents_mle,
                args=(log_y_tr, log_X_tr, gids_tr, unique_tr),
                jac=_neg_log_likelihood_grad,
                method="BFGS",
                options={"maxiter": 1000, "gtol": 1e-7},
            )
            exp_cv = res_cv.x

            # For the held-out geometry, estimate C_i from... we can't.
            # Use the mean of training C_i's as proxy
            raw_tr = log_y_tr - log_X_tr @ exp_cv
            train_means = []
            for tgid in unique_tr:
                tmask = gids_tr == tgid
                train_means.append(np.mean(raw_tr[tmask]))
            C_proxy = np.mean(train_means)

            loso_preds[test_mask] = C_proxy + log_X[test_mask] @ exp_cv
        except Exception:
            pass

    valid_loso = ~np.isnan(loso_preds)
    if np.sum(valid_loso) > 5:
        ss_res_cv = np.sum((log_y[valid_loso] - loso_preds[valid_loso])**2)
        ss_tot_cv = np.sum((log_y[valid_loso] - np.mean(log_y[valid_loso]))**2)
        r2_loso = 1.0 - ss_res_cv / ss_tot_cv if ss_tot_cv > 1e-30 else 0.0
    else:
        r2_loso = np.nan

    # --- Prefactors with CI (from bootstrap) ---
    prefactors = {}
    for gid in unique_groups:
        gid_s = str(gid)
        C_val = np.exp(log_C_map[gid_s])
        # Bootstrap CI for prefactor
        boot_log_C = []
        for b_idx in range(len(boot_valid)):
            raw_b = log_y - log_X @ boot_valid[b_idx]
            mask_g = group_ids == gid
            boot_log_C.append(np.mean(raw_b[mask_g]))
        if len(boot_log_C) >= 10:
            boot_log_C = np.array(boot_log_C)
            prefactors[gid_s] = {
                "C": float(C_val),
                "log_C": float(log_C_map[gid_s]),
                "ci_log_C": (
                    float(np.percentile(boot_log_C, 2.5)),
                    float(np.percentile(boot_log_C, 97.5)),
                ),
            }
        else:
            prefactors[gid_s] = {
                "C": float(C_val),
                "log_C": float(log_C_map[gid_s]),
                "ci_log_C": (float("nan"), float("nan")),
            }

    # --- Equation template ---
    terms = []
    for name, exp in zip(feature_names, exponents_mle):
        terms.append(f"{name}^{exp:.4f}")
    eq = "Cp = C_i * " + " * ".join(terms)

    return ClassSRResult(
        global_exponents={name: float(e) for name, e in zip(feature_names, exponents_mle)},
        ci_asymptotic=ci_asymptotic,
        ci_bootstrap=ci_bootstrap,
        prefactors=prefactors,
        sigma_per_geometry={str(g): float(sigma_map[str(g)]) for g in unique_groups},
        r2_logspace=float(r2_log),
        r2_original=float(r2_orig),
        r2_loso_cv=float(r2_loso),
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        n_params=n_params,
        n_points=n,
        n_geometries=n_geom,
        feature_names=feature_names,
        n_bootstrap=n_bootstrap,
        equation_template=eq,
    )


def select_best_model(
    log_y: np.ndarray,
    log_X_full: np.ndarray,
    group_ids: np.ndarray,
    all_feature_names: list[str],
    max_features: int = 4,
    criterion: str = "bic",
    seed: int = 42,
    y_original: Optional[np.ndarray] = None,
    X_original_full: Optional[np.ndarray] = None,
) -> tuple[ClassSRResult, list[str]]:
    """Try all combinations of features (1..max_features) and select by AIC/BIC.

    Parameters
    ----------
    log_X_full : array (n, p_full) — all candidate log features
    all_feature_names : list of str — names for all candidates
    max_features : int — max simultaneous features
    criterion : 'aic' or 'bic'

    Returns
    -------
    best_result : ClassSRResult
    best_features : list[str] — selected feature names
    """
    p_full = log_X_full.shape[1]
    best_score = np.inf
    best_result = None
    best_features = None

    for k in range(1, min(max_features + 1, p_full + 1)):
        for combo in itertools.combinations(range(p_full), k):
            cols = list(combo)
            names = [all_feature_names[c] for c in cols]
            log_X_sub = log_X_full[:, cols]

            try:
                res = class_sr_fit(
                    log_y, log_X_sub, group_ids, names,
                    n_bootstrap=0,  # skip bootstrap for selection
                    seed=seed,
                )
            except Exception:
                continue

            score = res.aic if criterion == "aic" else res.bic
            if score < best_score:
                best_score = score
                best_result = res
                best_features = (cols, names)

    if best_result is None:
        raise RuntimeError("No valid model found during selection")

    # Refit best model with full bootstrap
    cols, names = best_features
    log_X_best = log_X_full[:, cols]
    X_orig_best = X_original_full[:, cols] if X_original_full is not None else None
    best_result = class_sr_fit(
        log_y, log_X_best, group_ids, names,
        n_bootstrap=2000,  # reduced from 10000 due to memory constraint
        seed=seed,
        y_original=y_original,
        X_original=X_orig_best,
    )

    return best_result, names


def result_to_dict(res: ClassSRResult) -> dict:
    """Serialize ClassSRResult to JSON-safe dict."""
    return {
        "global_exponents": res.global_exponents,
        "ci_asymptotic": {k: list(v) for k, v in res.ci_asymptotic.items()},
        "ci_bootstrap": {k: list(v) for k, v in res.ci_bootstrap.items()},
        "prefactors": res.prefactors,
        "sigma_per_geometry": res.sigma_per_geometry,
        "r2_logspace": res.r2_logspace,
        "r2_original": res.r2_original,
        "r2_loso_cv": res.r2_loso_cv,
        "log_likelihood": res.log_likelihood,
        "aic": res.aic,
        "bic": res.bic,
        "n_params": res.n_params,
        "n_points": res.n_points,
        "n_geometries": res.n_geometries,
        "feature_names": res.feature_names,
        "n_bootstrap": res.n_bootstrap,
        "equation_template": res.equation_template,
    }
