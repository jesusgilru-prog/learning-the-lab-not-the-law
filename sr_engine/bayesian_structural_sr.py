"""Bayesian Structural Symbolic Regression.

Computes posterior probabilities over candidate functional forms (structures)
for the windage power coefficient model, accounting for hierarchical
per-geometry prefactors.

Structures:
  S1: Cp = C_i * Re^q                         (power law, current)
  S2: Cp = C_i * Re^q * Pi_gap^p              (+ geometry effect)
  S3: Cp = C_i * Re^q * (1 + a*log(Re))       (log correction)
  S4: Cp = C_i * Re^q * Pi_gap^p * Pi_block^r (full geometry)
  S5: Cp = C_i * (Re^q1 if Mach<m else Re^q2) (regime switch)
  S6: Cp = C_i                                (null/constant)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import inv as scipy_inv
from dataclasses import dataclass


@dataclass
class StructureResult:
    """Result of fitting a single structure."""
    name: str
    n_global_params: int
    n_total_params: int  # global + per-geometry prefactors + per-geometry sigma
    log_likelihood: float
    bic: float
    laplace_log_evidence: float
    params: dict


def _profile_prefactors_and_sigma(residuals_per_geom):
    """Profile out C_i and sigma_i given residuals per geometry."""
    log_C = {}
    sigma = {}
    nll_contribution = 0.0
    for gid, res in residuals_per_geom.items():
        n_i = len(res)
        mu_i = np.mean(res)
        log_C[gid] = mu_i
        centered = res - mu_i
        ss = np.sum(centered ** 2)
        sigma_sq = max(ss / n_i, 1e-12)
        sigma[gid] = np.sqrt(sigma_sq)
        nll_contribution += 0.5 * n_i * np.log(2 * np.pi * sigma_sq) + 0.5 * n_i
    return log_C, sigma, nll_contribution


def fit_structure(name, log_y, features_dict, group_ids, unique_groups, n_points):
    """Fit a single structure by MLE with profiled prefactors.

    Returns StructureResult with log-likelihood, BIC, Laplace evidence.
    """
    n = n_points
    n_geom = len(unique_groups)

    if name == "S6":
        # Null model: Cp = C_i (no global params)
        residuals_per_geom = {}
        for g in unique_groups:
            mask = group_ids == g
            residuals_per_geom[g] = log_y[mask]
        log_C, sigma, nll = _profile_prefactors_and_sigma(residuals_per_geom)
        n_global = 0
        n_total = n_geom * 2  # C_i + sigma_i
        return StructureResult(
            name=name, n_global_params=n_global, n_total_params=n_total,
            log_likelihood=-nll, bic=2 * nll + n_total * np.log(n),
            laplace_log_evidence=-nll - 0.5 * n_total * np.log(n),  # BIC approx
            params={"log_C": log_C}
        )

    elif name == "S1":
        # Cp = C_i * Re^q
        log_Re = features_dict["log_Re"]

        def neg_ll(params):
            q = params[0]
            resid_per_geom = {}
            for g in unique_groups:
                mask = group_ids == g
                resid_per_geom[g] = log_y[mask] - q * log_Re[mask]
            _, _, nll = _profile_prefactors_and_sigma(resid_per_geom)
            return nll

        res = minimize(neg_ll, x0=[-0.07], method="Nelder-Mead",
                       options={"xatol": 1e-8, "maxiter": 5000})
        q_opt = res.x[0]
        n_global = 1
        best_nll = res.fun
        params = {"q": float(q_opt)}

    elif name == "S2":
        # Cp = C_i * Re^q * Pi_gap^p
        log_Re = features_dict["log_Re"]
        log_Pi_gap = features_dict["log_Pi_gap"]

        def neg_ll(params):
            q, p = params
            resid_per_geom = {}
            for g in unique_groups:
                mask = group_ids == g
                resid_per_geom[g] = log_y[mask] - q * log_Re[mask] - p * log_Pi_gap[mask]
            _, _, nll = _profile_prefactors_and_sigma(resid_per_geom)
            return nll

        res = minimize(neg_ll, x0=[-0.07, -0.3], method="Nelder-Mead",
                       options={"xatol": 1e-8, "maxiter": 10000})
        n_global = 2
        best_nll = res.fun
        params = {"q": float(res.x[0]), "p_gap": float(res.x[1])}

    elif name == "S3":
        # Cp = C_i * Re^q * (1 + a*log(Re))
        # log form: log(Cp) = log(C_i) + q*log(Re) + log(1 + a*log(Re))
        log_Re = features_dict["log_Re"]

        def neg_ll(params):
            q, a = params
            correction = np.log(np.maximum(1 + a * log_Re, 1e-10))
            resid_per_geom = {}
            for g in unique_groups:
                mask = group_ids == g
                resid_per_geom[g] = log_y[mask] - q * log_Re[mask] - correction[mask]
            _, _, nll = _profile_prefactors_and_sigma(resid_per_geom)
            return nll

        res = minimize(neg_ll, x0=[-0.07, 0.01], method="Nelder-Mead",
                       options={"xatol": 1e-8, "maxiter": 10000})
        n_global = 2
        best_nll = res.fun
        params = {"q": float(res.x[0]), "a_log": float(res.x[1])}

    elif name == "S4":
        # Cp = C_i * Re^q * Pi_gap^p * Pi_blockage^r
        log_Re = features_dict["log_Re"]
        log_Pi_gap = features_dict["log_Pi_gap"]
        log_Pi_block = features_dict["log_Pi_block"]

        def neg_ll(params):
            q, p, r = params
            resid_per_geom = {}
            for g in unique_groups:
                mask = group_ids == g
                resid_per_geom[g] = (log_y[mask] - q * log_Re[mask]
                                     - p * log_Pi_gap[mask] - r * log_Pi_block[mask])
            _, _, nll = _profile_prefactors_and_sigma(resid_per_geom)
            return nll

        res = minimize(neg_ll, x0=[-0.07, -0.3, 0.1], method="Nelder-Mead",
                       options={"xatol": 1e-8, "maxiter": 15000})
        n_global = 3
        best_nll = res.fun
        params = {"q": float(res.x[0]), "p_gap": float(res.x[1]),
                  "r_block": float(res.x[2])}

    elif name == "S5":
        # Regime-switching: Cp = C_i * Re^q1 if Mach<m, Re^q2 if Mach>=m
        log_Re = features_dict["log_Re"]
        mach = features_dict["mach"]

        def neg_ll(params):
            q1, q2, m_thresh = params
            if m_thresh < 0.05 or m_thresh > 0.95:
                return 1e10
            low = mach < m_thresh
            high = ~low
            exponent = np.where(low, q1, q2)
            resid_per_geom = {}
            for g in unique_groups:
                mask = group_ids == g
                resid_per_geom[g] = log_y[mask] - exponent[mask] * log_Re[mask]
            _, _, nll = _profile_prefactors_and_sigma(resid_per_geom)
            return nll

        # Use differential evolution for the threshold
        bounds = [(-0.5, 0.1), (-0.5, 0.1), (0.1, 0.8)]
        res = differential_evolution(neg_ll, bounds, seed=42, maxiter=500,
                                     tol=1e-6, polish=True)
        n_global = 3
        best_nll = res.fun
        params = {"q1": float(res.x[0]), "q2": float(res.x[1]),
                  "m_threshold": float(res.x[2])}

    else:
        raise ValueError(f"Unknown structure: {name}")

    # Compute BIC and Laplace approximation
    n_total = n_global + n_geom * 2  # global params + C_i + sigma_i
    log_lik = -best_nll
    bic = 2 * best_nll + n_total * np.log(n)

    # Laplace approximation: log_evidence ≈ log_lik - 0.5*k*log(n)
    # (equivalent to BIC/2 for large n with unit-information prior)
    laplace_log_evidence = log_lik - 0.5 * n_total * np.log(n)

    return StructureResult(
        name=name, n_global_params=n_global, n_total_params=n_total,
        log_likelihood=log_lik, bic=bic,
        laplace_log_evidence=laplace_log_evidence,
        params=params
    )


def compute_structure_posterior(results, prior=None):
    """Compute posterior probabilities P(S_k | data) from log-evidences."""
    if prior is None:
        prior = np.ones(len(results)) / len(results)

    log_evidences = np.array([r.laplace_log_evidence for r in results])
    log_posterior = log_evidences + np.log(prior)
    # Log-sum-exp for normalization
    max_lp = np.max(log_posterior)
    log_norm = max_lp + np.log(np.sum(np.exp(log_posterior - max_lp)))
    log_posterior_normalized = log_posterior - log_norm
    posterior = np.exp(log_posterior_normalized)

    return posterior


def cluster_bootstrap_structure(df, structures, n_boot=500, seed=42):
    """Cluster bootstrap (2-level) over structures.

    Level 1: resample facilities
    Level 2: resample points within facility
    Returns posterior of each structure across bootstrap replicas.
    """
    rng = np.random.default_rng(seed)
    facilities = df["source"].unique()
    n_structures = len(structures)

    posteriors_boot = np.zeros((n_boot, n_structures))
    winning_structure = np.zeros(n_boot, dtype=int)

    for b in range(n_boot):
        # Level 1: resample facilities
        sampled_fac = rng.choice(facilities, size=len(facilities), replace=True)
        dfs = []
        for i, fac in enumerate(sampled_fac):
            sub = df[df["source"] == fac].copy()
            # Level 2: resample points
            idx = rng.choice(len(sub), size=len(sub), replace=True)
            sub_boot = sub.iloc[idx].copy()
            sub_boot["geometry_id"] = sub_boot["geometry_id"] + f"__{i}"
            dfs.append(sub_boot)
        boot_df = pd.concat(dfs, ignore_index=True)

        # Prepare features
        log_y = np.log(boot_df["Cp"].values)
        group_ids = boot_df["geometry_id"].values
        unique_groups = np.unique(group_ids)
        n_pts = len(boot_df)

        features_dict = {
            "log_Re": np.log(boot_df["Re_Omega"].values),
            "log_Pi_gap": np.log(boot_df["Pi_gap"].values),
            "log_Pi_block": np.log(boot_df["Pi_blockage"].values),
            "mach": boot_df["M_tip"].values,
        }

        # Fit all structures
        results = []
        for s_name in structures:
            try:
                r = fit_structure(s_name, log_y, features_dict, group_ids, unique_groups, n_pts)
                results.append(r)
            except Exception:
                # If fitting fails, assign very low evidence
                results.append(StructureResult(
                    name=s_name, n_global_params=0, n_total_params=100,
                    log_likelihood=-1e6, bic=1e6, laplace_log_evidence=-1e6,
                    params={}
                ))

        posterior = compute_structure_posterior(results)
        posteriors_boot[b] = posterior
        winning_structure[b] = np.argmax(posterior)

    return posteriors_boot, winning_structure


# Need pandas for bootstrap function
import pandas as pd
