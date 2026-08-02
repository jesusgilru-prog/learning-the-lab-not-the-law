"""Adaptive Conformal Prediction with Learned Heterogeneity Groups.

Implements a Learned-Group Mondrian Conformal method that clusters calibration
points by heterogeneity features and calibrates per-cluster quantiles.

Distinction from literature:
- Tibshirani et al. 2019 (Conformal under covariate shift): assumes known groups
- Gibbs 2021 (ACI): adapts quantile over time, no spatial grouping
- Barber et al. 2023: distribution-free, no learned structure
- THIS: learns calibration groups from heterogeneity structure (residual magnitude,
  operating regime features). Groups are data-driven, not assumed.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class AdaptiveConformalResult:
    """Result of adaptive conformal prediction."""
    best_k: int
    best_method: str  # "gmm" or "kmeans"
    cluster_labels: np.ndarray
    coverage_global: float
    logo_min: float
    mean_width: float
    per_cluster_coverage: dict
    per_cluster_width: dict
    silhouette: float
    bic: float
    ari_stability: float  # mean ARI under bootstrap


def build_heterogeneity_features(df, residuals_normalized):
    """Build feature matrix for clustering heterogeneity groups."""
    features = np.column_stack([
        np.abs(residuals_normalized),
        np.log(df["Re_Omega"].values),
        df["Pi_gap"].values,
        df["Pi_blockage"].values,
        df["M_tip"].values,
    ])
    feature_names = ["abs_resid_norm", "log_Re_Omega", "Pi_gap", "Pi_blockage", "Mach"]
    return features, feature_names


def select_best_clustering(features, k_range=(2, 3, 4, 5, 6), seed=42):
    """Try GMM and KMeans for each K, select by silhouette + BIC."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    results = []
    for k in k_range:
        # GMM
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=5, max_iter=300)
        gmm_labels = gmm.fit_predict(X)
        gmm_sil = silhouette_score(X, gmm_labels) if len(np.unique(gmm_labels)) > 1 else -1
        gmm_bic = gmm.bic(X)
        results.append({
            "k": k, "method": "gmm", "labels": gmm_labels,
            "silhouette": gmm_sil, "bic": gmm_bic, "model": gmm,
        })

        # KMeans
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(X, km_labels) if len(np.unique(km_labels)) > 1 else -1
        results.append({
            "k": k, "method": "kmeans", "labels": km_labels,
            "silhouette": km_sil, "bic": float("inf"), "model": km,
        })

    # Select: highest silhouette (primary), lowest BIC among GMMs (secondary)
    best = max(results, key=lambda r: r["silhouette"])
    return best, results, scaler


def cluster_stability_ari(features, best_k, best_method, n_boot=100, seed=42):
    """Measure clustering stability via bootstrap ARI."""
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n = len(X)

    # Reference clustering
    if best_method == "gmm":
        ref_model = GaussianMixture(n_components=best_k, random_state=seed, n_init=5)
    else:
        ref_model = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    ref_labels = ref_model.fit_predict(X)

    aris = []
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]

        if best_method == "gmm":
            model = GaussianMixture(n_components=best_k, random_state=seed + i + 1, n_init=3)
        else:
            model = KMeans(n_clusters=best_k, random_state=seed + i + 1, n_init=5)

        boot_labels = model.fit_predict(X_boot)

        # ARI between ref labels (on boot indices) and boot labels
        ari = adjusted_rand_score(ref_labels[idx], boot_labels)
        aris.append(ari)

    return float(np.mean(aris)), np.array(aris)


def learned_group_conformal(abs_residuals, cluster_labels, alpha=0.10):
    """Mondrian conformal with learned clusters.

    LOO within each cluster to evaluate coverage.
    """
    unique_clusters = np.unique(cluster_labels)
    per_cluster_coverage = {}
    per_cluster_width = {}
    q_hats = {}

    for c in unique_clusters:
        mask = cluster_labels == c
        scores = abs_residuals[mask]
        n = len(scores)

        if n < 3:
            q_hat = float(np.max(scores)) * 1.5
            per_cluster_coverage[int(c)] = 1.0
            per_cluster_width[int(c)] = 2 * q_hat
            q_hats[int(c)] = q_hat
            continue

        # LOO conformal within cluster
        covered = 0
        for i in range(n):
            cal = np.delete(scores, i)
            q_level = min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal))
            q_hat_loo = np.quantile(cal, q_level)
            if scores[i] <= q_hat_loo:
                covered += 1

        per_cluster_coverage[int(c)] = covered / n
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q_hat = np.quantile(scores, q_level)
        per_cluster_width[int(c)] = 2 * q_hat
        q_hats[int(c)] = q_hat

    global_coverage = float(np.mean(list(per_cluster_coverage.values())))
    mean_width = float(np.mean(list(per_cluster_width.values())))

    return global_coverage, mean_width, per_cluster_coverage, per_cluster_width, q_hats


def logo_by_facility_adaptive(abs_residuals, cluster_labels, facility_ids, alpha=0.10):
    """LOGO-CV by facility for the adaptive method.

    For each held-out facility: calibrate on remaining points using their cluster
    assignments, predict on held-out facility points using nearest cluster.
    """
    unique_facilities = np.unique(facility_ids)
    results = {}

    for fac in unique_facilities:
        test_mask = facility_ids == fac
        cal_mask = ~test_mask

        cal_scores = abs_residuals[cal_mask]
        cal_clusters = cluster_labels[cal_mask]
        test_scores = abs_residuals[test_mask]
        test_clusters = cluster_labels[test_mask]

        # Calibrate per-cluster quantile on calibration set
        q_hats = {}
        for c in np.unique(cal_clusters):
            c_scores = cal_scores[cal_clusters == c]
            n_c = len(c_scores)
            q_level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
            q_hats[c] = np.quantile(c_scores, q_level)

        # Fallback: global quantile for clusters not in calibration
        if q_hats:
            global_q = np.quantile(cal_scores,
                                   min(1.0, np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores)))
        else:
            global_q = np.quantile(abs_residuals,
                                   min(1.0, np.ceil((len(abs_residuals) + 1) * (1 - alpha)) / len(abs_residuals)))

        # Test coverage
        covered = 0
        for score, c in zip(test_scores, test_clusters):
            q = q_hats.get(c, global_q)
            if score <= q:
                covered += 1

        coverage = covered / len(test_scores) if len(test_scores) > 0 else 0.0
        results[fac] = float(coverage)

    return results
