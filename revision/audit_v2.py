"""Version 2 (post-review corrections, see CORRECTIONS_v2.md) of the implementation of the audit protocol (Algorithms 1-2, Table `tab:config`)
used for the round-1 revision benchmarks of algorithms-4569489.

Every decision rule below is taken from the submitted manuscript's final protocol
and is FROZEN: the SHA-256 of this file is recorded in FROZEN_PROTOCOL.md before any
benchmark scenario is generated, and the file is not edited afterwards.

Model: class-SR backbone, log y = log C_g + sum_j theta_j x_j + eps, with a free
intercept and a free noise variance per geometry g (profile maximum likelihood,
exactly the likelihood of remediation_experiments.fit_structure_simple). Fitted by
iteratively reweighted least squares, which converges to the same maximum.

S1 = single shared Reynolds exponent; S5 = two Reynolds exponents switching at a
Mach threshold m. The S5 likelihood is piecewise constant in m, so the free
threshold is optimised EXACTLY over the midpoints between consecutive observed
Mach values (the published code used differential evolution on the same objective).
A minimum of MIN_SIDE points on each side is required.
"""
import numpy as np

# ------------------------------------------------------------------ frozen settings
MIN_SIDE = 3                 # smallest side for a candidate threshold
M_LO, M_HI = 0.05, 0.80      # threshold search range (as in the published S5 search)
T1_SINGLE = 0.90             # Test 1(a): one facility holds >= 90% of a side
T1_MINORITY = 0.10           # Test 1(b): minority side < 10% of all points
T2_BF = 1.0                  # Test 2: BF(S5 at tau_phys / S1) must exceed 1
T3_ALPHA = 0.05              # Test 3: permutation p < 0.05
LOSO_R2_FLAG = 0.0           # Component 1: LOSO R^2 <= 0 -> confounding-suspected
LOGO_MIN_OK = 0.80           # Component 4 acceptance
CONF_ALPHA = 0.10            # nominal 90% coverage
CANONICAL_Q = (-0.20, -0.10, -0.05)
BOOT_SELECT = 0.50           # bootstrap-only ablation: S5 chosen in >= 50% replicates


# ------------------------------------------------------------------ core fitting
def _design(groups_idx, n_groups, X):
    G = np.zeros((len(groups_idx), n_groups))
    G[np.arange(len(groups_idx)), groups_idx] = 1.0
    return np.hstack([G, X]) if X is not None and X.shape[1] else G


def fit_classsr(log_y, X, groups_idx, n_groups, iters=40, tol=1e-10):
    """Profile ML with per-group intercept and variance. Returns (loglik, beta, sigma2)."""
    A = _design(groups_idx, n_groups, X)
    present = np.bincount(groups_idx, minlength=n_groups) > 0
    if not present.all():  # drop empty groups' columns
        keep = np.concatenate([present, np.ones(A.shape[1] - n_groups, bool)])
        A = A[:, keep]
    w = np.ones(len(log_y))
    ll_old = -np.inf
    for _ in range(iters):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], log_y * sw, rcond=None)
        r = log_y - A @ beta
        ss = np.bincount(groups_idx, weights=r * r, minlength=n_groups)
        ng = np.bincount(groups_idx, minlength=n_groups)
        s2 = np.where(ng > 0, np.maximum(ss / np.maximum(ng, 1), 1e-12), np.nan)
        ll = -0.5 * np.nansum(ng * (np.log(2 * np.pi * s2) + 1.0))
        w = 1.0 / s2[groups_idx]
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    n_int = int(present.sum())
    slopes = beta[n_int:]
    return ll, slopes, s2, beta[:n_int], present


def laplace(ll, n_global, n_groups_present, n):
    k = n_global + 2 * n_groups_present
    return ll - 0.5 * k * np.log(n), k


def fit_s1(log_y, log_re, gi, ng):
    ll, sl, s2, ints, pres = fit_classsr(log_y, log_re[:, None], gi, ng)
    lap, k = laplace(ll, 1, pres.sum(), len(log_y))
    return dict(ll=ll, lap=lap, k=k, q=sl[0], s2=s2, ints=ints, present=pres)


def fit_s5_at(log_y, log_re, mach, m, gi, ng, n_global=3):
    """n_global=3 when m is the estimated threshold, 2 when m is fixed externally."""
    low = mach < m
    X = np.column_stack([log_re * low, log_re * (~low)])
    ll, sl, s2, ints, pres = fit_classsr(log_y, X, gi, ng)
    lap, k = laplace(ll, n_global, pres.sum(), len(log_y))
    return dict(ll=ll, lap=lap, k=k, q1=sl[0], q2=sl[1], m=m)


def candidate_thresholds(mach):
    u = np.unique(mach)
    mids = (u[:-1] + u[1:]) / 2
    mids = mids[(mids >= M_LO) & (mids <= M_HI)]
    out = []
    for m in mids:
        nl = np.sum(mach < m)
        if nl >= MIN_SIDE and len(mach) - nl >= MIN_SIDE:
            out.append(m)
    return np.array(out)


def fit_s5_free(log_y, log_re, mach, gi, ng):
    best = None
    for m in candidate_thresholds(mach):
        r = fit_s5_at(log_y, log_re, mach, m, gi, ng)
        if best is None or r["ll"] > best["ll"]:
            best = r
    return best


def s5_selected(log_y, log_re, mach, gi, ng):
    """Model selection by BIC (equivalently Laplace evidence) between S1 and S5."""
    s1 = fit_s1(log_y, log_re, gi, ng)
    s5 = fit_s5_free(log_y, log_re, mach, gi, ng)
    if s5 is None:
        return False, s1, None, 0.0
    log_bf = s5["lap"] - s1["lap"]
    return log_bf > 0, s1, s5, log_bf


# ------------------------------------------------------------------ Track B tests
def test1(mach, fac, m, single=T1_SINGLE, minority=T1_MINORITY):
    """Returns True if Test 1 PASSES (threshold does not look facility/support driven)."""
    low = mach < m
    n = len(mach)
    for side in (low, ~low):
        ns = side.sum()
        if ns == 0:
            return False
        _, counts = np.unique(fac[side], return_counts=True)
        if counts.max() / ns >= single:
            return False
    if min(low.sum(), (~low).sum()) / n < minority:
        return False
    return True


def test2(log_y, log_re, mach, gi, ng, s1, tau_phys):
    nl = np.sum(mach < tau_phys)
    if nl == 0 or nl == len(mach):
        return False, -np.inf
    r = fit_s5_at(log_y, log_re, mach, tau_phys, gi, ng, n_global=2)
    log_bf = r["lap"] - s1["lap"]
    return log_bf > np.log(T2_BF), log_bf


def test3(log_y, log_re, mach, gi, ng, s1, log_bf_obs, rng, n_perm):
    exceed = 0
    for _ in range(n_perm):
        pm = rng.permutation(mach)
        r = fit_s5_free(log_y, log_re, pm, gi, ng)
        lb = (r["lap"] - s1["lap"]) if r is not None else -np.inf
        exceed += lb >= log_bf_obs
    p = (1 + exceed) / (n_perm + 1)
    return p < T3_ALPHA, p


def test4(log_y, log_re, mach, gi, fac):
    ok = True
    sel = []
    for f in np.unique(fac):
        keep = fac != f
        g2, gi2 = np.unique(gi[keep], return_inverse=True)
        s, *_ = s5_selected(log_y[keep], log_re[keep], mach[keep], gi2, len(g2))
        sel.append(bool(s))
        ok &= bool(s)
    return ok, sel


def cluster_bootstrap_selection(log_y, log_re, mach, gi, fac, rng, B):
    facs = np.unique(fac)
    hits = 0
    for _ in range(B):
        idx = []
        for f in rng.choice(facs, size=len(facs), replace=True):
            pts = np.where(fac == f)[0]
            idx.append(rng.choice(pts, size=len(pts), replace=True))
        idx = np.concatenate(idx)
        g2, gi2 = np.unique(gi[idx], return_inverse=True)
        s, *_ = s5_selected(log_y[idx], log_re[idx], mach[idx], gi2, len(g2))
        hits += bool(s)
    return hits / B


# ------------------------------------------------------------------ Track A
def loso_r2(log_y, log_re, gi, ng, fac):
    """Leave-one-facility-out pooled R^2 of the S1 law; an unseen geometry is
    predicted with the mean of the training intercepts."""
    pred = np.empty_like(log_y)
    for f in np.unique(fac):
        te = fac == f
        g2, gi2 = np.unique(gi[~te], return_inverse=True)
        r = fit_s1(log_y[~te], log_re[~te], gi2, len(g2))
        pred[te] = np.mean(r["ints"]) + r["q"] * log_re[te]
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def im_interval(log_y, log_re, gi, fac, level=0.95):
    """Ibragimov-Mueller: per-facility exponent estimates, t-interval with F-1 df."""
    from scipy.stats import t as tdist
    qs = []
    for f in np.unique(fac):  # OLS with geometry fixed effects, as in the published Exp. 2
        m = fac == f
        g2, gi2 = np.unique(gi[m], return_inverse=True)
        Xf = _design(gi2, len(g2), log_re[m][:, None])
        qs.append(np.linalg.lstsq(Xf, log_y[m], rcond=None)[0][-1])
    qs = np.array(qs)
    F = len(qs)
    se = qs.std(ddof=1) / np.sqrt(F)
    h = tdist.ppf(0.5 + level / 2, F - 1) * se
    return qs.mean() - h, qs.mean() + h


def logo_min_coverage(log_y, log_re, gi, ng, rng, n_inner=200):
    """Nested cross-fitted normalised LOGO conformal (same construction as
    conformal_logo_nested_crossfit.py, fewer inner splits for cost)."""
    covs = []
    for g in range(ng):
        te = gi == g
        if te.sum() == 0:
            continue
        tr = np.where(~te)[0]
        g2, gi2 = np.unique(gi[tr], return_inverse=True)
        r = fit_s1(log_y[tr], log_re[tr], gi2, len(g2))
        sig = max(float(np.nanmean(np.sqrt(r["s2"]))), 0.01)  # mean sigma, as in the paper
        test_scores = np.abs(log_y[te] - (np.mean(r["ints"]) + r["q"] * log_re[te])) / sig
        pooled = []
        for _ in range(n_inner):
            perm = rng.permutation(len(tr))
            a, b = tr[perm[: len(tr) // 2]], tr[perm[len(tr) // 2:]]
            ga = set(gi[a])
            b = b[np.array([x in ga for x in gi[b]])]
            if len(b) == 0:
                continue
            g3, gi3 = np.unique(gi[a], return_inverse=True)
            ri = fit_s1(log_y[a], log_re[a], gi3, len(g3))
            pos = {gg: k for k, gg in enumerate(g3)}
            ib = np.array([pos[x] for x in gi[b]])
            pr = ri["ints"][ib] + ri["q"] * log_re[b]
            sg = np.maximum(np.sqrt(ri["s2"][ib]), 0.01)
            pooled.extend(np.abs(log_y[b] - pr) / sg)
        pooled = np.array(pooled)
        lvl = min(1.0, np.ceil((len(pooled) + 1) * (1 - CONF_ALPHA)) / len(pooled))
        qh = np.quantile(pooled, lvl)
        covs.append(np.mean(test_scores <= qh))
    return float(np.min(covs))


# ------------------------------------------------------------------ full audit
def run_audit(log_y, log_re, mach, geom, fac, tau_phys, rng, n_perm=199, n_boot=100,
              n_inner=200):
    """Runs the frozen protocol and returns every intermediate quantity, so that
    ablations can be computed from the same run."""
    g_lab, gi = np.unique(geom, return_inverse=True)
    ng = len(g_lab)
    out = {}
    # ---- Track B
    sel, s1, s5, log_bf = s5_selected(log_y, log_re, mach, gi, ng)
    out.update(selected=bool(sel), log_bf=float(log_bf),
               m_hat=float(s5["m"]) if s5 else np.nan, q_s1=float(s1["q"]))
    if sel:
        out["t1"] = test1(mach, fac, s5["m"])
        out["t2"], out["t2_logbf"] = test2(log_y, log_re, mach, gi, ng, s1, tau_phys)
        out["t3"], out["t3_p"] = test3(log_y, log_re, mach, gi, ng, s1, log_bf, rng, n_perm)
        out["t4"], out["t4_sel"] = test4(log_y, log_re, mach, gi, fac)
        out["boot_sel"] = cluster_bootstrap_selection(log_y, log_re, mach, gi, fac, rng, n_boot)
        # sensitivity grid of Test 1 thresholds (does not affect the frozen verdict)
        out["t1_grid"] = {f"{a:.2f}/{b:.2f}": test1(mach, fac, s5["m"], a, b)
                          for a in (0.80, 0.85, 0.90, 0.95) for b in (0.05, 0.10, 0.15)}
        if out["t3"] and out["t4"]:
            if out["t1"] and out["t2"]:
                vb = "STRUCTURE-SUPPORTED"
            elif (not out["t1"]) and (not out["t2"]):
                vb = "SUPPORTED-UNATTRIBUTED"
            else:
                vb = "INCONCLUSIVE"
        else:
            vb = "INCONCLUSIVE"
    else:
        vb = "NO-STRUCTURE"
    # ---- Track A
    out["loso_r2"] = float(loso_r2(log_y, log_re, gi, ng, fac))
    lo, hi = im_interval(log_y, log_re, gi, fac)
    out["im_lo"], out["im_hi"] = float(lo), float(hi)
    out["logo_min"] = logo_min_coverage(log_y, log_re, gi, ng, rng, n_inner)
    # Kish effective size and largest weight share of the calibration weights w_i = 1/sigma_g(i)
    s1full = fit_s1(log_y, log_re, gi, ng)
    w = 1.0 / np.maximum(np.sqrt(s1full["s2"][gi]), 0.01)
    out["n_eff"] = float(w.sum() ** 2 / (w ** 2).sum())
    out["max_w_share"] = float(w.max() / w.sum())
    excl0 = hi < 0 or lo > 0
    n_can = sum(lo <= c <= hi for c in CANONICAL_Q)
    out["im_excl0"], out["im_n_canonical"] = bool(excl0), int(n_can)
    conformal_fail = (out["logo_min"] < LOGO_MIN_OK or out["n_eff"] < 30
                      or out["max_w_share"] >= 0.5)
    if out["loso_r2"] <= LOSO_R2_FLAG:
        va = "NON-TRANSPORTABLE"
    elif not excl0 or n_can != 1:  # resolved to exactly one canonical regime
        va = "REPORT-WITH-DISCLAIMER"
    else:
        va = "REPORT-WITH-DISCLAIMER" if conformal_fail else "REPORT-AS-PHYSICS"
    # discriminant label (before caps) and final capped verdict of S
    out["verdict_B_disc"] = vb
    if vb == "STRUCTURE-SUPPORTED":
        vb = "STRUCTURE-WITH-DISCLAIMER" if conformal_fail else "STRUCTURE-AS-PHYSICS"
    out["verdict_A"], out["verdict_B"] = va, vb
    out["conformal_fail"] = bool(conformal_fail)
    return out
