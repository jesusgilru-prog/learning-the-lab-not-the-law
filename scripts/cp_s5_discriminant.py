"""CP-S5-DISCRIMINANT: Is the S5 regime switch at Mach=0.127 physical or artifact?

Tests:
1. Does threshold separate facilities? (contingency + Cramér's V)
2. Fixed physical threshold (0.30) vs free vs S1
3. Permutation test of the threshold
4. Leave-one-facility-out structure selection
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sr_engine.bayesian_structural_sr import (
    fit_structure, compute_structure_posterior
)

DATASET = "data/processed/cross_rotor_dataset_v3.parquet"
SR_RESULTS = "data/processed/class_sr_results.json"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

STRUCTURES = ["S1", "S2", "S3", "S4", "S5", "S6"]
MACH_THRESHOLD_FREE = 0.127
MACH_THRESHOLD_PHYSICAL = 0.30


def fit_s5_fixed_threshold(log_y, features_dict, group_ids, unique_groups, n_pts, threshold):
    """Fit S5 with a FIXED Mach threshold (not optimized)."""
    from scipy.optimize import minimize

    log_Re = features_dict["log_Re"]
    mach = features_dict["mach"]

    def neg_ll(params):
        q1, q2 = params
        low = mach < threshold
        exponent = np.where(low, q1, q2)
        resid_per_geom = {}
        for g in unique_groups:
            mask = group_ids == g
            resid_per_geom[g] = log_y[mask] - exponent[mask] * log_Re[mask]
        nll = 0.0
        for g, res in resid_per_geom.items():
            n_i = len(res)
            mu_i = np.mean(res)
            ss = np.sum((res - mu_i) ** 2)
            sigma_sq = max(ss / n_i, 1e-12)
            nll += 0.5 * n_i * np.log(2 * np.pi * sigma_sq) + 0.5 * n_i
        return nll

    res = minimize(neg_ll, x0=[-0.05, -0.06], method="Nelder-Mead",
                   options={"xatol": 1e-8, "maxiter": 5000})

    n_geom = len(unique_groups)
    n_global = 2  # q1, q2 (threshold is fixed, not a param)
    n_total = n_global + n_geom * 2
    log_lik = -res.fun
    bic = 2 * res.fun + n_total * np.log(n_pts)
    laplace = log_lik - 0.5 * n_total * np.log(n_pts)

    return {
        "log_likelihood": log_lik,
        "bic": bic,
        "laplace_log_evidence": laplace,
        "params": {"q1": float(res.x[0]), "q2": float(res.x[1]), "threshold": threshold},
        "n_total_params": n_total,
    }


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_parquet(DATASET)
    with open(SR_RESULTS) as f:
        sr = json.load(f)

    print("=" * 60)
    print("CP-S5-DISCRIMINANT — Is Mach=0.127 transition physical?")
    print("=" * 60)

    log_y = np.log(df["Cp"].values)
    group_ids = df["geometry_id"].values
    unique_groups = np.unique(group_ids)
    n_pts = len(df)
    mach = df["M_tip"].values
    facilities = df["source"].values

    features_dict = {
        "log_Re": np.log(df["Re_Omega"].values),
        "log_Pi_gap": np.log(df["Pi_gap"].values),
        "log_Pi_block": np.log(df["Pi_blockage"].values),
        "mach": mach,
    }

    # ══════════════════════════════════════════════════════════
    # TEST 1: Does threshold separate facilities?
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("TEST 1 — ¿El threshold Mach=0.127 separa facilities?")
    print(f"{'═'*60}")

    low_mask = mach < MACH_THRESHOLD_FREE
    high_mask = ~low_mask

    contingency_data = []
    unique_fac = np.unique(facilities)
    print(f"\n  {'Facility':<30} {'Mach<0.127':<12} {'Mach≥0.127':<12} {'% low'}")
    print(f"  {'-'*65}")
    for fac in unique_fac:
        fac_mask = facilities == fac
        n_low = np.sum(fac_mask & low_mask)
        n_high = np.sum(fac_mask & high_mask)
        pct_low = n_low / (n_low + n_high) * 100
        print(f"  {fac:<30} {n_low:<12} {n_high:<12} {pct_low:.1f}%")
        contingency_data.append([n_low, n_high])

    contingency_table = np.array(contingency_data)

    # Cramér's V
    chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
    n_total_ct = contingency_table.sum()
    k = min(contingency_table.shape)
    cramers_v = np.sqrt(chi2 / (n_total_ct * (k - 1)))

    print(f"\n  Chi² = {chi2:.2f}, p = {p_chi2:.4e}, dof = {dof}")
    print(f"  Cramér's V = {cramers_v:.4f}")
    print(f"  Interpretation: {'STRONG association' if cramers_v > 0.3 else 'MODERATE' if cramers_v > 0.15 else 'WEAK association'}")

    # Is any facility almost entirely on one side?
    separates_facilities = any(
        contingency_data[i][0] / max(sum(contingency_data[i]), 1) > 0.90 or
        contingency_data[i][1] / max(sum(contingency_data[i]), 1) > 0.90
        for i in range(len(unique_fac))
    )
    print(f"  Any facility >90% on one side? {separates_facilities}")

    test1_pass = cramers_v < 0.3 and not separates_facilities

    # Save contingency
    ct_df = pd.DataFrame(contingency_data, columns=["Mach_lt_0127", "Mach_ge_0127"],
                         index=unique_fac)
    ct_df.to_csv(os.path.join(RESULTS_DIR, "s5_facility_threshold_contingency.csv"))

    print(f"\n  >>> TEST 1 {'PASS' if test1_pass else 'FAIL'}: "
          f"Cramér's V={cramers_v:.4f}, separates={separates_facilities}")

    # ══════════════════════════════════════════════════════════
    # TEST 2: Fixed physical threshold vs free vs S1
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("TEST 2 — Threshold libre vs threshold físico (0.30) vs S1")
    print(f"{'═'*60}")

    # S1 (baseline)
    s1_result = fit_structure("S1", log_y, features_dict, group_ids, unique_groups, n_pts)

    # S5 free (already known)
    s5_free = fit_structure("S5", log_y, features_dict, group_ids, unique_groups, n_pts)

    # S5 fixed at 0.30
    s5_phys = fit_s5_fixed_threshold(log_y, features_dict, group_ids, unique_groups,
                                      n_pts, MACH_THRESHOLD_PHYSICAL)

    print(f"\n  {'Model':<25} {'logL':<10} {'BIC':<10} {'Laplace':<10}")
    print(f"  {'-'*55}")
    print(f"  {'S1 (pure power)':<25} {s1_result.log_likelihood:<10.2f} "
          f"{s1_result.bic:<10.2f} {s1_result.laplace_log_evidence:<10.2f}")
    print(f"  {'S5 (thr=0.127 free)':<25} {s5_free.log_likelihood:<10.2f} "
          f"{s5_free.bic:<10.2f} {s5_free.laplace_log_evidence:<10.2f}")
    print(f"  {'S5 (thr=0.30 fixed)':<25} {s5_phys['log_likelihood']:<10.2f} "
          f"{s5_phys['bic']:<10.2f} {s5_phys['laplace_log_evidence']:<10.2f}")

    # Does S5(0.30) beat S1?
    bf_phys_vs_s1 = np.exp(s5_phys["laplace_log_evidence"] - s1_result.laplace_log_evidence)
    bf_free_vs_phys = np.exp(s5_free.laplace_log_evidence - s5_phys["laplace_log_evidence"])

    print(f"\n  BF S5(0.30)/S1 = {bf_phys_vs_s1:.4f}")
    print(f"  BF S5(0.127)/S5(0.30) = {bf_free_vs_phys:.4f}")

    s5_phys_beats_s1 = s5_phys["laplace_log_evidence"] > s1_result.laplace_log_evidence
    print(f"\n  S5(0.30) beats S1? {s5_phys_beats_s1}")
    test2_pass = s5_phys_beats_s1  # Physical threshold also works

    print(f"\n  >>> TEST 2 {'PASS' if test2_pass else 'FAIL'}: "
          f"S5(0.30) {'>' if s5_phys_beats_s1 else '<='} S1")

    # ══════════════════════════════════════════════════════════
    # TEST 3: Permutation test
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("TEST 3 — Permutation test del threshold (500 permutaciones)")
    print(f"{'═'*60}")

    rng = np.random.default_rng(42)
    n_perm = 501
    observed_bf = np.exp(s5_free.laplace_log_evidence - s1_result.laplace_log_evidence)
    print(f"  Observed BF S5/S1 = {observed_bf:.2f}")

    perm_bfs = []
    t_perm = time.time()
    for i in range(n_perm):
        # Permute Mach labels
        perm_mach = rng.permutation(mach)
        perm_features = features_dict.copy()
        perm_features["mach"] = perm_mach

        # Fit S5 with permuted Mach
        try:
            s5_perm = fit_structure("S5", log_y, perm_features, group_ids, unique_groups, n_pts)
            bf_perm = np.exp(s5_perm.laplace_log_evidence - s1_result.laplace_log_evidence)
        except Exception:
            bf_perm = 1.0  # neutral
        perm_bfs.append(bf_perm)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_perm} done ({time.time()-t_perm:.1f}s)")

    perm_bfs = np.array(perm_bfs)
    n_exceed = int(np.sum(perm_bfs >= observed_bf))
    # Finite-sample-corrected Monte-Carlo p-value (never exactly 0):
    # p = (1 + #{permuted >= observed}) / (n_perm + 1)
    p_value = (1 + n_exceed) / (n_perm + 1)
    p_value_str = (f"< 1/{n_perm+1}" if n_exceed == 0 else f"= {p_value:.4f}")
    print(f"\n  Permutation BFs: median={np.median(perm_bfs):.4f}, "
          f"max={np.max(perm_bfs):.4f}, p95={np.percentile(perm_bfs, 95):.4f}")
    print(f"  Observed BF = {observed_bf:.2f}")
    print(f"  p-value (finite-sample corrected): {p_value_str}")

    test3_pass = p_value < 0.05
    print(f"\n  >>> TEST 3 {'PASS' if test3_pass else 'FAIL'}: p{p_value_str}")

    # ══════════════════════════════════════════════════════════
    # TEST 4: Leave-one-facility-out structure selection
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("TEST 4 — Leave-one-facility-out structure selection")
    print(f"{'═'*60}")

    lofo_winners = {}
    for fac in unique_fac:
        sub_df = df[df["source"] != fac].copy()
        sub_log_y = np.log(sub_df["Cp"].values)
        sub_group_ids = sub_df["geometry_id"].values
        # Ensure geometry IDs are unique after removing facility
        sub_unique_groups = np.unique(sub_group_ids)
        sub_n = len(sub_df)

        sub_features = {
            "log_Re": np.log(sub_df["Re_Omega"].values),
            "log_Pi_gap": np.log(sub_df["Pi_gap"].values),
            "log_Pi_block": np.log(sub_df["Pi_blockage"].values),
            "mach": sub_df["M_tip"].values,
        }

        results = []
        for s_name in STRUCTURES:
            try:
                r = fit_structure(s_name, sub_log_y, sub_features,
                                  sub_group_ids, sub_unique_groups, sub_n)
                results.append(r)
            except Exception:
                from sr_engine.bayesian_structural_sr import StructureResult
                results.append(StructureResult(
                    name=s_name, n_global_params=0, n_total_params=100,
                    log_likelihood=-1e6, bic=1e6, laplace_log_evidence=-1e6, params={}
                ))

        posterior = compute_structure_posterior(results)
        winner_idx = np.argmax(posterior)
        winner = STRUCTURES[winner_idx]
        lofo_winners[fac] = {
            "winner": winner,
            "posterior": {STRUCTURES[i]: float(posterior[i]) for i in range(len(STRUCTURES))},
        }
        print(f"  Without {fac:<15}: winner={winner} (P={posterior[winner_idx]:.4f})")

    # Does S5 win in all 4?
    s5_wins_all = all(v["winner"] == "S5" for v in lofo_winners.values())
    s5_win_count = sum(1 for v in lofo_winners.values() if v["winner"] == "S5")
    print(f"\n  S5 wins in {s5_win_count}/4 LOFO combinations")

    # Which facility, if removed, causes S5 to lose?
    s5_depends_on = [fac for fac, v in lofo_winners.items() if v["winner"] != "S5"]
    if s5_depends_on:
        print(f"  S5 loses when {s5_depends_on} removed → S5 depends on these facilities")

    test4_pass = s5_win_count >= 3  # S5 wins in at least 3/4
    print(f"\n  >>> TEST 4 {'PASS' if test4_pass else 'FAIL'}: "
          f"S5 wins {s5_win_count}/4 LOFO")

    # ══════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════
    tests_passed = sum([test1_pass, test2_pass, test3_pass, test4_pass])
    tests_failed = 4 - tests_passed

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    print(f"  Test 1 (threshold ≠ facilities):   {'PASS' if test1_pass else 'FAIL'} — Cramér's V={cramers_v:.4f}")
    print(f"  Test 2 (physical thr works):       {'PASS' if test2_pass else 'FAIL'} — S5(0.30) vs S1: BF={bf_phys_vs_s1:.4f}")
    print(f"  Test 3 (permutation significant):  {'PASS' if test3_pass else 'FAIL'} — p{p_value_str}")
    print(f"  Test 4 (LOFO stable):              {'PASS' if test4_pass else 'FAIL'} — S5 wins {s5_win_count}/4")
    print(f"\n  Tests passed: {tests_passed}/4")

    if tests_passed >= 4:
        verdict = "S5-PHYSICAL"
        verdict_text = ("S5 sobrevive los 4 tests. El threshold no separa facilities, "
                        "funciona también cerca de Mach=0.3, BF significativo en permutation, "
                        "estable en LOFO. → Paper fuerte: transición de régimen identificada "
                        "por inferencia bayesiana. Narrativa de descubrimiento.")
    elif tests_passed <= 1:
        verdict = "S5-ARTIFACT"
        verdict_text = (f"S5 falla {tests_failed} tests. S5 es artefacto del covariate shift. "
                        "NO usar como descubrimiento. Volver a CASE B con framing "
                        "framework+protocolo. El intento S5 se reporta como ablación honesta.")
    else:
        verdict = "S5-PARTIAL" if tests_passed >= 3 else "S5-ARTIFACT"
        if tests_passed >= 3:
            verdict_text = ("S5 sobrevive 3/4 tests. Resultado sugerente pero no concluyente. "
                            "Reportar S5 como 'hipótesis estructural sugerente pero no concluyente', "
                            "no como hallazgo central. Paper intermedio.")
        else:
            verdict_text = (f"S5 falla {tests_failed}/4 tests. S5 es artefacto del covariate shift. "
                            "NO usar como descubrimiento. Volver a CASE B con framing "
                            "framework+protocolo. El intento S5 se reporta como ablación honesta.")

    print(f"\n>>> VERDICT: {verdict} <<<")
    print(f"  {verdict_text}")

    # ══════════════════════════════════════════════════════════
    # OUTPUTS
    # ══════════════════════════════════════════════════════════

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 1: Permutation null distribution
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(np.log10(np.maximum(perm_bfs, 1e-10)), bins=40, color="steelblue",
                alpha=0.8, edgecolor="black", label="Permutation null")
        ax.axvline(np.log10(observed_bf), color="red", linewidth=2, linestyle="--",
                   label=f"Observed (log10={np.log10(observed_bf):.2f})")
        ax.set_xlabel("log10(Bayes Factor S5/S1)")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation test of S5 threshold (p{p_value_str})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "s5_permutation_null.png"), dpi=150)
        plt.close()

        # Fig 2: LOFO structure posterior
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, (fac, info) in enumerate(lofo_winners.items()):
            ax = axes[idx // 2, idx % 2]
            posts = [info["posterior"].get(s, 0) for s in STRUCTURES]
            colors = ["red" if s == "S5" else "steelblue" for s in STRUCTURES]
            ax.bar(STRUCTURES, posts, color=colors, alpha=0.8, edgecolor="black")
            ax.set_title(f"Without {fac}")
            ax.set_ylabel("P(S_k)")
            ax.set_ylim(0, 1.1)
        plt.suptitle("Leave-One-Facility-Out Structure Selection")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "s5_lofo_structure.png"), dpi=150)
        plt.close()

        print("\n  Figures saved.")
    except Exception as e:
        print(f"  Figure error: {e}")

    # Summary markdown
    summary = f"""# CP-S5-DISCRIMINANT — Is Mach=0.127 transition physical?

## Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
## Elapsed: {time.time() - t0:.1f}s

---

## TEST 1 — Does threshold separate facilities?

| Facility | Mach<0.127 | Mach≥0.127 | % low |
|----------|-----------|-----------|-------|
"""
    for i, fac in enumerate(unique_fac):
        n_low, n_high = contingency_data[i]
        pct = n_low / max(n_low + n_high, 1) * 100
        summary += f"| {fac} | {n_low} | {n_high} | {pct:.1f}% |\n"

    summary += f"""
- Chi² = {chi2:.2f}, p = {p_chi2:.4e}
- **Cramér's V = {cramers_v:.4f}** ({'STRONG' if cramers_v > 0.3 else 'MODERATE' if cramers_v > 0.15 else 'WEAK'})
- Any facility >90% on one side? {separates_facilities}
- **TEST 1: {'PASS' if test1_pass else 'FAIL'}**

---

## TEST 2 — Physical threshold (0.30) vs free (0.127) vs S1

| Model | logL | BIC | Laplace |
|-------|------|-----|---------|
| S1 (pure power) | {s1_result.log_likelihood:.2f} | {s1_result.bic:.2f} | {s1_result.laplace_log_evidence:.2f} |
| S5 (thr=0.127 free) | {s5_free.log_likelihood:.2f} | {s5_free.bic:.2f} | {s5_free.laplace_log_evidence:.2f} |
| S5 (thr=0.30 fixed) | {s5_phys['log_likelihood']:.2f} | {s5_phys['bic']:.2f} | {s5_phys['laplace_log_evidence']:.2f} |

- BF S5(0.30)/S1 = {bf_phys_vs_s1:.4f}
- S5(0.30) beats S1? {s5_phys_beats_s1}
- **TEST 2: {'PASS' if test2_pass else 'FAIL'}**

---

## TEST 3 — Permutation test (500 permutations)

- Observed BF S5/S1 = {observed_bf:.2f}
- Permutation distribution: median={np.median(perm_bfs):.4f}, p95={np.percentile(perm_bfs, 95):.4f}
- **p-value {p_value_str}**
- **TEST 3: {'PASS' if test3_pass else 'FAIL'}**

---

## TEST 4 — Leave-one-facility-out

| Removed facility | Winner | P(winner) |
|-----------------|--------|-----------|
"""
    for fac, info in lofo_winners.items():
        winner = info["winner"]
        p_win = info["posterior"][winner]
        summary += f"| {fac} | {winner} | {p_win:.4f} |\n"

    summary += f"""
- S5 wins {s5_win_count}/4 combinations
- S5 depends on facilities: {s5_depends_on if s5_depends_on else 'none'}
- **TEST 4: {'PASS' if test4_pass else 'FAIL'}**

---

## VERDICT

| Test | Result |
|------|--------|
| 1. Threshold ≠ facilities | {'PASS' if test1_pass else 'FAIL'} (Cramér's V={cramers_v:.4f}) |
| 2. Physical threshold works | {'PASS' if test2_pass else 'FAIL'} (BF={bf_phys_vs_s1:.4f}) |
| 3. Permutation significant | {'PASS' if test3_pass else 'FAIL'} (p{p_value_str}) |
| 4. LOFO stable | {'PASS' if test4_pass else 'FAIL'} ({s5_win_count}/4) |

**Tests passed: {tests_passed}/4**

### >>> {verdict} <<<

{verdict_text}
"""

    with open(os.path.join(RESULTS_DIR, "s5_discriminant.md"), "w") as f:
        f.write(summary)

    # Update STATUS.md
    with open("data/checkpoints/plan_a/STATUS.md", "a") as f:
        f.write(f"""
### CP-S5-DISCRIMINANT Results
- Test 1 (facilities): {'PASS' if test1_pass else 'FAIL'} (Cramér's V={cramers_v:.4f})
- Test 2 (physical thr): {'PASS' if test2_pass else 'FAIL'} (BF={bf_phys_vs_s1:.4f})
- Test 3 (permutation): {'PASS' if test3_pass else 'FAIL'} (p{p_value_str})
- Test 4 (LOFO): {'PASS' if test4_pass else 'FAIL'} ({s5_win_count}/4)
- **VERDICT: {verdict}**
""")

    print(f"\n{'═'*60}")
    print(f"DONE. Total: {time.time() - t0:.1f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
