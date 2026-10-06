# Pre-registration of the revision benchmark (algorithms-4569489, round 1)

Frozen on 2026-10-01T03:33:25+02:00, BEFORE any benchmark scenario was generated
(only the real corpus was run, to check that the implementation reproduces the
published numbers, and one smoke-test job per regime was timed without
inspecting verdicts).

| file | sha256 |
|---|---|
| audit_frozen.py | 3c12f4b5df2bac206098a8ddb0bace209a7c0bd9951fdacba1b00020af0f4ea9 |
| benchmark_factorial.py | 4a7aebca929c3d7293435c15a3685435633dee473b51acab6084125c4aa9d7bc |

## Implementation check on the real corpus (validate_real.py)
S1 exponent -0.074512 (published -0.074512); S5 split puts 8/114 points below the
threshold (published 8/114; the likelihood is constant between consecutive Mach
values 0.1082 and 0.1290, the grid returns the midpoint 0.1186, differential
evolution returned 0.127); BF_S5/S1 = 5.80e5 (published 5.8e5); LOSO R^2 = -1.001
(published Table 6: -1.001); IM 95% CI [-0.362, 0.071] (published [-0.362, 0.071]);
Tests 1/2 FAIL, Tests 3/4 PASS, S5 selected in 4/4 LOFO fits; verdicts
NON-TRANSPORTABLE (g-hat) and statistically-supported/physically-unattributed (S),
as published. Known differences: (i) the bootstrap-selection statistic here is
S5-vs-S1 only (published 69.2% is the S5 share among S1-S6); (ii) the conformal
component uses LOGO-min only (n_eff and max-weight conditions are not computed),
with 20 inner splits instead of 200; (iii) permutations 199 (published 501).

## Frozen rules (Table tab:config of the submitted manuscript)
- Structure selected iff BIC(S5) < BIC(S1).
- Test 1 FAIL iff one facility holds >= 90% of the points on either side of the
  fitted threshold, or the minority side holds < 10% of all points.
- Test 2 PASS iff BF(S5 at tau_phys = 0.20 / S1) > 1 (Laplace).
- Test 3 PASS iff permutation p < 0.05 (199 permutations of Mach, threshold re-optimised).
- Test 4 PASS iff S5 is selected in every leave-one-facility-out refit.
- Track B: T3&T4&T1&T2 -> STRUCTURE-SUPPORTED; T3&T4&!T1&!T2 -> SUPPORTED-UNATTRIBUTED
  (the verdict called STATISTICALLY-REAL-SPURIOUS in the submission); otherwise INCONCLUSIVE;
  not selected -> NO-STRUCTURE.
- Track A: LOSO R^2 <= 0 -> NON-TRANSPORTABLE; else IM CI must exclude 0 and contain at
  most one canonical regime (-0.20, -0.10, -0.05), else REPORT-WITH-DISCLAIMER; else
  conformal cap (LOGO-min < 0.80 -> REPORT-WITH-DISCLAIMER), else REPORT-AS-PHYSICS.

## Design
5 regimes (R1, R2, R3, R4a, R4b; see benchmark_factorial.py docstring) x effect level
{low, high} x noise {0.03, 0.08} x sample-size scale {1, 2} x F {4, 6}; K = 30
replicates per cell; 2400 datasets; master seed 20261001.

## Pre-registered expected verdicts
| regime | Track B correct | Track B dangerous error | Track A truth |
|---|---|---|---|
| R1 | NO-STRUCTURE or INCONCLUSIVE | STRUCTURE-SUPPORTED | transportable |
| R2 | NO-STRUCTURE or INCONCLUSIVE | STRUCTURE-SUPPORTED | not transportable |
| R3 | STRUCTURE-SUPPORTED | (miss) | not scored |
| R4a | SUPPORTED-UNATTRIBUTED (INCONCLUSIVE = safe miss) | STRUCTURE-SUPPORTED | not scored |
| R4b | SUPPORTED-UNATTRIBUTED (INCONCLUSIVE = safe miss) | STRUCTURE-SUPPORTED | not scored |

Primary Track-B metric: "claims a physical structure". Positive class = R3.
Sensitivity = P(claim | R3); specificity = P(no claim | R1, R2, R4a, R4b);
FPR = 1 - specificity, reported per regime. Secondary: exact-verdict accuracy
per the table. Track A: sensitivity of NON-TRANSPORTABLE in R2, FPR in R1.

## Pre-registered ablation (computed on the same runs)
Track B, "claims a physical structure" when the structure is selected AND:
1. model selection only (conventional pipeline): nothing else
2. Test 3 only: permutation p < 0.05
3. Test 4 only: LOFO-stable
4. cluster bootstrap only: S5 chosen in >= 50% of 100 facility->point resamples
5. Test 1 only: Test 1 passes
6. Test 2 only: Test 2 passes
7. Tests 1+2: both pass
8. Tests 3+4: both pass
9. full protocol: all four pass
Stage 0 takes no response input; all benchmark designs are identifiable by
construction, so as a stand-alone component it passes every dataset (row reported
for completeness); it is evaluated separately in the Stage 0 phase study.
Track A, "flags non-transportability": (a) LOSO only; (b) IM only (CI includes 0 or
more than one canonical regime); (c) conformal only (LOGO-min < 0.80); (d) full Track A
(verdict != REPORT-AS-PHYSICS counts as a flag for (b)-(d) comparisons; NON-TRANSPORTABLE
is the full-protocol flag for (a)).
