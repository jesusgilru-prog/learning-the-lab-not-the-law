# Corrections to the frozen implementation (version 2) — recorded 2026-10-01T04:56:17+02:00

The frozen record (FROZEN_PROTOCOL.md, audit_frozen.py, benchmark_factorial.py, bench_K30.jsonl)
is kept unchanged. An external review (Codex, gpt-6-astra) of the K=30 run found that the frozen
implementation departed from the rules stated in the manuscript. Version 2 corrects these
departures. These corrections were made AFTER the K=30 results had been seen; they restore the
written rules and do not tune any threshold.

| file | sha256 |
|---|---|
| audit_v2.py | 7f59247f75b2a5e87bf03e674c43a458eacf18fac4eebbfed65a81bf71b64501 |
| benchmark_v2.py | 677b0be8f7696ec2bd3f6f8be82f65722f532c43fde67a8143d12c51bdf2b1b9 |

## Corrections (each restores a rule stated in the submitted manuscript)
1. Test 2 penalised the externally fixed threshold as an estimated parameter (3 global
   parameters instead of 2), understating its Bayes factor by sqrt(n). Fixed: n_global=2.
   Real corpus: BF(S5 at 0.30 / S1) 0.009 -> 0.098 (published 0.098).
2. The conformal cap was not applied to Track B. Fixed: the discriminant label
   STRUCTURE-SUPPORTED becomes STRUCTURE-AS-PHYSICS, or STRUCTURE-WITH-DISCLAIMER when the
   conformal check fails. Both the discriminant label (verdict_B_disc) and the final verdict
   (verdict_B) are recorded.
3. The conformal acceptance used LOGO-min only. Fixed: also n_eff >= 30 and largest calibration
   weight share < 0.5 (weights 1/sigma_g). Real corpus: n_eff 75.17 (published 75.2), max share
   3.25% (published 3.25%).
4. Held-out scale used sqrt(mean sigma^2); the paper specifies mean sigma. Fixed.
5. Inner conformal splits 20 -> 200, as in the paper.
6. Track A REPORT-AS-PHYSICS required "at most one" canonical regime in the IM interval;
   Table 1 states "resolved to one canonical regime". Fixed: exactly one.
Not implemented (benchmark scope, stated in the paper): model selection is S1 vs S5 (not S1-S6);
Stage 0 and its cap (benchmark designs are identifiable by construction); the F=1 branch.

## Added regimes (added after the K=30 results, at the reviewer's request; labelled as such)
- R3w weak genuine transition at tau_phys=0.20 (jump 0.005 / 0.015).
- R3s genuine transition at 0.11 on sparse support (~8% of the points of EVERY facility below 0.11);
  reference threshold 0.11 (correct). This is the physical counterpart of R4b.
- R3m genuine transition at 0.25 while the reference supplied to Test 2 is 0.20 (misspecified).
- R4c artefact added to one facility's points above 0.20 (balanced support, artefact at the
  canonical threshold).
Design: 9 regimes x 16 cells x K=20 = 2880 corpora, master seed 20261002.

## Expected outcomes (recorded before the v2 run)
- R1, R2, R4a, R4b, R4c: correct = no STRUCTURE-* verdict (discriminant or final).
  R4b exact expected label SUPPORTED-UNATTRIBUTED; R4c expected INCONCLUSIVE (Test 4 should fail).
- R3, R3w, R3m: correct = discriminant STRUCTURE-SUPPORTED (final AS-PHYSICS or WITH-DISCLAIMER).
  Anticipated: R3w partly missed; R3m may fail Test 2.
- R3s: correct = STRUCTURE-SUPPORTED, but the minority rule of Test 1 is expected to FAIL it:
  anticipated outcome INCONCLUSIVE, i.e. a false negative by design (the protocol cannot separate
  a genuine sparse-support transition from R4b on these tests alone).
Endpoints reported: (a) discriminant label; (b) final capped verdict; (c) Track A.

## Erratum (2026-10-06)
The docstring of `audit_v2.py` (lines 4-6) is inherited from `audit_frozen.py` and wrongly says that
its SHA-256 is recorded in `FROZEN_PROTOCOL.md` before any scenario was generated and that the file
is not edited afterwards. The hash of `audit_v2.py` is recorded in this file (table above), after the
K=30 results had been seen; `FROZEN_PROTOCOL.md` records only the hashes of the first implementation.
`audit_v2.py` is left unchanged so that its hash stays valid. Also, `FROZEN_PROTOCOL.md` (lines 3-6)
records that one smoke-test job per regime was timed before the first freeze (verdicts not inspected).
