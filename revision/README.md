# Revision 1 (algorithms-4569489): simulation benchmark (version 2) and Stage 0 phase study

**Version 2 (reported in the revised paper)**
- `audit_v2.py` — implementation of Algorithms 1–2 with the corrections listed in `CORRECTIONS_v2.md` (SHA-256 recorded there, after the first results had been seen; see the erratum on the stale docstring).
- `benchmark_v2.py` — 9 regimes × 16 cells × 20 replicates (2,880 corpora); output `bench_v2_K20.jsonl`.
  Run: `OMP_NUM_THREADS=1 python3 benchmark_v2.py 20 <workers> bench_v2_K20.jsonl`.
- `analyze_v2.py` — discriminant and final (capped) endpoints, ablation, Track A, sensitivity; `gen_S13.py` regenerates Supplement S13 tables.
- `worked_example_v2.py` / `.json` — re-runs two corpora from their seeds and prints every decision node (Table 15).
- `stage0_phase_v2.py`, `stage0_phase_trials_v2.csv`, `fig_stage0_phase.py` — Stage 0 phase study (14,400 designs) and Figure 4.
- `validate_real.py` — reproduces case-study values.

**Version 1 (first implementation; kept for the record, NOT reported)**
- `audit_frozen.py`, `FROZEN_PROTOCOL.md`, `benchmark_factorial.py`, `bench_K30.jsonl`, `analyze_benchmark.py`, `analysis_K30.txt`.
  An external review found that this implementation departed from the rules written in the paper
  (Test 2 parameter count, conformal cap not applied to the structure verdict, incomplete conformal
  acceptance, scale and inner-split settings). See `CORRECTIONS_v2.md`.

The benchmark is not a pre-registered test: the corrections and four of the nine regimes were fixed after the version-1 results had been seen.
