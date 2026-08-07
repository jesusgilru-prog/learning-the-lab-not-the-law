#!/usr/bin/env bash
# Regenerates every number in Supplement S4-S6 (supplement.pdf), as promised
# in the supplement's own opening paragraph. Requires: numpy, scipy, pandas,
# pyarrow (for the .parquet checkpoint), and gplearn (for S6's naive-SR
# replication only -- pip install gplearn if scripts/naive_sr_replication.py
# is not needed).
#
# Does NOT re-run the full multi-experiment battery in
# scripts/remediation_experiments.py (Experiments 1-5, including S1-S6
# discriminant tests, IRM/GroupDRO, and the threshold sweep behind Table 6/7)
# -- that script still assumes the original research repository's directory
# layout and re-running it fresh risks silently diverging numbers already
# cited in the main text if the environment differs even slightly. Its
# already-computed outputs are archived in data/processed_checkpoints/
# (model_comparison_table.csv, threshold_sweep_bf.csv) instead. See
# README.md for the full breakdown of what each script does and does not
# cover.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

echo "=== S4: per-facility MAPE/RMSE (facility_mape_rmse.csv) ==="
python3 scripts/facility_mape_rmse.py

echo
echo "=== S6: structural-confound detection power (structural_confound_power.csv) ==="
python3 scripts/structural_confound_power.py

echo
echo "=== S6: naive/blind SR replication (naive_sr_replication.json) ==="
echo "(requires gplearn: pip install gplearn)"
python3 scripts/naive_sr_replication.py

echo
echo "Done. Outputs in results/."
