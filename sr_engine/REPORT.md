# SR Engine — Benchmark Report

## Recovery Rates on SRSD Power-Law Subset

| Difficulty | Exact (NED<0.1) | Total | Rate  | Avg NED | Avg R²   | Avg Time |
|------------|-----------------|-------|-------|---------|----------|----------|
| Easy       | 24              | 24    | 100%  | 0.0077  | 0.9993   | 0.24s    |
| Medium     | 26              | 26    | 100%  | 0.0027  | 0.9966   | 0.29s    |
| Hard       | 23              | 23    | 100%  | 0.0030  | 0.9968   | 0.30s    |
| **Total**  | **73**          | **73**| **100%** | **0.0043** | **0.9976** | **0.28s** |

Total benchmark time: ~20s for all 73 problems.

## Methodology

1. **Log-space transformation**: `log(y) = log(C) + Σ aᵢ·log(xᵢ)`
2. **Multi-method regression**: OLS, LassoCV, RidgeCV, HuberRegressor, RANSACRegressor
3. **Best model selection**: By R² in log-space
4. **Exponent rounding**: To nearest simple fraction (halves, thirds, quarters, etc.)
5. **Bootstrap CI**: 200 resamples for confidence intervals

## Comparison with Literature

| Method            | SRSD-Feynman Easy | Source                |
|-------------------|-------------------|-----------------------|
| **This engine**   | **100%**          | Power-law subset only |
| PySR              | ~70%              | SRSD paper (all forms)|
| Φ-SO              | ~55%              | Tenachi et al. 2023   |
| LaSR (GPT-4o)     | ~45%              | LLM-SR-Bench 2024     |

**Important caveat**: Our 100% rate is on the *power-law subset* (73/120 equations
that are well-approximated by power laws in log-space, R²>0.95). The full SRSD
benchmark includes transcendental functions (sin, exp, sqrt compositions) that
our current engine cannot handle. PySR's ~70% is on the full set including those.

## Execution Environment

- Python 3.13, scikit-learn 1.7, sympy, numpy
- Single-threaded, no GPU
- 64GB RAM system (uses <500MB)

## Limitations

1. **Power-laws only**: Cannot discover transcendental functions (sin, cos, exp, log).
   The subset filter (R²>0.95 in log-space) pre-selects tractable problems.
2. **No additive terms**: Model is strictly multiplicative (y = C·∏xᵢᵃⁱ).
   Equations like `y = x₁ + x₂²` are outside scope.
3. **Positive data required**: Log transformation requires y>0 and X>0.
   Uses |y|, |X| fallback but sign information is lost.
4. **No dimensional data in SRSD**: The Buckingham Pi module is implemented
   and tested but not used in the SRSD benchmark (dataset lacks dimension annotations).
   It will be used for the windage/scaling-law datasets.
5. **Rounding heuristic**: Exponent rounding to simple fractions may fail for
   irrational exponents (e.g., √2 ≈ 1.414). Current tolerance is 0.08.

## Next Steps (Out of Scope for This Phase)

- Conformal prediction for uncertainty quantification
- PySR/Φ-SO comparison pipeline
- Extension to additive/compositional forms
- Integration with windage experimental data (liu2024, garnier2007)
