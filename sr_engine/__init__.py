"""HyperScale-CHIEF Symbolic Regression Engine.

Dimensionally-constrained sparse regression for power-law discovery.
"""

from .pipeline import audit_law, AuditResult
from .buckingham_pi import find_pi_groups
from .library import build_monomial_library
from .sparse_fit import sparse_fit_logspace
from .metrics import r_squared, normalized_exponent_distance, complexity_score

__all__ = [
    "audit_law",
    "AuditResult",
    "find_pi_groups",
    "build_monomial_library",
    "sparse_fit_logspace",
    "r_squared",
    "normalized_exponent_distance",
    "complexity_score",
]
