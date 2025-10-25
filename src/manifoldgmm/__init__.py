"""
ManifoldGMM: Generalized Method of Moments estimation on Riemannian manifolds.

The package currently provides geometry primitives, autodiff helpers, and
econometrics-layer abstractions for moment restrictions.
"""

from .autodiff import jacobian_from_pymanopt, jacobian_operator
from .econometrics import MomentRestriction
from .geometry import Manifold, ManifoldPoint

__all__ = [
    "jacobian_operator",
    "jacobian_from_pymanopt",
    "Manifold",
    "ManifoldPoint",
    "MomentRestriction",
]
