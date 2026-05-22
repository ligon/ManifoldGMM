"""
ManifoldGMM: Generalized Method of Moments estimation on Riemannian manifolds.

The package currently provides geometry primitives, autodiff helpers, and
econometrics-layer abstractions for moment restrictions.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("manifoldgmm")
except PackageNotFoundError:
    __version__ = "0.0.0"

from ._warnings import ConvergenceWarning
from .autodiff import jacobian_from_pymanopt, jacobian_operator
from .econometrics import GMM, GMMResult, KStatisticResult, MomentRestriction
from .econometrics.bootstrap import MomentWildBootstrap, geodesic_mahalanobis_distance
from .econometrics.simulation import monte_carlo
from .geometry import Manifold, ManifoldPoint

__all__ = [
    "jacobian_operator",
    "jacobian_from_pymanopt",
    "Manifold",
    "ManifoldPoint",
    "GMM",
    "GMMResult",
    "KStatisticResult",
    "MomentRestriction",
    "MomentWildBootstrap",
    "geodesic_mahalanobis_distance",
    "monte_carlo",
    "ConvergenceWarning",
]
