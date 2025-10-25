"""
ManifoldGMM: Generalized Method of Moments estimation on Riemannian manifolds.

This package currently provides geometry primitives and autodiff helpers.
See the documentation in ``docs/`` for design notes.
"""

from .autodiff import jacobian_from_pymanopt, jacobian_operator
from .geometry import Manifold, ManifoldPoint

__all__ = ["jacobian_operator", "jacobian_from_pymanopt", "Manifold", "ManifoldPoint"]
