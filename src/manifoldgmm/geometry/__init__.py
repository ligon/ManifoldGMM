"""Geometry primitives for ManifoldGMM."""

from .manifold import Manifold, PointProjectionFn, TangentProjectionFn
from .point import ManifoldPoint

__all__ = [
    "Manifold",
    "ManifoldPoint",
    "TangentProjectionFn",
    "PointProjectionFn",
]
