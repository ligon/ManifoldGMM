"""
Lightweight manifold wrapper that delegates projections to an underlying
pymanopt manifold or custom projection callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class TangentProjectionFn(Protocol):
    """Protocol for projecting an ambient vector onto the tangent space."""

    def __call__(self, point_value: Any, ambient_vector: Any) -> Any:
        ...


class PointProjectionFn(Protocol):
    """Protocol for projecting an ambient point back onto the manifold."""

    def __call__(self, ambient_point: Any) -> Any:
        ...


def _identity_point_projection(value: Any) -> Any:
    """Return the supplied point unchanged (Euclidean manifold default)."""
    return value


@dataclass(frozen=True)
class Manifold:
    """
    Wraps the projection routines needed by :class:`ManifoldPoint`.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g., ``"Stiefel(n=10, p=3)"``).
    projection:
        Callable implementing Π_θ(v). Often ``pymanopt_manifold.proj``.
    project_point:
        Callable that projects an ambient point back onto the manifold. When
        omitted the identity map is used, which is appropriate for Euclidean
        manifolds.
    data:
        Arbitrary metadata (e.g., the underlying pymanopt manifold instance).
    """

    name: str
    projection: TangentProjectionFn
    project_point: PointProjectionFn | None = None
    data: Any | None = None

    def project_tangent(self, point_value: Any, ambient_vector: Any) -> Any:
        """Project a vector onto the tangent space at ``point_value``."""
        return self.projection(point_value, ambient_vector)

    def project(self, ambient_point: Any) -> Any:
        """Project an ambient point back onto the manifold."""
        projector = self.project_point or _identity_point_projection
        return projector(ambient_point)
