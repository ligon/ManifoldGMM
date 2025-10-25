"""Manifold wrappers building on top of pymanopt manifolds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from pymanopt.manifolds.manifold import Manifold as PymanoptManifold
else:  # pragma: no cover - runtime fallback when type hints are unavailable
    PymanoptManifold = object

try:  # pragma: no cover - optional dependency already declared in pyproject
    from pymanopt.manifolds.manifold import Manifold as _PymanoptManifoldRuntime
except ImportError:  # pragma: no cover
    _PymanoptManifoldRuntime = None


class TangentProjectionFn(Protocol):
    """Protocol for projecting an ambient vector onto the tangent space."""

    def __call__(self, point_value: Any, ambient_vector: Any) -> Any: ...


class PointProjectionFn(Protocol):
    """Protocol for projecting an ambient point back onto the manifold."""

    def __call__(self, ambient_point: Any) -> Any: ...


def _identity_point_projection(value: Any) -> Any:
    """Return the supplied point unchanged (Euclidean manifold default)."""
    return value


_ProjectPointFn = Callable[[Any], Any]
_ProjectionFn = Callable[[Any, Any], Any]

Pymanopt = TypeVar("Pymanopt")


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

    def random_point(self) -> Any:
        """
        Draw a random point on the manifold.

        Returns
        -------
        Any
            Ambient representation sampled from the wrapped manifold.
        """

        rand_fn = getattr(self.data, "random_point", None)
        if callable(rand_fn):
            return rand_fn()
        raise AttributeError("Underlying manifold does not expose random_point()")

    def random_tangent(self, base_point: Any | None = None) -> Any:
        """
        Draw a random tangent vector at ``base_point``.

        Parameters
        ----------
        base_point:
            Point on the manifold. If omitted, a new random point is sampled
            first and used as the base.
        """

        tangent_fn = getattr(self.data, "random_tangent_vector", None)
        if not callable(tangent_fn):
            raise AttributeError(
                "Underlying manifold does not expose random_tangent_vector()"
            )
        if base_point is None:
            base_point = self.random_point()
        return tangent_fn(base_point)

    @classmethod
    def from_pymanopt(
        cls,
        manifold: PymanoptManifold,
        *,
        project_point: PointProjectionFn | None = None,
    ) -> Manifold:
        """
        Wrap a ``pymanopt`` manifold so it can be used with :class:`ManifoldPoint`.

        Parameters
        ----------
        manifold:
            Instance of :class:`pymanopt.manifolds.manifold.Manifold`.
        project_point:
            Optional callable that projects ambient points onto ``manifold``.
            If omitted, points are assumed to already satisfy the manifold
            constraints.
        """

        if _PymanoptManifoldRuntime is None:  # pragma: no cover
            raise RuntimeError(
                "pymanopt is required to construct a Manifold from a pymanopt manifold"
            )
        if not isinstance(manifold, _PymanoptManifoldRuntime):
            raise TypeError(
                "Expected a pymanopt.manifolds.manifold.Manifold instance; "
                f"got {type(manifold)!r}"
            )
        if not hasattr(manifold, "projection"):
            raise AttributeError(
                "pymanopt manifold does not expose a 'projection' method required "
                "for tangent projections."
            )
        projection = cast(TangentProjectionFn, manifold.projection)
        return cls(
            name=str(manifold),
            projection=projection,
            project_point=project_point,
            data=manifold,
        )
