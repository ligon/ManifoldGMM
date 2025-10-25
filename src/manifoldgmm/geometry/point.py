"""ManifoldPoint represents a location on a manifold with projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    from pymanopt.manifolds.manifold import Manifold as PymanoptManifold
except ImportError:  # pragma: no cover
    PymanoptManifold = None  # type: ignore[assignment]

from .manifold import Manifold, PointProjectionFn


@dataclass(frozen=True)
class ManifoldPoint:
    """
    Bundle a manifold with a point living on it.

    Parameters
    ----------
    manifold:
        Underlying manifold description with projection routines.
    value:
        Ambient representation of the point. This is projected onto the
        manifold during initialisation.
    """

    manifold: Manifold
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.manifold.project(self.value))

    def with_value(self, value: Any) -> "ManifoldPoint":
        """
        Return a new point on the same manifold with updated coordinates.

        Parameters
        ----------
        value:
            Ambient representation to associate with the new point.
        """
        data = self.manifold.project(value)
        return ManifoldPoint(self.manifold, data)

    def project_tangent(self, ambient_vector: Any) -> Any:
        """Project an ambient vector onto the tangent space at this point."""
        return self.manifold.project_tangent(self.value, ambient_vector)

    @classmethod
    def from_pymanopt(
        cls,
        manifold: "PymanoptManifold",
        value: Any,
        *,
        project_point: PointProjectionFn | None = None,
    ) -> "ManifoldPoint":
        """
        Create a :class:`ManifoldPoint` from a ``pymanopt`` manifold instance.

        Parameters
        ----------
        manifold:
            A ``pymanopt`` manifold providing tangent projections.
        value:
            Ambient representation of the point (must already satisfy manifold
            constraints unless ``project_point`` is provided).
        project_point:
            Optional callable to project ``value`` back onto the manifold
            before storing it.
        """

        if PymanoptManifold is None:  # pragma: no cover
            raise RuntimeError(
                "pymanopt is required to construct ManifoldPoint objects from pymanopt manifolds"
            )
        wrapper = Manifold.from_pymanopt(manifold, project_point=project_point)
        return cls(wrapper, value)

    def as_pymanopt_data(self) -> Any:
        """
        Return the ambient representation for use with ``pymanopt`` APIs.

        This is a thin wrapper around the stored value to avoid exposing
        internal dataclass attributes directly in user code.
        """

        return self.value
