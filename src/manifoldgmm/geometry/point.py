"""ManifoldPoint represents a location on a manifold with projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .manifold import Manifold


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
