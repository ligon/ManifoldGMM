from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .manifold import Manifold

if TYPE_CHECKING:  # pragma: no cover
    from pymanopt.manifolds.manifold import Manifold as PymanoptManifold
else:  # pragma: no cover
    PymanoptManifold = object

try:  # pragma: no cover
    from pymanopt.manifolds.manifold import Manifold as _PymanoptManifoldRuntime
except ImportError:  # pragma: no cover
    _PymanoptManifoldRuntime = None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, tuple | list)


def _as_array(value: Any) -> np.ndarray:
    if _is_sequence(value):
        flattened = [np.asarray(component).ravel() for component in value]
        return np.concatenate(flattened)
    return np.asarray(value).ravel()


def _scale(value: Any, factor: float) -> Any:
    if _is_sequence(value):
        return type(value)(_scale(component, factor) for component in value)
    return np.asarray(value) * factor


def _difference_norm(a: Any, b: Any) -> float:
    return float(np.linalg.norm(_as_array(a) - _as_array(b)))


def _ensure_same_manifold(lhs: Manifold, rhs: Manifold) -> None:
    if lhs is not rhs:
        raise ValueError("Points belong to different manifolds")


class ManifoldPoint:
    """Immutable point living on a :class:`Manifold`."""

    __slots__ = ("manifold", "_value")

    manifold: Manifold
    _value: Any

    def __init__(self, manifold: Manifold, value: Any):
        object.__setattr__(self, "manifold", manifold)
        projected = manifold.project(value)
        canonical = manifold.canonicalize(projected)
        object.__setattr__(self, "_value", canonical)

    @property
    def value(self) -> Any:
        """Ambient representation of the point."""

        return self._value

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        shape = np.asarray(self._value).shape
        return f"ManifoldPoint(name={self.manifold.name!r}, shape={shape})"

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        raise AttributeError("ManifoldPoint is immutable")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pymanopt(
        cls,
        manifold: PymanoptManifold,
        value: Any,
        *,
        project_point: Any | None = None,
        canonical_point: Any | None = None,
    ) -> ManifoldPoint:
        if _PymanoptManifoldRuntime is None:  # pragma: no cover
            raise RuntimeError(
                "pymanopt is required to construct ManifoldPoint objects"
            )
        if not isinstance(manifold, _PymanoptManifoldRuntime):
            raise TypeError(
                "Expected a pymanopt.manifolds.manifold.Manifold instance; "
                f"got {type(manifold)!r}"
            )
        wrapper = Manifold.from_pymanopt(
            manifold,
            project_point=project_point,
            canonical_point=canonical_point,
        )
        return cls(wrapper, value)

    def with_value(self, value: Any) -> ManifoldPoint:
        """Return a new point on the same manifold with updated coordinates."""

        return ManifoldPoint(self.manifold, value)

    def copy(self) -> ManifoldPoint:
        """Clone the point."""

        return ManifoldPoint(self.manifold, self._value)

    def components(self) -> tuple[Any, ...]:
        """Return components when the manifold is a product."""

        if _is_sequence(self._value):
            return tuple(self._value)
        return (self._value,)

    def component(self, index: int) -> Any:
        """Return the ``index``-th component for product manifolds."""

        if not _is_sequence(self._value):
            if index != 0:
                raise IndexError("Single component manifold: index must be 0")
            return self._value
        return self._value[index]

    # ------------------------------------------------------------------
    # Membership & canonicalisation
    # ------------------------------------------------------------------
    def is_on_manifold(self, tol: float = 1e-8) -> bool:
        """Check whether the stored value satisfies manifold constraints."""

        projected = self.manifold.project(self._value)
        return _difference_norm(projected, self._value) <= tol

    def canonicalize(self) -> ManifoldPoint:
        """Return a new point mapped to the canonical representative."""

        canonical = self.manifold.canonicalize(self._value)
        return ManifoldPoint(self.manifold, canonical)

    # ------------------------------------------------------------------
    # Tangent operations
    # ------------------------------------------------------------------
    def project_tangent(self, ambient_vector: Any) -> Any:
        """Project an ambient vector onto the tangent space at this point."""

        return self.manifold.project_tangent(self._value, ambient_vector)

    def retract(self, tangent_vector: Any, *, step: float = 1.0) -> ManifoldPoint:
        """Retract along ``tangent_vector`` to obtain a new point."""

        retraction = getattr(self.manifold.data, "retraction", None)
        if retraction is None:
            retraction = getattr(self.manifold.data, "retract", None)
        if retraction is None:
            raise AttributeError("Underlying manifold does not expose retraction()")
        scaled = _scale(tangent_vector, step)
        if np.linalg.norm(_as_array(scaled)) == 0.0:
            return self.copy()
        new_value = retraction(self._value, scaled)
        return ManifoldPoint(self.manifold, new_value)

    def exp(self, tangent_vector: Any, *, step: float = 1.0) -> ManifoldPoint:
        """Exponential map along ``tangent_vector`` if available."""

        exponential = getattr(self.manifold.data, "exp", None)
        if exponential is None:
            raise AttributeError("Underlying manifold does not expose exp()")
        scaled = _scale(tangent_vector, step)
        new_value = exponential(self._value, scaled)
        return ManifoldPoint(self.manifold, new_value)

    def transport(self, to_point: ManifoldPoint, tangent_vector: Any) -> Any:
        """Transport ``tangent_vector`` from this point to ``to_point``."""

        _ensure_same_manifold(self.manifold, to_point.manifold)
        transport_fn = getattr(self.manifold.data, "transport", None)
        if transport_fn is None:
            raise AttributeError("Underlying manifold does not expose transport()")
        return transport_fn(self._value, to_point._value, tangent_vector)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def as_pymanopt_data(self) -> Any:
        """Return the stored ambient representation."""

        return self._value

    def __eq__(self, other: object) -> bool:  # pragma: no cover - convenience
        if not isinstance(other, ManifoldPoint):
            return NotImplemented
        try:
            _ensure_same_manifold(self.manifold, other.manifold)
        except ValueError:
            return False
        return np.allclose(_as_array(self._value), _as_array(other._value))
