"""
JAX-backed Jacobian utilities for vector-valued moment maps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..geometry.point import ManifoldPoint

VectorFunction = Callable[[ManifoldPoint], Any]


@dataclass(frozen=True)
class JacobianOperator:
    """
    Linear operator representation of a Jacobian.

    Attributes
    ----------
    shape:
        Tuple ``(ell, ambient_dim)`` giving the flattened dimensions of the
        moment map output and manifold coordinates.
    matvec:
        Callable mapping tangent vectors (same PyTree structure as θ) to
        ℝ^ℓ outputs (same structure as the moment map).
    T_matvec:
        Callable mapping ℝ^ℓ covectors back into the tangent space.
    """

    shape: tuple[int, int]
    matvec: Callable[[Any], Any]
    T_matvec: Callable[[Any], Any]


def jacobian_operator(
    function: VectorFunction, point: ManifoldPoint
) -> JacobianOperator:
    """
    Construct a Jacobian linear operator for a vector-valued function at ``point``.

    Parameters
    ----------
    function:
        Callable f(θ) returning ℝ^ℓ (or a PyTree thereof) at ``point``.
    point:
        Base point θ on the manifold.

    Returns
    -------
    JacobianOperator
        Linear operator exposing =matvec= and =T_matvec= closures.
    """
    try:
        import jax
        from jax.flatten_util import ravel_pytree
    except ImportError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            "JAX is required for the jacobian_operator backend. "
            "Install ManifoldGMM with the 'jax' extra."
        ) from exc

    def _is_sequence(obj: Any) -> bool:
        return isinstance(obj, Sequence) and not isinstance(obj, str | bytes)

    def _to_canonical_structure(obj: Any) -> Any:
        if _is_sequence(obj):
            return [_to_canonical_structure(element) for element in obj]
        return obj

    def _restore_structure(template: Any, obj: Any) -> Any:
        if _is_sequence(template):
            if not _is_sequence(obj):
                raise TypeError("Structure mismatch when restoring manifold value")
            restored_children = [
                _restore_structure(t_child, o_child)
                for t_child, o_child in zip(template, obj, strict=False)
            ]
            if isinstance(template, tuple):
                return tuple(restored_children)
            try:
                return type(template)(restored_children)
            except TypeError:
                return restored_children
        return obj

    canonical_point = _to_canonical_structure(point.value)

    def wrapped(value: Any) -> Any:
        restored_value = _restore_structure(point.value, value)
        theta = point.with_value(restored_value)
        return function(theta)

    primal_output, jvp = jax.linearize(wrapped, canonical_point)
    _, vjp = jax.vjp(wrapped, point.value)
    flat_output, _ = ravel_pytree(primal_output)
    flat_point, _ = ravel_pytree(point.value)

    def matvec(tangent: Any) -> Any:
        projected = point.project_tangent(tangent)
        canonical_tangent = _to_canonical_structure(projected)
        return jvp(canonical_tangent)

    def T_matvec(covector: Any) -> Any:
        (tangent,) = vjp(covector)
        restored_tangent = _restore_structure(point.value, tangent)
        return point.project_tangent(restored_tangent)

    operator_shape = (flat_output.shape[0], flat_point.shape[0])

    return JacobianOperator(
        shape=operator_shape,
        matvec=matvec,
        T_matvec=T_matvec,
    )
