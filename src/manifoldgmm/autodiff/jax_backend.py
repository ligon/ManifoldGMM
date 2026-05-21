"""
JAX-backed Jacobian utilities for vector-valued moment maps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    matrix_in_basis:
        Optional callable that, given a list of tangent basis vectors of
        length ``p``, returns the dense ``(ell, p)`` Jacobian matrix in a
        single call.  When provided, this is typically a batched
        (e.g., ``jax.vmap``-based) implementation that amortises the
        per-direction Python-side overhead of looping over ``matvec``.
        ``MomentRestriction.jacobian_matrix`` uses this fast path when
        available and falls back to the ``matvec`` loop otherwise.
    """

    shape: tuple[int, int]
    matvec: Callable[[Any], Any]
    T_matvec: Callable[[Any], Any]
    matrix_in_basis: Callable[[list[Any]], np.ndarray] | None = field(default=None)


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
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
    except ImportError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            "JAX is required for the jacobian_operator backend. "
            "Run `poetry install` to install all required dependencies."
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

    def matrix_in_basis(basis_vectors: list[Any]) -> np.ndarray:
        """Batched (ell, p) Jacobian via a single ``jax.vmap(jvp)`` call.

        Builds a stacked PyTree from the basis directions, calls the
        linearised JVP closure under ``vmap``, then concatenates the
        per-leaf outputs into the dense matrix.  Each basis vector is
        first projected onto the tangent space (idempotent on a valid
        basis) and canonicalised to match the structure ``jvp`` expects,
        matching the per-direction ``matvec`` behaviour byte-for-byte.
        """

        p = len(basis_vectors)
        ell = operator_shape[0]
        if p == 0:
            return np.zeros((ell, 0), dtype=float)
        if ell == 0:
            return np.zeros((0, p), dtype=float)

        canonical_basis = [
            _to_canonical_structure(point.project_tangent(direction))
            for direction in basis_vectors
        ]
        # Stack the basis into a batched PyTree.  Each leaf gains a
        # leading axis of length p.  ``jax.tree.map`` here works because
        # every basis vector shares the structure of ``point.value``
        # (verified by ``project_tangent``).
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *canonical_basis)
        batched_output = jax.vmap(jvp)(stacked)
        # Each leaf now has shape (p, *leaf_shape).  Flatten the leaf
        # axes and concatenate so the row major order matches what the
        # loop-based ``matvec`` path produces: per-direction outputs are
        # raveled in PyTree-leaf order.
        leaves = jax.tree.leaves(batched_output)
        per_row = [np.asarray(leaf).reshape(p, -1) for leaf in leaves]
        # Each entry of `per_row` has shape (p, leaf_size); concatenation
        # along axis=1 produces (p, ell).
        dense_T = np.concatenate(per_row, axis=1)
        return dense_T.T  # (ell, p)

    return JacobianOperator(
        shape=operator_shape,
        matvec=matvec,
        T_matvec=T_matvec,
        matrix_in_basis=matrix_in_basis,
    )
