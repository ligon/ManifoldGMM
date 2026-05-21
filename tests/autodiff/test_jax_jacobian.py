"""Tests for the JAX Jacobian helper."""

from __future__ import annotations

import jax.numpy as jnp
from manifoldgmm.autodiff import jacobian_operator
from manifoldgmm.geometry import Manifold, ManifoldPoint


def identity_projection(point, vector):
    return vector


euclidean = Manifold(name="Euclidean", projection=identity_projection)


def vector_function(point: ManifoldPoint):
    x = point.value
    return jnp.stack([x[0] + x[1], x[0] * x[1]])


def test_matvec_matches_jacobian():
    theta = ManifoldPoint(euclidean, jnp.array([1.5, -0.25]))
    jac = jacobian_operator(vector_function, theta)
    tangent = jnp.array([0.1, -0.2])
    result = jac.matvec(tangent)
    expected = jnp.array(
        [
            tangent[0] + tangent[1],
            theta.value[0] * tangent[1] + theta.value[1] * tangent[0],
        ]
    )
    assert jnp.allclose(result, expected)


def test_transpose_maps_back_to_tangent():
    theta = ManifoldPoint(euclidean, jnp.array([0.3, 0.7]))
    jac = jacobian_operator(vector_function, theta)
    covector = jnp.array([2.0, -1.0])
    tangent = jac.T_matvec(covector)
    expected = jnp.array(
        [
            covector[0] + covector[1] * theta.value[1],
            covector[0] + covector[1] * theta.value[0],
        ]
    )
    assert jnp.allclose(tangent, expected)
    projected = theta.project_tangent(tangent)
    assert jnp.allclose(projected, tangent)


def test_matrix_in_basis_matches_matvec_loop():
    """Batched matrix_in_basis must agree with the matvec loop column-by-column."""

    import numpy as np

    theta = ManifoldPoint(euclidean, jnp.array([1.5, -0.25]))
    jac = jacobian_operator(vector_function, theta)
    basis = [jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])]

    expected_cols = [np.asarray(jac.matvec(b), dtype=float).reshape(-1) for b in basis]
    expected = np.column_stack(expected_cols)

    assert jac.matrix_in_basis is not None
    actual = jac.matrix_in_basis(basis)

    assert actual.shape == expected.shape == (2, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_matrix_in_basis_empty():
    """Empty basis returns an (ell, 0) matrix."""

    import numpy as np

    theta = ManifoldPoint(euclidean, jnp.array([1.0, 1.0]))
    jac = jacobian_operator(vector_function, theta)
    assert jac.matrix_in_basis is not None
    out = jac.matrix_in_basis([])
    assert out.shape == (2, 0)
    assert isinstance(out, np.ndarray)
