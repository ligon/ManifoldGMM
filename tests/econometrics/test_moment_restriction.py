from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from datamat import DataMat
from manifoldgmm import Manifold, ManifoldPoint, MomentRestriction

jax = None  # type: Any
jnp = None  # type: Any
try:  # pragma: no cover - runtime optional dependency
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
except ModuleNotFoundError:
    pass

try:  # pragma: no cover - optional dependency resolved at runtime
    _manifolds_mod: Any = importlib.import_module("pymanopt.manifolds")
    PymanoptEuclidean = _manifolds_mod.Euclidean
    PymanoptProduct = _manifolds_mod.Product
    SymmetricPositiveDefinite = _manifolds_mod.SymmetricPositiveDefinite
except ImportError:
    PymanoptEuclidean = None
    PymanoptProduct = None
    SymmetricPositiveDefinite = None


def _identity_projection(
    _point_value: np.ndarray, ambient_vector: np.ndarray
) -> np.ndarray:
    return ambient_vector


def test_moment_restriction_numpy_workflow():
    data = DataMat({"y": [1.0, 2.0, 3.0]})

    def gi(theta, sample):
        theta_value = np.asarray(theta, dtype=float).reshape(-1)[0]
        residual = sample["y"] - theta_value
        stacked = np.column_stack([residual.values, (2.0 * residual).values])
        columns = pd.MultiIndex.from_product([["moments"], ["g1", "g2"]])
        return DataMat(stacked, index=residual.index, columns=columns)

    def jacobian(theta, sample):
        np.asarray(theta)
        np.asarray(sample)
        values = np.array([[-1.0], [-2.0]])
        return DataMat(values, index=["g1", "g2"], columns=["theta"])

    restriction = MomentRestriction.from_datamat(
        gi,
        data=data,
        jacobian_datamat=jacobian,
        backend="numpy",
    )
    theta = np.array([2.0])

    moments = restriction.gi(theta)
    assert moments.shape == (3, 2)

    expected_mean = np.array([0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(restriction.g_bar(theta)).ravel(), expected_mean
    )
    np.testing.assert_allclose(np.asarray(restriction.gN(theta)).ravel(), expected_mean)

    expected_cov = np.array([[2.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]])
    omega = restriction.omega_hat(theta)
    np.testing.assert_allclose(np.asarray(omega), expected_cov)
    np.testing.assert_allclose(np.asarray(restriction.Omega_hat(theta)), expected_cov)

    eigenvalues = np.linalg.eigvalsh(np.asarray(omega, dtype=float))
    assert np.all(eigenvalues >= -1e-12)

    operator = restriction.jacobian(theta)
    matvec_result = operator.matvec(np.array([1.0]))
    np.testing.assert_allclose(matvec_result.reshape(-1), np.array([-1.0, -2.0]))

    covector = np.array([1.0, 2.0])
    T_matvec_result = operator.T_matvec(covector)
    np.testing.assert_allclose(T_matvec_result.reshape(-1), np.array([-5.0]))

    np.testing.assert_allclose(
        restriction.jacobian_operator(theta, euclidean=True), np.array([[-1.0], [-2.0]])
    )

    assert restriction.parameter_dimension == 1
    np.testing.assert_array_equal(restriction.observation_counts, np.array([3.0, 3.0]))
    assert restriction.num_moments == 2
    assert restriction.num_observations == 3


@pytest.mark.skipif(jnp is None, reason="JAX is required for autodiff Jacobian test")
def test_moment_restriction_jacobian_autodiff():
    euclidean = Manifold(name="R1", projection=_identity_projection)
    raw_data = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float64)
    data = DataMat({"y": [1.0, 2.0, 4.0]})

    def gi(theta, sample):
        theta_value = float(np.asarray(theta).reshape(-1)[0])
        residual = sample["y"] - theta_value
        stacked = np.column_stack([residual.values, (residual.values) ** 2])
        columns = pd.MultiIndex.from_product([["moments"], ["g1", "g2"]])
        return DataMat(stacked, index=residual.index, columns=columns)

    def jacobian(theta, sample):
        theta_value = float(np.asarray(theta).reshape(-1)[0])
        residual = sample["y"] - theta_value
        mean_residual = float(residual.to_numpy(dtype=float).mean())
        values = np.array([[-1.0], [-2.0 * mean_residual]])
        return DataMat(
            values,
            index=[("moments", "g1"), ("moments", "g2")],
            columns=["theta"],
        )

    restriction = MomentRestriction.from_datamat(
        gi,
        data=data,
        jacobian_datamat=jacobian,
        manifold=euclidean,
        backend="jax",
    )
    theta_point = ManifoldPoint(euclidean, jnp.array([1.5], dtype=jnp.float64))

    operator = restriction.jacobian(theta_point)

    tangent = jnp.array([0.1], dtype=jnp.float64)
    residual_mean = jnp.mean(raw_data - theta_point.value[0])
    matvec_expected = jnp.array([-0.1, -0.2 * residual_mean])
    assert jnp.allclose(operator.matvec(tangent), matvec_expected)

    covector = jnp.array([2.0, -1.5], dtype=jnp.float64)
    tangent_expected = jnp.array([-2.0 + 3.0 * residual_mean])
    assert jnp.allclose(operator.T_matvec(covector), tangent_expected)

    jacobian_matrix = restriction.jacobian_operator(theta_point, euclidean=True)
    expected_jacobian = jnp.array([[-1.0], [-2.0 * float(residual_mean)]])
    np.testing.assert_allclose(
        np.asarray(jacobian_matrix), np.asarray(expected_jacobian)
    )

    matrix_dense = restriction.jacobian_matrix(theta_point)
    np.testing.assert_allclose(np.asarray(matrix_dense), np.asarray(expected_jacobian))

    omega = restriction.omega_hat(theta_point)
    residual = raw_data - theta_point.value[0]
    stacked = jnp.stack([residual, residual**2], axis=1)
    centered = stacked - jnp.mean(stacked, axis=0)
    scale = jnp.sqrt(stacked.shape[0])
    expected_omega = (centered / scale).T @ (centered / scale)
    np.testing.assert_allclose(np.asarray(omega), np.asarray(expected_omega))


@pytest.mark.skipif(jnp is None or jax is None, reason="JAX is required")
def test_moment_restriction_gi_jax():
    data = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float64)

    def gi_jax(theta, observation):
        residual = observation - theta[0]
        return jnp.array([residual, residual**2])

    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        backend="jax",
    )

    manifold = Manifold(name="R1", projection=_identity_projection)
    theta_point = ManifoldPoint(manifold, jnp.array([1.2], dtype=jnp.float64))

    def g_bar_reference(theta_array):
        residual = data - theta_array[0]
        return jnp.array([residual.mean(), jnp.mean(residual**2)])

    expected_g_bar = g_bar_reference(theta_point.value)
    np.testing.assert_allclose(
        np.asarray(restriction.g_bar(theta_point)),
        np.asarray(expected_g_bar),
    )

    stacked = jnp.stack(
        [data - theta_point.value[0], (data - theta_point.value[0]) ** 2], axis=1
    )
    centered = stacked - stacked.mean(axis=0)
    expected_omega = centered.T @ centered / stacked.shape[0]
    np.testing.assert_allclose(
        np.asarray(restriction.omega_hat(theta_point)),
        np.asarray(expected_omega),
    )

    operator = restriction.jacobian(theta_point)
    expected_jacobian = jax.jacrev(g_bar_reference)(theta_point.value)
    tangent = jnp.array([0.05], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(operator.matvec(tangent)),
        np.asarray(expected_jacobian @ tangent),
    )


def test_moment_restriction_tracks_missing_data_counts():
    data = np.array([1.0, 2.0, 3.0])

    def gi(theta, sample):
        theta_value = float(np.asarray(theta).reshape(-1)[0])
        residual = sample - theta_value
        stacked = np.column_stack([residual, residual])
        stacked[1, 1] = np.nan
        return stacked

    restriction = MomentRestriction(gi, data=data)
    restriction.g_bar(np.array([2.0]))

    np.testing.assert_array_equal(restriction.observation_counts, np.array([3.0, 2.0]))
    assert restriction.num_moments == 2
    assert restriction.num_observations == 3


def test_moment_restriction_accepts_manifold_points_and_custom_adapter():
    euclidean = Manifold(name="R1", projection=_identity_projection)

    data = np.array([1.0, 2.0, 3.0])

    def gi_manifold(point: ManifoldPoint, sample):
        value = np.asarray(point.value).reshape(-1)[0]
        residual = sample - value
        return residual[:, np.newaxis]

    restriction = MomentRestriction(
        gi_manifold,
        data=data,
        manifold=euclidean,
        argument_adapter=lambda point: point,
    )

    theta_point = ManifoldPoint(euclidean, np.array([2.0]))

    moments = restriction.gi(theta_point)
    assert moments.shape == (3, 1)
    np.testing.assert_allclose(restriction.g_bar(theta_point), np.array([0.0]))
    assert restriction.parameter_dimension == 1

    restriction_no_jac = MomentRestriction(gi_manifold, data=data)
    with pytest.raises(NotImplementedError):
        restriction_no_jac.jacobian(np.array([2.0]))
    with pytest.raises(NotImplementedError):
        restriction_no_jac.jacobian_operator(np.array([2.0]), euclidean=True)


def test_moment_restriction_tangent_basis_product_manifold():
    if PymanoptProduct is None:
        pytest.skip("pymanopt is required for tangent basis generation")

    product_manifold = PymanoptProduct(
        (PymanoptEuclidean(2), SymmetricPositiveDefinite(2))
    )
    manifold = Manifold.from_pymanopt(product_manifold)

    def gi(theta, _data=None):
        mu, sigma = theta
        np.asarray(mu)
        np.asarray(sigma)
        return np.zeros((1, 1))

    restriction = MomentRestriction(gi, manifold=manifold)
    theta = (np.zeros(2), np.eye(2))

    basis = restriction.tangent_basis(theta)
    assert len(basis) == 5

    def _flatten(value: Any) -> np.ndarray:
        if isinstance(value, tuple | list):
            parts = [_flatten(component) for component in value]
            return np.concatenate(parts) if parts else np.array([], dtype=float)
        return np.asarray(value, dtype=float).reshape(-1)

    matrix = np.column_stack([_flatten(direction) for direction in basis])
    assert np.linalg.matrix_rank(matrix) == len(basis)

    mu_directions = [
        direction for direction in basis if np.linalg.norm(direction[0]) > 1e-12
    ]
    sigma_directions = [
        direction for direction in basis if np.linalg.norm(direction[1]) > 1e-12
    ]

    assert len(mu_directions) == 2
    assert len(sigma_directions) == 3

    for _, sigma_direction in sigma_directions:
        np.testing.assert_allclose(sigma_direction, sigma_direction.T)
