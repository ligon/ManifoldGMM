from __future__ import annotations

import numpy as np
import pytest
from manifoldgmm import Manifold, ManifoldPoint, MomentRestriction

try:
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jnp = None


def _identity_projection(_point_value: np.ndarray, ambient_vector: np.ndarray) -> np.ndarray:
    return ambient_vector


def test_moment_restriction_numpy_workflow():
    data = np.array([1.0, 2.0, 3.0])

    def gi(theta, sample):
        theta_value = np.asarray(theta, dtype=float).reshape(-1)[0]
        residual = sample - theta_value
        return np.column_stack([residual, 2.0 * residual])

    def jacobian(theta, sample):
        np.asarray(theta)  # ensure the adapter delivered array-like input
        np.asarray(sample)
        return np.array([[-1.0], [-2.0]])

    restriction = MomentRestriction(gi, data=data, jacobian_map=jacobian)
    theta = np.array([2.0])

    moments = restriction.gi(theta)
    assert moments.shape == (3, 2)

    expected_mean = np.array([0.0, 0.0])
    np.testing.assert_allclose(restriction.g_bar(theta), expected_mean)
    np.testing.assert_allclose(restriction.gN(theta), expected_mean)

    expected_cov = np.array([[2.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]])
    omega = restriction.omega_hat(theta)
    np.testing.assert_allclose(omega, expected_cov)
    np.testing.assert_allclose(restriction.Omega_hat(theta), expected_cov)

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
    data = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float64)

    def gi(theta, sample):
        theta_value = theta[0]
        residual = sample - theta_value
        return jnp.stack([residual, residual**2], axis=1)

    restriction = MomentRestriction(gi, data=data, manifold=euclidean)
    theta_point = ManifoldPoint(euclidean, jnp.array([1.5], dtype=jnp.float64))

    operator = restriction.jacobian(theta_point)

    tangent = jnp.array([0.1], dtype=jnp.float64)
    residual_mean = jnp.mean(data - theta_point.value[0])
    matvec_expected = jnp.array([-0.1, -0.2 * residual_mean])
    assert jnp.allclose(operator.matvec(tangent), matvec_expected)

    covector = jnp.array([2.0, -1.5], dtype=jnp.float64)
    tangent_expected = jnp.array([-2.0 + 3.0 * residual_mean])
    assert jnp.allclose(operator.T_matvec(covector), tangent_expected)

    with pytest.raises(NotImplementedError):
        restriction.jacobian_operator(theta_point, euclidean=True)


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
