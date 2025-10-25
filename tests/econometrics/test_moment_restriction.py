from __future__ import annotations

import numpy as np
import pytest

from manifoldgmm import Manifold, ManifoldPoint, MomentRestriction


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
    np.testing.assert_allclose(restriction.omega_hat(theta), expected_cov)
    np.testing.assert_allclose(restriction.Omega_hat(theta), expected_cov)

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
