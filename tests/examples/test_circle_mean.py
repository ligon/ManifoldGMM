from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from datamat import DataMat, DataVec
from jax.scipy.special import ndtri
from manifoldgmm import GMM, GMMResult, Manifold, MomentRestriction

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("cloudpickle")
try:  # pragma: no cover - optional dependency
    from pymanopt.manifolds import Sphere
except ImportError:  # pragma: no cover
    Sphere = None


ROT90 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)


def gi_jax(theta: Any, observation: Any) -> Any:
    theta_perp = ROT90 @ theta
    return jnp.array([jnp.dot(theta_perp, observation)], dtype=jnp.float64)


@pytest.mark.skipif(Sphere is None, reason="pymanopt is required for circle manifold")
def test_circle_mean_inference_matches_sandwich(tmp_path):
    mu_0 = np.pi / 2
    angles = DataVec.random(256, rng=2025, name="phi", idxnames="obs") + mu_0
    observations_dm = DataMat(
        np.column_stack(
            [
                np.cos(np.asarray(angles)),
                np.sin(np.asarray(angles)),
            ]
        ),
        index=angles.index,
        columns=["cos_phi", "sin_phi"],
        idxnames="obs",
    )
    observations = jnp.asarray(observations_dm.to_jax().values)

    manifold = Manifold.from_pymanopt(Sphere(2))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=observations,
        manifold=manifold,
        backend="jax",
    )

    estimator = GMM(
        restriction,
        initial_point=jnp.array([1.0, 0.0], dtype=jnp.float64),
    )
    result = estimator.estimate()
    theta_hat = result.theta

    basis = restriction.tangent_basis(theta_hat)
    assert len(basis) == 1

    tangent_cov = result.tangent_covariance(basis=basis)
    assert tangent_cov.shape == (1, 1)
    assert np.isfinite(tangent_cov).all()

    jacobian_chart = np.column_stack(
        [
            np.asarray(restriction._array_adapter(direction), dtype=float).reshape(-1)
            for direction in basis
        ]
    )
    ambient_cov_manual = jacobian_chart @ tangent_cov @ jacobian_chart.T

    ambient_cov = result.manifold_covariance(basis=basis)
    assert ambient_cov.shape == ambient_cov_manual.shape == (2, 2)
    np.testing.assert_allclose(ambient_cov, ambient_cov_manual)
    standard_error = np.sqrt(np.diag(ambient_cov))
    assert np.all(standard_error > 0)

    def gaussian_quantile(confidence: float = 0.95) -> float:
        upper_tail = 0.5 + 0.5 * confidence
        return float(jnp.asarray(ndtri(upper_tail)))

    z = gaussian_quantile(0.95)
    ci = np.vstack(
        [
            np.asarray(result.theta) - z * standard_error,
            np.asarray(result.theta) + z * standard_error,
        ]
    )
    assert np.all(ci[0] <= np.asarray(result.theta))
    assert np.all(ci[1] >= np.asarray(result.theta))

    artifact = tmp_path / "result.pkl"
    result.to_pickle(artifact)
    restored = GMMResult.from_pickle(artifact)
    np.testing.assert_allclose(
        np.asarray(restored.theta, dtype=float), np.asarray(theta_hat, dtype=float)
    )
    assert restored.weighting_info == result.weighting_info
