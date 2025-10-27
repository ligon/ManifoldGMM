from __future__ import annotations

import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean as PymanoptEuclidean
from pymanopt.manifolds import Product as PymanoptProduct

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _build_simple_restriction(
    backend: str = "jax",
) -> tuple[MomentRestriction, float]:
    data = jnp.array([1.0, 2.0, 3.0])

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend=backend,
    )
    true_mean = float(np.mean(np.asarray(data)))
    return restriction, true_mean


def test_gmm_estimate_matches_sample_mean() -> None:
    restriction, true_mean = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate()

    estimate = result.theta
    assert np.allclose(np.asarray(estimate), np.array([true_mean]), atol=1e-8)
    assert np.allclose(np.asarray(result.g_bar), np.zeros_like(result.g_bar), atol=1e-8)
    assert result.degrees_of_freedom == 0


def test_gmm_two_step_sets_flag_and_updates_weighting() -> None:
    restriction, true_mean = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate(two_step=True)

    assert result.two_step is True
    assert result.weighting_info.get("two_step") is True
    assert np.allclose(np.asarray(result.theta), np.array([true_mean]), atol=1e-8)


def test_exposed_helpers_match_restriction_evaluations() -> None:
    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    theta = jnp.array([1.5])
    np.testing.assert_allclose(
        np.asarray(gmm.g_bar(theta)), np.asarray(restriction.g_bar(theta))
    )
    np.testing.assert_allclose(
        np.asarray(gmm.gN(theta)), np.asarray(restriction.gN(theta))
    )


def test_gmm_handles_product_manifold_initial_points() -> None:
    data = jnp.array([[1.0, 2.0], [1.5, 1.8]])

    def gi_jax(theta, observation):
        mu, alpha = theta
        return jnp.array([observation[0] - mu[0], observation[1] - alpha[0]])

    manifold = Manifold.from_pymanopt(
        PymanoptProduct((PymanoptEuclidean(1), PymanoptEuclidean(1)))
    )
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
    )

    true_params = (
        jnp.array([jnp.mean(data[:, 0])]),
        jnp.array([jnp.mean(data[:, 1])]),
    )

    estimator = GMM(
        restriction,
        weighting=np.eye(2),
        initial_point=(jnp.zeros(1), jnp.zeros(1)),
    )

    result = estimator.estimate()
    theta_hat = result.theta
    np.testing.assert_allclose(
        np.asarray(theta_hat[0]), np.asarray(true_params[0]), atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(theta_hat[1]), np.asarray(true_params[1]), atol=1e-8
    )
