from __future__ import annotations

from itertools import combinations_with_replacement

import jax.numpy as jnp
import numpy as np
import pytest
from datamat import DataMat, DataVec
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean, Product, SymmetricPositiveDefinite

pytestmark = pytest.mark.slow


def _build_dataset(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = DataMat.random(
        (512, 2),
        rng=rng,
        columns=["x1", "x2"],
        colnames="p",
        idxnames="i",
    )

    S = DataMat([[1.0, 0.4], [0.4, 1.1]], index=X.columns, columns=X.columns)
    mu = DataVec([0.3, -0.2], index=X.columns)
    transformed = X @ S + mu
    sigma_true = S @ S.T

    observations_array = jnp.asarray(transformed.to_numpy(dtype=float))
    mu_true = jnp.asarray(mu.to_numpy(dtype=float))
    sigma_true_array = jnp.asarray(sigma_true.to_numpy(dtype=float))
    return observations_array, mu_true, sigma_true_array


def _build_restrictions(observations_array: jnp.ndarray):
    combos = list(combinations_with_replacement(range(observations_array.shape[1]), 3))

    def gi_jax(theta, observation):
        mu, sigma = theta
        e = observation - mu
        e2 = e[:, None] @ e[None, :] - sigma
        flatten_idx = jnp.array(combos) @ jnp.array(
            [observations_array.shape[1] ** 2, observations_array.shape[1], 1]
        )
        e3 = jnp.kron(e, jnp.kron(e, e))[flatten_idx]
        tri_rows = jnp.triu_indices(observations_array.shape[1])
        return jnp.concatenate([e, e2[tri_rows], e3])

    geometries = {
        "euclidean": Manifold.from_pymanopt(Product((Euclidean(2), Euclidean(2, 2)))),
        "product": Manifold.from_pymanopt(
            Product((Euclidean(2), SymmetricPositiveDefinite(2)))
        ),
    }

    restrictions = {
        key: MomentRestriction(
            gi_jax=gi_jax,
            data=observations_array,
            manifold=geom,
            backend="jax",
        )
        for key, geom in geometries.items()
    }
    return restrictions


def _run_cue_gmm(restrictions):
    return {
        "euclidean": GMM(
            restrictions["euclidean"],
            initial_point=(jnp.zeros(2), jnp.zeros((2, 2))),
        ).estimate(),
        "product": GMM(
            restrictions["product"],
            initial_point=(jnp.zeros(2), jnp.eye(2)),
        ).estimate(),
    }


def test_gaussian_example_estimation_produces_psd_covariance():
    observations_array, mu_true, sigma_true = _build_dataset(seed=7)
    restrictions = _build_restrictions(observations_array)
    cue_results = _run_cue_gmm(restrictions)

    result = cue_results["product"]
    mu_hat, sigma_hat = result.theta.value

    observations_np = np.asarray(observations_array, dtype=float)
    sample_mean = observations_np.mean(axis=0)
    centered = observations_np - sample_mean
    sample_cov = centered.T @ centered / observations_np.shape[0]

    np.testing.assert_allclose(np.asarray(mu_hat), sample_mean, atol=1e-2)
    np.testing.assert_allclose(np.asarray(sigma_hat), sample_cov, atol=5e-2)

    np.testing.assert_allclose(np.asarray(mu_hat), np.asarray(mu_true), atol=2e-1)
    np.testing.assert_allclose(np.asarray(sigma_hat), np.asarray(sigma_true), atol=2e-1)

    eigenvalues = np.linalg.eigvalsh(np.asarray(sigma_hat))
    assert np.all(eigenvalues >= -1e-8)

    residual_norm = float(jnp.linalg.norm(restrictions["product"].g_bar(result.theta)))
    assert residual_norm < 1.0
