"""v1 vs v2 parity: point estimate theta_hat under clustering and overidentification.

The v2 omega_hat-on-DGP migration (PR #49) pinned byte-parity for
``omega_hat`` itself; this file lifts that pin to the full
``GMM.estimate()`` pipeline.  Three scenarios:

1. iid + just-identified -- sanity baseline (no clusters, no extra
   weighting).
2. clustered + just-identified -- exercises ``ClusteredSampling`` on
   the v2 side vs ``with_clusters`` on the v1 side.  Just-identified
   so the result is independent of the weighting matrix.
3. clustered + overidentified -- two moments, one parameter, so the
   weighting matrix matters; exercises Omega-hat through the cluster
   sandwich on both paths.

Tolerance is 1e-8 on ``theta_array``, matching the existing
``test_v1_and_v2_point_estimates_agree_on_iid_data`` baseline.
"""

from __future__ import annotations

import warnings

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_panel(seed: int = 2030):
    """3-dim panel with cluster structure.  Mirrors test_omega_hat_delegates."""

    rng = np.random.default_rng(seed)
    n_clusters = 20
    rows_per_cluster = 3
    n = n_clusters * rows_per_cluster
    k = 3
    cluster_offsets = 0.6 * rng.standard_normal(size=(n_clusters, k))
    within = 0.4 * rng.standard_normal(size=(n, k))
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    obs = cluster_offsets[cluster_ids] + within
    return jnp.asarray(obs), cluster_ids


def _make_overid_sample(seed: int = 7):
    """1-d Normal(mu_true, 1) sample, returned as (N, 1) data."""

    rng = np.random.default_rng(seed)
    mu_true = 0.7
    n = 200
    x = rng.standard_normal(size=n) + mu_true
    return jnp.asarray(x.reshape(-1, 1))


def _location_g(theta, data):
    """``g(theta, X) = X - theta`` for theta of shape (k,) and X of (N, k)."""

    return data - theta[None, :]


def _normal_two_moment_g(theta, data):
    """For X ~ Normal(theta[0], 1): mean + variance moments.

    Returns shape (N, 2).  Just-identification fails (2 moments, 1
    param), so the weighting matrix is exercised.
    """

    eps = data[:, 0] - theta[0]
    return jnp.stack([eps, eps**2 - 1.0], axis=-1)


def _M3():
    return Manifold.from_pymanopt(Euclidean(3))


def _M1():
    return Manifold.from_pymanopt(Euclidean(1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_iid_just_identified_theta_matches() -> None:
    """Baseline: iid, just-identified, default weighting."""

    X, _ = _make_panel()
    M = _M3()
    theta_0 = jnp.zeros(3)

    r1 = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    gmm_v1 = GMM(r1, initial_point=theta_0)
    result_v1 = gmm_v1.estimate()

    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm_v2 = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=theta_0,
    )
    result_v2 = gmm_v2.estimate()

    np.testing.assert_allclose(
        np.asarray(result_v1.theta_array, dtype=float),
        np.asarray(result_v2.theta_array, dtype=float),
        atol=1e-8,
    )


def test_clustered_just_identified_theta_matches() -> None:
    """Clustered v1 (with_clusters) vs v2 (ClusteredSampling), just-identified.

    With identity weighting and just-identification, theta_hat is the
    sample mean regardless of cluster structure -- but the v1/v2 paths
    *route* the cluster information differently, so this still
    exercises the cluster-passing plumbing on both sides.
    """

    X, cluster_ids = _make_panel()
    M = _M3()
    theta_0 = jnp.zeros(3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r1 = MomentRestriction(
            g=_location_g, data=X, manifold=M, backend="jax"
        ).with_clusters(cluster_ids)
    gmm_v1 = GMM(r1, initial_point=theta_0)
    result_v1 = gmm_v1.estimate()

    dgp = dp.EmpiricalDGP(
        observation=X,
        sampling=dp.ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )
    gmm_v2 = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=theta_0,
    )
    result_v2 = gmm_v2.estimate()

    np.testing.assert_allclose(
        np.asarray(result_v1.theta_array, dtype=float),
        np.asarray(result_v2.theta_array, dtype=float),
        atol=1e-8,
    )


def test_clustered_overidentified_two_step_theta_matches() -> None:
    """Overidentified + clustered + two-step: weighting matters.

    Two moments (mean, variance) on a 1-d Normal(mu_true, 1) sample.
    Two-step uses identity then ``Omega_hat^{-1}`` -- so theta_hat
    depends on the cluster-aware Omega-hat on both v1 and v2.
    """

    X = _make_overid_sample()
    # Synthetic cluster structure (5 clusters of 40 obs each).
    cluster_ids = np.repeat(np.arange(5), 40)
    M = _M1()
    theta_0 = jnp.array([0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r1 = MomentRestriction(
            g=_normal_two_moment_g, data=X, manifold=M, backend="jax"
        ).with_clusters(cluster_ids)
    gmm_v1 = GMM(r1, initial_point=theta_0)
    result_v1 = gmm_v1.estimate(two_step=True)

    dgp = dp.EmpiricalDGP(
        observation=X,
        sampling=dp.ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )
    gmm_v2 = GMM(
        moment_func=_normal_two_moment_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=theta_0,
    )
    result_v2 = gmm_v2.estimate(two_step=True)

    np.testing.assert_allclose(
        np.asarray(result_v1.theta_array, dtype=float),
        np.asarray(result_v2.theta_array, dtype=float),
        atol=1e-7,  # two-step amplifies floating-point drift slightly
    )
