"""v1 vs v2 parity: J / K / Wald inference statistics on shared theta_hat.

These three helpers live on ``GMMResult`` and are not aware of which
construction path produced the result.  If v2 routes anything
differently through ``omega_hat`` or the Jacobian, the inference
numbers will diverge.  These tests pin that they don't.

Also includes a parity test for ``k_statistic_bootstrap_for_result``:
on a clustered v2 GMMResult, the bootstrap must source cluster ids
from ``restriction._dgp.sampling.cluster_ids`` (the v2 surface), not
from ``restriction.clusters`` (the v1 surface that v2 callers don't
populate).  Without this, the cluster-wild Rademacher path silently
degrades to per-observation iid signs.

Fixture: overidentified 1-d Normal(theta, 1) with two moments (mean
and centered second moment).  Overidentification is required for the
J-statistic to be non-degenerate.
"""

from __future__ import annotations

import warnings

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics.bootstrap import k_statistic_bootstrap_for_result
from pymanopt.manifolds import Euclidean


def _normal_two_moment_g(theta, data):
    """X ~ Normal(theta[0], 1): mean and variance moments.  Shape (N, 2)."""

    eps = data[:, 0] - theta[0]
    return jnp.stack([eps, eps**2 - 1.0], axis=-1)


def _make_overid_sample(seed: int = 11):
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.standard_normal(size=n) + 0.5
    return jnp.asarray(x.reshape(-1, 1))


def _M1():
    return Manifold.from_pymanopt(Euclidean(1))


def _build_results():
    X = _make_overid_sample()
    M = _M1()
    theta_0_init = jnp.array([0.0])

    r1 = MomentRestriction(g=_normal_two_moment_g, data=X, manifold=M, backend="jax")
    gmm_v1 = GMM(r1, initial_point=theta_0_init)
    res_v1 = gmm_v1.estimate(two_step=True)

    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm_v2 = GMM(
        moment_func=_normal_two_moment_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=theta_0_init,
    )
    res_v2 = gmm_v2.estimate(two_step=True)

    # Sanity: shared theta_hat is the premise for the inference tests.
    np.testing.assert_allclose(
        np.asarray(res_v1.theta_array, dtype=float),
        np.asarray(res_v2.theta_array, dtype=float),
        atol=1e-7,
    )
    return res_v1, res_v2


def test_j_statistic_v1_v2_match() -> None:
    """``criterion_value`` (the GMM J target) agrees v1 vs v2."""

    res_v1, res_v2 = _build_results()
    # criterion_value is N * g_bar' W g_bar; both should compute it the
    # same way regardless of construction route.
    assert np.isclose(
        float(res_v1.criterion_value),
        float(res_v2.criterion_value),
        atol=1e-8,
    ), f"v1={res_v1.criterion_value}, v2={res_v2.criterion_value}"


def test_k_statistic_v1_v2_match() -> None:
    """``result.k_statistic(theta_0)`` returns matching K, S, J numbers."""

    res_v1, res_v2 = _build_results()
    theta_0 = jnp.array([0.5])  # the true mean
    k_v1 = res_v1.k_statistic(theta_0=theta_0)
    k_v2 = res_v2.k_statistic(theta_0=theta_0)

    assert np.isclose(float(k_v1.K), float(k_v2.K), atol=1e-7)
    assert np.isclose(float(k_v1.S), float(k_v2.S), atol=1e-7)
    assert np.isclose(float(k_v1.J), float(k_v2.J), atol=1e-7)


def test_wald_test_v1_v2_match() -> None:
    """``result.wald_test(constraint)`` returns matching statistic + p_value."""

    res_v1, res_v2 = _build_results()

    # Constraint h(theta) = theta[0] - 0.5.  The wald_test contract
    # passes a ``ManifoldPoint``; access the ambient array via
    # ``theta_point.value`` (see tests/econometrics/test_wald_test.py).
    def _constraint(theta_point):
        return theta_point.value[0] - 0.5

    w_v1 = res_v1.wald_test(_constraint, q=1)
    w_v2 = res_v2.wald_test(_constraint, q=1)

    assert np.isclose(float(w_v1.statistic), float(w_v2.statistic), atol=1e-7)
    assert np.isclose(float(w_v1.p_value), float(w_v2.p_value), atol=1e-7)
    assert w_v1.degrees_of_freedom == w_v2.degrees_of_freedom


def test_k_statistic_bootstrap_clustered_v1_v2_match() -> None:
    """``k_statistic_bootstrap_for_result`` on clustered v1 vs v2 GMMResult.

    Same root cause as the wild bootstrap parity: this helper resolves
    cluster ids and the v1 fallback (``restriction.clusters``) is None
    on v2 results.  The fix in ``bootstrap.py`` adds a v2 fallback to
    ``restriction._dgp.sampling.cluster_ids``; this test pins it.

    Byte-for-byte equality of the bootstrap arrays at fixed RNG seed.
    """

    X = _make_overid_sample()
    M = _M1()
    theta_0_init = jnp.array([0.0])
    # Synthetic cluster structure: 6 clusters of 50 obs each.
    cluster_ids = np.repeat(np.arange(6), 50)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r1 = MomentRestriction(
            g=_normal_two_moment_g, data=X, manifold=M, backend="jax"
        ).with_clusters(cluster_ids)
    res_v1 = GMM(r1, initial_point=theta_0_init).estimate(two_step=True)

    dgp = dp.EmpiricalDGP(
        observation=X,
        sampling=dp.ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )
    res_v2 = GMM(
        moment_func=_normal_two_moment_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=theta_0_init,
    ).estimate(two_step=True)

    theta_0 = jnp.array([0.5])
    bs_v1 = k_statistic_bootstrap_for_result(
        res_v1, theta_0=theta_0, n_replicates=20, rng=37
    )
    bs_v2 = k_statistic_bootstrap_for_result(
        res_v2, theta_0=theta_0, n_replicates=20, rng=37
    )

    # Precondition: both bootstraps must have detected clustering.  The
    # v1 path reports source ``"restriction"``; the v2 path (fixed by
    # this PR) reports source ``"dgp_sampling"``.
    assert bs_v1.method == "cluster_wild", bs_v1.method
    assert bs_v2.method == "cluster_wild", bs_v2.method
    info_v1 = bs_v1.cluster_info
    info_v2 = bs_v2.cluster_info
    assert info_v1 is not None and info_v2 is not None
    assert info_v1["source"] == "restriction"
    assert info_v2["source"] == "dgp_sampling"
    assert info_v1["num_clusters"] == info_v2["num_clusters"]

    # Byte parity on the bootstrap arrays.  Same seed, same cluster
    # signs, same projection -> identical replicates.
    np.testing.assert_allclose(bs_v1.K_bootstrap, bs_v2.K_bootstrap, atol=1e-10)
    np.testing.assert_allclose(bs_v1.S_bootstrap, bs_v2.S_bootstrap, atol=1e-10)
    np.testing.assert_allclose(bs_v1.J_bootstrap, bs_v2.J_bootstrap, atol=1e-10)
