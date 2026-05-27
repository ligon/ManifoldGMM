"""v1 vs v2 parity: ``MomentWildBootstrap`` on a v1- vs v2-constructed GMMResult.

``MomentWildBootstrap`` is the residual-based wild bootstrap; it
consumes a ``GMMResult`` and refits on Rademacher-reweighted moment
errors at the fitted theta.  Because it reads ``result.theta``,
``result.weighting``, and ``result.restriction`` -- and because the
v2 GMM synthesizes a restriction internally -- a regression in the
v2 synthesis path would corrupt bootstrap replicates relative to v1.

These tests pin replicate-by-replicate byte parity at fixed
``base_seed``: same Rademacher weights drawn, same refit theta in
each replicate, on iid and clustered designs.
"""

from __future__ import annotations

import warnings

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction, MomentWildBootstrap
from pymanopt.manifolds import Euclidean


def _make_panel(seed: int = 2030):
    rng = np.random.default_rng(seed)
    n_clusters = 12
    rows_per_cluster = 3
    n = n_clusters * rows_per_cluster
    k = 3
    cluster_offsets = 0.6 * rng.standard_normal(size=(n_clusters, k))
    within = 0.4 * rng.standard_normal(size=(n, k))
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    obs = cluster_offsets[cluster_ids] + within
    return jnp.asarray(obs), cluster_ids


def _location_g(theta, data):
    return data - theta[None, :]


def _M3():
    return Manifold.from_pymanopt(Euclidean(3))


def _v1_result(X, cluster_ids=None):
    M = _M3()
    r1 = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    if cluster_ids is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r1 = r1.with_clusters(cluster_ids)
    gmm = GMM(r1, initial_point=jnp.zeros(3))
    return gmm.estimate()


def _v2_result(X, cluster_ids=None):
    M = _M3()
    # Note: explicitly passing ``sampling=None`` bypasses the dataclass
    # default_factory and crashes inside ``omega_hat``; pass the kwarg
    # only when we actually want a non-default sampling design.
    if cluster_ids is not None:
        dgp = dp.EmpiricalDGP(
            observation=X,
            sampling=dp.ClusteredSampling(cluster_ids=cluster_ids),
            seed=0,
        )
    else:
        dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=jnp.zeros(3),
    )
    return gmm.estimate()


def _replicate_thetas(boot) -> np.ndarray:
    """Run all bootstrap tasks serially and return the (B, k) theta matrix.

    ``MomentWildBootstrap`` tasks return a ``BootstrapResult`` whose
    refit is ``theta_star`` (a ``ManifoldPoint``), not ``theta_array``.
    """

    rows = []
    for task in boot.tasks():
        r = task.run()
        rows.append(np.asarray(r.theta_star.value, dtype=float))
    return np.vstack(rows)


def test_wild_bootstrap_iid_byte_parity_v1_v2_result() -> None:
    """MomentWildBootstrap on v1 vs v2 GMMResult: identical refits, iid."""

    X, _ = _make_panel()
    res_v1 = _v1_result(X)
    res_v2 = _v2_result(X)

    # Precondition: same theta_hat going in.  Otherwise the bootstrap
    # parity claim would be meaningless.
    np.testing.assert_allclose(
        np.asarray(res_v1.theta_array, dtype=float),
        np.asarray(res_v2.theta_array, dtype=float),
        atol=1e-8,
    )

    boot_v1 = MomentWildBootstrap(res_v1, n_bootstrap=4, base_seed=11)
    boot_v2 = MomentWildBootstrap(res_v2, n_bootstrap=4, base_seed=11)
    thetas_v1 = _replicate_thetas(boot_v1)
    thetas_v2 = _replicate_thetas(boot_v2)

    np.testing.assert_allclose(thetas_v1, thetas_v2, atol=1e-8)


def test_wild_bootstrap_clustered_byte_parity_v1_v2_result() -> None:
    """Same as above but with clustered sampling on both sides."""

    X, cluster_ids = _make_panel()
    res_v1 = _v1_result(X, cluster_ids=cluster_ids)
    res_v2 = _v2_result(X, cluster_ids=cluster_ids)

    np.testing.assert_allclose(
        np.asarray(res_v1.theta_array, dtype=float),
        np.asarray(res_v2.theta_array, dtype=float),
        atol=1e-8,
    )

    boot_v1 = MomentWildBootstrap(res_v1, n_bootstrap=4, base_seed=23)
    boot_v2 = MomentWildBootstrap(res_v2, n_bootstrap=4, base_seed=23)
    thetas_v1 = _replicate_thetas(boot_v1)
    thetas_v2 = _replicate_thetas(boot_v2)

    np.testing.assert_allclose(thetas_v1, thetas_v2, atol=1e-8)
