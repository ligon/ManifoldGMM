"""Phase B-minimal: MomentRestriction.omega_hat delegates to DGP when attached.

Verifies that when a v2-constructed GMM attaches a DGP to its
synthesized MomentRestriction, ``restriction.omega_hat(theta)``
delegates to
``dgp.sample_distribution.moment_covariance(theta, gi_jax)``
instead of computing the v1 formula on
``self._clusters`` / ``self._weights``.

The DGP-side analytic in ``dgp_protocol/sampling.py`` was ported
byte-parity from this restriction's v1 formula (verified at 1e-12 by
``DGP_Protocol/tests/test_moment_covariance_estimator.py``), so the
delegation must produce byte-identical results to the v1 path on
shared inputs.  This file pins that property end-to-end through the
ManifoldGMM consumer.
"""

from __future__ import annotations

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean


def _location_g(theta, data):
    """``g(theta, X) = X - theta``: just-identified location moment."""

    return data - theta[None, :]


# ---------------------------------------------------------------------------
# Shared fixture: a small (N=60, k=3) panel with cluster structure.
# ---------------------------------------------------------------------------
def _make_panel(seed: int = 2030):
    rng = np.random.default_rng(seed)
    n_clusters = 20
    rows_per_cluster = 3
    n = n_clusters * rows_per_cluster
    k = 3
    cluster_offsets = 0.6 * rng.standard_normal(size=(n_clusters, k))
    within = 0.4 * rng.standard_normal(size=(n, k))
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    obs = cluster_offsets[cluster_ids] + within
    return obs, cluster_ids


def _build_manifold(k: int = 3) -> Manifold:
    return Manifold.from_pymanopt(Euclidean(k))


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_omega_hat_iid_v2_matches_v1() -> None:
    """v2 (DGP-attached) omega_hat = v1 (clusters=None) omega_hat for iid data."""

    obs, _ = _make_panel()
    theta = jnp.asarray(obs.mean(axis=0))

    # v1: classic restriction with data + no clusters.
    v1_restriction = MomentRestriction(
        g=_location_g,
        data=jnp.asarray(obs),
        manifold=_build_manifold(),
        backend="jax",
    )
    omega_v1 = np.asarray(v1_restriction.omega_hat(theta))

    # v2: GMM constructor synthesizes a restriction + attaches the DGP.
    iid_dgp = dp.EmpiricalDGP(observation=jnp.asarray(obs))
    gmm = GMM(
        moment_func=_location_g,
        dgp=iid_dgp,
        manifold=_build_manifold(),
        backend="jax",
    )
    omega_v2 = np.asarray(gmm._restriction.omega_hat(theta))

    # Byte-parity (1e-12 tolerance, same as DGP_Protocol's parity tests).
    np.testing.assert_allclose(omega_v2, omega_v1, atol=1e-12)


def test_omega_hat_clustered_v2_matches_v1() -> None:
    """v2 (DGP-attached, ClusteredSampling) omega_hat = v1 (clusters=ids) omega_hat."""

    obs, cluster_ids = _make_panel()
    theta = jnp.asarray(obs.mean(axis=0))

    # v1: with_clusters on the restriction.
    v1_restriction = MomentRestriction(
        g=_location_g,
        data=jnp.asarray(obs),
        manifold=_build_manifold(),
        backend="jax",
    ).with_clusters(cluster_ids)
    omega_v1 = np.asarray(v1_restriction.omega_hat(theta))

    # v2: clusters live on the DGP's SamplingDesign.
    cdgp = dp.EmpiricalDGP(
        observation=jnp.asarray(obs),
        sampling=dp.ClusteredSampling(cluster_ids=cluster_ids),
    )
    gmm = GMM(
        moment_func=_location_g,
        dgp=cdgp,
        manifold=_build_manifold(),
        backend="jax",
    )
    omega_v2 = np.asarray(gmm._restriction.omega_hat(theta))

    np.testing.assert_allclose(omega_v2, omega_v1, atol=1e-12)


def test_v1_path_unchanged_no_dgp_attached() -> None:
    """v1 callers (no DGP) hit the original v1 formula, unchanged."""

    obs, cluster_ids = _make_panel()
    theta = jnp.asarray(obs.mean(axis=0))

    v1_restriction = MomentRestriction(
        g=_location_g,
        data=jnp.asarray(obs),
        manifold=_build_manifold(),
        backend="jax",
    ).with_clusters(cluster_ids)

    # No _dgp attached -> falls through to the existing v1 formula.
    assert not hasattr(v1_restriction, "_dgp") or v1_restriction._dgp is None
    omega = np.asarray(v1_restriction.omega_hat(theta))

    # Same shape, finite, symmetric.
    assert omega.shape == (3, 3)
    np.testing.assert_allclose(omega, omega.T, atol=1e-12)
    assert np.linalg.eigvalsh(omega).min() >= -1e-10


def test_v2_centered_kwarg_propagates() -> None:
    """``centered=False`` flows through the delegation to the DGP."""

    obs, _ = _make_panel()
    theta = jnp.asarray(obs.mean(axis=0))

    v1_restriction = MomentRestriction(
        g=_location_g,
        data=jnp.asarray(obs),
        manifold=_build_manifold(),
        backend="jax",
    )
    omega_v1_uncentered = np.asarray(v1_restriction.omega_hat(theta, centered=False))

    iid_dgp = dp.EmpiricalDGP(observation=jnp.asarray(obs))
    gmm = GMM(
        moment_func=_location_g,
        dgp=iid_dgp,
        manifold=_build_manifold(),
        backend="jax",
    )
    omega_v2_uncentered = np.asarray(gmm._restriction.omega_hat(theta, centered=False))

    np.testing.assert_allclose(omega_v2_uncentered, omega_v1_uncentered, atol=1e-12)
