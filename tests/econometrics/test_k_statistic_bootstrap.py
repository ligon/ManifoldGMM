"""Tests for the bootstrap K-statistic (#25).

Covers the six acceptance criteria filed against #25:

1. Returns finite ``K_observed`` and arrays of the right length.
2. On a well-conditioned synthetic fixture, bootstrap and asymptotic
   p-values agree to within Monte Carlo error.
3. On an ill-conditioned fixture where ``ridge_inverse`` fires its
   cap, the bootstrap p-value is well-defined.
4. Penalty independence: same numerical p-values on a penalised
   ``GMMResult`` vs unpenalised ``GMMResult`` fit on the same data.
5. Cluster-aware bootstrap with cluster size 1 reproduces the
   iid variant (degenerate sanity check).
6. Pickle round-trip on ``KStatBootstrapResult``.
"""

from __future__ import annotations

import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics.bootstrap import (
    KStatBootstrapResult,
    rademacher_signs,
)
from manifoldgmm.econometrics.gmm import FixedWeighting
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _well_conditioned_fit():
    """``g_i(theta) = y_i - theta``, N=100, Euclidean(1).  cond(D'WD) ~ O(1)."""

    rng = np.random.default_rng(20260523)
    data = jnp.asarray(rng.standard_normal(size=100).astype(np.float64))

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta"],
    )
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
    )
    return gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )


# ---------------------------------------------------------------------------
# rademacher_signs helper
# ---------------------------------------------------------------------------
def test_rademacher_signs_returns_pm_one() -> None:
    """``rademacher_signs(n, rng)`` returns values in ``{-1, +1}``."""

    rng = np.random.default_rng(1)
    signs = rademacher_signs(1000, rng)
    assert signs.shape == (1000,)
    assert set(np.unique(signs).tolist()).issubset({-1.0, 1.0})
    # Empirical mean ~ 0 for n=1000 with high probability
    assert abs(signs.mean()) < 0.1


# ---------------------------------------------------------------------------
# Acceptance criterion 1: returns finite, right shape
# ---------------------------------------------------------------------------
def test_bootstrap_returns_correct_shape_and_finite() -> None:
    """Bootstrap returns finite ``K_observed`` and arrays of length ``n_replicates``."""

    result = _well_conditioned_fit()

    boot = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]), n_replicates=50, rng=42
    )

    assert isinstance(boot, KStatBootstrapResult)
    assert np.isfinite(boot.K_observed)
    assert np.isfinite(boot.J_observed)
    assert boot.S_observed >= 0.0
    assert boot.K_bootstrap.shape == (50,)
    assert boot.J_bootstrap.shape == (50,)
    assert boot.S_bootstrap.shape == (50,)
    assert np.all(np.isfinite(boot.K_bootstrap))
    assert np.all(np.isfinite(boot.J_bootstrap))
    assert np.all(boot.S_bootstrap >= 0.0)
    assert boot.df_K == 1
    assert boot.df_S == 0  # ell - p == 1 - 1 == 0
    assert 0.0 <= boot.p_K_bootstrap <= 1.0
    assert boot.n_replicates == 50
    assert boot.method == "iid"
    assert boot.cluster_info is None


# ---------------------------------------------------------------------------
# Acceptance criterion 2: bootstrap vs asymptotic agreement on
# a well-conditioned fixture
# ---------------------------------------------------------------------------
def test_bootstrap_agrees_with_asymptotic_on_well_conditioned_fixture() -> None:
    """``p_K_bootstrap`` and ``p_K_asymptotic`` agree within MC error on iid data.

    With N=100 standard-normal data and a scalar mean model, the
    K-statistic at ``theta_0 = 0.5`` (the truth here is ~0; ``theta_0``
    is meaningfully off the data mean so K is non-trivial)
    has a well-known chi^2(1) reference asymptotically.  With 500
    bootstrap replicates the percentile p-value should land within
    ~0.05 of the asymptotic p-value.
    """

    result = _well_conditioned_fit()
    boot = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]), n_replicates=500, rng=20260523
    )

    # The two p-values should both be in (0, 1); their distance bounds
    # Monte Carlo error of the bootstrap.
    assert 0.0 < boot.p_K_asymptotic < 1.0
    assert 0.0 < boot.p_K_bootstrap < 1.0
    # On well-conditioned data the gap should be small.  Allow a
    # generous 0.1 tolerance to keep the test robust to the rng seed.
    assert abs(boot.p_K_bootstrap - boot.p_K_asymptotic) < 0.1, (
        f"Bootstrap p_K={boot.p_K_bootstrap:.3f} differs from "
        f"asymptotic p_K={boot.p_K_asymptotic:.3f} by more than 0.1; "
        "either the bootstrap is misimplemented or the fixture is "
        "ill-conditioned."
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 3: ill-conditioned regime returns well-defined p-value
# ---------------------------------------------------------------------------
def test_bootstrap_well_defined_when_ridge_inverse_fires_cap() -> None:
    """Under-id design (rank-deficient D): bootstrap p-value still finite.

    With ``g_i = y_i - (theta1 + theta2)`` and a 2-D Euclidean
    parameter, ``D'WD`` is rank 1.  ``ridge_inverse`` will fire its
    cap on the second inversion (``D' Omega^{-1} D``) with
    ``ridge_condition`` smaller than the matrix's effective conditioning.
    The bootstrap should still produce finite K, S statistics under
    the warning regime.
    """

    rng = np.random.default_rng(20260523)
    data = jnp.asarray(rng.standard_normal(size=80).astype(np.float64))

    def gi_jax(theta, observation):
        return observation - (theta[0] + theta[1])

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(2))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta1", "theta2"],
    )
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0, 0.0]),
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    # ``ridge_condition`` deliberately small (1e3) so ridge_inverse's
    # cap fires on the rank-1 D'WD; the bootstrap should silently
    # tolerate that and return finite stats.
    boot = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5, 0.5]),
        n_replicates=100,
        rng=42,
        ridge_condition=1e3,
    )

    assert np.isfinite(boot.K_observed)
    assert np.all(np.isfinite(boot.K_bootstrap))
    assert 0.0 <= boot.p_K_bootstrap <= 1.0


# ---------------------------------------------------------------------------
# Acceptance criterion 4: penalty independence
# ---------------------------------------------------------------------------
def test_bootstrap_is_penalty_invariant() -> None:
    """Same restriction, same theta_0: bootstrap matches across penalty status.

    With the same RNG seed, the bootstrap on a penalised
    ``GMMResult`` and an unpenalised one (fit on the same data with
    the same initial point) should produce numerically-identical K,
    S, J distributions because the bootstrap formula doesn't
    reference the optimiser at all.
    """

    rng = np.random.default_rng(20260523)
    data = jnp.asarray(rng.standard_normal(size=50).astype(np.float64))

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta"],
    )

    gmm_un = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
    )
    gmm_pen = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=lambda theta: 0.5 * theta[0] ** 2,
    )

    res_un = gmm_un.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )
    res_pen = gmm_pen.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    boot_un = res_un.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]), n_replicates=100, rng=7
    )
    boot_pen = res_pen.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]), n_replicates=100, rng=7
    )

    # Observed K, S, J at the same theta_0: bit-identical.
    assert np.isclose(boot_un.K_observed, boot_pen.K_observed, atol=1e-10)
    assert np.isclose(boot_un.J_observed, boot_pen.J_observed, atol=1e-10)
    assert np.isclose(boot_un.S_observed, boot_pen.S_observed, atol=1e-10)
    # Bootstrap distributions: same seed → same signs → same K* per replicate.
    assert np.allclose(boot_un.K_bootstrap, boot_pen.K_bootstrap, atol=1e-10)
    assert np.allclose(boot_un.J_bootstrap, boot_pen.J_bootstrap, atol=1e-10)
    # p-values match too.
    assert boot_un.p_K_bootstrap == boot_pen.p_K_bootstrap
    assert boot_un.p_K_asymptotic == boot_pen.p_K_asymptotic


# ---------------------------------------------------------------------------
# Acceptance criterion 5: cluster-aware degenerate case
# ---------------------------------------------------------------------------
def test_cluster_aware_with_cluster_size_one_matches_iid() -> None:
    """``cluster_index = np.arange(N)`` (singleton clusters) == iid bootstrap.

    With each observation in its own cluster, the cluster-wild
    bootstrap draws an independent sign per observation -- exactly
    the same scheme as the iid bootstrap.  Same RNG seed should give
    bit-identical bootstrap distributions.
    """

    result = _well_conditioned_fit()
    N = 100

    boot_iid = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]), n_replicates=100, rng=11
    )
    boot_singleton = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]),
        n_replicates=100,
        cluster_index=np.arange(N),
        rng=11,
    )

    # Same RNG seed + same effective scheme ⇒ identical signs ⇒
    # identical bootstrap distributions.  Note that the *method*
    # label differs ("iid" vs "cluster_wild"), but the numerics match.
    assert boot_iid.method == "iid"
    assert boot_singleton.method == "cluster_wild"
    assert boot_singleton.cluster_info is not None
    assert boot_singleton.cluster_info["num_clusters"] == N
    assert np.allclose(boot_iid.K_bootstrap, boot_singleton.K_bootstrap, atol=1e-10)


# ---------------------------------------------------------------------------
# Acceptance criterion 6: pickle round-trip
# ---------------------------------------------------------------------------
def test_kstat_bootstrap_result_pickles_roundtrip(tmp_path) -> None:
    """``KStatBootstrapResult`` survives a pickle round-trip."""

    result = _well_conditioned_fit()
    boot = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]), n_replicates=50, rng=42
    )

    pickle_path = tmp_path / "kstat_boot.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(boot, f)
    with pickle_path.open("rb") as f:
        restored = pickle.load(f)

    assert isinstance(restored, KStatBootstrapResult)
    assert restored.K_observed == boot.K_observed
    assert restored.J_observed == boot.J_observed
    assert np.allclose(restored.K_bootstrap, boot.K_bootstrap)
    assert np.allclose(restored.J_bootstrap, boot.J_bootstrap)
    assert restored.df_K == boot.df_K
    assert restored.df_S == boot.df_S
    assert restored.p_K_bootstrap == boot.p_K_bootstrap
    assert restored.p_K_asymptotic == boot.p_K_asymptotic
    assert restored.n_replicates == boot.n_replicates
    assert restored.method == boot.method


# ---------------------------------------------------------------------------
# Additional coverage: input validation
# ---------------------------------------------------------------------------
def test_bootstrap_requires_explicit_theta_0() -> None:
    """``theta_0`` is required; calling without it raises ``TypeError``."""

    result = _well_conditioned_fit()
    with pytest.raises(TypeError):
        result.k_statistic_bootstrap(n_replicates=10)


def test_bootstrap_cluster_index_length_must_match_N() -> None:
    """Wrong-length ``cluster_index`` raises ``ValueError`` with a clear message."""

    result = _well_conditioned_fit()
    bad_clusters = np.arange(7)  # N is 100
    with pytest.raises(ValueError, match="cluster_index has 7"):
        result.k_statistic_bootstrap(
            theta_0=jnp.array([0.5]),
            n_replicates=10,
            cluster_index=bad_clusters,
        )


def test_cluster_aware_two_cluster_design_has_correct_info() -> None:
    """Two-cluster fixture surfaces ``cluster_info`` correctly."""

    result = _well_conditioned_fit()
    N = 100
    # First half in cluster 0, second half in cluster 1.
    clusters = np.concatenate([np.zeros(N // 2), np.ones(N - N // 2)])

    boot = result.k_statistic_bootstrap(
        theta_0=jnp.array([0.5]),
        n_replicates=20,
        cluster_index=clusters,
        rng=0,
    )

    assert boot.method == "cluster_wild"
    assert boot.cluster_info is not None
    assert boot.cluster_info["num_clusters"] == 2
    assert boot.cluster_info["source"] == "argument"
