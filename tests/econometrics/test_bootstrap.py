"""Tests for the moment wild bootstrap module.

Covers weight moment properties, weighted mean correctness, task serialization,
single-task execution, and end-to-end integration on Euclidean(1).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, ManifoldPoint, MomentRestriction
from manifoldgmm.econometrics.bootstrap import (
    BootstrapResult,
    BootstrapTask,
    MomentWildBootstrap,
    exponential_weights,
    geodesic_mahalanobis_distance,
    mammen_weights,
    rademacher_weights,
)
from pymanopt.manifolds import Euclidean as PymanoptEuclidean

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _simple_restriction_and_result(
    data: Any | None = None,
) -> tuple[MomentRestriction, Any, Any]:
    """Build a simple Euclidean(1) mean-estimation problem and solve it."""

    if data is None:
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()
    return restriction, result, float(jnp.mean(data))


# -----------------------------------------------------------------------
# 1. Weight moment tests
# -----------------------------------------------------------------------


class TestWeightMoments:
    """Verify E[w] ~ 1 and Var[w] ~ 1 for each weight generator."""

    N_DRAWS = 100_000
    ATOL = 0.02  # 2% tolerance for Monte Carlo

    def test_rademacher_mean_and_variance(self) -> None:
        rng = np.random.default_rng(42)
        w = rademacher_weights(self.N_DRAWS, rng)
        np.testing.assert_allclose(w.mean(), 1.0, atol=self.ATOL)
        np.testing.assert_allclose(w.var(), 1.0, atol=self.ATOL)
        # Rademacher values are {0, 2}
        assert set(np.unique(w)) == {0.0, 2.0}

    def test_mammen_mean_and_variance(self) -> None:
        rng = np.random.default_rng(42)
        w = mammen_weights(self.N_DRAWS, rng)
        np.testing.assert_allclose(w.mean(), 1.0, atol=self.ATOL)
        np.testing.assert_allclose(w.var(), 1.0, atol=self.ATOL)
        # Mammen skewness E[(w-1)^3] = 1
        np.testing.assert_allclose(((w - 1.0) ** 3).mean(), 1.0, atol=0.05)

    def test_exponential_mean_and_variance(self) -> None:
        rng = np.random.default_rng(42)
        w = exponential_weights(self.N_DRAWS, rng)
        np.testing.assert_allclose(w.mean(), 1.0, atol=self.ATOL)
        np.testing.assert_allclose(w.var(), 1.0, atol=self.ATOL)
        # All positive
        assert (w > 0).all()


# -----------------------------------------------------------------------
# 2. Weighted mean tests
# -----------------------------------------------------------------------


class TestWeightedMean:
    """Verify restriction.with_weights(w).g_bar(theta) matches hand computation."""

    def test_weighted_mean_matches_manual(self) -> None:
        data = jnp.array([1.0, 2.0, 3.0])

        def gi_jax(theta: Any, observation: Any) -> Any:
            return observation - theta[0]

        manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
        restriction = MomentRestriction(
            gi_jax=gi_jax,
            data=data,
            manifold=manifold,
            backend="jax",
        )

        theta = jnp.array([0.0])
        sqrt_N = np.sqrt(3.0)
        # Unweighted: mean(data - 0) = 2.0, g_bar = √N * mean
        unweighted = np.asarray(restriction.g_bar(theta)).ravel()
        np.testing.assert_allclose(unweighted, [sqrt_N * 2.0], atol=1e-10)

        # Weighted: w = [2, 0, 1], g_i = data - 0 = [1, 2, 3]
        # weighted mean = (2*1 + 0*2 + 1*3) / 3 = 5/3
        # g_bar = √N * weighted_mean
        weights = np.array([2.0, 0.0, 1.0])
        weighted_restriction = restriction.with_weights(weights)
        weighted_g_bar = np.asarray(weighted_restriction.g_bar(theta)).ravel()
        np.testing.assert_allclose(weighted_g_bar, [sqrt_N * 5.0 / 3.0], atol=1e-10)

    def test_with_weights_shares_data(self) -> None:
        """Shallow copy should share the dataset and manifold."""

        data = jnp.array([1.0, 2.0])

        def gi_jax(theta: Any, observation: Any) -> Any:
            return observation - theta[0]

        manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
        restriction = MomentRestriction(
            gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
        )

        clone = restriction.with_weights(np.array([1.0, 1.0]))
        assert clone.manifold is restriction.manifold
        assert clone._gi_map is restriction._gi_map

    def test_original_unaffected_by_with_weights(self) -> None:
        """with_weights should not mutate the original restriction."""

        data = jnp.array([1.0, 2.0, 3.0])

        def gi_jax(theta: Any, observation: Any) -> Any:
            return observation - theta[0]

        manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
        restriction = MomentRestriction(
            gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
        )

        restriction.with_weights(np.array([2.0, 0.0, 1.0]))
        assert restriction.weights is None


# -----------------------------------------------------------------------
# 3. Serialization round-trip
# -----------------------------------------------------------------------


class TestSerialization:
    """BootstrapTask.to_bytes() / from_bytes() round-trip."""

    def test_round_trip_preserves_fields(self) -> None:
        restriction, result, _ = _simple_restriction_and_result()
        W = np.eye(1)
        task = BootstrapTask(
            restriction=restriction,
            weighting_matrix=W,
            initial_point=result.theta_array,
            seed=42,
            weight_scheme="rademacher",
            optimizer_class=None,
            optimizer_kwargs={},
            task_id=7,
        )

        data = task.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = BootstrapTask.from_bytes(data)
        assert restored.task_id == 7
        assert restored.seed == 42
        assert restored.weight_scheme == "rademacher"


# -----------------------------------------------------------------------
# 4. Single task execution
# -----------------------------------------------------------------------


class TestSingleTaskExecution:
    """BootstrapTask.run() on Euclidean(1) returns a valid BootstrapResult."""

    def test_task_run_returns_result(self) -> None:
        restriction, result, true_mean = _simple_restriction_and_result()

        # Extract W from result
        theta_hat = result.theta_point
        weighting = result.weighting
        if hasattr(weighting, "matrix"):
            W = np.asarray(weighting.matrix(theta_hat), dtype=float)
        else:
            W = np.eye(1)

        task = BootstrapTask(
            restriction=restriction,
            weighting_matrix=W,
            initial_point=result.theta_array,
            seed=123,
            weight_scheme="rademacher",
            optimizer_class=None,
            optimizer_kwargs={},
            task_id=0,
        )

        br = task.run()
        assert isinstance(br, BootstrapResult)
        assert br.task_id == 0
        assert br.seed == 123
        assert isinstance(br.theta_star, ManifoldPoint)
        assert isinstance(br.criterion_value, float)
        assert isinstance(br.converged, bool)


# -----------------------------------------------------------------------
# 5. End-to-end integration
# -----------------------------------------------------------------------


class TestEndToEndIntegration:
    """MomentWildBootstrap on Euclidean(1) mean estimation."""

    def test_full_workflow(self) -> None:
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, result, true_mean = _simple_restriction_and_result(data)

        boot = MomentWildBootstrap(
            result,
            n_bootstrap=50,
            weight_scheme="rademacher",
            base_seed=42,
        )

        # tasks()
        task_list = boot.tasks()
        assert len(task_list) == 50
        assert all(isinstance(t, BootstrapTask) for t in task_list)
        # Seeds are sequential
        assert task_list[0].seed == 42
        assert task_list[49].seed == 91

        # run_sequential()
        results = boot.run_sequential()
        assert len(results) == 50
        assert all(isinstance(r, BootstrapResult) for r in results)

        # geodesic_distances()
        d2 = boot.geodesic_distances()
        assert d2.shape == (50,)
        assert (d2 >= 0).all()

        # critical_value()
        cv = boot.critical_value(alpha=0.05)
        assert isinstance(cv, float)
        assert cv >= 0

        # in_confidence_region: theta_hat should be inside
        assert boot.in_confidence_region(result.theta_point, alpha=0.05)

        # summary()
        s = boot.summary()
        assert s["n_collected"] == 50
        assert "distances_quantiles" in s

    def test_collect_appends_results(self) -> None:
        _, result, _ = _simple_restriction_and_result()

        boot = MomentWildBootstrap(result, n_bootstrap=5, base_seed=0)
        tasks = boot.tasks()
        # Run first two manually
        r1 = tasks[0].run()
        r2 = tasks[1].run()
        boot.collect([r1, r2])

        s = boot.summary()
        assert s["n_collected"] == 2

        # Collect more
        boot.collect([tasks[2].run()])
        s = boot.summary()
        assert s["n_collected"] == 3

    def test_exponential_weights_end_to_end(self) -> None:
        """Verify alternative weight scheme works through the full pipeline."""

        _, result, _ = _simple_restriction_and_result()

        boot = MomentWildBootstrap(
            result,
            n_bootstrap=20,
            weight_scheme="exponential",
            base_seed=7,
        )
        results = boot.run_sequential()
        assert len(results) == 20
        d2 = boot.geodesic_distances()
        assert d2.shape == (20,)
        assert (d2 >= 0).all()

    def test_invalid_weight_scheme_raises(self) -> None:
        _, result, _ = _simple_restriction_and_result()

        with pytest.raises(ValueError, match="Unknown weight_scheme"):
            MomentWildBootstrap(result, weight_scheme="unknown")


# -----------------------------------------------------------------------
# 7. Cluster-aware bootstrap: structural + API
# -----------------------------------------------------------------------


def _clustered_restriction_and_result(
    cluster_ids: np.ndarray,
    data: Any | None = None,
    *,
    attach_clusters: bool = False,
) -> tuple[Any, Any, np.ndarray]:
    """Build a clustered Euclidean(1) mean-estimation problem and solve it.

    ``cluster_ids`` is returned as the canonical cluster assignment.  When
    ``attach_clusters=True`` the restriction itself carries the assignment;
    otherwise the restriction is unclustered (the bootstrap will need an
    explicit ``clusters=`` argument).
    """

    if data is None:
        n = cluster_ids.shape[0]
        data = jnp.arange(1.0, n + 1.0, dtype=jnp.float64)

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    kwargs: dict[str, Any] = {
        "gi_jax": gi_jax,
        "data": data,
        "manifold": manifold,
        "backend": "jax",
    }
    if attach_clusters:
        kwargs["clusters"] = cluster_ids
    restriction = MomentRestriction(**kwargs)
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()
    return restriction, result, cluster_ids


class TestClusterStructural:
    """Structural and API behaviour of the cluster-aware bootstrap."""

    def test_clusters_size_one_byte_identical_to_iid(self) -> None:
        """Regression invariant: G = N (singleton clusters) reproduces the i.i.d. path.

        With cluster ids ``0..N-1`` and the same ``base_seed`` and
        ``weight_scheme``, the cluster path draws ``G = N`` weights from the
        same RNG state and broadcasts them via ``codes = arange(N)``.  The
        ``theta_star`` trajectory must match the unclustered path exactly.
        """

        _, result, _ = _simple_restriction_and_result()
        n = result.restriction.num_observations

        boot_iid = MomentWildBootstrap(result, n_bootstrap=8, base_seed=42)
        res_iid = boot_iid.run_sequential()

        cluster_ids = np.arange(n)
        boot_clus = MomentWildBootstrap(
            result, n_bootstrap=8, clusters=cluster_ids, base_seed=42
        )
        res_clus = boot_clus.run_sequential()

        iid_thetas = np.array(
            [float(np.asarray(r.theta_star.value).ravel()[0]) for r in res_iid]
        )
        clus_thetas = np.array(
            [float(np.asarray(r.theta_star.value).ravel()[0]) for r in res_clus]
        )
        np.testing.assert_array_equal(iid_thetas, clus_thetas)

    def test_callable_scheme_called_with_G_in_cluster_mode(self) -> None:
        """When ``clusters=`` is set, the weight generator is called with G, not N.

        Indirect verification of the broadcast: the per-cluster scheme draws
        ``G`` weights; the unclustered path draws ``N``.
        """

        cluster_ids = np.repeat(np.arange(3), 4)  # 12 obs in 3 clusters of size 4
        _, result, _ = _clustered_restriction_and_result(cluster_ids)

        captured: list[int] = []

        def recording_scheme(n: int, rng: np.random.Generator) -> np.ndarray:
            captured.append(n)
            return rng.uniform(size=n)

        boot = MomentWildBootstrap(
            result,
            n_bootstrap=2,
            clusters=cluster_ids,
            weight_scheme=recording_scheme,
            base_seed=0,
        )
        boot.run_sequential()

        assert captured == [3, 3], (
            "scheme should be called with G=3 once per replicate, "
            f"got {captured}"
        )

    def test_weights_constant_within_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The weights handed to ``with_weights`` are cluster-constant."""

        cluster_ids = np.repeat(np.arange(3), 4)  # 12 obs / 3 clusters / size 4
        _, result, _ = _clustered_restriction_and_result(cluster_ids)

        from manifoldgmm.econometrics import moment_restriction as mr_mod

        original_with_weights = mr_mod.MomentRestriction.with_weights
        captured_weights: list[np.ndarray] = []

        def spy_with_weights(self: Any, w: Any) -> Any:
            captured_weights.append(np.asarray(w).copy())
            return original_with_weights(self, w)

        monkeypatch.setattr(
            mr_mod.MomentRestriction, "with_weights", spy_with_weights
        )

        boot = MomentWildBootstrap(
            result, n_bootstrap=2, clusters=cluster_ids, base_seed=0
        )
        boot.run_sequential()

        assert len(captured_weights) == 2
        for w in captured_weights:
            assert w.shape == (12,)
            for c in range(3):
                mask = cluster_ids == c
                unique_vals = np.unique(w[mask])
                assert unique_vals.size == 1, (
                    f"weights not cluster-constant for cluster {c}: "
                    f"{w[mask]}"
                )

    def test_auto_default_from_restriction_clusters(self) -> None:
        """If the restriction carries ``.clusters``, the bootstrap auto-uses it."""

        cluster_ids = np.repeat(np.arange(4), 3)  # 12 obs in 4 clusters
        _, result, _ = _clustered_restriction_and_result(
            cluster_ids, attach_clusters=True
        )

        # No explicit clusters= argument -- should still pick up the parent
        boot = MomentWildBootstrap(result, n_bootstrap=3, base_seed=0)

        assert boot._cluster_codes is not None
        assert boot._cluster_codes.shape == (12,)
        assert boot._num_clusters == 4

        # Verify it's actually used by checking the scheme is called with G=4
        captured: list[int] = []

        def recording_scheme(n: int, rng: np.random.Generator) -> np.ndarray:
            captured.append(n)
            return rng.uniform(size=n)

        boot2 = MomentWildBootstrap(
            result,
            n_bootstrap=2,
            weight_scheme=recording_scheme,
            base_seed=0,
        )
        boot2.run_sequential()
        assert captured == [4, 4]

    def test_explicit_clusters_override_restriction(self) -> None:
        """An explicit ``clusters=`` argument overrides any inherited assignment."""

        inherited_ids = np.repeat(np.arange(4), 3)  # G=4
        override_ids = np.repeat(np.arange(2), 6)  # G=2
        _, result, _ = _clustered_restriction_and_result(
            inherited_ids, attach_clusters=True
        )

        captured: list[int] = []

        def recording_scheme(n: int, rng: np.random.Generator) -> np.ndarray:
            captured.append(n)
            return rng.uniform(size=n)

        boot = MomentWildBootstrap(
            result,
            n_bootstrap=2,
            clusters=override_ids,
            weight_scheme=recording_scheme,
            base_seed=0,
        )
        boot.run_sequential()
        assert captured == [2, 2], (
            "explicit clusters= should take precedence over inherited; "
            f"got n={captured}"
        )

    def test_task_restriction_carries_clusters(self) -> None:
        """``BootstrapTask.restriction.clusters`` is set in clustered mode.

        This ensures the ``with_clusters`` chain in ``BootstrapTask.run``
        fires and the replicate :math:`\\hat\\Omega` is cluster-robust.
        """

        cluster_ids = np.repeat(np.arange(3), 4)
        _, result, _ = _clustered_restriction_and_result(cluster_ids)

        boot = MomentWildBootstrap(
            result, n_bootstrap=1, clusters=cluster_ids, base_seed=0
        )
        task = boot.tasks()[0]
        assert task.restriction.clusters is not None
        assert task.cluster_codes is not None
        assert task.num_clusters == 3

    def test_callable_scheme_pickle_round_trip(self) -> None:
        """A callable ``weight_scheme`` survives ``BootstrapTask.to_bytes``."""

        cluster_ids = np.repeat(np.arange(3), 4)
        _, result, _ = _clustered_restriction_and_result(cluster_ids)

        # Use the module-level helper (picklable) as the scheme
        boot = MomentWildBootstrap(
            result,
            n_bootstrap=1,
            clusters=cluster_ids,
            weight_scheme=exponential_weights,
            base_seed=11,
        )
        task = boot.tasks()[0]

        # Round trip
        data = task.to_bytes()
        revived = BootstrapTask.from_bytes(data)
        assert callable(revived.weight_scheme)
        # Same RNG state should yield identical weights
        n_clusters = revived.num_clusters
        assert n_clusters is not None
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        w1 = task.weight_scheme(n_clusters, rng1)  # type: ignore[operator]
        w2 = revived.weight_scheme(n_clusters, rng2)  # type: ignore[operator]
        np.testing.assert_array_equal(w1, w2)

        # End-to-end run also succeeds via the revived task
        revived_result = revived.run()
        assert isinstance(revived_result, BootstrapResult)

    def test_callable_scheme_replaces_registry(self) -> None:
        """A user-defined callable bypasses ``_WEIGHT_GENERATORS`` entirely.

        Regression guard for the loky-fork hazard: the worker path must not
        depend on parent-process mutations of the module-level registry.
        """

        cluster_ids = np.repeat(np.arange(2), 3)  # 6 obs / 2 clusters / size 3
        _, result, _ = _clustered_restriction_and_result(cluster_ids)

        # A scheme NOT in _WEIGHT_GENERATORS; bootstrap accepts it directly.
        def custom_scheme(n: int, rng: np.random.Generator) -> np.ndarray:
            # E[w] = 1, Var[w] = 1; values in {1 - sqrt(3), 1, 1 + sqrt(3)}
            return rng.choice(
                [1.0 - np.sqrt(3.0), 1.0, 1.0 + np.sqrt(3.0)],
                size=n,
                p=[1 / 6, 2 / 3, 1 / 6],
            )

        boot = MomentWildBootstrap(
            result,
            n_bootstrap=3,
            clusters=cluster_ids,
            weight_scheme=custom_scheme,
            base_seed=0,
        )
        results = boot.run_sequential()
        assert len(results) == 3
        # All replicates should produce finite theta_star values
        for r in results:
            assert np.isfinite(float(np.asarray(r.theta_star.value).ravel()[0]))

    def test_invalid_weight_scheme_type_raises(self) -> None:
        """Non-str, non-callable ``weight_scheme`` raises TypeError."""

        _, result, _ = _simple_restriction_and_result()

        with pytest.raises(TypeError, match="weight_scheme must be"):
            MomentWildBootstrap(result, weight_scheme=42)  # type: ignore[arg-type]

    def test_clusters_wrong_shape_raises(self) -> None:
        """``clusters=`` with shape mismatched to N raises ValueError."""

        _, result, _ = _simple_restriction_and_result()

        with pytest.raises(ValueError, match="clusters length"):
            MomentWildBootstrap(
                result, clusters=np.arange(99), n_bootstrap=1, base_seed=0
            )

    def test_clusters_non_1d_raises(self) -> None:
        """``clusters=`` with ndim != 1 raises ValueError."""

        _, result, _ = _simple_restriction_and_result()

        with pytest.raises(ValueError, match="1-D array"):
            MomentWildBootstrap(
                result,
                clusters=np.zeros((5, 2)),
                n_bootstrap=1,
                base_seed=0,
            )


# -----------------------------------------------------------------------
# 8. Cluster-aware bootstrap: variance inflation smoke test
# -----------------------------------------------------------------------


class TestClusterVarianceInflation:
    """Smoke-grade: cluster bootstrap distribution is wider than i.i.d. on clustered data."""

    def test_cluster_sd_exceeds_iid_sd(self) -> None:
        """On clustered DGP, the bootstrap SD of ``theta_star`` is larger
        under the cluster scheme than the i.i.d. scheme.

        This is the bootstrap-distribution analogue of the SE inflation
        established for ``omega_hat`` in PR #5; it is a necessary (but not
        sufficient) condition for correct CR coverage.  The full coverage
        check is in :class:`TestClusterMonteCarlo` below, marked slow.
        """

        rng = np.random.default_rng(0)
        G, m = 20, 8  # 160 observations
        sigma_u, sigma_e = 1.0, 0.5
        u = rng.normal(scale=sigma_u, size=G)
        e = rng.normal(scale=sigma_e, size=(G, m))
        X = (u[:, None] + e).ravel()
        cluster_ids = np.repeat(np.arange(G), m)

        data = jnp.array(X)
        _, result, _ = _clustered_restriction_and_result(cluster_ids, data=data)

        boot_iid = MomentWildBootstrap(result, n_bootstrap=200, base_seed=1)
        boot_iid.run_sequential()
        iid_sd = float(
            np.std(
                [
                    float(np.asarray(r.theta_star.value).ravel()[0])
                    for r in boot_iid._results
                ]
            )
        )

        boot_clus = MomentWildBootstrap(
            result, n_bootstrap=200, clusters=cluster_ids, base_seed=1
        )
        boot_clus.run_sequential()
        clus_sd = float(
            np.std(
                [
                    float(np.asarray(r.theta_star.value).ravel()[0])
                    for r in boot_clus._results
                ]
            )
        )

        # Theoretical ratio: sqrt((sigma_u^2 + sigma_e^2/m)*N / (sigma_u^2 + sigma_e^2))
        # = sqrt((1.0 + 0.25/8) * 160 / 1.25) ~= sqrt(132) ~= 11.5... no wait,
        # the i.i.d. bootstrap recovers an estimator of (sigma_u^2 + sigma_e^2)/N,
        # whereas the cluster bootstrap targets (sigma_u^2 + sigma_e^2/m)/G.
        # Ratio of SDs:
        #   cluster / iid = sqrt(((sigma_u^2 + sigma_e^2/m)/G)
        #                        / ((sigma_u^2 + sigma_e^2)/N))
        #                 = sqrt((1.03125/20) / (1.25/160))
        #                 = sqrt(0.0515625 / 0.0078125)
        #                 = sqrt(6.6) ~= 2.57
        # Use a loose lower bound of 1.5 to absorb 200-replicate noise.
        assert clus_sd > 1.5 * iid_sd, (
            f"cluster SD ({clus_sd:.4f}) should exceed 1.5x iid SD "
            f"({iid_sd:.4f}); ratio {clus_sd / iid_sd:.2f}"
        )


# -----------------------------------------------------------------------
# 9. Cluster-aware bootstrap: size, power, and i.i.d.-under-coverage MC
# -----------------------------------------------------------------------


def _draw_clustered_panel(
    rng: np.random.Generator,
    mu: float,
    G: int,
    m: int,
    sigma_u: float,
    sigma_e: float,
) -> tuple[Any, np.ndarray]:
    """Draw a panel with G clusters of size m and within-cluster correlation.

    ``X_{c,i} = mu + u_c + e_{c,i}`` with ``u_c ~ N(0, sigma_u^2)``,
    ``e_{c,i} ~ N(0, sigma_e^2)``.  Returns ``(data_jax, cluster_ids)``.
    """

    u = rng.normal(scale=sigma_u, size=G)
    e = rng.normal(scale=sigma_e, size=(G, m))
    X = mu + u[:, None] + e
    cluster_ids = np.repeat(np.arange(G), m)
    return jnp.array(X.ravel(), dtype=jnp.float64), cluster_ids


def _fit_and_bootstrap(
    data: Any,
    cluster_ids: np.ndarray,
    *,
    n_bootstrap: int,
    base_seed: int,
    use_clusters_in_bootstrap: bool,
) -> tuple[Any, Any, np.ndarray]:
    """Fit cluster-robust GMM on the panel and run a wild bootstrap.

    The parent GMM always carries cluster ids so that the analytic
    ``Sigma`` returned by ``tangent_covariance`` is the cluster-robust
    (true) one.  Only the bootstrap weight scheme is toggled by
    ``use_clusters_in_bootstrap``: when False, the wild draws are i.i.d.
    per observation (the pre-fix bug); when True, they are cluster-
    constant.

    Returns ``(result, boot, sigma_cluster)``.  Callers should use
    ``sigma_cluster`` as the explicit ``covariance=`` argument to
    ``geodesic_mahalanobis_distance`` and ``boot.geodesic_distances`` so
    that the cluster-vs-iid contrast isolates the bootstrap scheme
    rather than entangling it with the analytic-Sigma estimator.
    """

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        clusters=cluster_ids,
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()
    sigma_cluster = np.asarray(
        result.tangent_covariance().to_numpy(dtype=float)
    )

    if use_clusters_in_bootstrap:
        boot = MomentWildBootstrap(
            result, n_bootstrap=n_bootstrap, base_seed=base_seed
        )
    else:
        # Build a parallel result whose restriction lacks clusters so the
        # bootstrap's auto-default falls back to per-observation draws.
        # We do NOT use this result's analytic Sigma below; callers pass
        # ``sigma_cluster`` explicitly to keep the contrast clean.
        from dataclasses import replace as dc_replace

        bare_restriction = restriction.with_clusters(None)  # type: ignore[arg-type]
        bare_result = dc_replace(result, restriction=bare_restriction)
        boot = MomentWildBootstrap(
            bare_result, n_bootstrap=n_bootstrap, base_seed=base_seed
        )

    boot.run_sequential()
    return result, boot, sigma_cluster


class TestClusterMonteCarlo:
    """Slow MC tests: size, power, and i.i.d.-under-coverage on a clustered DGP.

    Parameters chosen so that the i.i.d. wild bootstrap is severely
    under-sized (intra-cluster correlation rho ~= 0.8) and the cluster
    bootstrap recovers correct coverage.  All seeds fixed.

    Compute budget: with the rep counts below (~10,000 GMM fits total
    across the three tests) and Euclidean(1) data of size
    ``N = G*M = 100``, the suite runs in well under one hour
    single-threaded.  Coverage SE at 50 outer reps is ~0.07 so the
    assertion bands are intentionally wide; the cluster-vs-iid contrast
    is huge (true coverage ~0.95 vs ~0.55) and easily resolved.
    """

    # Common DGP -- intentionally small to keep the slow-test budget
    # bounded while preserving a clear cluster-vs-iid signal.
    G = 20
    M = 5
    SIGMA_U = 1.0
    SIGMA_E = 0.5
    MU_TRUE = 0.0
    N_BOOT = 99
    ALPHA = 0.05
    N_OUTER_SIZE = 50
    N_OUTER_POWER = 20

    @staticmethod
    def _is_in_cr(
        result: Any,
        boot: Any,
        sigma: np.ndarray,
        point: Any,
        alpha: float,
    ) -> bool:
        """CR membership using an explicit cluster-robust ``Sigma``.

        Mirrors :meth:`MomentWildBootstrap.in_confidence_region` but pins
        the covariance matrix on both sides of the inequality so the
        cluster-vs-iid contrast isolates the bootstrap weight scheme.
        """

        d2_boots = boot.geodesic_distances(covariance=sigma)
        cv = float(np.quantile(d2_boots, 1.0 - alpha))
        d2_point = geodesic_mahalanobis_distance(
            result, point, covariance=sigma
        )
        return d2_point <= cv

    @pytest.mark.slow
    def test_cluster_bootstrap_size(self) -> None:
        """Empirical coverage of the cluster wild bootstrap CR is near 1 - alpha."""

        n_reps = self.N_OUTER_SIZE
        rng = np.random.default_rng(2026)
        contained = 0

        for rep in range(n_reps):
            data, cids = _draw_clustered_panel(
                rng,
                mu=self.MU_TRUE,
                G=self.G,
                m=self.M,
                sigma_u=self.SIGMA_U,
                sigma_e=self.SIGMA_E,
            )
            result, boot, sigma = _fit_and_bootstrap(
                data,
                cids,
                n_bootstrap=self.N_BOOT,
                base_seed=rep,
                use_clusters_in_bootstrap=True,
            )
            if self._is_in_cr(
                result, boot, sigma, jnp.array([self.MU_TRUE]), self.ALPHA
            ):
                contained += 1

        coverage = contained / n_reps
        # Nominal 0.95; with 50 reps SE ~= sqrt(0.95*0.05/50) ~= 0.031.
        # Wide band absorbs MC noise and any small bootstrap finite-sample bias.
        assert 0.85 <= coverage <= 1.0, (
            f"cluster wild bootstrap coverage {coverage:.3f} "
            f"outside [0.85, 1.0] band (nominal 0.95)"
        )

    @pytest.mark.slow
    def test_cluster_bootstrap_power(self) -> None:
        """Against an alternative the bootstrap CR rejects with sufficient power."""

        n_reps = self.N_OUTER_POWER
        # Pick mu_alt at roughly 3 cluster-SDs from 0; SD of theta_hat is
        # sqrt((sigma_u^2 + sigma_e^2/m)/G) = sqrt((1 + 0.05)/20) ~= 0.229.
        # mu_alt = 0.7 -> z ~= 3.06 -> theoretical power ~= 0.93.
        mu_alt = 0.7
        rng = np.random.default_rng(7)
        rejected = 0

        for rep in range(n_reps):
            data, cids = _draw_clustered_panel(
                rng,
                mu=mu_alt,
                G=self.G,
                m=self.M,
                sigma_u=self.SIGMA_U,
                sigma_e=self.SIGMA_E,
            )
            result, boot, sigma = _fit_and_bootstrap(
                data,
                cids,
                n_bootstrap=self.N_BOOT,
                base_seed=1000 + rep,
                use_clusters_in_bootstrap=True,
            )
            # H0: mu = 0 (we know the truth is mu = mu_alt)
            in_cr = self._is_in_cr(
                result, boot, sigma, jnp.array([self.MU_TRUE]), self.ALPHA
            )
            if not in_cr:
                rejected += 1

        power = rejected / n_reps
        # Theoretical power ~= 0.95; with 30 reps the worst case (true power
        # 0.8) has SE ~= 0.073.  Floor of 0.7 is comfortably above noise.
        assert power >= 0.7, f"cluster bootstrap power {power:.3f} below 0.7"

    @pytest.mark.slow
    def test_iid_bootstrap_under_covers_clustered_dgp(self) -> None:
        """Pre-fix bug regression check: per-obs wild bootstrap under-covers.

        Uses the cluster-robust ``Sigma`` (same as the size test) so the
        contrast against ``test_cluster_bootstrap_size`` isolates the
        bootstrap weight scheme.  Confirms that per-observation wild
        draws produce a bootstrap distribution too narrow for clustered
        data, dropping coverage well below the nominal 0.95.  This is
        the on-disk documentation of the bug this PR fixes.
        """

        n_reps = self.N_OUTER_SIZE
        rng = np.random.default_rng(2026)  # same seed as the size test
        contained = 0

        for rep in range(n_reps):
            data, cids = _draw_clustered_panel(
                rng,
                mu=self.MU_TRUE,
                G=self.G,
                m=self.M,
                sigma_u=self.SIGMA_U,
                sigma_e=self.SIGMA_E,
            )
            result, boot, sigma = _fit_and_bootstrap(
                data,
                cids,
                n_bootstrap=self.N_BOOT,
                base_seed=rep,
                use_clusters_in_bootstrap=False,
            )
            if self._is_in_cr(
                result, boot, sigma, jnp.array([self.MU_TRUE]), self.ALPHA
            ):
                contained += 1

        coverage = contained / n_reps
        # With Sigma held fixed at the cluster-robust value and the
        # bootstrap drawing i.i.d. weights, the bootstrap critical value
        # is roughly the chi^2_1 quantile rescaled by (s_iid_boot / s_cluster)^2.
        # For this DGP that ratio is ~ 1/2.6, so the effective alpha is
        # ~ 2 * (1 - Phi(1.96/2.6)) ~= 0.45; coverage ~= 0.55.  Threshold
        # at 0.85 leaves wide MC margin.
        assert coverage < 0.85, (
            f"per-obs wild bootstrap coverage {coverage:.3f} "
            f"unexpectedly close to nominal 0.95; either the DGP is too "
            f"weak or the bootstrap is not actually using i.i.d. draws"
        )
