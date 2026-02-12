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
    mammen_weights,
    rademacher_weights,
)
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _simple_restriction_and_result(
    data: np.ndarray | None = None,
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
        # Unweighted: g_bar = mean(data - 0) = 2.0
        unweighted = np.asarray(restriction.g_bar(theta)).ravel()
        np.testing.assert_allclose(unweighted, [2.0], atol=1e-10)

        # Weighted: w = [2, 0, 1], g_i = data - 0 = [1, 2, 3]
        # weighted mean = (2*1 + 0*2 + 1*3) / 3 = 5/3
        weights = np.array([2.0, 0.0, 1.0])
        weighted_restriction = restriction.with_weights(weights)
        weighted_mean = np.asarray(weighted_restriction.g_bar(theta)).ravel()
        np.testing.assert_allclose(weighted_mean, [5.0 / 3.0], atol=1e-10)

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
