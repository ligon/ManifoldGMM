"""Tests for the :class:`LoggingTrustRegions` optimizer subclass."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.optimizers import LoggingTrustRegions
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


def _simple_fit() -> Any:
    """Run the Euclidean(1) sample-mean fit used as a regression fixture."""

    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    return gmm.estimate()


def test_default_optimizer_is_logging_trust_regions_compatible() -> None:
    """Unconfigured estimate() produces an optimizer_report with the new log."""

    result = _simple_fit()
    log = result.optimizer_report.get("log")
    assert log is not None, "default optimizer should populate optimizer_report['log']"
    assert "inner_stop_counts" in log, (
        "LoggingTrustRegions should attach inner_stop_counts; " f"saw keys: {list(log)}"
    )
    assert "gradient_norms" in log, (
        "LoggingTrustRegions should attach gradient_norms; " f"saw keys: {list(log)}"
    )


def test_inner_stop_counts_sum_equals_outer_iterations() -> None:
    """Every outer iteration runs exactly one inner CG and bumps the counter."""

    result = _simple_fit()
    log = result.optimizer_report["log"]
    counts = log["inner_stop_counts"]
    total = sum(counts.values())
    n_outer = result.optimizer_report["iterations"]
    assert total == n_outer, (
        f"inner_stop_counts sum {total} should equal outer iterations {n_outer}; "
        f"counts: {counts}"
    )


def test_gradient_norms_include_initial_and_per_iter() -> None:
    """One initial gradient evaluation plus one per accepted outer step.

    Pymanopt's TrustRegions evaluates the gradient once before the loop
    and once on each accepted iteration; rejected steps reuse the
    previous gradient.  On this trivial problem every step is accepted,
    so the list length is ``1 + n_outer``.
    """

    result = _simple_fit()
    log = result.optimizer_report["log"]
    norms = log["gradient_norms"]
    assert isinstance(norms, list)
    n_outer = result.optimizer_report["iterations"]
    # Allow exact equality (all-accepted) or off-by-one tolerance for
    # the rare case of a rejected first step.
    assert (
        len(norms) >= n_outer
    ), f"expected at least {n_outer} recorded gradient norms, got {len(norms)}"
    assert all(float(n) >= 0.0 for n in norms)


def test_gradient_attribute_restored_after_run() -> None:
    """LoggingTrustRegions must not leak its wrapper on the Problem.

    Failing this test would mean a Problem reused across estimates
    keeps recording into the *first* run's counter -- a silent
    correctness bug.
    """

    from pymanopt import Problem

    data = jnp.array([1.0, 2.0, 3.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )

    # Pull the cost function the same way GMM does, then build a
    # Problem ourselves so we can inspect ``_riemannian_gradient`` after
    # the optimizer returns.
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    cost = gmm._build_cost(gmm._weighting)
    assert manifold.data is not None  # narrowed for mypy
    problem = Problem(cost=cost, manifold=manifold.data)

    # Resolve the gradient once so the lazy fallback writes a callable
    # into ``_riemannian_gradient`` (otherwise it stays None before the
    # property is accessed).
    original = problem.riemannian_gradient

    opt = LoggingTrustRegions(verbosity=0, log_verbosity=1)
    opt.run(problem, initial_point=jnp.array([0.0]))

    # The wrapper should have been removed in the finally clause.
    assert problem._riemannian_gradient is original, (
        "LoggingTrustRegions leaked its recording_gradient wrapper onto "
        "the Problem; reusing the Problem would silently aggregate "
        "telemetry across fits."
    )


def test_repeat_runs_do_not_aggregate_telemetry() -> None:
    """Re-running on the same optimizer resets per-run telemetry."""

    from pymanopt import Problem

    data = jnp.array([1.0, 2.0, 3.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    cost = gmm._build_cost(gmm._weighting)
    assert manifold.data is not None
    problem = Problem(cost=cost, manifold=manifold.data)

    opt = LoggingTrustRegions(verbosity=0, log_verbosity=1)
    first = opt.run(problem, initial_point=jnp.array([0.0]))
    second = opt.run(problem, initial_point=jnp.array([0.0]))

    # The second run's telemetry should match the first, not be the sum.
    first_total = sum(first.log["inner_stop_counts"].values())
    second_total = sum(second.log["inner_stop_counts"].values())
    assert second_total == first_total, (
        f"second run total inner stops {second_total} should equal "
        f"first run total {first_total} (telemetry should reset, not aggregate)"
    )
    assert len(second.log["gradient_norms"]) == len(first.log["gradient_norms"])


def test_gmm_result_optimizer_health_surfaces_telemetry() -> None:
    """GMMResult.optimizer_health condenses the log into the expected keys."""

    result = _simple_fit()
    health = result.optimizer_health

    assert health["n_outer_iters"] >= 1
    assert isinstance(health["inner_stop_counts"], dict)
    assert health["n_inner_cap_hits"] >= 0
    assert health["inner_cap_hit_frac"] is not None
    assert 0.0 <= health["inner_cap_hit_frac"] <= 1.0
    # Healthy fit -> non-zero number of recorded norms -> slope is set.
    assert health["tail_grad_slope"] is not None
    # Healthy convergence: slope of log|grad| is strongly negative.
    assert health["tail_grad_slope"] < 0.0, (
        f"expected strongly negative tail slope on a converged Euclidean(1) "
        f"fit; got {health['tail_grad_slope']}"
    )
    assert health["tail_window"] >= 2


def test_compute_hessian_cond_finite_and_well_conditioned() -> None:
    """The trivial Euclidean(1) Hessian is the identity (up to W); cond ~ 1."""

    result = _simple_fit()
    cond = result.compute_hessian_cond()
    assert np.isfinite(cond)
    assert cond > 0.0
    # 1-dimensional manifold -> 1x1 Hessian -> condition number is exactly 1.
    np.testing.assert_allclose(cond, 1.0, atol=1e-10)


def test_compute_hessian_cond_matches_hand_computed_value() -> None:
    """``compute_hessian_cond`` is exactly ``cond(D' W D)`` from cached pieces.

    Avoids the convergence behaviour of a multi-parameter fit (which is
    irrelevant to the diagnostic in question).  Instead we hand-build a
    ``GMMResult`` where ``canonical_jacobian`` and ``weighting`` are
    known and check the closed-form ratio.  ``D' W D = diag(1, 4)`` ->
    ``cond = 4``.
    """

    from dataclasses import replace as dc_replace

    from manifoldgmm.econometrics.gmm import FixedWeighting

    base = _simple_fit()

    D = np.array([[1.0, 0.0], [0.0, 2.0]])  # canonical-basis Jacobian
    W = np.eye(2)

    forged = dc_replace(base, weighting=FixedWeighting(W, label="hand-set"))
    forged._cached_jacobian = D
    forged._cached_jacobian_basis = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]

    cond = forged.compute_hessian_cond()
    np.testing.assert_allclose(cond, 4.0, rtol=1e-10)


def test_tail_log_grad_slope_helper_unit() -> None:
    """The slope helper handles empty, all-zero, and known-line inputs."""

    from manifoldgmm.econometrics.gmm import _tail_log_grad_slope

    # Empty
    slope, window = _tail_log_grad_slope([])
    assert slope is None and window == 0

    # Single positive value
    slope, window = _tail_log_grad_slope([1.0])
    assert slope is None and window == 1

    # All zeros -> no positive values to log
    slope, window = _tail_log_grad_slope([0.0, 0.0, 0.0])
    assert slope is None

    # Pure exponential decay: |grad_k| = exp(-2 k) -> log slope is -2.
    series = [float(np.exp(-2.0 * k)) for k in range(10)]
    slope, window = _tail_log_grad_slope(series)
    assert slope is not None
    np.testing.assert_allclose(slope, -2.0, atol=1e-10)
    assert window == 10

    # Long input clamped to max_window
    long_series = [float(np.exp(-0.5 * k)) for k in range(100)]
    slope, window = _tail_log_grad_slope(long_series, max_window=20)
    assert slope is not None
    assert window == 20
    np.testing.assert_allclose(slope, -0.5, atol=1e-10)


def test_optimizer_health_handles_third_party_optimizer() -> None:
    """A user-supplied optimizer that doesn't populate the log still works."""

    import types

    from pymanopt.optimizers.optimizer import Optimizer

    class StubOptimizer(Optimizer):
        def run(self, problem: Any, *, initial_point: Any) -> Any:
            return types.SimpleNamespace(
                point=initial_point,
                cost=0.0,
                iterations=0,
                stopping_criterion="custom: did not run",
                time=0.0,
                gradient_norm=None,
                log=None,
            )

    data = jnp.array([1.0, 2.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )
    gmm = GMM(
        restriction,
        optimizer=StubOptimizer,
        initial_point=jnp.array([0.0]),
    )
    result = gmm.estimate()
    health = result.optimizer_health

    # Without a log, the cap fraction and tail slope must be None
    # rather than crashing or pretending to a value.
    assert health["inner_cap_hit_frac"] is None
    assert health["tail_grad_slope"] is None
    assert health["n_inner_cap_hits"] == 0
    assert health["inner_stop_counts"] == {}


def test_user_optimizer_kwargs_can_override_log_verbosity() -> None:
    """The default sets log_verbosity=1; users can override to 0."""

    data = jnp.array([1.0, 2.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate(optimizer_kwargs={"log_verbosity": 0})

    log = result.optimizer_report["log"]
    # Even at log_verbosity=0, our subclass still attaches its own
    # telemetry to ``result.log`` -- the upstream "iterations" sub-dict
    # is None at log_verbosity=0 but ``inner_stop_counts`` and
    # ``gradient_norms`` are populated unconditionally.
    assert log is not None
    assert log.get("inner_stop_counts") is not None
    assert log.get("gradient_norms") is not None


def test_compute_hessian_cond_handles_missing_weighting() -> None:
    """compute_hessian_cond raises a clear error if weighting is None."""

    from dataclasses import replace as dc_replace

    result = _simple_fit()
    bare = dc_replace(result, weighting=None)
    with pytest.raises(ValueError, match="weighting"):
        bare.compute_hessian_cond()
