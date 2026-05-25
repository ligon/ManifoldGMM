"""Tests for the :class:`LoggingTrustRegions` optimizer subclass."""

from __future__ import annotations

import warnings
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
    health = result.diagnostics.optimizer_health

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
    cond = result.diagnostics.hessian_cond()
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

    cond = forged.diagnostics.hessian_cond()
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
    health = result.diagnostics.optimizer_health

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
        bare.diagnostics.hessian_cond()


# -----------------------------------------------------------------------
# ConvergenceWarning on stall (#10 PR 2)
# -----------------------------------------------------------------------


def _force_optimizer_report(
    result: Any,
    *,
    log: dict[str, Any],
    converged: bool | None,
    stopping_criterion: str,
    iterations: int,
) -> Any:
    """Replace ``optimizer_report`` on a real GMMResult with a synthetic one.

    Used to test ``_maybe_warn_optimizer_health`` without having to drive
    an actual estimator into the stall pathology.
    """

    fake_report = dict(result.optimizer_report)
    fake_report.update(
        log=log,
        converged=converged,
        stopping_criterion=stopping_criterion,
        iterations=iterations,
    )
    object.__setattr__(result, "optimizer_report", fake_report)
    return result


def test_healthy_fit_emits_no_convergence_warning() -> None:
    """A clean Euclidean(1) fit should not trigger the stall warning."""

    from manifoldgmm import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        # If estimate() emits ConvergenceWarning the simplefilter("error")
        # promotes it to a raised exception, failing the test.
        _ = _simple_fit()


def test_convergence_warning_fires_on_synthetic_stall() -> None:
    """All three stall conditions present -> ConvergenceWarning is emitted."""

    from manifoldgmm import ConvergenceWarning

    base = _simple_fit()
    fake_log = {
        # 14 cap hits out of 16 inner solves -> cap_frac = 0.875 > 0.5.
        "inner_stop_counts": {
            "maximum inner iterations": 14,
            "exceeded trust region": 2,
        },
        # Flat tail: gradient norm barely moves over ~14 iters.
        # log slope on these is approximately -0.003 per step,
        # well above (less negative than) the -0.01 threshold.
        "gradient_norms": [
            4.20e-3,
            4.18e-3,
            4.16e-3,
            4.15e-3,
            4.13e-3,
            4.11e-3,
            4.10e-3,
            4.09e-3,
            4.08e-3,
            4.07e-3,
            4.06e-3,
            4.05e-3,
            4.04e-3,
            4.03e-3,
            4.02e-3,
        ],
    }
    stalled = _force_optimizer_report(
        base,
        log=fake_log,
        converged=False,
        stopping_criterion=(
            "Terminated - max iterations reached after 200.00 seconds."
        ),
        iterations=14,
    )

    # We invoke the warning helper directly with the patched result
    # (estimate() already returned; the in-flight warning has fired or
    # not based on the *original* report).
    from manifoldgmm.econometrics.gmm import _maybe_warn_optimizer_health

    with pytest.warns(ConvergenceWarning) as record:
        _maybe_warn_optimizer_health(stalled)

    assert len(record) == 1
    message = str(record[0].message)
    # The remediation hint should mention both maxinner and
    # the hessian-cond diagnostic per issue #10's PR 2 spec.  The
    # message text was updated to point at the new diagnostics
    # namespace (``result.diagnostics.hessian_cond()``) but the
    # substring ``hessian_cond`` still appears.
    assert "maxinner" in message
    assert "hessian_cond" in message
    # And the diagnostics that drove the decision should be visible.
    assert "inner_cap_hit_frac" in message
    assert "tail_grad_slope" in message


def test_convergence_warning_silent_when_telemetry_missing() -> None:
    """No warning when the log lacks the inputs the helper needs.

    A third-party optimizer (or a user run at ``log_verbosity=0`` against
    plain ``TrustRegions``) won't have ``inner_stop_counts`` or
    ``gradient_norms``; the helper must short-circuit rather than fire
    spuriously.
    """

    from manifoldgmm import ConvergenceWarning
    from manifoldgmm.econometrics.gmm import _maybe_warn_optimizer_health

    base = _simple_fit()
    bare = _force_optimizer_report(
        base,
        log={},  # no inner_stop_counts, no gradient_norms
        converged=False,
        stopping_criterion="Terminated - max iterations reached.",
        iterations=14,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        _maybe_warn_optimizer_health(bare)


def test_convergence_warning_silent_on_tolerance_stop() -> None:
    """A converged=True run skips the warning even with high cap_frac.

    Cap-hit fraction can plausibly be > 0.5 on hard problems that still
    reach a tolerance.  The discriminating signal for "stuck" is that
    the outer loop ran out of budget, so converged=True suppresses.
    """

    from manifoldgmm import ConvergenceWarning
    from manifoldgmm.econometrics.gmm import _maybe_warn_optimizer_health

    base = _simple_fit()
    high_cap = _force_optimizer_report(
        base,
        log={
            "inner_stop_counts": {
                "maximum inner iterations": 9,
                "exceeded trust region": 1,
            },
            "gradient_norms": [1.0, 0.1, 0.01],  # healthy slope ~ -2.3 per iter
        },
        converged=True,
        stopping_criterion="Terminated - min grad norm reached.",
        iterations=3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        _maybe_warn_optimizer_health(high_cap)


def test_convergence_warning_silent_when_slope_is_descending() -> None:
    """No warning when the gradient is still falling, even if the cap fires.

    Two of three conditions met (cap_frac high, max_iters stop) but the
    gradient is still genuinely descending -- the optimiser was making
    progress and just hit the iteration budget.  Bumping max_iterations,
    not maxinner, is the right remediation, so the stall warning would
    be misleading.
    """

    from manifoldgmm import ConvergenceWarning
    from manifoldgmm.econometrics.gmm import _maybe_warn_optimizer_health

    base = _simple_fit()
    descending = _force_optimizer_report(
        base,
        log={
            "inner_stop_counts": {
                "maximum inner iterations": 12,
                "exceeded trust region": 2,
            },
            "gradient_norms": [
                10 ** (-i) for i in range(14)
            ],  # slope = -ln(10) per iter
        },
        converged=False,
        stopping_criterion="Terminated - max iterations reached.",
        iterations=14,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        _maybe_warn_optimizer_health(descending)


# -----------------------------------------------------------------------
# Adaptive maxinner policy (#10 PR 3)
# -----------------------------------------------------------------------


def test_adaptive_maxinner_off_by_default() -> None:
    """Default LoggingTrustRegions matches PR #12 behaviour exactly."""

    opt = LoggingTrustRegions()
    assert opt._adaptive_maxinner is False
    # No adaptive log keys when adaptive is off.
    result = _simple_fit()
    log = result.optimizer_report["log"]
    assert "maxinner_history" not in log
    assert "numit_history" not in log


def test_adaptive_maxinner_records_history_on_real_fit() -> None:
    """When opted in, the adaptive log keys appear and are sane."""

    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate(optimizer_kwargs={"adaptive_maxinner": True})

    log = result.optimizer_report["log"]
    history = log["maxinner_history"]
    numit_history = log["numit_history"]
    assert len(history) == len(numit_history)
    # Healthy convergence on Euclidean(1): maxinner should stay constant
    # at the starting value (manifold.dim = 1).  No doublings expected.
    assert all(m == history[0] for m in history), (
        f"healthy Euclidean(1) fit should not trigger adaptive doubling; "
        f"saw history {history}"
    )


def test_adaptive_maxinner_doubles_when_window_caps() -> None:
    """The doubling logic fires when ``adaptive_window`` recent iters cap.

    Unit-tests the policy by hand-driving ``_maybe_double_maxinner`` rather
    than running an optimizer to the stuck state.
    """

    opt = LoggingTrustRegions(
        adaptive_maxinner=True,
        adaptive_window=5,
        adaptive_threshold=0.6,
        adaptive_ceiling=float("inf"),
    )
    opt._adaptive_starting_maxinner = 4
    opt._current_maxinner = 4

    # 5 consecutive iterations each hitting numit == maxinner == 4.
    for _ in range(5):
        opt._numit_history.append(4)
        opt._maxinner_history.append(4)
    opt._maybe_double_maxinner(4)
    assert opt._current_maxinner == 8, (
        f"after 5/5 cap hits the policy should double 4 -> 8; "
        f"got {opt._current_maxinner}"
    )


def test_adaptive_maxinner_holds_when_window_short() -> None:
    """The policy waits for a full window before evaluating cap-hit rate."""

    opt = LoggingTrustRegions(
        adaptive_maxinner=True,
        adaptive_window=5,
        adaptive_ceiling=float("inf"),
    )
    opt._adaptive_starting_maxinner = 4
    opt._current_maxinner = 4

    # 4 cap hits but window is 5; should not yet trigger.
    for _ in range(4):
        opt._numit_history.append(4)
        opt._maxinner_history.append(4)
    opt._maybe_double_maxinner(4)
    assert opt._current_maxinner == 4, (
        f"adaptive should not act with fewer than window entries; "
        f"got {opt._current_maxinner}"
    )


def test_adaptive_maxinner_respects_ceiling() -> None:
    """Doublings stop at ``adaptive_ceiling``."""

    opt = LoggingTrustRegions(
        adaptive_maxinner=True,
        adaptive_window=3,
        adaptive_threshold=0.5,
        adaptive_ceiling=10,
    )
    opt._adaptive_starting_maxinner = 4
    opt._current_maxinner = 8  # already past one doubling

    for _ in range(3):
        opt._numit_history.append(8)
        opt._maxinner_history.append(8)
    opt._maybe_double_maxinner(8)
    # Proposed 16, capped at 10.
    assert opt._current_maxinner == 10


def test_adaptive_maxinner_default_ceiling_is_8x_starting() -> None:
    """Unspecified ``adaptive_ceiling`` resolves to 8 * starting_maxinner."""

    opt = LoggingTrustRegions(
        adaptive_maxinner=True,
        adaptive_window=3,
        adaptive_threshold=0.5,
        # adaptive_ceiling left at default (None)
    )
    opt._adaptive_starting_maxinner = 4  # 8x = 32
    opt._current_maxinner = 4

    # Drive the policy until it saturates.
    for _ in range(20):
        opt._numit_history.append(opt._current_maxinner)
        opt._maxinner_history.append(opt._current_maxinner)
        opt._maybe_double_maxinner(opt._current_maxinner)

    # Final value must be at most 8 * starting_maxinner = 32.
    assert opt._current_maxinner <= 32
    # And -- having had ample window to ratchet -- should reach the ceiling.
    assert opt._current_maxinner == 32


def test_adaptive_maxinner_no_action_when_under_threshold() -> None:
    """Cap-hit fraction below threshold leaves maxinner alone."""

    opt = LoggingTrustRegions(
        adaptive_maxinner=True,
        adaptive_window=5,
        adaptive_threshold=0.6,
        adaptive_ceiling=float("inf"),
    )
    opt._adaptive_starting_maxinner = 4
    opt._current_maxinner = 4

    # 2 cap hits out of 5 = 0.4 < 0.6.
    opt._numit_history.extend([4, 4, 1, 1, 1])
    opt._maxinner_history.extend([4, 4, 4, 4, 4])
    opt._maybe_double_maxinner(4)
    assert opt._current_maxinner == 4


def test_adaptive_maxinner_off_does_not_log_history() -> None:
    """``adaptive_maxinner=False`` (default) keeps the log key set unchanged."""

    data = jnp.array([1.0, 2.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    # Default: adaptive_maxinner not passed -> off.
    result = gmm.estimate()
    log = result.optimizer_report["log"]
    assert "maxinner_history" not in log
    assert "numit_history" not in log


def test_adaptive_maxinner_uses_strict_predicate() -> None:
    """Cap-hit predicate is ``numit >= maxinner``, not the stop code.

    Pymanopt pre-assumes ``stop_code = MAX_INNER_ITER`` and only
    overwrites on early termination, so the stop code falsely reads
    cap-hit when the inner loop body never runs (``numit == 0``).  The
    adaptive policy must use the strict ``numit >= maxinner`` predicate
    to avoid doubling on those false positives.
    """

    opt = LoggingTrustRegions(
        adaptive_maxinner=True,
        adaptive_window=5,
        adaptive_threshold=0.6,
        adaptive_ceiling=float("inf"),
    )
    opt._adaptive_starting_maxinner = 4
    opt._current_maxinner = 4

    # Every iteration has stop_code = MAX_INNER_ITER (pymanopt's
    # pre-assumption) but numit == 0 (loop body never ran).  Strict
    # predicate says 0 < 4 -> no cap hits -> no doubling.
    opt._numit_history.extend([0, 0, 0, 0, 0])
    opt._maxinner_history.extend([4, 4, 4, 4, 4])
    # Simulate the stop-code Counter saying "5 cap hits" -- we ignore
    # it in the predicate.
    from pymanopt.optimizers import TrustRegions

    opt._inner_stop_counts[TrustRegions.MAX_INNER_ITER] = 5

    opt._maybe_double_maxinner(4)
    assert (
        opt._current_maxinner == 4
    ), "strict predicate should not count num_inner=0 false positives"
