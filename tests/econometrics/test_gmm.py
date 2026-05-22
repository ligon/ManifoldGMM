from __future__ import annotations

import pickle
import types
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from datamat import DataVec
from manifoldgmm import GMM, Manifold, ManifoldPoint, MomentRestriction
from manifoldgmm.econometrics.gmm import GMMResult
from pymanopt.manifolds import Euclidean as PymanoptEuclidean
from pymanopt.manifolds import Product as PymanoptProduct
from pymanopt.optimizers.optimizer import Optimizer


def _build_simple_restriction(
    backend: str = "jax",
) -> tuple[MomentRestriction, float]:
    data = jnp.array([1.0, 2.0, 3.0])

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend=backend,
        parameter_labels=["theta"],
    )
    true_mean = float(np.mean(np.asarray(data)))
    return restriction, true_mean


def test_gmm_estimate_matches_sample_mean() -> None:
    restriction, true_mean = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate()

    estimate_point = result.theta
    assert isinstance(estimate_point, ManifoldPoint)
    assert np.allclose(
        np.asarray(estimate_point.value), np.array([true_mean]), atol=1e-8
    )
    labeled = result.theta_labeled
    assert isinstance(labeled, DataVec)
    assert np.allclose(labeled.values, np.array([true_mean]), atol=1e-8)
    assert isinstance(estimate_point + 0.0, DataVec)
    assert np.allclose(np.asarray(result.g_bar), np.zeros_like(result.g_bar), atol=1e-8)
    assert result.degrees_of_freedom == 0


def test_gmm_two_step_sets_flag_and_updates_weighting() -> None:
    restriction, true_mean = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate(two_step=True)

    assert result.two_step is True
    assert result.weighting_info.get("two_step") is True
    assert np.allclose(np.asarray(result.theta.value), np.array([true_mean]), atol=1e-8)


def test_gmm_two_step_iteration_diagnostics() -> None:
    """``two_step=True`` should record exactly one reweighting stage."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate(two_step=True)

    assert result.weighting_info["iterations"] == 1
    assert len(result.weighting_info["theta_path"]) == 1
    assert result.weighting_info["converged"] is True


def test_gmm_weighting_iterations_runs_requested_stages() -> None:
    """``weighting_iterations=k`` runs exactly ``k`` reweightings."""

    restriction, true_mean = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate(weighting_iterations=3)

    assert result.weighting_info["iterations"] == 3
    assert len(result.weighting_info["theta_path"]) == 3
    # Iterated runs should still recover the sample mean on this restriction.
    assert np.allclose(np.asarray(result.theta.value), np.array([true_mean]), atol=1e-8)
    # Successive stages should remain in the same neighbourhood (this
    # restriction has Ω̂ independent of θ once g_bar = 0, so distances after
    # the first reweighting collapse quickly).
    assert result.weighting_info["theta_path"][-1] < 1e-6


def test_gmm_weighting_iterations_converge_terminates_under_tol() -> None:
    """``"converge"`` mode stops once the manifold distance falls under tol."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate(
        weighting_iterations="converge",
        weighting_tol=1e-6,
        max_weighting_iterations=10,
    )

    assert result.weighting_info["converged"] is True
    # Should converge well before the cap on this trivial restriction.
    assert result.weighting_info["iterations"] <= 5
    assert result.weighting_info["theta_path"][-1] < 1e-6


def test_gmm_weighting_iterations_converge_respects_cap(monkeypatch) -> None:
    """When the distance never falls below tol, the loop stops at the cap.

    The simple sample-mean restriction converges in a single reweighting
    (the second-stage θ̂ exactly matches the first), so we monkeypatch the
    distance metric to force a non-converging path and verify the cap
    actually triggers.
    """

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    monkeypatch.setattr(gmm, "_theta_distance", lambda *_args, **_kwargs: 1.0)

    result = gmm.estimate(
        weighting_iterations="converge",
        weighting_tol=1e-30,
        max_weighting_iterations=2,
    )

    assert result.weighting_info["iterations"] == 2
    assert result.weighting_info["converged"] is False
    assert len(result.weighting_info["theta_path"]) == 2


def test_gmm_weighting_iterations_validates_inputs() -> None:
    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    with pytest.raises(TypeError):
        gmm.estimate(weighting_iterations=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        gmm.estimate(weighting_iterations=-1)


def test_gmm_iterated_picks_up_cluster_aware_omega() -> None:
    """Iterated GMM threads ``with_clusters`` through to its reweighting.

    Build a four-observation single-moment restriction in which the
    cluster pattern ``[0, 0, 1, 1]`` lumps adjacent residuals together --
    their within-cluster sums no longer cancel, so the cluster-robust Ω̂
    differs from the i.i.d. version, and so the iterated weighting matrix
    stored on ``GMMResult.weighting`` differs as well.
    """

    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
    )

    gmm_iid = GMM(restriction, initial_point=jnp.array([0.0]))
    iid_result = gmm_iid.estimate(weighting_iterations=2)

    clustered = restriction.with_clusters(np.array([0, 0, 1, 1]))
    gmm_clust = GMM(clustered, initial_point=jnp.array([0.0]))
    clust_result = gmm_clust.estimate(weighting_iterations=2)

    iid_weighting: Any = iid_result.weighting
    clust_weighting: Any = clust_result.weighting
    assert iid_weighting is not None and clust_weighting is not None
    iid_W = np.asarray(iid_weighting.matrix(iid_result.theta))
    clust_W = np.asarray(clust_weighting.matrix(clust_result.theta))

    # Both runs target the sample mean (2.5) but with different weighting
    # matrices; we should still recover the optimum, yet the matrices the
    # iterated loop installed should not agree.
    assert np.isclose(float(np.asarray(iid_result.theta.value)[0]), 2.5, atol=1e-6)
    assert np.isclose(float(np.asarray(clust_result.theta.value)[0]), 2.5, atol=1e-6)
    assert not np.allclose(iid_W, clust_W)


def test_exposed_helpers_match_restriction_evaluations() -> None:
    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    theta = jnp.array([1.5])
    np.testing.assert_allclose(
        np.asarray(gmm.g_bar(theta)), np.asarray(restriction.g_bar(theta))
    )
    np.testing.assert_allclose(
        np.asarray(gmm.gN(theta)), np.asarray(restriction.gN(theta))
    )


def test_gmm_handles_product_manifold_initial_points() -> None:
    data = jnp.array([[1.0, 2.0], [1.5, 1.8]])

    def gi_jax(theta, observation):
        mu, alpha = theta
        return jnp.array([observation[0] - mu[0], observation[1] - alpha[0]])

    manifold = Manifold.from_pymanopt(
        PymanoptProduct((PymanoptEuclidean(1), PymanoptEuclidean(1)))
    )
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
    )

    true_params = (
        jnp.array([jnp.mean(data[:, 0])]),
        jnp.array([jnp.mean(data[:, 1])]),
    )

    estimator = GMM(
        restriction,
        weighting=np.eye(2),
        initial_point=(jnp.zeros(1), jnp.zeros(1)),
    )

    result = estimator.estimate()
    theta_hat = result.theta_array
    np.testing.assert_allclose(
        np.asarray(theta_hat[0]), np.asarray(true_params[0]), atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(theta_hat[1]), np.asarray(true_params[1]), atol=1e-8
    )


def test_default_initial_point_uses_manifold_random_point(monkeypatch) -> None:
    restriction, _ = _build_simple_restriction()
    sentinel = jnp.array([3.5])

    assert restriction.manifold is not None

    monkeypatch.setattr(
        restriction.manifold.data,
        "random_point",
        types.MethodType(lambda self: sentinel, restriction.manifold.data),
    )

    gmm = GMM(restriction)
    theta0 = gmm._default_initial_point()

    np.testing.assert_allclose(np.asarray(theta0), np.asarray(sentinel))


def test_default_initial_point_falls_back_to_noise(monkeypatch) -> None:
    data = np.array([1.0, 2.0, 3.0])

    def g_map(theta, dataset):
        return dataset - theta

    restriction = MomentRestriction(g=g_map, data=data, backend="numpy")
    restriction.g_bar(np.array([0.0]))  # populate parameter metadata

    class DummyRNG:
        def normal(self, *, loc: float = 0.0, scale: float = 1.0, size: int | None):
            assert loc == 0.0
            assert size is not None
            return np.full(size, scale * 0.5)

    monkeypatch.setattr(np.random, "default_rng", lambda: DummyRNG())

    gmm = GMM(restriction)
    theta0 = gmm._default_initial_point()

    assert isinstance(theta0, np.ndarray)
    assert theta0.shape == (1,)
    np.testing.assert_allclose(theta0, np.full((1,), 0.5e-3))


def test_gmm_estimate_passes_verbose_flag_to_optimizer() -> None:
    restriction, _ = _build_simple_restriction()

    class RecordingOptimizer(Optimizer):
        last_kwargs: dict[str, Any] | None = None

        def __init__(self, **kwargs: Any) -> None:
            type(self).last_kwargs = dict(kwargs)
            super().__init__(**kwargs)

        def run(self, problem: Any, *, initial_point: Any) -> Any:
            return types.SimpleNamespace(
                point=initial_point,
                iterations=0,
                converged=True,
                stopping_reason="recording",
            )

    gmm = GMM(
        restriction,
        optimizer=RecordingOptimizer,
        initial_point=jnp.array([0.0]),
    )

    gmm.estimate(verbose=True)

    assert RecordingOptimizer.last_kwargs is not None
    assert RecordingOptimizer.last_kwargs.get("verbosity") == 2


def test_gmm_estimate_updates_preconfigured_optimizer_verbosity() -> None:
    restriction, _ = _build_simple_restriction()

    class RecordingOptimizerInstance(Optimizer):
        def __init__(self) -> None:
            super().__init__()

        def run(self, problem: Any, *, initial_point: Any) -> Any:
            return types.SimpleNamespace(
                point=initial_point,
                iterations=0,
                converged=True,
                stopping_reason="instance",
            )

    optimizer = RecordingOptimizerInstance()
    optimizer.verbosity = 3  # ensure overwritten

    gmm = GMM(
        restriction,
        optimizer=optimizer,
        initial_point=jnp.array([0.0]),
    )

    gmm.estimate(verbose=False)

    assert optimizer.verbosity == 0


# -----------------------------------------------------------------------
# Pickle round-trip (standard library only — no cloudpickle)
# -----------------------------------------------------------------------


def _module_level_gi_jax(theta: Any, observation: Any) -> Any:
    """Module-level moment function — picklable by stdlib pickle."""
    return observation - theta[0]


def _build_restriction_with_module_level_gi() -> tuple[MomentRestriction, float]:
    """Like ``_build_simple_restriction`` but uses a module-level gi_jax."""
    data = jnp.array([1.0, 2.0, 3.0])
    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=_module_level_gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta"],
    )
    true_mean = float(np.mean(np.asarray(data)))
    return restriction, true_mean


def test_gmm_result_pickle_without_cloudpickle(tmp_path, monkeypatch) -> None:
    """GMMResult.to_pickle/from_pickle must work without cloudpickle.

    Uses a module-level gi_jax (the normal case for stdlib pickle).
    This guards against unpicklable internal state (e.g. module
    references on MomentRestriction._xp/_linalg) leaking into the
    serialized result.  The cloudpickle fallback is disabled via
    monkeypatch so only stdlib pickle is exercised.
    """
    import manifoldgmm.econometrics.gmm as gmm_mod

    monkeypatch.setattr(gmm_mod, "cloudpickle", None)

    restriction, true_mean = _build_restriction_with_module_level_gi()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    path = tmp_path / "result.pkl"
    result.to_pickle(path)

    restored = GMMResult.from_pickle(path)
    np.testing.assert_allclose(
        np.asarray(restored.theta.value),
        np.asarray(result.theta.value),
        atol=1e-12,
    )
    assert restored.degrees_of_freedom == result.degrees_of_freedom


def test_gmm_result_pickle_local_gi_needs_cloudpickle(tmp_path, monkeypatch) -> None:
    """A local gi_jax closure cannot be pickled by stdlib pickle alone.

    Verify that to_pickle raises when cloudpickle is absent and the
    moment function is a local closure (the scenario from the original
    bug report).
    """
    import manifoldgmm.econometrics.gmm as gmm_mod

    monkeypatch.setattr(gmm_mod, "cloudpickle", None)

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    import pytest

    with pytest.raises((AttributeError, TypeError)):
        result.to_pickle(tmp_path / "should_fail.pkl")


def test_moment_restriction_pickle_round_trip() -> None:
    """MomentRestriction survives a stdlib pickle round-trip.

    Uses a module-level gi_jax to isolate the test to internal state
    (backend module references, metadata caches) rather than
    user-provided closures.
    """
    restriction, _ = _build_restriction_with_module_level_gi()

    # Ensure metadata is populated before pickling
    theta = jnp.array([1.5])
    _ = restriction.g_bar(theta)

    data = pickle.dumps(restriction, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(data)

    np.testing.assert_allclose(
        np.asarray(restored.g_bar(theta)),
        np.asarray(restriction.g_bar(theta)),
    )


# -----------------------------------------------------------------------
# optimizer_report attribute surface (#10 part 1)
# -----------------------------------------------------------------------


def test_optimizer_report_surfaces_pymanopt_fields() -> None:
    """A real fit populates the actual pymanopt OptimizerResult fields.

    Prior to the part-1 fix on #10, ``_run_stage`` read ``converged`` and
    ``stopping_reason`` via ``getattr`` with a ``None`` fallback.
    Pymanopt's ``OptimizerResult`` defines neither attribute, so both
    fields came back silently ``None`` for every fit -- in particular
    flipping ``BootstrapResult.converged`` permanently to ``False`` (it
    bool-casts the report value).  This test pins the corrected surface.
    """

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    report = result.optimizer_report

    # Canonical key from pymanopt.
    assert "stopping_criterion" in report
    sc = report["stopping_criterion"]
    assert isinstance(sc, str) and sc, "stopping_criterion should be a non-empty string"
    assert "Terminated" in sc, f"unexpected stopping_criterion: {sc!r}"

    # Synthesised convergence flag: this trivial mean-estimation problem
    # converges to the optimum (min grad norm reached), so the flag must
    # be True -- not None and not False.
    assert report["converged"] is True, (
        f"converged should be True for a trivial fit; got {report['converged']!r} "
        f"with stopping_criterion={sc!r}"
    )

    # Pulled-through pymanopt fields all present and sensible.  We're
    # deliberately permissive about the concrete numeric type -- the
    # JAX backend returns ``jax.Array`` for cost and gradient norm
    # while the numpy path returns Python scalars; both are valid.
    assert isinstance(report["iterations"], int) and report["iterations"] >= 0
    assert report["cost"] is not None
    assert np.isfinite(float(report["cost"]))
    assert report["gradient_norm"] is not None
    assert float(report["gradient_norm"]) >= 0.0
    assert "time" in report and float(report["time"]) >= 0.0
    # log is optional (only populated at log_verbosity>=1); just check it
    # exists as a key.
    assert "log" in report

    # The legacy ``stopping_reason`` key is gone -- callers should
    # migrate to ``stopping_criterion``.  Make this explicit so a future
    # accidental reintroduction is caught.
    assert "stopping_reason" not in report


def test_classify_converged_recognises_pymanopt_strings() -> None:
    """The convergence classifier maps the standard pymanopt strings."""

    from manifoldgmm.econometrics.gmm import _classify_converged

    # Tolerance-style stops -> converged
    assert (
        _classify_converged(
            "Terminated - min grad norm reached after 9 iterations, 1.50 seconds."
        )
        is True
    )
    assert (
        _classify_converged(
            "Terminated - min step_size reached after 12 iterations, 0.30 seconds."
        )
        is True
    )

    # Budget-style stops -> not converged
    assert (
        _classify_converged("Terminated - max iterations reached after 60.00 seconds.")
        is False
    )
    assert (
        _classify_converged("Terminated - max time reached after 200 iterations.")
        is False
    )
    assert (
        _classify_converged("Terminated - max cost evals reached after 5.00 seconds.")
        is False
    )

    # Missing or unrecognised -> None (don't pretend to know)
    assert _classify_converged(None) is None
    assert _classify_converged("") is None
    assert _classify_converged("some custom optimizer message") is None


def test_bootstrap_converged_no_longer_permanently_false() -> None:
    """Bootstrap replicates now inherit a real converged flag.

    Pre-fix, ``optimizer_report['converged']`` was always ``None`` so
    ``BootstrapResult.converged`` was always ``False`` (the bootstrap
    bool-casts the report value).  Post-fix, a successful Euclidean(1)
    replicate fit should report ``converged=True``.
    """

    from manifoldgmm.econometrics.bootstrap import BootstrapTask

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    weighting: Any = result.weighting
    assert weighting is not None
    W = np.asarray(weighting.matrix(result.theta)).astype(float)
    task = BootstrapTask(
        restriction=restriction,
        weighting_matrix=W,
        initial_point=result.theta_array,
        seed=0,
        weight_scheme="rademacher",
        optimizer_class=None,
        optimizer_kwargs={},
        task_id=0,
    )
    br = task.run()
    assert br.converged is True, (
        f"Bootstrap replicate should report converged=True for a trivial "
        f"Euclidean(1) fit; got {br.converged!r}"
    )


# -----------------------------------------------------------------------
# LoggingTrustRegions + optimizer_health (#10 PR 1, remaining)
# -----------------------------------------------------------------------


def test_default_optimizer_is_logging_trust_regions() -> None:
    """``GMM.estimate`` should default to ``LoggingTrustRegions``.

    PR #11 fixed the report fields; this remaining piece of #10 PR 1
    wires the logging-aware optimizer in as the default so
    ``GMMResult.optimizer_health`` has its telemetry without callers
    having to opt in.
    """

    from manifoldgmm.optimizers import LoggingTrustRegions
    from pymanopt.optimizers import TrustRegions

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    # The default ``estimate`` call should resolve through
    # ``LoggingTrustRegions``; ``optimizer_report`` reflects whichever
    # class actually ran, so we trip an opt-in log_verbosity to make the
    # log shape visible.
    result = gmm.estimate(optimizer_kwargs={"log_verbosity": 1})
    log = result.optimizer_report.get("log") or {}
    iters = log.get("iterations") if isinstance(log, dict) else None
    # The base ``TrustRegions`` never populates these keys; their
    # presence is the signature of the logging-aware subclass running.
    assert iters is not None
    assert "inner_stop_code" in iters
    assert "num_inner" in iters
    # And that subclass is a TrustRegions, so legacy isinstance checks
    # on user code still work.
    assert issubclass(LoggingTrustRegions, TrustRegions)


def test_logging_trust_regions_records_inner_state() -> None:
    """At ``log_verbosity=1`` the per-iteration log carries inner-CG state."""

    from manifoldgmm.optimizers import LoggingTrustRegions

    restriction, _ = _build_simple_restriction()
    gmm = GMM(
        restriction,
        initial_point=jnp.array([0.0]),
        optimizer=LoggingTrustRegions,
    )

    result = gmm.estimate(optimizer_kwargs={"log_verbosity": 1})

    iters = result.optimizer_report["log"]["iterations"]
    # Each per-iter list has length equal to the number of recorded
    # iterations (defaultdict semantics: a key that was never appended
    # is just absent).
    n = len(iters["iteration"])
    assert n >= 1
    assert len(iters["num_inner"]) == n
    assert len(iters["inner_stop_code"]) == n
    assert len(iters["gradient_norm"]) == n
    # Inner stop codes are valid ints in TrustRegions' code range.
    from pymanopt.optimizers import TrustRegions

    valid = {
        TrustRegions.NEGATIVE_CURVATURE,
        TrustRegions.EXCEEDED_TR,
        TrustRegions.REACHED_TARGET_LINEAR,
        TrustRegions.REACHED_TARGET_SUPERLINEAR,
        TrustRegions.MAX_INNER_ITER,
        TrustRegions.MODEL_INCREASED,
    }
    for code in iters["inner_stop_code"]:
        assert code in valid


def test_optimizer_health_returns_none_without_log() -> None:
    """At default ``log_verbosity`` the health fields are all None."""

    from manifoldgmm import OptimizerHealth

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    # log_verbosity defaults to 0 -> log["iterations"] is None.
    result = gmm.estimate()
    health = result.optimizer_health()

    assert isinstance(health, OptimizerHealth)
    assert health.n_iterations is None
    assert health.inner_cap_hit_frac is None
    assert health.tail_grad_slope is None
    assert health.tail_window is None


def test_optimizer_health_populated_under_logging() -> None:
    """At ``log_verbosity=1`` the health fields are populated."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))

    result = gmm.estimate(optimizer_kwargs={"log_verbosity": 1})
    health = result.optimizer_health()

    assert health.n_iterations is not None and health.n_iterations >= 1
    # On this trivial Euclidean(1) fit the inner CG should converge
    # well below the cap on every iteration -- cap-hit frac is 0.
    assert health.inner_cap_hit_frac == 0.0
    # Gradient norm should be falling, so the LS slope of log|grad| is
    # negative.  (Don't pin a tight value -- pymanopt's exact iteration
    # count is implementation-dependent.)
    assert health.tail_grad_slope is not None
    assert health.tail_grad_slope < 0.0
    # tail_window is clamped to the available log length.
    assert health.tail_window is not None
    assert 2 <= health.tail_window <= health.n_iterations


def test_optimizer_health_tail_window_clamps_to_log_length() -> None:
    """``tail_window`` is clamped to the available log length."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate(optimizer_kwargs={"log_verbosity": 1})

    n = result.optimizer_health().n_iterations
    assert n is not None

    # tail_window > n_iterations clamps to n_iterations.  The returned
    # window may still be smaller if the tail contains non-positive
    # gradient values that get masked out (the trivial Euclidean(1) fit
    # can produce a zero gradient on the final iteration), so we test
    # only the upper bound.
    h_large = result.optimizer_health(tail_window=10_000)
    assert h_large.tail_window is not None
    assert h_large.tail_window <= n


def test_optimizer_health_handles_synthetic_cap_hit_log() -> None:
    """``inner_cap_hit_frac`` is computed correctly from a synthetic log.

    Bypass the optimizer: hand-build a ``GMMResult`` (well, the bits the
    method reads) and verify the metric arithmetic.
    """

    from manifoldgmm import OptimizerHealth

    # Build the bare minimum of a GMMResult-shaped object that
    # optimizer_health needs.  Real GMMResult instances have many more
    # fields; we test the helper in isolation here.
    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    base = gmm.estimate(optimizer_kwargs={"log_verbosity": 1})

    # Replace the log with a synthetic one: 10 iters, 4 of them cap-hit
    # (num_inner == maxinner == 20), the rest with num_inner < maxinner.
    # gradient norm 1e0, 1e-1, ..., 1e-9 (clean log-linear decay of
    # slope -ln(10)).
    fake_log = {
        "iterations": {
            "iteration": list(range(1, 11)),
            "num_inner": [20] * 4 + [5] * 6,
            "maxinner": [20] * 10,
            "gradient_norm": [10.0 ** (-i) for i in range(10)],
        }
    }
    object.__setattr__(
        base, "optimizer_report", dict(base.optimizer_report, log=fake_log)
    )

    health = base.optimizer_health(tail_window=10)
    assert isinstance(health, OptimizerHealth)
    assert health.n_iterations == 10
    assert health.inner_cap_hit_frac == pytest.approx(0.4)
    # Tail-window slope of log(10**-i) on i=0..9 is -ln(10) ≈ -2.3026.
    assert health.tail_grad_slope == pytest.approx(-np.log(10.0), rel=1e-6)


# -----------------------------------------------------------------------
# Canonical-Jacobian cache (#4)
# -----------------------------------------------------------------------


def test_canonical_jacobian_caches_result() -> None:
    """canonical_jacobian returns the same ndarray object on repeat calls."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    first = result.canonical_jacobian()
    second = result.canonical_jacobian()
    # Object identity confirms the cache fired; equality alone could be
    # satisfied by recomputing an identical matrix.
    assert first is second


def test_canonical_jacobian_matches_uncached() -> None:
    """The cached matrix equals what restriction.jacobian_matrix returns."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    cached = result.canonical_jacobian()
    basis = restriction.tangent_basis(result.theta_point)
    uncached = restriction.jacobian_matrix(result.theta_point, basis=basis)

    np.testing.assert_allclose(cached, uncached, atol=1e-12)


def test_tangent_covariance_uses_cached_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two tangent_covariance calls trigger jacobian_matrix only once."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    call_count = {"n": 0}
    original = type(restriction).jacobian_matrix

    def counting_jacobian_matrix(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(restriction), "jacobian_matrix", counting_jacobian_matrix)

    _ = result.tangent_covariance()
    _ = result.tangent_covariance()

    assert call_count["n"] == 1, (
        f"jacobian_matrix should be called exactly once across two "
        f"tangent_covariance calls; called {call_count['n']} times"
    )


def test_k_statistic_at_theta_hat_uses_cached_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """k_statistic(theta_0=None) reuses the cached Jacobian from tangent_covariance."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    # Warm the cache via tangent_covariance.
    _ = result.tangent_covariance()

    call_count = {"n": 0}
    original = type(restriction).jacobian_matrix

    def counting_jacobian_matrix(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(restriction), "jacobian_matrix", counting_jacobian_matrix)

    _ = result.k_statistic()  # theta_0=None -> uses cache

    assert call_count["n"] == 0, (
        "k_statistic at theta_hat should reuse the cached Jacobian; "
        f"jacobian_matrix called {call_count['n']} times"
    )


def test_k_statistic_at_theta_0_recomputes_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """k_statistic(theta_0=<other>) bypasses the cache and recomputes."""

    restriction, _ = _build_simple_restriction()
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    # Warm the cache.
    _ = result.canonical_jacobian()

    call_count = {"n": 0}
    original = type(restriction).jacobian_matrix

    def counting_jacobian_matrix(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(restriction), "jacobian_matrix", counting_jacobian_matrix)

    # Use a point that's structurally a different ManifoldPoint instance
    # but happens to coincide with theta_hat numerically -- ``is`` check
    # in k_statistic still misses, so we expect a recomputation.
    theta_0 = ManifoldPoint(result.theta_point.manifold, jnp.array([2.5]))
    _ = result.k_statistic(theta_0=theta_0)

    assert call_count["n"] == 1, (
        "k_statistic at a custom theta_0 should recompute the Jacobian "
        f"once; called {call_count['n']} times"
    )
