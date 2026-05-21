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
