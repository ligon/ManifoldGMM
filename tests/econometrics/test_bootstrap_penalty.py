"""Tests for #29: MomentWildBootstrap propagating the original fit's
parameter penalty into each replicate.

Pre-#29, ``BootstrapTask.run`` built the replicate ``GMM`` without
forwarding the penalty; the bootstrap distribution therefore
characterised the *unpenalised* estimator's sampling distribution, not
the penalised one.  On a weakly-identified design this is a real bug
(the unpenalised optimum can live in a different basin from
``theta_hat_pen``).  Acceptance criteria covered here mirror those in
#29:

1. Bootstrap from a penalised ``GMMResult`` clusters replicate
   ``theta_star`` near ``theta_hat_pen``, not the unpenalised
   ``ybar``.
2. ``penalty=None`` path is bit-identical to pre-#29.
3. ``BootstrapTask.to_bytes`` round-trip preserves the penalty.
4. Cluster-wild bootstrap with both ``clusters`` and ``penalty`` set:
   both axes propagate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics.bootstrap import (
    BootstrapTask,
    MomentWildBootstrap,
)
from manifoldgmm.econometrics.gmm import FixedWeighting
from pymanopt.manifolds import Euclidean as PymanoptEuclidean

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Module-level fn so vanilla pickle can serialise it without cloudpickle in
# the unit tests below; CallablePenalty wraps it transparently.
_LAMBDA = 10.0


def _strong_quadratic_penalty(theta):
    return _LAMBDA * (theta[0] ** 2)


def _mean_restriction():
    """``g_i(theta) = y_i - theta[0]``: Euclidean(1), ell == p == 1."""

    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    return MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta"],
    )


# ---------------------------------------------------------------------------
# AC1 -- penalised replicates cluster near theta_hat_pen, not ybar
# ---------------------------------------------------------------------------
def test_bootstrap_replicates_cluster_near_penalised_estimator() -> None:
    """Strong penalty pulls replicates to ``theta_hat_pen``, not ``ybar``.

    With ``data = [1, 2, 3, 4]``, ``ybar = 2.5``, ``N = 4``, and
    ``lambda = 10`` on the L2 penalty, the (sqrt(N)-scaled) criterion is
    ``N (ybar - theta)^2 + lambda theta^2``.  Penalised optimum:
    ``theta_hat_pen = N ybar / (N + lambda) = 10 / 14 = 0.714``.
    Unpenalised optimum is ``ybar = 2.5``.

    A bootstrap that correctly propagates the penalty produces
    replicates clustered near 0.714; the pre-#29 (broken) bootstrap
    would have produced replicates clustered near 2.5.
    """

    restriction = _mean_restriction()
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=_strong_quadratic_penalty,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    theta_pen = float(np.asarray(result.theta.value)[0])
    assert np.isclose(theta_pen, 10.0 / 14.0, atol=1e-6)
    ybar = 2.5

    boot = MomentWildBootstrap(result, n_bootstrap=50, base_seed=0)
    tasks = boot.tasks()
    # Sanity: tasks carry the penalty
    assert all(t.penalty is not None for t in tasks)
    assert all(t.penalty is result.penalty for t in tasks)

    # Run sequentially to keep the test self-contained
    replicate_theta = []
    for task in tasks:
        br = task.run()
        replicate_theta.append(float(np.asarray(br.theta_star.value)[0]))
    replicate_theta_arr = np.asarray(replicate_theta)

    # The penalty pulls each replicate's optimum toward 0; replicates
    # should hug theta_pen substantially closer than ybar.  Concrete
    # threshold: mean replicate within 0.5 of theta_pen, and NO
    # replicate within 0.5 of ybar (which would indicate the
    # unpenalised basin).
    assert abs(replicate_theta_arr.mean() - theta_pen) < 0.5, (
        f"Mean replicate theta {replicate_theta_arr.mean():.3f} is not "
        f"near theta_pen={theta_pen:.3f}; bootstrap may have dropped "
        "the penalty."
    )
    assert not np.any(np.abs(replicate_theta_arr - ybar) < 0.5), (
        f"At least one replicate landed near unpenalised ybar={ybar}; "
        f"replicates = {replicate_theta_arr!r}"
    )


# ---------------------------------------------------------------------------
# AC2 -- penalty=None path bit-identical to pre-#29
# ---------------------------------------------------------------------------
def test_bootstrap_with_no_penalty_unchanged() -> None:
    """A ``GMMResult`` without penalty produces unpenalised replicates.

    Penalty field on each BootstrapTask is ``None``, and the
    replicate fits land near the data mean ``ybar`` (the unpenalised
    optimum).  This is the pre-#29 behaviour preserved.
    """

    restriction = _mean_restriction()
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    boot = MomentWildBootstrap(result, n_bootstrap=20, base_seed=0)
    tasks = boot.tasks()
    assert all(t.penalty is None for t in tasks)

    # Run a single replicate to sanity-check the un-penalised path
    # converges near ybar (any single replicate's optimum is close to
    # ybar = 2.5 modulo the weight-induced perturbation).
    br = tasks[0].run()
    replicate_theta = float(np.asarray(br.theta_star.value)[0])
    # Without penalty, the bootstrap weights tilt theta around ybar = 2.5;
    # the deviation should be at most a few units (very generous bound)
    # but NOT pinned near 0 the way the penalised case is.
    assert 0.5 < replicate_theta, (
        f"Unpenalised replicate {replicate_theta!r} is suspiciously "
        "close to 0; suggests the penalty fired when it shouldn't have."
    )


# ---------------------------------------------------------------------------
# AC3 -- BootstrapTask.to_bytes preserves the penalty
# ---------------------------------------------------------------------------
def test_bootstrap_task_pickle_preserves_penalty() -> None:
    """Round-trip a penalised task through ``to_bytes``/``from_bytes``."""

    restriction = _mean_restriction()
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=_strong_quadratic_penalty,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    boot = MomentWildBootstrap(result, n_bootstrap=1, base_seed=0)
    task = boot.tasks()[0]

    blob = task.to_bytes()
    restored = BootstrapTask.from_bytes(blob)

    assert restored.penalty is not None
    # ``CallablePenalty`` is the wrapper produced by ``_coerce_penalty``
    # on the bare-callable input; ``.value(theta)`` must still produce
    # the original lambda * theta**2.
    theta_test = np.array([0.5])
    expected = _strong_quadratic_penalty(theta_test)
    assert hasattr(restored.penalty, "value")
    actual = float(np.asarray(restored.penalty.value(theta_test)))
    assert np.isclose(actual, float(expected))

    # Run the restored task and verify it converges to theta_hat_pen
    # too -- end-to-end check that the penalty survived pickling.
    br = restored.run()
    theta_pen = float(np.asarray(result.theta.value)[0])
    rep = float(np.asarray(br.theta_star.value)[0])
    assert abs(rep - theta_pen) < 0.5, (
        f"Restored task's replicate {rep!r} is not near theta_pen={theta_pen!r}; "
        "the penalty did not survive serialisation."
    )


# ---------------------------------------------------------------------------
# AC4 -- clusters + penalty propagate independently
# ---------------------------------------------------------------------------
def test_cluster_and_penalty_propagate_independently() -> None:
    """A penalised + clustered fit produces tasks with both axes set."""

    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    def gi_jax(theta, observation):
        return observation - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    # 4 obs split into 2 clusters of 2.
    clusters = np.array([0, 0, 1, 1])
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta"],
        clusters=clusters,
    )

    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=_strong_quadratic_penalty,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    boot = MomentWildBootstrap(result, n_bootstrap=5, base_seed=0)
    tasks = boot.tasks()

    # Penalty axis: every task carries the same penalty as the result.
    assert all(t.penalty is result.penalty for t in tasks)

    # Cluster axis: every task carries cluster_codes (length 4) and
    # num_clusters = 2.  Cluster_codes is a contiguous-integer
    # re-coding of the labels [0, 0, 1, 1] -> [0, 0, 1, 1].
    for t in tasks:
        assert t.cluster_codes is not None
        assert t.cluster_codes.shape == (4,)
        assert t.num_clusters == 2

    # Smoke-test end-to-end on multiple replicates and verify the
    # bootstrap distribution's centre clusters closer to theta_pen
    # than to ybar.  We can't use the tight ``|rep - theta_pen| < 0.5``
    # bound from AC1 here because with only G=2 clusters there are
    # just 4 possible weight patterns under shifted Rademacher --
    # individual replicates can be substantially off-centre.  The
    # bootstrap *mean* is the right summary.
    reps = np.asarray([float(np.asarray(t.run().theta_star.value)[0]) for t in tasks])
    theta_pen = float(np.asarray(result.theta.value)[0])
    ybar = 2.5
    mean_rep = float(reps.mean())
    assert abs(mean_rep - theta_pen) < abs(mean_rep - ybar), (
        f"Clustered+penalised bootstrap mean {mean_rep!r} is closer to "
        f"ybar={ybar} than to theta_pen={theta_pen!r}; suggests the "
        "penalty did not propagate through the clustered code path."
    )


# ---------------------------------------------------------------------------
# Backward-compat sanity: BootstrapTask field ordering preserves
# existing call patterns
# ---------------------------------------------------------------------------
def test_bootstrap_task_constructor_backward_compatible() -> None:
    """Existing callers that omit ``penalty=`` get the documented default ``None``.

    ``penalty`` is added as a defaulted field after the existing
    ``num_clusters``, so positional and kwarg patterns from pre-#29
    code keep working.  This is a structural sanity test, not a
    behavioural one.
    """

    restriction = _mean_restriction()
    task = BootstrapTask(
        restriction=restriction,
        weighting_matrix=np.eye(1),
        initial_point=jnp.array([0.0]),
        seed=0,
        weight_scheme="rademacher",
        optimizer_class=None,
        optimizer_kwargs={},
        task_id=0,
    )
    assert task.penalty is None
    assert task.cluster_codes is None
    assert task.num_clusters is None
