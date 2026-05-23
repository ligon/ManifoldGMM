"""Regression tests: ``GMM.estimate(optimizer_kwargs=...)`` actually
forwards both ``__init__``- and ``run()``-flavoured keys.

The partition logic lives in ``GMM._split_optimizer_kwargs`` and
``GMM._resolve_optimizer``.  A regression that silently swallowed any
of these keys would degrade tunability without any other signal --
fits would just stop responding to the knobs documented on
``GMM.estimate``.

Two paths are covered:

1. ``__init__``-flavoured kwarg (``min_gradient_norm``) lands on the
   constructed ``LoggingTrustRegions`` and shows up in the resulting
   optimizer's attribute.
2. ``run()``-flavoured kwarg (``maxinner``) is honoured at run time
   -- we wrap the default optimizer in a sentinel that captures the
   ``maxinner`` keyword on ``.run`` and assert the captured value.
3. The opt-in ``adaptive_maxinner`` flag populates the
   ``maxinner_history`` log key (a side-effect that only fires when
   the adaptive policy is active in
   :class:`LoggingTrustRegions._truncated_conjugate_gradient`).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.optimizers import LoggingTrustRegions
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


def _build_gmm() -> GMM:
    data = jnp.array([1.0, 2.0, 3.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=Manifold.from_pymanopt(PymanoptEuclidean(1)),
        backend="jax",
        parameter_labels=["theta"],
    )
    return GMM(restriction, initial_point=jnp.array([0.0]))


def test_init_kwarg_lands_on_optimizer() -> None:
    """``min_gradient_norm`` is a ``TrustRegions.__init__`` kwarg.  It
    must be set on the constructed optimizer."""

    gmm = _build_gmm()
    custom = 3.14e-5
    optimizer_seen: list[LoggingTrustRegions] = []

    class _Capturing(LoggingTrustRegions):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            optimizer_seen.append(self)

    gmm._optimizer = _Capturing
    gmm.estimate(optimizer_kwargs={"min_gradient_norm": custom})

    assert len(optimizer_seen) >= 1
    assert optimizer_seen[0]._min_gradient_norm == custom


def test_run_kwarg_reaches_optimizer_run() -> None:
    """``maxinner`` is a ``TrustRegions.run`` kwarg.  It must be
    captured by ``run()`` rather than silently dropped on the floor."""

    gmm = _build_gmm()
    captured: dict[str, Any] = {}

    class _CapturingRun(LoggingTrustRegions):
        def run(self, problem: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return super().run(problem, **kwargs)

    gmm._optimizer = _CapturingRun
    gmm.estimate(optimizer_kwargs={"maxinner": 7})

    assert captured.get("maxinner") == 7


def test_adaptive_maxinner_populates_history_log() -> None:
    """The opt-in adaptive policy only writes ``maxinner_history`` /
    ``numit_history`` when active."""

    gmm = _build_gmm()
    res = gmm.estimate(optimizer_kwargs={"adaptive_maxinner": True})

    log = res.optimizer_report.get("log") or {}
    assert "maxinner_history" in log
    assert "numit_history" in log
    assert len(log["maxinner_history"]) == len(log["numit_history"])
