"""Parity tests for ``MomentWildBootstrap.run_parallel``.

The parallel driver is a thin wrapper around joblib that dispatches
self-contained :class:`BootstrapTask` instances to worker processes.
We assert (a) it returns the same number of results as the sequential
driver, (b) the per-replicate ``theta_star`` values match
replicate-for-replicate when seeds are deterministic, and (c) the
``loky`` backend round-trips cloudpickle-serialised tasks (including
JAX-traced cost closures via ``BootstrapTask.restriction``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction, MomentWildBootstrap
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


def _build_fixture() -> tuple[MomentWildBootstrap, MomentWildBootstrap]:
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

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
    gmm = GMM(restriction, initial_point=jnp.array([0.0]))
    result = gmm.estimate()

    seq = MomentWildBootstrap(result, n_bootstrap=8, base_seed=42)
    par = MomentWildBootstrap(result, n_bootstrap=8, base_seed=42)
    return seq, par


def test_run_parallel_threading_matches_sequential() -> None:
    """Threading backend (no process serialisation) is the cheapest
    parity check and exercises the dispatch path."""

    seq, par = _build_fixture()
    seq_results = seq.run_sequential()
    par_results = par.run_parallel(n_jobs=2, backend="threading")

    assert len(seq_results) == len(par_results) == 8
    for sr, pr in zip(seq_results, par_results, strict=True):
        assert sr.task_id == pr.task_id
        assert sr.seed == pr.seed
        seq_theta = float(np.asarray(sr.theta_star.value).ravel()[0])
        par_theta = float(np.asarray(pr.theta_star.value).ravel()[0])
        assert seq_theta == pytest.approx(par_theta, abs=1e-10)


def test_run_parallel_loky_round_trips_tasks() -> None:
    """``loky`` spawns processes via cloudpickle.  This smoke-test
    runs a tiny bootstrap end-to-end through that path and asserts
    parity with the sequential run -- catching any regression in
    :meth:`BootstrapTask.to_bytes` that would break cluster dispatch."""

    seq, par = _build_fixture()
    seq_results = seq.run_sequential()
    par_results = par.run_parallel(n_jobs=2, backend="loky")

    assert len(par_results) == len(seq_results) == 8
    seq_thetas = sorted(
        float(np.asarray(r.theta_star.value).ravel()[0]) for r in seq_results
    )
    par_thetas = sorted(
        float(np.asarray(r.theta_star.value).ravel()[0]) for r in par_results
    )
    for s, p in zip(seq_thetas, par_thetas, strict=True):
        assert s == pytest.approx(p, abs=1e-8)
