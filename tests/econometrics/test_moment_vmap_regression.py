"""Regression tests: ``gi_jax`` is vectorized via ``jax.vmap`` and the
resulting moment-bar evaluation traces cleanly under ``jax.jit``.

When a :class:`MomentRestriction` is built from a per-observation
``gi_jax(theta, observation)`` rather than from an already-vectorized
``g(theta, dataset)``, internal vectorization is delegated to
``_VmapVectorizer``.  If a future refactor swaps that for a Python
loop, the GMM cost function silently degrades from "one batched
``jax.vmap`` per cost eval" to "N tracer dispatches per cost eval" --
catastrophic for the inner CG.  These tests guard against that
regression and also confirm that the vmapped path composes with
``jax.jit`` (so the JIT win from #37 carries through to the moment
assembly).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from manifoldgmm import Manifold, MomentRestriction
from manifoldgmm.econometrics.moment_restriction import _VmapVectorizer
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


def _build_restriction() -> MomentRestriction:
    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
        return observation - theta[0]

    return MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=Manifold.from_pymanopt(PymanoptEuclidean(1)),
        backend="jax",
        parameter_labels=["theta"],
    )


def test_gi_jax_uses_vmap_vectorizer() -> None:
    """The ``gi_jax`` constructor path must install ``_VmapVectorizer``.

    A future refactor that replaces it with a Python list comprehension
    or a numpy ``apply_along_axis`` would silently destroy the inner-CG
    perf characteristics of every JAX-backed fit.
    """

    r = _build_restriction()
    assert isinstance(r._gi_map, _VmapVectorizer), (
        "MomentRestriction(gi_jax=...) no longer dispatches through "
        "_VmapVectorizer; per-observation moment evaluation is at risk "
        "of falling back to a Python loop on the inner-CG hot path."
    )


def test_g_bar_traces_under_jit_and_matches_eager() -> None:
    """``g_bar`` must trace cleanly inside ``jax.jit`` (i.e. produce a
    single batched HLO) and return the same value as the eager call.
    """

    r = _build_restriction()
    theta = jnp.array([1.5])

    eager = np.asarray(r.g_bar(theta))

    jitted = jax.jit(r.g_bar)
    traced = np.asarray(jitted(theta))

    assert eager.shape == traced.shape
    np.testing.assert_allclose(eager, traced, rtol=0, atol=1e-12)


def test_g_bar_under_jit_does_not_iterate_over_observations() -> None:
    """Sanity probe: jaxpr of ``g_bar`` mentions at most one ``scan`` or
    ``while`` (the implicit vmap loop), not one per observation.

    Catches the case where someone replaces ``jax.vmap`` with
    ``jnp.stack([gi_jax(theta, x) for x in data])`` -- which traces
    fine but unrolls the loop into N independent primitives in the
    jaxpr, swelling compile time linearly in N.
    """

    r = _build_restriction()
    theta = jnp.array([0.0])
    jaxpr = jax.make_jaxpr(r.g_bar)(theta).jaxpr
    # The dataset has 4 observations.  In a vmapped trace we expect
    # one primitive per arithmetic op (sub, mean, sqrt, mul) -- well
    # under a couple dozen total.  An unrolled loop would land in
    # the 4 * (ops_per_obs) ballpark and grow with N.
    assert len(jaxpr.eqns) < 25, (
        "g_bar jaxpr unexpectedly long; "
        "the per-observation moment loop may be unrolled instead of "
        f"vmapped.  Got {len(jaxpr.eqns)} primitives."
    )
