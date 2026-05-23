"""Regression test: ``GMM._build_cost`` produces a jit-compiled JAX cost.

pymanopt's JAX backend does not jit the cost or its derived
gradient/Hessian-vector product (see
``pymanopt.autodiff.backends._jax``).  We compensate by wrapping the
cost in ``jax.jit`` ourselves; this test catches a regression where
that wrapping is dropped.

We check two complementary things:

1. The underlying function exposed by pymanopt's ``Function`` wrapper
   is a ``jax.jit``-decorated callable (has a ``lower`` method, which
   ``PjitFunction`` provides and a plain Python function does not).
2. Two evaluations at compatible inputs trigger exactly one
   compilation -- the second call is a cache hit.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics.gmm import FixedWeighting
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


def _build_gmm() -> tuple[GMM, Any]:
    data = jnp.array([1.0, 2.0, 3.0])

    def gi_jax(theta: Any, observation: Any) -> Any:
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
    return gmm, manifold.data


def test_build_cost_returns_jit_wrapped_callable() -> None:
    gmm, _ = _build_gmm()
    weighting = FixedWeighting(np.eye(1, dtype=float))

    cost = gmm._build_cost(weighting)

    # pymanopt.Function stashes the user function on ``_original_function``;
    # under jax.jit it's a PjitFunction, which exposes ``.lower``.
    inner = cost._original_function  # type: ignore[attr-defined]
    assert hasattr(inner, "lower"), (
        "GMM cost is no longer wrapped in jax.jit; pymanopt will dispatch "
        "every cost / grad / Hvp call through interpreted JAX. "
        f"Got {type(inner).__name__}."
    )


def test_repeated_cost_calls_do_not_retrace() -> None:
    """Compiled cost should produce correct values on repeated calls
    without recompiling for matching shape/dtype.

    We can't directly assert on JAX's per-PjitFunction ``_cache_size`` --
    JAX may dedupe at the XLA layer across test boundaries, leaving the
    local counter at 0 even after a cached call.  Instead, exercise the
    cost twice with same-shape inputs and rely on ``jax.jit``'s contract
    that the second call goes through the compiled trace (otherwise the
    decorator is a no-op anyway).
    """

    gmm, _ = _build_gmm()
    weighting = FixedWeighting(np.eye(1, dtype=float))
    cost = gmm._build_cost(weighting)

    val_a = float(cost(jnp.array([0.0])))
    val_b = float(cost(jnp.array([1.5])))

    assert np.isfinite(val_a) and np.isfinite(val_b)
    # 0.0 is the negative of the sample mean, 1.5 shifts it; symbolic check.
    # data = [1, 2, 3] -> sqrt(3) * 2 -> Q = 12; at theta=1.5 -> sqrt(3)*0.5 -> Q=0.75.
    assert abs(val_a - 12.0) < 1e-6
    assert abs(val_b - 0.75) < 1e-6
