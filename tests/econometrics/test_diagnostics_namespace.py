"""Locks in the ``GMMResult.diagnostics`` namespace and the deprecation
contract on the old top-level methods (``optimizer_health``,
``compute_hessian_cond``, ``check_inference_validity``).

The namespace consolidation separates *optimization / numerical-quality
diagnostics* from *inference methods* (``tangent_covariance``,
``wald_test``, ``k_statistic``, ...).  These tests assert:

1. ``result.diagnostics`` returns a :class:`Diagnostics` instance.
2. Each new path returns the expected value.
3. Each deprecated top-level alias still works, emits exactly one
   :class:`DeprecationWarning`, and returns numerically identical
   values to its diagnostics-side counterpart.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics.gmm import Diagnostics, FixedWeighting
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


def _simple_fit():
    """Smallest possible fit: Euclidean(1) mean estimation."""

    data = jnp.array([1.0, 2.0, 3.0, 4.0])

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
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
    )
    return gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )


# ---------------------------------------------------------------------------
# Namespace shape
# ---------------------------------------------------------------------------
def test_diagnostics_property_returns_diagnostics_instance() -> None:
    """``result.diagnostics`` returns a :class:`Diagnostics` wrapping the result."""

    result = _simple_fit()
    diag = result.diagnostics
    assert isinstance(diag, Diagnostics)
    # The wrapper holds a reference to the underlying result.
    assert diag._result is result


def test_diagnostics_methods_present() -> None:
    """The namespace exposes the three expected diagnostics surfaces."""

    diag = _simple_fit().diagnostics
    # Three callables / properties; each callable in its expected shape.
    assert isinstance(diag.optimizer_health, dict)
    assert isinstance(diag.hessian_cond(), float)
    assert isinstance(diag.check_inference_validity(warn=False), dict)


# ---------------------------------------------------------------------------
# Deprecation aliases
# ---------------------------------------------------------------------------
def test_optimizer_health_alias_warns_and_matches() -> None:
    """Old ``result.optimizer_health`` still works; emits one DeprecationWarning."""

    result = _simple_fit()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        old = result.optimizer_health

    relevant = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(relevant) == 1
    assert "result.diagnostics.optimizer_health" in str(relevant[0].message)
    new = result.diagnostics.optimizer_health
    # Same keys, same values (in/out-of order doesn't matter for the
    # dict comparison, but for the cap-hit fields a None-equality check
    # works either way).
    assert old == new


def test_compute_hessian_cond_alias_warns_and_matches() -> None:
    """Old ``result.compute_hessian_cond(...)`` still works; emits DeprecationWarning."""

    result = _simple_fit()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        old = result.compute_hessian_cond()

    relevant = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(relevant) == 1
    assert "result.diagnostics.hessian_cond" in str(relevant[0].message)
    new = result.diagnostics.hessian_cond()
    assert np.isclose(old, new)


def test_check_inference_validity_alias_warns_and_matches() -> None:
    """Old ``result.check_inference_validity(...)`` still works; emits DeprecationWarning."""

    result = _simple_fit()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        old = result.check_inference_validity(warn=False)

    relevant = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(relevant) == 1
    assert "result.diagnostics.check_inference_validity" in str(relevant[0].message)
    new = result.diagnostics.check_inference_validity(warn=False)
    assert old == new


# ---------------------------------------------------------------------------
# Kwarg forwarding via the alias
# ---------------------------------------------------------------------------
def test_compute_hessian_cond_alias_forwards_kwargs() -> None:
    """Deprecated alias forwards ``data_only`` and ``exclude_gauge`` kwargs."""

    result = _simple_fit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_data = result.compute_hessian_cond(data_only=True)
        old_excl = result.compute_hessian_cond(exclude_gauge=True)

    new_data = result.diagnostics.hessian_cond(data_only=True)
    new_excl = result.diagnostics.hessian_cond(exclude_gauge=True)
    assert np.isclose(old_data, new_data)
    assert np.isclose(old_excl, new_excl)
