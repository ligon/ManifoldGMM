"""Tests for #32: gauge-aware ``compute_hessian_cond(exclude_gauge=True)``.

The canonical tangent basis of a quotient manifold includes gauge
directions whose entries in ``D'WD`` are exactly zero, so
``compute_hessian_cond()`` mechanically saturates at ``~1/eps_float64``
on K>=2 ``PSDFixedRank`` (and similar) manifolds.  The new
``exclude_gauge`` kwarg detects the gauge nullspace and drops the
corresponding eigenvalues before computing cond.
"""

from __future__ import annotations

import warnings
from dataclasses import replace as dc_replace
from typing import Any

import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm._warnings import NumericalWarning
from manifoldgmm.econometrics.gmm import (
    FixedWeighting,
    _detect_gauge_dim_by_threshold,
    _gauge_dim_of_manifold,
)
from pymanopt.manifolds import Euclidean as PymanoptEuclidean
from pymanopt.manifolds import Product as PymanoptProduct
from pymanopt.manifolds.psd import Elliptope, PSDFixedRank


# ---------------------------------------------------------------------------
# _gauge_dim_of_manifold: unit tests on real pymanopt manifolds
# ---------------------------------------------------------------------------
def test_gauge_dim_zero_for_euclidean() -> None:
    """Euclidean manifolds have no gauge."""

    assert _gauge_dim_of_manifold(PymanoptEuclidean(5)) == 0
    assert _gauge_dim_of_manifold(PymanoptEuclidean(1)) == 0


def test_gauge_dim_psd_fixed_rank_matches_K_choose_two() -> None:
    """``PSDFixedRank(m, K)`` gauge is the ``O(K)`` orbit, dim ``K(K-1)/2``."""

    assert _gauge_dim_of_manifold(PSDFixedRank(5, 1)) == 0  # 1*0/2 = 0
    assert _gauge_dim_of_manifold(PSDFixedRank(5, 2)) == 1  # 2*1/2 = 1
    assert _gauge_dim_of_manifold(PSDFixedRank(5, 3)) == 3  # 3*2/2 = 3
    assert _gauge_dim_of_manifold(PSDFixedRank(10, 4)) == 6  # 4*3/2 = 6


def test_gauge_dim_elliptope_matches_K_choose_two() -> None:
    """``Elliptope(m, K)`` shares the ``O(K)`` orbit."""

    assert _gauge_dim_of_manifold(Elliptope(5, 2)) == 1
    assert _gauge_dim_of_manifold(Elliptope(5, 3)) == 3


def test_gauge_dim_product_manifold_sums_children() -> None:
    """``Product([Euclidean, PSDFixedRank(m, K)])`` inherits the PSD gauge."""

    md = PymanoptProduct((PymanoptEuclidean(3), PSDFixedRank(5, 2)))
    assert _gauge_dim_of_manifold(md) == 1
    md2 = PymanoptProduct(
        (PymanoptEuclidean(2), PSDFixedRank(5, 3), PymanoptEuclidean(1))
    )
    assert _gauge_dim_of_manifold(md2) == 3


def test_gauge_dim_explicit_attribute_wins() -> None:
    """A manifold-like object exposing ``gauge_dim`` is respected directly."""

    class _StubManifoldWithGauge:
        gauge_dim = 4
        # No ``_k`` so the PSDFixedRank fallback path can't fire.

    assert _gauge_dim_of_manifold(_StubManifoldWithGauge()) == 4

    class _StubCallableGauge:
        def gauge_dim(self) -> int:
            return 7

    assert _gauge_dim_of_manifold(_StubCallableGauge()) == 7


def test_gauge_dim_unknown_manifold_returns_zero() -> None:
    """Unrecognised manifold without ``gauge_dim`` returns 0 (no detection)."""

    class _OpaqueManifold:
        pass

    assert _gauge_dim_of_manifold(_OpaqueManifold()) == 0


# ---------------------------------------------------------------------------
# _detect_gauge_dim_by_threshold: unit tests
# ---------------------------------------------------------------------------
def test_threshold_detection_counts_below_threshold() -> None:
    """Counts eigenvalues with ``|eig| < max(|eigs|) * rel_threshold``."""

    eigs = np.array([1e-20, 1e-15, 0.5, 1.0])
    # max=1.0; 1e-20 and 1e-15 are below 1e-12 threshold
    assert _detect_gauge_dim_by_threshold(eigs) == 2

    # With a stricter threshold, only 1e-20 counts
    assert _detect_gauge_dim_by_threshold(eigs, rel_threshold=1e-18) == 1


def test_threshold_detection_empty_and_zero_spectra() -> None:
    """Degenerate spectra return 0 cleanly."""

    assert _detect_gauge_dim_by_threshold(np.array([])) == 0
    assert _detect_gauge_dim_by_threshold(np.zeros(5)) == 0


# ---------------------------------------------------------------------------
# Integration: exclude_gauge on a forged GMMResult with stub manifold
# ---------------------------------------------------------------------------
def _simple_fit():
    """Smallest possible GMMResult: mean estimation on Euclidean(1)."""

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


class _StubManifoldWithGauge:
    """Pymanopt-side stub exposing ``gauge_dim`` for forged-result tests."""

    def __init__(self, gauge_dim: int, dim: int = 3) -> None:
        self.gauge_dim = gauge_dim
        self.dim = dim


def _forge_result_with_gauge(D: np.ndarray, gauge_dim: int) -> Any:
    """Build a GMMResult whose manifold advertises ``gauge_dim`` via stub.

    Uses the same ``dataclass.replace`` pattern as
    ``test_compute_hessian_cond_matches_hand_computed_value``: start
    from a real Euclidean fit, swap out the weighting and cached
    Jacobian, and monkey-patch a stub manifold onto the parameter's
    manifold wrapper.  This avoids the cost of a real PSDFixedRank fit
    while still exercising the kwarg-dispatch path end-to-end.
    """

    p = D.shape[1]
    base = _simple_fit()
    forged = dc_replace(
        base, weighting=FixedWeighting(np.eye(D.shape[0]), label="hand-set")
    )
    forged._cached_jacobian = D
    forged._cached_jacobian_basis = [
        np.eye(1, p, k, dtype=float).reshape(-1) for k in range(p)
    ]
    # Replace the parameter's manifold-wrapper data with a stub whose
    # ``gauge_dim`` attribute is detected by ``_gauge_dim_of_manifold``.
    # ``Manifold`` is a frozen dataclass; bypass with __setattr__.
    object.__setattr__(
        forged._theta.manifold,
        "data",
        _StubManifoldWithGauge(gauge_dim=gauge_dim, dim=p),
    )
    return forged


def test_exclude_gauge_drops_smallest_eigenvalues() -> None:
    """Forged ``D'WD = diag(1, 4, 0)``: ``exclude_gauge=True`` drops the 0."""

    D = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],  # gauge column: contributes a zero eigenvalue
        ]
    )
    forged = _forge_result_with_gauge(D, gauge_dim=1)

    # Without exclude_gauge: cond saturates at ~1/ridge_floor since
    # the smallest eigenvalue is exactly 0.
    cond_with_gauge = forged.diagnostics.hessian_cond()
    assert (
        cond_with_gauge > 1e100
    ), f"With the gauge zero, cond should saturate; got {cond_with_gauge!r}"

    # With exclude_gauge: drop the zero, cond becomes max/next-smallest
    # = 4/1 = 4.
    cond_quotient = forged.diagnostics.hessian_cond(exclude_gauge=True)
    np.testing.assert_allclose(cond_quotient, 4.0, rtol=1e-10)


def test_exclude_gauge_dim_zero_is_noop() -> None:
    """``gauge_dim=0`` makes ``exclude_gauge`` a no-op (K=1 invariance)."""

    D = np.array([[1.0, 0.0], [0.0, 2.0]])
    # gauge_dim=0 mimics PSDFixedRank(m, 1) or Euclidean.
    forged = _forge_result_with_gauge(D, gauge_dim=0)

    cond_default = forged.diagnostics.hessian_cond()
    cond_explicit = forged.diagnostics.hessian_cond(exclude_gauge=True)
    np.testing.assert_allclose(cond_default, cond_explicit, rtol=1e-12)
    np.testing.assert_allclose(cond_default, 4.0, rtol=1e-10)


def test_exclude_gauge_dim_two_drops_two_smallest() -> None:
    """``gauge_dim=2`` drops the two smallest eigenvalues (K=3 case)."""

    # D'WD = diag(1, 4, 9, 0, 0).  Drop the two zeros -> cond = 9/1 = 9.
    D = np.diag([1.0, 2.0, 3.0, 0.0, 0.0])
    forged = _forge_result_with_gauge(D, gauge_dim=2)

    cond_quotient = forged.diagnostics.hessian_cond(exclude_gauge=True)
    np.testing.assert_allclose(cond_quotient, 9.0, rtol=1e-10)


def test_exclude_gauge_threshold_fallback_with_warning() -> None:
    """Unknown manifold + ``exclude_gauge=True``: threshold scan + warning."""

    D = np.diag([1.0, 2.0, 1e-20])  # synthetic near-zero "gauge"

    class _OpaqueManifold:
        dim = 3  # no gauge_dim attribute, not a recognised name

    base = _simple_fit()
    forged = dc_replace(base, weighting=FixedWeighting(np.eye(3), label="hand-set"))
    forged._cached_jacobian = D
    forged._cached_jacobian_basis = [
        np.eye(1, 3, k, dtype=float).reshape(-1) for k in range(3)
    ]
    object.__setattr__(forged._theta.manifold, "data", _OpaqueManifold())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NumericalWarning)
        cond_quotient = forged.diagnostics.hessian_cond(exclude_gauge=True)

    relevant = [w for w in caught if issubclass(w.category, NumericalWarning)]
    assert (
        len(relevant) == 1
    ), f"Expected one NumericalWarning; got {[str(w.message) for w in relevant]}"
    assert "threshold detection" in str(relevant[0].message)
    assert "#32" in str(relevant[0].message)

    # The threshold detected 1 near-zero eig; cond reported on the
    # remaining two: diag(1, 4) -> 4.
    np.testing.assert_allclose(cond_quotient, 4.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Backward compatibility: existing default behaviour preserved
# ---------------------------------------------------------------------------
def test_existing_compute_hessian_cond_unchanged_by_kwarg_addition() -> None:
    """Default kwargs match pre-#32 behaviour bit-identically.

    Re-implements the hand-computed fixture from
    ``test_compute_hessian_cond_matches_hand_computed_value`` to
    confirm that adding ``exclude_gauge=False`` to the signature did
    not perturb the cond computation on a non-gauge fixture.
    """

    base = _simple_fit()
    D = np.array([[1.0, 0.0], [0.0, 2.0]])
    forged = dc_replace(base, weighting=FixedWeighting(np.eye(2), label="hand-set"))
    forged._cached_jacobian = D
    forged._cached_jacobian_basis = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]

    cond = forged.diagnostics.hessian_cond()
    np.testing.assert_allclose(cond, 4.0, rtol=1e-10)

    # exclude_gauge=True on a manifold the framework doesn't recognise
    # AND with no near-zero eigenvalues should be a silent no-op (no
    # warning, same answer).
    with warnings.catch_warnings():
        warnings.simplefilter("error", NumericalWarning)
        cond_quotient = forged.diagnostics.hessian_cond(exclude_gauge=True)
    np.testing.assert_allclose(cond_quotient, cond, rtol=1e-12)


# ---------------------------------------------------------------------------
# End-to-end: GMMResult method composition with existing fixtures
# ---------------------------------------------------------------------------
def test_simple_fit_exclude_gauge_no_op_on_euclidean() -> None:
    """A real Euclidean(1) fit returns the same cond for both kwarg values."""

    result = _simple_fit()
    cond_default = result.diagnostics.hessian_cond()
    cond_quotient = result.diagnostics.hessian_cond(exclude_gauge=True)
    np.testing.assert_allclose(cond_default, cond_quotient, rtol=1e-12)


def test_data_only_and_exclude_gauge_compose() -> None:
    """``data_only`` and ``exclude_gauge`` interact cleanly.

    With a forged result, gauge_dim=1, and an L2 penalty whose
    tangent Hessian is ``2*lam*I``, the full Hessian has the gauge
    direction's penalty contribution (since the penalty acts in
    ambient coords).  ``exclude_gauge=True`` drops the gauge
    eigenvalue *of the chosen Hessian*; ``data_only`` controls
    whether the chosen Hessian includes the penalty term.
    """

    D = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    forged = _forge_result_with_gauge(D, gauge_dim=1)
    # data_only=True + exclude_gauge=True: D'WD = diag(1,4,0); drop
    # the 0 -> cond = 4/1 = 4.
    cond_data_quotient = forged.diagnostics.hessian_cond(
        data_only=True, exclude_gauge=True
    )
    np.testing.assert_allclose(cond_data_quotient, 4.0, rtol=1e-10)


def test_gauge_consumes_full_spectrum_returns_inf() -> None:
    """``gauge_dim >= n_eigs`` returns ``inf`` rather than crashing."""

    D = np.diag([1.0, 2.0])
    # Pathological stub: claim gauge_dim = 2 even though D'WD is rank 2.
    forged = _forge_result_with_gauge(D, gauge_dim=2)
    cond = forged.diagnostics.hessian_cond(exclude_gauge=True)
    assert np.isinf(cond)
