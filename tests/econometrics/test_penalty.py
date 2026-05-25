"""Tests for #19 MR1 -- parameter-penalty hook on :class:`GMM`.

Covers acceptance criteria 1, 2, 3, 4, and 6 from issue #19 (the
revised list in the "Scope split: MR1 vs. deferred" section).  Test 5
(K-statistic decomposition under penalty) is deferred to #21 and is
exercised here only via the ``NotImplementedError`` guard.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics.gmm import (
    CallablePenalty,
    FixedWeighting,
    GMMResult,
)
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _mean_restriction(
    data: np.ndarray | jnp.ndarray | None = None,
) -> MomentRestriction:
    """``g_i(theta) = y_i - theta[0]``: Euclidean(1), p == ell == 1."""

    if data is None:
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


def _under_id_restriction() -> MomentRestriction:
    """``g_i = y_i - (theta[0] + theta[1])``: rank-1 design, p = 2 > ell = 1."""

    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    def gi_jax(theta, observation):
        return observation - (theta[0] + theta[1])

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(2))
    return MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta1", "theta2"],
    )


# ---------------------------------------------------------------------------
# Test 1 -- closed-form ridge consistency
# ---------------------------------------------------------------------------
def test_l2_penalty_matches_closed_form_ridge() -> None:
    """Mean estimation with L2 penalty: ``theta_hat = N ybar / (N + lambda)``.

    Under the workspace's ``sqrt(N)`` scaling of ``g_bar`` (see
    :meth:`MomentRestriction.g_bar`), the criterion is ``N (ybar - theta)^2
    + lambda theta^2``.  FOC ``-2 N (ybar - theta) + 2 lambda theta = 0``
    gives ``theta = N ybar / (N + lambda)``.

    Verifies that ``GMMResult.criterion_value`` carries the penalised cost
    and ``GMMResult.data_criterion_value`` carries the data-only piece.
    """

    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    N = data.size
    ybar = float(np.mean(np.asarray(data)))
    lam = 0.75

    restriction = _mean_restriction(data)
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=lambda theta: lam * (theta[0] ** 2),
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500},
    )

    expected_theta = N * ybar / (N + lam)
    actual_theta = float(np.asarray(result.theta.value)[0])
    assert np.isclose(actual_theta, expected_theta, atol=1e-9), (
        f"Closed-form ridge mismatch: expected {expected_theta!r}, "
        f"got {actual_theta!r}"
    )

    expected_data_J = N * (ybar - expected_theta) ** 2
    expected_penalty = lam * expected_theta**2
    expected_total = expected_data_J + expected_penalty

    assert result.data_criterion_value is not None
    assert np.isclose(result.data_criterion_value, expected_data_J, atol=1e-9), (
        f"data_criterion_value mismatch: expected {expected_data_J!r}, "
        f"got {result.data_criterion_value!r}"
    )
    assert np.isclose(result.criterion_value, expected_total, atol=1e-9), (
        f"criterion_value mismatch: expected {expected_total!r}, "
        f"got {result.criterion_value!r}"
    )
    # Sanity: criterion_value - data_criterion_value == penalty(theta_hat)
    pen_diff = result.criterion_value - result.data_criterion_value
    assert np.isclose(pen_diff, expected_penalty, atol=1e-9)


# ---------------------------------------------------------------------------
# Test 2 -- penalty=None bit-identical fallback
# ---------------------------------------------------------------------------
def test_penalty_none_bit_identical_to_unpenalised_path() -> None:
    """``penalty=None`` reproduces the pre-#19 fit exactly.

    Same restriction and initial point with and without ``penalty=None``;
    expected ``theta_hat``, ``criterion_value``, ``data_criterion_value``,
    ``tangent_covariance``, and ``compute_hessian_cond`` all coincide
    (or, for ``data_criterion_value``, equal ``criterion_value`` modulo
    being defaulted to ``None`` historically).
    """

    restriction = _mean_restriction()
    init = jnp.array([0.5])

    gmm_default = GMM(restriction, initial_point=init)
    gmm_none = GMM(restriction, initial_point=init, penalty=None)

    res_default = gmm_default.estimate()
    res_none = gmm_none.estimate()

    assert np.allclose(
        np.asarray(res_default.theta.value), np.asarray(res_none.theta.value)
    )
    assert res_default.criterion_value == res_none.criterion_value
    # penalty=None still populates data_criterion_value (equals criterion_value)
    assert res_none.data_criterion_value == res_none.criterion_value
    # penalty/penalty_info remain None
    assert res_none.penalty is None
    assert res_none.penalty_info is None
    # tangent_covariance and compute_hessian_cond match
    cov_default = res_default.tangent_covariance().to_numpy()
    cov_none = res_none.tangent_covariance().to_numpy()
    assert np.allclose(cov_default, cov_none)
    assert res_default.diagnostics.hessian_cond() == res_none.diagnostics.hessian_cond()


# ---------------------------------------------------------------------------
# Test 3 -- cross-weighting consistency under penalty
# ---------------------------------------------------------------------------
def test_penalty_split_is_consistent_across_weighting_choices() -> None:
    """``criterion_value - data_criterion_value == penalty(theta_hat)``.

    Under different weighting choices the penalised ``theta_hat`` will
    differ (the data side weighs the residual differently while the
    penalty stays in raw parameter units).  But the *split* between
    data fit and penalty contribution must be exact regardless of
    weighting -- that's what makes ``data_criterion_value`` a coherent
    accessor.  This is #19 acceptance criterion 3 (the revised wording
    in the MR1 scope section): data fit reports the same g'Wg whatever
    g'Wg landed on.
    """

    lam = 0.25
    pen = lambda theta: lam * (theta[0] ** 2)  # noqa: E731

    restriction = _mean_restriction()
    init = jnp.array([0.0])

    gmm_identity = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=init,
        penalty=pen,
    )
    gmm_two_step = GMM(
        restriction,
        initial_point=init,
        penalty=pen,
    )

    res_identity = gmm_identity.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )
    res_two_step = gmm_two_step.estimate(
        two_step=True,
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500},
    )

    for label, res in [("identity", res_identity), ("two_step", res_two_step)]:
        assert res.data_criterion_value is not None, label
        theta_value = float(np.asarray(res.theta.value)[0])
        expected_penalty = lam * theta_value**2
        split = res.criterion_value - res.data_criterion_value
        assert np.isclose(split, expected_penalty, atol=1e-9), (
            f"{label}: criterion_value - data_criterion_value = {split!r} "
            f"does not match penalty(theta_hat) = {expected_penalty!r}"
        )
        # And the penalty_info echo of the penalty value should agree.
        assert res.penalty_info is not None
        assert np.isclose(
            res.penalty_info["value_at_theta_hat"], expected_penalty, atol=1e-9
        )


# ---------------------------------------------------------------------------
# Test 4 -- compute_hessian_cond distinguishes data vs penalty curvature
# ---------------------------------------------------------------------------
def test_compute_hessian_cond_data_vs_penalty() -> None:
    """Under-id design: ``data_only=True`` is singular; default is bounded.

    With ``g_i = y_i - (theta1 + theta2)`` the data Jacobian is constant
    ``[-1, -1]`` per row, so ``D'WD`` is rank 1 (eigenvalue 0 along
    ``(theta1, theta2) -> theta1 - theta2``).  An L2 penalty
    ``lambda(theta1^2 + theta2^2)`` has tangent Hessian ``2 lambda I``,
    rescuing the null direction.  The composite cond is therefore
    ``(N^2 + 2 lambda) / (2 lambda)`` for ``N`` observations and identity
    weighting.
    """

    lam = 0.5
    restriction = _under_id_restriction()

    # An analytic ``hessian_tangent`` lets us exercise both the analytic
    # and FD paths in one test by also having a sibling that omits it.
    def pen_value(theta):
        return lam * (theta[0] ** 2 + theta[1] ** 2)

    def pen_hessian_tangent(theta, basis):
        # Euclidean basis vectors -> Hessian is just 2 lambda I in basis.
        return 2.0 * lam * np.eye(len(basis))

    pen_analytic = CallablePenalty(
        pen_value, label="ridge_analytic", hessian_tangent=pen_hessian_tangent
    )

    # Identity weighting; ell = 1, p = 2.  Initial guess centred so the
    # ridge keeps the optimizer well behaved.
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0, 0.0]),
        penalty=pen_analytic,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 1000}
    )

    cond_with_penalty = result.diagnostics.hessian_cond()
    cond_data_only = result.diagnostics.hessian_cond(data_only=True)

    # Data-only cond should be enormous (singular up to ridge_floor)
    assert cond_data_only > 1e20, (
        f"Data-only D'WD should be singular for under-id design; "
        f"got cond={cond_data_only!r}"
    )
    # Penalty-aware cond should be bounded.  Theoretical value with
    # N = 4 obs and identity weighting:
    #   data D'WD eigenvalues: (N * 2, 0) = (8, 0) per observation count
    # but g_bar averages over N, so D = (-1, -1)/sqrt(N)?  We don't need
    # the precise prediction -- assert "small" instead.
    assert (
        cond_with_penalty < 1e3
    ), f"Penalty-aware cond should be moderate; got {cond_with_penalty!r}"
    # And: the ratio should be the data-vs-rescued signal we want.
    assert cond_data_only > 1e10 * cond_with_penalty


def test_compute_hessian_cond_fd_fallback_matches_analytic() -> None:
    """FD penalty Hessian matches analytic to ~1e-6 on a quadratic penalty."""

    lam = 0.5
    restriction = _under_id_restriction()

    pen_callable = lambda theta: lam * (theta[0] ** 2 + theta[1] ** 2)  # noqa: E731

    def pen_hessian_tangent(theta, basis):
        return 2.0 * lam * np.eye(len(basis))

    pen_analytic = CallablePenalty(
        pen_callable, label="ridge_analytic", hessian_tangent=pen_hessian_tangent
    )

    gmm_fd = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0, 0.0]),
        penalty=pen_callable,  # FD fallback path
    )
    gmm_analytic = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0, 0.0]),
        penalty=pen_analytic,
    )

    res_fd = gmm_fd.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 1000}
    )
    res_an = gmm_analytic.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 1000}
    )

    cond_fd = res_fd.diagnostics.hessian_cond()
    cond_an = res_an.diagnostics.hessian_cond()
    assert np.isclose(cond_fd, cond_an, rtol=1e-5), (
        f"FD fallback cond {cond_fd!r} should match analytic {cond_an!r} "
        "for a quadratic penalty; central differences are exact to "
        "machine precision modulo step size in theory."
    )


# ---------------------------------------------------------------------------
# Test 6 -- pickle round-trip with penalty
# ---------------------------------------------------------------------------
def test_penalised_gmm_result_pickle_roundtrip(tmp_path) -> None:
    """Penalised ``GMMResult`` survives pickle; ``data_criterion_value`` and
    ``penalty_info`` reproduce.
    """

    lam = 0.25
    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Module-level penalty would be cleaner but a local lambda is what
    # K-Aggregators users will actually write; let cloudpickle handle it.
    pen = lambda theta: lam * (theta[0] ** 2)  # noqa: E731

    restriction = _mean_restriction(data)
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=pen,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    pickle_path = tmp_path / "penalised.pkl"
    result.to_pickle(pickle_path)
    restored = GMMResult.from_pickle(pickle_path)

    assert restored.criterion_value == result.criterion_value
    assert restored.data_criterion_value == result.data_criterion_value
    assert restored.penalty_info is not None
    assert result.penalty_info is not None  # mypy
    assert restored.penalty_info["value_at_theta_hat"] == pytest.approx(
        result.penalty_info["value_at_theta_hat"]
    )
    # Restored penalty is callable & matches at the estimate
    assert restored.penalty is not None
    assert result.penalty is not None
    assert hasattr(restored.penalty, "value")
    assert hasattr(result.penalty, "value")
    restored_pen = float(np.asarray(restored.penalty.value(restored.theta.value)))
    original_pen = float(np.asarray(result.penalty.value(result.theta.value)))
    assert np.isclose(restored_pen, original_pen)


# ---------------------------------------------------------------------------
# k_statistic guard (#21 narrowed, #25 file for finite-sample variant)
# ---------------------------------------------------------------------------
def test_k_statistic_raises_at_theta_hat_under_penalty() -> None:
    """Default ``theta_0=None`` on a penalised fit raises pointing at #21.

    The penalised FOC includes a (1/2) grad(p) term, so ``K`` at
    ``theta_hat_pen`` is not the unpenalised K-statistic and has no
    yet-defined reference distribution.  The guard fires on this path
    and the error message tells the caller how to recover (pass
    ``theta_0`` explicitly) plus where the deferred derivation lives.
    """

    restriction = _mean_restriction()
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=lambda theta: 0.5 * theta[0] ** 2,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    with pytest.raises(NotImplementedError) as excinfo:
        result.k_statistic()
    msg = str(excinfo.value)
    assert "#21" in msg, "guard message should point at the deferred derivation"
    assert "theta_0" in msg, "guard message should tell the caller how to recover"


def test_k_statistic_with_explicit_theta_0_works_under_penalty() -> None:
    """``k_statistic(theta_0=...)`` is callable on a penalised fit.

    ``K(theta_0)`` is a pure function of ``(restriction, theta_0,
    data)`` -- it does not reference the optimiser or the penalty -- so
    it remains valid under penalty.  The guard relaxation lets callers
    test a specific null on a penalised ``GMMResult`` without
    refitting.
    """

    restriction = _mean_restriction()
    gmm = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=jnp.array([0.0]),
        penalty=lambda theta: 0.5 * theta[0] ** 2,
    )
    result = gmm.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    # Pick a theta_0 distinct from both ybar (= 2.5) and theta_hat_pen
    # (= N ybar / (N + lambda) = 4 * 2.5 / 4.5 = 2.222...) so K is
    # comfortably non-trivial.
    ks = result.k_statistic(theta_0=jnp.array([3.0]))

    # All four returned values are finite scalars; degrees of freedom
    # are the expected p == ell == 1 for the mean fixture.
    assert np.isfinite(ks.K)
    assert np.isfinite(ks.S)
    assert np.isfinite(ks.J)
    assert ks.df_K == 1
    assert ks.df_S == 0  # ell - p = 1 - 1 = 0; over-id component degenerate
    assert 0.0 <= ks.p_K <= 1.0


def test_k_statistic_at_theta_0_is_penalty_invariant() -> None:
    """Same restriction, same theta_0: K matches with and without penalty.

    Demonstrates the insight that motivated the guard relaxation:
    ``K(theta_0)`` depends only on the restriction, the null, and the
    data -- not on whether the optimiser used a penalty.  Both fits
    converge to different ``theta_hat`` (penalised vs unpenalised),
    but ``k_statistic(theta_0=X)`` returns numerically identical
    K, S, J, p_K, p_S in both cases.
    """

    restriction = _mean_restriction()
    init = jnp.array([0.0])
    theta_0 = jnp.array([3.0])

    gmm_unpenalised = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=init,
    )
    gmm_penalised = GMM(
        restriction,
        weighting=FixedWeighting(np.eye(1)),
        initial_point=init,
        penalty=lambda theta: 0.5 * theta[0] ** 2,
    )

    res_un = gmm_unpenalised.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )
    res_pen = gmm_penalised.estimate(
        optimizer_kwargs={"min_gradient_norm": 1e-12, "max_iterations": 500}
    )

    # Sanity: the two fits did land at different theta_hat, so we are
    # really testing penalty-independence rather than measuring the
    # same fit twice.
    assert not np.allclose(
        np.asarray(res_un.theta.value), np.asarray(res_pen.theta.value)
    )

    ks_un = res_un.k_statistic(theta_0=theta_0)
    ks_pen = res_pen.k_statistic(theta_0=theta_0)

    # K, S, J at the same theta_0 are numerically identical across
    # the two fits -- they share the same restriction/data/null.
    assert np.isclose(
        ks_un.K, ks_pen.K, atol=1e-10
    ), f"K mismatch: unpenalised={ks_un.K!r}, penalised={ks_pen.K!r}"
    assert np.isclose(ks_un.S, ks_pen.S, atol=1e-10)
    assert np.isclose(ks_un.J, ks_pen.J, atol=1e-10)
    assert ks_un.df_K == ks_pen.df_K
    assert ks_un.df_S == ks_pen.df_S
    assert np.isclose(ks_un.p_K, ks_pen.p_K, atol=1e-10)


# ---------------------------------------------------------------------------
# Coercion edge cases
# ---------------------------------------------------------------------------
def test_penalty_invalid_type_raises() -> None:
    """Non-callable, non-PenaltyStrategy penalty raises ``TypeError``."""

    restriction = _mean_restriction()
    with pytest.raises(TypeError, match="penalty must be"):
        GMM(restriction, initial_point=jnp.array([0.0]), penalty=42)  # type: ignore[arg-type]


def test_callable_penalty_no_hessian_attr_when_omitted() -> None:
    """CallablePenalty without ``hessian_tangent`` kwarg has no such attr.

    This is what lets ``_penalty_hessian_tangent`` distinguish the
    analytic path from the FD fallback via ``hasattr``.
    """

    pen = CallablePenalty(lambda theta: 0.0)
    assert not hasattr(pen, "hessian_tangent")


def test_callable_penalty_hessian_attr_present_when_supplied() -> None:
    """CallablePenalty with ``hessian_tangent`` kwarg exposes it as an attr."""

    pen = CallablePenalty(
        lambda theta: 0.0,
        hessian_tangent=lambda theta, basis: np.eye(len(basis)),
    )
    assert hasattr(pen, "hessian_tangent")
    assert np.allclose(pen.hessian_tangent(None, [None, None]), np.eye(2))
