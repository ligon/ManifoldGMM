"""Tests for the linearity autodetection and closed-form GMM path.

Covers:

- :func:`is_affine_in_theta` correctly classifies linear vs nonlinear
  moment functions (including JAX-numerical-derivative blind-spots
  like the bump function constructed via ``jax.lax.cond``).
- ``assume_linear=True`` short-circuits jaxpr-walker; raises clear
  error when the assertion is wrong (sanity check trips).
- ``detect_linear=True`` (the default) autodetects via jaxpr.
- ``detect_linear=False`` suppresses detection.
- ``verbosity=1`` reports linearity status to stdout.
- The closed-form warm start produces the right point estimate on
  linear-moment problems with ``IdentityWeighting``.
- The fast path is correctly skipped on nonlinear-moment problems
  and on CUE-weighted problems (which can't be expressed as
  closed-form).
"""

from __future__ import annotations

import warnings

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from manifoldgmm.econometrics._linearity import (
    affine_coefficients,
    is_affine_in_theta,
    is_flat_manifold,
    solve_linear_gmm,
)
from manifoldgmm.econometrics.gmm import (
    CUEWeighting,
    FixedWeighting,
    IdentityWeighting,
    _weighting_is_theta_independent,
)
from pymanopt.manifolds import Euclidean, Sphere

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)


# --------------------------------------------------------------------------
# is_affine_in_theta: linear and nonlinear cases
# --------------------------------------------------------------------------
def _g_linear(theta, data):
    """Linear in theta: g(theta, X) = X - theta."""

    return data - theta


def _g_quadratic(theta, data):
    """Nonlinear in theta: g(theta, X) = X - theta**2."""

    return data - theta**2


def _g_exp(theta, data):
    """Nonlinear in theta: g(theta, X) = X - exp(theta)."""

    return data - jnp.exp(theta)


def _g_matmul_linear(theta, data):
    """Linear in theta via matmul: g(theta, X) = X - theta[None, :].

    data has shape (N, p); theta has shape (p,); result has shape (N, p).
    Exercises matmul/broadcast paths in the jaxpr walker.
    """

    return data - theta[None, :]


def _make_panel(N: int = 20) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal(size=(N, 1)))


def test_jaxpr_walker_detects_linear() -> None:
    data = _make_panel()
    assert is_affine_in_theta(_g_linear, jnp.array([0.5]), data) is True


def test_jaxpr_walker_detects_quadratic_nonlinear() -> None:
    data = _make_panel()
    assert is_affine_in_theta(_g_quadratic, jnp.array([0.5]), data) is False


def test_jaxpr_walker_detects_exp_nonlinear() -> None:
    data = _make_panel()
    assert is_affine_in_theta(_g_exp, jnp.array([0.5]), data) is False


def test_jaxpr_walker_detects_matmul_linear() -> None:
    """Multi-d linear moment: matmul of theta-derived var with a constant."""

    data = jnp.asarray(np.random.default_rng(0).standard_normal(size=(20, 3)))
    assert (
        is_affine_in_theta(_g_matmul_linear, jnp.array([0.1, 0.2, 0.3]), data) is True
    )


# --------------------------------------------------------------------------
# is_flat_manifold
# --------------------------------------------------------------------------
def test_is_flat_manifold_euclidean() -> None:
    M = Manifold.from_pymanopt(Euclidean(3))
    assert is_flat_manifold(M) is True


def test_is_flat_manifold_sphere() -> None:
    M = Manifold.from_pymanopt(Sphere(3))
    assert is_flat_manifold(M) is False


def test_is_flat_manifold_none() -> None:
    assert is_flat_manifold(None) is False


# --------------------------------------------------------------------------
# Closed-form solver
# --------------------------------------------------------------------------
def test_affine_coefficients_extracts_a_and_B() -> None:
    """``affine_coefficients`` recovers ``a`` and ``B`` for a linear moment."""

    data = _make_panel()
    a, B = affine_coefficients(_g_linear, jnp.array([0.5]), data)
    # Linear: a = g_bar(0) = data.mean(); B = -1.
    np.testing.assert_allclose(float(a[0]), float(data.mean()), atol=1e-6)
    np.testing.assert_allclose(float(B[0, 0]), -1.0, atol=1e-6)


def test_affine_coefficients_raises_on_nonlinear() -> None:
    """Sanity check trips when the moment isn't actually affine."""

    data = _make_panel()
    with pytest.raises(ValueError, match="affinely"):
        affine_coefficients(_g_quadratic, jnp.array([0.5]), data)


def test_solve_linear_gmm_matches_analog() -> None:
    """Closed-form solve recovers the empirical mean for the location model."""

    data = _make_panel()
    W = jnp.eye(1)
    theta_hat, a, B = solve_linear_gmm(_g_linear, data, W, jnp.array([0.5]))
    np.testing.assert_allclose(float(theta_hat[0]), float(data.mean()), atol=1e-6)


# --------------------------------------------------------------------------
# _weighting_is_theta_independent
# --------------------------------------------------------------------------
def test_identity_weighting_is_theta_independent() -> None:
    assert _weighting_is_theta_independent(IdentityWeighting(3)) is True


def test_fixed_weighting_is_theta_independent() -> None:
    W = FixedWeighting(np.eye(3))
    assert _weighting_is_theta_independent(W) is True


def test_cue_weighting_is_theta_dependent() -> None:
    # CUEWeighting depends on theta via the restriction.
    rest = MomentRestriction(g=_g_linear, data=_make_panel())
    assert _weighting_is_theta_independent(CUEWeighting(rest)) is False


# --------------------------------------------------------------------------
# GMM integration: warm start fires when applicable
# --------------------------------------------------------------------------
def test_gmm_uses_closed_form_on_linear_with_identity_weighting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbosity=1 reports the autodetected closed-form warm start."""

    data = _make_panel()
    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    gmm = GMM(
        moment_func=_g_linear,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(1),
        weighting=IdentityWeighting(1),
        verbosity=1,
    )
    gmm.estimate(optimizer_kwargs={"verbosity": 0})
    captured = capsys.readouterr()
    assert "linear moment in theta detected" in captured.out


def test_gmm_assume_linear_short_circuits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``assume_linear=True`` skips the walker and reports the assertion."""

    data = _make_panel()
    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    gmm = GMM(
        moment_func=_g_linear,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(1),
        weighting=IdentityWeighting(1),
        verbosity=1,
        assume_linear=True,
    )
    gmm.estimate(optimizer_kwargs={"verbosity": 0})
    captured = capsys.readouterr()
    assert "asserted by `assume_linear=True`" in captured.out


def test_gmm_detect_linear_off_is_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``detect_linear=False`` produces no linearity message."""

    data = _make_panel()
    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    gmm = GMM(
        moment_func=_g_linear,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(1),
        weighting=IdentityWeighting(1),
        verbosity=1,
        detect_linear=False,
    )
    gmm.estimate(optimizer_kwargs={"verbosity": 0})
    captured = capsys.readouterr()
    assert "linear moment" not in captured.out
    assert "asserted by" not in captured.out


def test_gmm_skips_fast_path_for_nonlinear_moment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nonlinear moment is correctly identified; no warm-start message."""

    rng = np.random.default_rng(0)
    # Generate data consistent with g_quadratic = X - theta^2 = 0,
    # i.e., X = theta_true^2 + noise.
    theta_true = 0.7
    data = jnp.asarray(theta_true**2 + rng.standard_normal(size=(50, 1)) * 0.1)
    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    gmm = GMM(
        moment_func=_g_quadratic,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.array([0.5]),
        weighting=IdentityWeighting(1),
        verbosity=1,
    )
    gmm.estimate(optimizer_kwargs={"verbosity": 0})
    captured = capsys.readouterr()
    assert "linear moment" not in captured.out
    assert "asserted by" not in captured.out


def test_gmm_skips_per_stage_closed_form_for_cue_weighting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CUE weighting on a 1-step estimate doesn't use per-stage closed form.

    Linearity *is* still detected (the moment doesn't change just
    because the weighting is CUE) and the v=1 message fires.  But
    the *per-stage* closed-form solve doesn't fire for the (sole)
    CUE-weighted stage; verbosity=2 would normally print
    ``GMM stage closed-form: ...`` per closed-form stage, and we
    verify that line is absent here.
    """

    data = _make_panel()
    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    # Default weighting is CUE (None -> CUEWeighting); leave it.
    gmm = GMM(
        moment_func=_g_linear,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(1),
        verbosity=2,
    )
    gmm.estimate(optimizer_kwargs={"verbosity": 0})
    captured = capsys.readouterr()
    # Detection still fires (CUE is orthogonal to whether the
    # moment is affine in theta).
    assert "linear moment in theta detected" in captured.out
    # ... but the per-stage closed-form solve does not, because the
    # 1-step CUE-weighted stage is theta-dependent.
    assert "stage closed-form" not in captured.out


def test_gmm_two_step_with_cue_kwarg_uses_per_stage_closed_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``two_step=True`` overrides stage 1 to identity and computes Omega^-1
    for stage 2; both stages are then theta-independent and trigger the
    closed-form per-stage path, regardless of the user-supplied
    ``weighting`` (CUE here).  This is the 3SLS-equivalent fast path.
    """

    # Over-identified linear moment: y = data with 2 columns, theta
    # is a scalar.  Use g(theta, X) = X - theta (broadcast to (N, 2)).
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.standard_normal(size=(40, 2)) + 1.5)

    def g_overid(theta, x):
        return x - theta  # shape (N, 2); k=2 moments, p=1 param

    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    gmm = GMM(
        moment_func=g_overid,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(1),
        verbosity=2,
    )
    result = gmm.estimate(two_step=True, optimizer_kwargs={"verbosity": 0})
    captured = capsys.readouterr()
    # Linearity detected ...
    assert "linear moment in theta detected" in captured.out
    # ... and *both* stages solve in closed form (two messages).
    assert captured.out.count("stage closed-form") == 2

    # And the estimate is sensible.  The over-identified analog
    # under identity weighting is the average over both columns; the
    # efficient (Omega^-1) weighting tightens this further but still
    # near the mean of the data.
    theta_hat = float(np.asarray(result.theta_array)[0])
    assert abs(theta_hat - float(data.mean())) < 0.2


def test_gmm_closed_form_recovers_correct_point_estimate() -> None:
    """End-to-end: closed-form warm start lands at the analog (empirical mean)."""

    data = _make_panel()
    dgp = dp.EmpiricalDGP(observation=data, seed=0)
    M = Manifold.from_pymanopt(Euclidean(1))
    gmm = GMM(
        moment_func=_g_linear,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(1),
        weighting=IdentityWeighting(1),
    )
    result = gmm.estimate(optimizer_kwargs={"verbosity": 0})
    np.testing.assert_allclose(
        float(np.asarray(result.theta_array)[0]),
        float(data.mean()),
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Bump function: pathological case (jaxpr correctly catches the exp)
# --------------------------------------------------------------------------
def test_jaxpr_walker_rejects_bump_function() -> None:
    """The classic bump function evaluated at ``x <= 0`` has Hessian zero
    on (-inf, 0) but isn't affine globally; the jaxpr walker sees the
    ``exp`` in the true branch of the ``cond`` and rejects.
    """

    import jax

    def g_bump(theta, data):
        # g(theta, data) = data - bump(theta_scalar)
        # bump(t) = exp(-1/t) for t > 0, else 0.
        def true_branch(t):
            return jnp.exp(-1.0 / t)

        def false_branch(t):
            return jnp.zeros_like(t)

        t = theta[0]
        bump = jax.lax.cond(t > 0, true_branch, false_branch, t)
        return data - bump

    data = jnp.asarray(np.array([1.0, 2.0, 3.0]))
    # Trace at theta = -1 (false branch).  Numerical Hessian would
    # falsely conclude "affine"; the jaxpr walker should still see
    # the ``exp`` in the true branch and reject.
    assert is_affine_in_theta(g_bump, jnp.array([-1.0]), data) is False
