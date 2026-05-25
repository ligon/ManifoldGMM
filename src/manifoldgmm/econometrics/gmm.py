"""High-level GMM estimator built on top of :class:`MomentRestriction`."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from datamat import DataMat

try:  # Optional dependency for richer pickling support
    import cloudpickle
except ImportError:  # pragma: no cover - optional
    cloudpickle = None

from pymanopt import Problem
from pymanopt.function import jax as pymanopt_jax_function
from pymanopt.function import numpy as pymanopt_numpy_function
from pymanopt.optimizers.optimizer import Optimizer

from .._warnings import ConvergenceWarning, NumericalWarning
from ..geometry import Manifold, ManifoldPoint
from ..optimizers import LoggingTrustRegions
from .moment_restriction import MomentRestriction

# Default threshold for "tail_grad_slope ≈ 0".  Slope is in
# log-gradient-norm per outer iteration; healthy convergence has slope
# well below this (e.g., -0.5 near a quadratic optimum).  The
# motivating #10 trace had slope ≈ -0.003 over 14 iterations; -0.01
# is a tight "not actually descending" line that still admits slow
# but real progress.
_STALL_TAIL_SLOPE_THRESHOLD: float = -0.01

# Minimum cap-hit fraction that flips the warning on.  Aligned with the
# >0.5 phrasing in issue #10's PR 2 sketch.
_STALL_CAP_HIT_FRAC_THRESHOLD: float = 0.5


def _maybe_warn_optimizer_health(result: GMMResult) -> None:
    """Emit a :class:`ConvergenceWarning` if the optimiser stalled.

    Three diagnostics jointly identify the stuck-optimizer pathology
    issue #10 was opened against:

    1. Inner truncated-CG repeatedly hit ``MAX_INNER_ITER``
       (``optimizer_health["inner_cap_hit_frac"] > 0.5``).
    2. Outer loop terminated on a budget criterion, not a tolerance
       (``optimizer_report["converged"] is not True``).
    3. Gradient norm stopped descending on the tail of the trace
       (``optimizer_health["tail_grad_slope"] > -0.01``).

    Any field being ``None`` (no telemetry, fewer than two recorded
    gradient norms, missing stopping criterion) skips the check
    entirely -- the warning fires only when all three signals are
    available and aligned.

    Scope and limitations
    ---------------------
    The predicate above targets **MAX_INNER_ITER plateaus** -- runs
    where the inner truncated-CG repeatedly hits its iteration cap
    without progress and the outer loop terminates on a resource
    budget.  It is **not** designed to flag *post-convergence runaway*
    pathologies where the optimizer terminates *successfully* on a
    tolerance (``stopping_criterion = "min grad norm"``) but lands at
    an iterate the user wouldn't want to use -- e.g., the K-Aggregators
    exp-link runaway documented in #19's empirical comment, where
    ``c_0 = 41.5`` and ``theta(0) = 9.3e18`` with zero cap-hits and a
    steeply negative ``tail_grad_slope``.  That pathology is a
    property of the *result*, not the trajectory, and this warning
    correctly does not fire on it.

    For post-convergence runaway / weak-identification detection,
    inspect :meth:`Diagnostics.hessian_cond` (with
    ``exclude_gauge=True`` on K>=2 quotient manifolds per #32) -- a
    large condition number paired with a healthy trajectory is the
    runaway signature.  See #10 / #32 for the empirical mapping
    between diagnostic and pathology.
    """

    health = result.diagnostics.optimizer_health
    cap_frac = health.get("inner_cap_hit_frac")
    slope = health.get("tail_grad_slope")
    if cap_frac is None or slope is None:
        return
    if cap_frac <= _STALL_CAP_HIT_FRAC_THRESHOLD:
        return
    if slope <= _STALL_TAIL_SLOPE_THRESHOLD:
        return

    converged = (result.optimizer_report or {}).get("converged")
    if converged is True:
        return

    stopping = (result.optimizer_report or {}).get("stopping_criterion") or ""
    n_outer = health.get("n_outer_iters")
    n_cap_hits = health.get("n_inner_cap_hits")
    tail_window = health.get("tail_window")

    message = (
        "Optimisation may have stalled before reaching a tolerance:\n"
        f"  inner_cap_hit_frac = {cap_frac:.2f}  "
        f"({n_cap_hits} of {sum(health.get('inner_stop_counts', {}).values())} "
        f"inner solves hit MAX_INNER_ITER)\n"
        f"  tail_grad_slope    = {slope:+.4f}  "
        f"(d log|grad| / d iter over the last {tail_window} iters; "
        f"healthy convergence is more negative than "
        f"{_STALL_TAIL_SLOPE_THRESHOLD})\n"
        f"  outer iterations   = {n_outer}, stopping_criterion = "
        f"{stopping!r}\n"
        "Remediation:\n"
        "  - Raise the inner-CG iteration cap, e.g.\n"
        "      gmm.estimate(optimizer_kwargs={'maxinner': <larger>})\n"
        "    pymanopt's default is manifold.dim; doubling it is a cheap first try.\n"
        "  - Inspect curvature with result.diagnostics.hessian_cond(); large\n"
        "    values (>> 1e6) suggest the geometry itself is poorly\n"
        "    identified rather than the optimiser being underpowered."
    )
    warnings.warn(message, ConvergenceWarning, stacklevel=3)


class WeightingStrategy(Protocol):
    """Protocol for objects returning a weighting matrix W(θ)."""

    def matrix(self, theta: Any) -> Any:
        """Return the ℓ×ℓ weighting matrix evaluated at ``theta``."""

    def info(self) -> Mapping[str, Any]:  # pragma: no cover - default impl used
        """Metadata describing the weighting strategy."""


class FixedWeighting:
    """Always return the same weighting matrix regardless of θ."""

    # Pure / theta-independent: safe to evaluate inside a jax.jit trace.
    # Theta-dependent strategies (CUE, generic callables) opt out by
    # leaving this False so their Python-side diagnostics (e.g.
    # CUEWeighting._last_ridge) still run on every cost call.
    _jit_safe: bool = True

    def __init__(self, matrix: Any, *, label: str | None = None) -> None:
        self._matrix = matrix
        self._label = label or "fixed"

    def matrix(self, theta: Any) -> Any:  # noqa: D401 - simple delegation
        return self._matrix

    def info(self) -> Mapping[str, Any]:
        return {"type": self._label}


class CallableWeighting:
    """Wrap a callable ``theta -> W`` as a :class:`WeightingStrategy`."""

    def __init__(self, fn: Callable[[Any], Any], *, label: str | None = None) -> None:
        self._fn = fn
        self._label = label or "callable"

    def matrix(self, theta: Any) -> Any:  # noqa: D401 - simple delegation
        try:
            return self._fn(theta)
        except TypeError:
            if isinstance(theta, ManifoldPoint):
                return self._fn(theta.value)
            raise

    def info(self) -> Mapping[str, Any]:
        return {"type": self._label}


class CUEWeighting:
    """Continuously updated weighting based on Ω̂(θ)⁻¹.

    Parameters
    ----------
    restriction : MomentRestriction
        The moment restriction providing omega_hat(theta).
    ridge : float, default 0.0
        Minimum ridge regularization to add: W = (Ω + ridge·I)⁻¹.
    target_condition : float or None, default None
        If set, adaptively choose ridge at each evaluation to keep
        cond(Ω + ridge·I) ≤ target_condition. This handles cases where
        Ω(θ) becomes ill-conditioned as θ moves through the parameter space.
        The effective ridge is max(ridge, adaptive_ridge).

    Notes
    -----
    ``ridge`` / ``target_condition`` regularise the *weighting matrix*
    ``Ω̂(θ)``.  An independent parameter-space regulariser is available
    on :class:`GMM` via the ``penalty`` keyword (since #19 MR1) -- see
    :class:`PenaltyStrategy`.  The two knobs stack arithmetically but
    firing both non-trivially is usually a sign that one of them should
    be off: the weighting ridge addresses ``Ω̂`` conditioning, while
    ``penalty`` addresses identification weakness in ``θ`` itself.
    See the *Basin-shifting risk* warning below for the specific
    failure mode of the **adaptive** ridge path.

    .. warning::

        **Basin-shifting risk under adaptive ridge.**  When
        ``target_condition`` is set, the effective ridge ``λ(θ)``
        depends on ``θ``, so the CUE criterion
        ``Q(θ) = g_bar(θ)' [Ω̂(θ) + λ(θ) I]^{-1} g_bar(θ)`` has an
        additional ``∂λ/∂θ`` term in its first-order conditions that
        the unridged CUE does not.  Two consequences:

        1. **Stationary points move.**  The location of ``θ̂`` under
           the ridged criterion is not the same as under the unridged
           criterion, even when the adaptive ridge happens to be
           small at ``θ̂``.  How far they move depends on how
           reactive ``λ`` is to ``θ``.

        2. **Ill-conditioned regions become attractive.**  Where
           unridged CUE refuses to live (objective explodes as
           ``Ω̂(θ)`` approaches singular), ridged CUE stays bounded.
           If the surrounding cost surface is flatter at extreme
           ``θ`` -- common in weak-identification regions -- the
           optimiser is *attracted* toward those regions rather than
           repelled.  The protective effect at ``θ̂`` is not
           symmetric with the global behaviour.

        Structurally analogous to Newey and Windmeijer (2009) on CUE
        with many moments: re-evaluating the weighting through ``θ``
        adds an ``∂Ω̂/∂θ`` term that destabilises the criterion;
        adaptive ridge adds a second such term ``∂λ/∂θ``.

        **Practical recommendation.**  Use ``target_condition`` only
        when a clear numerical-conditioning failure at ``θ̂`` needs
        regularising; prefer a small fixed ``ridge`` over an adaptive
        ``target_condition`` if the conditioning issue is local; and
        inspect :meth:`Diagnostics.check_inference_validity`'s
        ``ridge_ratio`` -- a large ratio at the reported optimum is a
        warning sign that the optimiser may have settled into a
        ridge-stabilised region rather than a true unridged stationary
        point.

        See #16 for the original discussion thread and proposed
        empirical confirmation experiment.
    """

    def __init__(
        self,
        restriction: MomentRestriction,
        ridge: float = 0.0,
        target_condition: float | None = None,
    ) -> None:
        self._restriction = restriction
        self._ridge = ridge
        self._target_condition = target_condition
        # Track diagnostics for inference validity
        self._last_ridge: float = ridge
        self._last_condition: float = 1.0
        self._last_lambda_min: float = 1.0  # smallest eigenvalue of Ω (before ridge)

    def matrix(self, theta: Any) -> Any:
        omega = self._restriction.omega_hat(theta)
        xp = getattr(self._restriction, "_xp", np)
        linalg = getattr(self._restriction, "_linalg", np.linalg)
        omega_array = xp.asarray(omega)

        if self._target_condition is not None:
            is_jax = hasattr(xp, "where")

            if is_jax:
                # JAX path: compute eigenvalues (needed for tracing, can't branch)
                # eigvalsh is O(n³) same as cond, and we need eigenvalues for ridge
                eigvals = linalg.eigvalsh(omega_array)
                lambda_max = eigvals[-1]
                lambda_min = eigvals[0]
                current_cond = lambda_max / (xp.abs(lambda_min) + 1e-15)

                adaptive_ridge = lambda_max / self._target_condition - lambda_min
                adaptive_ridge = xp.maximum(adaptive_ridge, self._ridge)
                ridge = xp.where(
                    current_cond > self._target_condition,
                    adaptive_ridge,
                    xp.asarray(self._ridge, dtype=omega_array.dtype),
                )

                # Store diagnostics (skip during tracing)
                try:
                    self._last_condition = float(current_cond)
                    self._last_ridge = float(ridge)
                    self._last_lambda_min = float(xp.abs(lambda_min))
                except (TypeError, ValueError):
                    pass

                # Always add ridge for JAX (value may be 0)
                omega_array = omega_array + ridge * xp.eye(
                    omega_array.shape[0], dtype=omega_array.dtype
                )
            else:
                # NumPy path: use cond() first, only compute eigenvalues if needed
                current_cond = float(linalg.cond(omega_array))
                self._last_condition = current_cond

                if current_cond > self._target_condition:
                    eigvals = linalg.eigvalsh(omega_array)
                    lambda_max, lambda_min = float(eigvals[-1]), float(eigvals[0])
                    self._last_lambda_min = abs(lambda_min)
                    ridge = max(
                        lambda_max / self._target_condition - lambda_min,
                        self._ridge,
                    )
                    self._last_ridge = ridge
                    omega_array = omega_array + ridge * xp.eye(
                        omega_array.shape[0], dtype=omega_array.dtype
                    )
                elif self._ridge > 0.0:
                    self._last_ridge = self._ridge
                    # Compute lambda_min for diagnostic even when using fixed ridge
                    eigvals = linalg.eigvalsh(omega_array)
                    self._last_lambda_min = abs(float(eigvals[0]))
                    omega_array = omega_array + self._ridge * xp.eye(
                        omega_array.shape[0], dtype=omega_array.dtype
                    )
                else:
                    self._last_ridge = 0.0
                    self._last_lambda_min = 1.0  # Not computed when no ridge

        elif self._ridge > 0.0:
            # Fixed ridge (no adaptive) - compute lambda_min for diagnostic
            linalg = getattr(self._restriction, "_linalg", np.linalg)
            try:
                eigvals = linalg.eigvalsh(omega_array)
                self._last_lambda_min = abs(float(eigvals[0]))
                self._last_condition = float(eigvals[-1]) / (
                    self._last_lambda_min + 1e-15
                )
            except (TypeError, ValueError):
                pass  # Skip during JAX tracing
            self._last_ridge = self._ridge
            omega_array = omega_array + self._ridge * xp.eye(
                omega_array.shape[0], dtype=omega_array.dtype
            )

        return linalg.inv(omega_array)

    def info(self) -> Mapping[str, Any]:
        # Compute ridge_ratio: how much ridge dominates smallest eigenvalue
        # ridge_ratio > 0.1 suggests potential distortion of test statistics
        # ridge_ratio > 1.0 means ridge completely dominates λ_min
        # When lambda_min is near-zero (singular), ridge completely dominates
        if self._last_lambda_min < 1e-14:
            # Essentially singular - ridge dominates completely
            ridge_ratio = float("inf") if self._last_ridge > 0 else 0.0
        elif self._last_lambda_min > 0:
            ridge_ratio = self._last_ridge / self._last_lambda_min
        else:
            ridge_ratio = 0.0

        # Flag inference concerns
        inference_warning = None
        if ridge_ratio == float("inf") or ridge_ratio > 1.0:
            inference_warning = (
                f"Ridge ({self._last_ridge:.2e}) exceeds λ_min ({self._last_lambda_min:.2e}). "
                "Test statistics (J, Wald) may be substantially distorted."
            )
        elif ridge_ratio > 0.1:
            inference_warning = (
                f"Ridge is {ridge_ratio:.1%} of λ_min. "
                "Test statistics may have mild size distortion."
            )

        return {
            "type": "cue",
            "ridge": self._ridge,
            "target_condition": self._target_condition,
            "last_ridge": self._last_ridge,
            "last_condition": self._last_condition,
            "last_lambda_min": self._last_lambda_min,
            "ridge_ratio": ridge_ratio,
            "inference_warning": inference_warning,
        }


class IdentityWeighting(FixedWeighting):
    """Identity matrix weighting used for first-stage two-step GMM."""

    def __init__(self, dimension: int) -> None:
        super().__init__(np.eye(dimension, dtype=float), label="identity")


# ----------------------------------------------------------------------
# Parameter-space penalty (#19 MR1)
# ----------------------------------------------------------------------
class PenaltyStrategy(Protocol):
    """Protocol for a parameter-space penalty added to the GMM criterion.

    Implementations must expose ``value(theta) -> float``.  The penalty
    is a *cost addition*, not a moment -- it leaves :math:`\\hat\\Omega`
    and the data Jacobian untouched and composes naturally with every
    :class:`WeightingStrategy`.  See #19 for the motivating discussion.

    Composability with the CUE weighting ridge
    ------------------------------------------
    :class:`GMM` accepts ``cue_ridge`` / ``cue_target_condition`` (the
    ridge on :math:`\\hat\\Omega` used by :class:`CUEWeighting`) and
    ``penalty`` as **independent** regularisation knobs.  They stack
    arithmetically -- nothing in the framework prevents firing both at
    once -- but firing both non-trivially is usually a sign that one of
    them should be off:

    - ``penalty`` regularises the *parameters*; the right tool when
      :math:`\\hat\\theta` is poorly identified (the criterion is flat
      along some direction of the parameter space).  Moves the
      stationary point.
    - ``cue_ridge`` / ``cue_target_condition`` regularise the
      *weighting matrix*; the right tool when :math:`\\hat\\Omega(\\theta)`
      is poorly conditioned at the iterate.  Does not move the
      stationary point unless the ridge is adaptive (i.e.,
      :math:`\\lambda(\\theta)`).

    The basin-shifting consequences of stacking an adaptive CUE ridge
    in particular are discussed in #16.

    Optional methods, looked up via :func:`hasattr` at compute time:

    ``hessian_tangent(theta, basis) -> ndarray``
        Penalty Hessian projected onto the canonical tangent basis at
        ``theta``.  Returns a ``(len(basis), len(basis))`` symmetric
        matrix.  When this method is absent, downstream sandwich SEs
        and :meth:`Diagnostics.hessian_cond` fall back to a
        central-difference computation along the basis (accurate but
        ``O(p^2)`` penalty evaluations).
    """

    def value(self, theta: Any) -> float:
        """Return the scalar penalty at ``theta``."""

    def info(self) -> Mapping[str, Any]:  # pragma: no cover - default impl used
        """Metadata describing the penalty for serialisation/diagnostics."""


class CallablePenalty:
    """Wrap a bare callable ``theta -> float`` as a :class:`PenaltyStrategy`.

    Parameters
    ----------
    fn:
        Scalar-valued callable.  Must be backend-compatible with the
        :class:`MomentRestriction` it will be paired with (i.e. return a
        JAX scalar for a JAX restriction so autodiff propagates through
        the optimizer cost; a Python or NumPy float for a NumPy
        restriction).
    label:
        Optional short tag surfaced in :meth:`info` and :attr:`penalty_info`.
    hessian_tangent:
        Optional analytic Hessian in the canonical tangent basis, with
        the signature ``hessian_tangent(theta, basis) -> ndarray``.
        Supplying this avoids the central-difference fallback used by
        the inference-side methods on :class:`GMMResult`.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        label: str | None = None,
        hessian_tangent: Callable[[Any, list[Any]], Any] | None = None,
    ) -> None:
        self._fn = fn
        self._label = label or "callable"
        # Bind hessian_tangent as an instance attribute *only* when supplied,
        # so ``hasattr(penalty, "hessian_tangent")`` cleanly distinguishes
        # "analytic available" from "fall back to finite differences".
        if hessian_tangent is not None:
            self.hessian_tangent = hessian_tangent

    def value(self, theta: Any) -> Any:
        # Returns whatever the wrapped fn returns (JAX scalar or float).
        # The caller (cost builder, GMMResult.data_criterion_value, ...)
        # decides whether to coerce to a Python float.
        try:
            return self._fn(theta)
        except TypeError:
            if isinstance(theta, ManifoldPoint):
                return self._fn(theta.value)
            raise

    def info(self) -> Mapping[str, Any]:
        return {"type": self._label}


def _classify_converged(stopping_criterion: str | None) -> bool | None:
    """Heuristic boolean convergence flag from a pymanopt stopping string.

    Pymanopt's :class:`~pymanopt.optimizers.optimizer.OptimizerResult`
    exposes ``stopping_criterion`` as a free-form message rather than a
    structured enum.  This helper maps the standard messages produced by
    :meth:`Optimizer._check_stopping_criterion` to a boolean:

    - ``True`` when a tolerance was reached (``"min grad norm"`` or
      ``"min step_size"``) -- the optimizer is reporting the iterate
      satisfies a stationary-point criterion.
    - ``False`` when a resource budget was exhausted
      (``"max iterations"``, ``"max time"``, ``"max cost evals"``) --
      the iterate did *not* satisfy any tolerance.
    - ``None`` when the criterion string is missing or unrecognised, so
      downstream code that needs to distinguish "didn't converge" from
      "no information" can do so.  ``bool(None)`` is ``False``, so
      legacy callers that fall back to ``False`` on ``None`` keep their
      conservative behaviour.
    """

    if not stopping_criterion:
        return None
    lowered = stopping_criterion.lower()
    if (
        "min grad norm" in lowered
        or "min step_size" in lowered
        or "min step size" in lowered
    ):
        return True
    if (
        "max iterations" in lowered
        or "max time" in lowered
        or "max cost evals" in lowered
    ):
        return False
    return None


def _tail_log_grad_slope(
    grad_norms: list[float], *, max_window: int = 20
) -> tuple[float | None, int]:
    """LS slope of ``log|grad|`` over the last ``min(max_window, len)`` iters.

    Returns ``(slope, window_used)``.  A near-zero slope on a long tail
    suggests the optimizer is making no progress -- characteristic of
    the ``MAX_INNER_ITER`` plateau described in #10.  A strongly
    negative slope is the signature of healthy late-stage convergence.

    Norms equal to zero are dropped before the regression (the log is
    undefined); if fewer than two finite log-norms remain, the slope is
    reported as ``None`` rather than zero so callers can distinguish
    "the optimizer converged in one step" from "the optimizer stalled
    at a non-zero gradient".
    """

    if not grad_norms:
        return None, 0
    arr = np.asarray(grad_norms, dtype=float)
    window = min(max_window, arr.size)
    tail = arr[-window:]
    positive = tail[tail > 0.0]
    if positive.size < 2:
        return None, window
    log_norms = np.log(positive)
    x = np.arange(positive.size, dtype=float)
    # Closed-form slope; avoids polyfit overhead and its silent
    # numerical warnings on trivially-overdetermined regressions.
    x_mean = x.mean()
    y_mean = log_norms.mean()
    cov = float(np.sum((x - x_mean) * (log_norms - y_mean)))
    var = float(np.sum((x - x_mean) ** 2))
    if var <= 0.0:
        return None, window
    return cov / var, window


# ----------------------------------------------------------------------
# Gauge-dimension detection for quotient manifolds (#32)
# ----------------------------------------------------------------------
def _gauge_dim_of_manifold(manifold_data: Any) -> int:
    r"""Return the gauge nullspace dimension of a pymanopt manifold.

    Implemented for the cases this codebase exercises:

    1. **Explicit ``gauge_dim`` attribute** wins.  A custom manifold
       (or a future pymanopt manifold that learns to expose this) can
       advertise its gauge dimension directly; integer attribute or
       zero-arg callable both work.
    2. **PSDFixedRank / Elliptope family**: gauge dim is
       :math:`K(K-1)/2` from the ``O(K)`` rotation orbit.  Detected
       from ``type(...).__name__`` plus the ``_k`` attribute.
    3. **Product manifolds**: recurse over the constituent manifolds
       (accessed via the ``manifolds`` attribute) and sum.
    4. Otherwise return ``0`` (no detected gauge).  Callers that need
       a fallback for unrecognised manifolds can ask
       :func:`_detect_gauge_dim_by_threshold` to inspect the eigvals
       directly with a ``NumericalWarning``.
    """

    if manifold_data is None:
        return 0

    explicit = getattr(manifold_data, "gauge_dim", None)
    if explicit is not None:
        if callable(explicit):
            try:
                return int(explicit())
            except TypeError:
                pass
        else:
            return int(explicit)

    name = type(manifold_data).__name__
    if name in ("PSDFixedRank", "_PSDFixedRank", "PSDFixedRankComplex", "Elliptope"):
        k = getattr(manifold_data, "_k", None)
        if k is None:
            return 0
        k_int = int(k)
        return k_int * (k_int - 1) // 2

    children = getattr(manifold_data, "manifolds", None)
    if children is not None and not isinstance(children, str):
        try:
            return sum(_gauge_dim_of_manifold(child) for child in children)
        except TypeError:
            return 0

    return 0


def _detect_gauge_dim_by_threshold(
    eigvals: np.ndarray, *, rel_threshold: float = 1e-12
) -> int:
    r"""Count eigenvalues whose absolute magnitude is below ``rel_threshold`` x ``max(|eigvals|)``.

    Fallback for callers using ``diagnostics.hessian_cond(exclude_gauge=True)``
    when the manifold did not expose a ``gauge_dim``.  Returns ``0`` on
    empty input or all-zero spectra (degenerate cases handled by the
    caller).
    """

    if eigvals.size == 0:
        return 0
    abs_eigs = np.abs(eigvals)
    max_abs = float(abs_eigs.max())
    if max_abs == 0.0:
        return 0
    return int(np.sum(abs_eigs < max_abs * rel_threshold))


@dataclass
class WaldTestResult:
    """Result of a Wald test for H0: h(theta) = 0.

    Attributes:
        statistic: The Wald statistic W, asymptotically chi-squared distributed.
        degrees_of_freedom: The number of constraints q.
        p_value: The probability of observing a statistic > W under H0.
    """

    statistic: float
    degrees_of_freedom: int
    p_value: float


@dataclass
class KStatisticResult:
    """Result of the Kleibergen (2005) K-statistic decomposition.

    At a hypothesised value ``theta_0``, the efficient J-statistic
    decomposes as ``J(theta_0) = K(theta_0) + S(theta_0)``, where K
    and S are asymptotically independent under
    ``H0: theta = theta_0``.

    Attributes:
        K: K-statistic (score / LM component), chi2(df_K) under H0.
        S: S-statistic (overidentification complement), chi2(df_S) under H0.
        J: Efficient J-statistic (= K + S).
        df_K: Degrees of freedom for K (= manifold dimension p).
        df_S: Degrees of freedom for S (= ell - p).
        p_K: p-value for K under chi2(df_K).
        p_S: p-value for S under chi2(df_S).
    """

    K: float
    S: float
    J: float
    df_K: int
    df_S: int
    p_K: float
    p_S: float


@dataclass
class GMMResult:
    """Container returned by :meth:`GMM.estimate`."""

    _theta: ManifoldPoint
    criterion_value: float
    degrees_of_freedom: int
    weighting_info: Mapping[str, Any]
    weighting: WeightingStrategy | Callable[[Any], Any] | Any | None
    optimizer_report: Mapping[str, Any]
    restriction: MomentRestriction
    g_bar: Any
    two_step: bool
    # Penalty-aware fields (#19 MR1).  All three default so that
    # callers constructing a ``GMMResult`` manually (test_wald_test.py
    # does this) keep working untouched.  When ``penalty is None``
    # ``data_criterion_value == criterion_value``; when a penalty is
    # active ``criterion_value`` carries g'Wg + penalty(theta_hat) and
    # ``data_criterion_value`` carries the data-only g'Wg.
    data_criterion_value: float | None = None
    penalty: PenaltyStrategy | Callable[[Any], Any] | None = None
    penalty_info: Mapping[str, Any] | None = None
    _theta_labeled: Any | None = field(default=None, init=False, repr=False)
    # Lazy cache of the canonical Jacobian D bar g_N(theta_hat) in the
    # canonical tangent basis at theta_hat.  Computed on first access by
    # ``canonical_jacobian``; reused by ``tangent_covariance``,
    # ``k_statistic`` (when theta_0 is None or equals theta_hat), and any
    # external code calling ``canonical_jacobian`` directly.
    _cached_jacobian: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_jacobian_basis: list[Any] | None = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_pickle(
        self, path: str | Path, *, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        """Serialise the result to ``path`` using :mod:`pickle`."""

        file_path = Path(path)
        try:
            with file_path.open("wb") as handle:
                pickle.dump(self, handle, protocol=protocol)
            return
        except (pickle.PicklingError, TypeError, AttributeError):
            if cloudpickle is None:
                raise
        with file_path.open("wb") as handle:
            cloudpickle.dump(self, handle)

    @staticmethod
    def from_pickle(path: str | Path) -> GMMResult:
        """Load a pickled :class:`GMMResult` from ``path``."""

        file_path = Path(path)
        with file_path.open("rb") as handle:
            try:
                obj = pickle.load(handle)
            except Exception:
                if cloudpickle is None:
                    raise
                handle.seek(0)
                obj = cloudpickle.load(handle)
        if not isinstance(obj, GMMResult):  # pragma: no cover - safety check
            raise TypeError("Pickle does not contain a GMMResult")
        return obj

    # ------------------------------------------------------------------
    # Cached Jacobian at theta_hat
    # ------------------------------------------------------------------
    def canonical_jacobian(self, *, basis: list[Any] | None = None) -> np.ndarray:
        r"""Return the canonical Jacobian :math:`D\bar g_N(\hat\theta)`.

        The canonical basis at :math:`\hat\theta` is fixed once
        ``GMMResult`` is constructed, so the matrix is identical across
        ``tangent_covariance``, ``wald_test`` (via ``tangent_covariance``),
        and ``k_statistic`` (when ``theta_0`` is ``None``).  This method
        memoises the dense matrix on first access; subsequent callers
        reuse the cached array.

        Parameters
        ----------
        basis:
            Optional tangent basis.  When supplied, the cached value is
            used only if it matches the cached basis by object identity;
            otherwise the Jacobian is recomputed (and the cache is left
            untouched, since custom bases are typically a one-off).
            When ``None``, the canonical basis from
            :meth:`MomentRestriction.tangent_basis` is used and the
            result is cached.

        Returns
        -------
        numpy.ndarray
            Dense ``(ell, p)`` Jacobian in the chosen basis.

        Notes
        -----
        For large ``N`` (e.g., :math:`N \sim 10^5`) the Jacobian
        computation can dominate the cost of ``tangent_covariance``,
        ``wald_test``, and ``k_statistic``.  See #4 for context; this
        cache is the small-footprint half of that fix, paired with the
        ``jax.vmap`` batched assembly in
        :meth:`MomentRestriction.jacobian_matrix`.
        """

        if basis is not None:
            # Custom basis: reuse the cache only if the caller hands us the
            # very list we cached earlier (object identity).  Otherwise we
            # compute fresh without disturbing the canonical cache.
            if (
                self._cached_jacobian is not None
                and basis is self._cached_jacobian_basis
            ):
                return self._cached_jacobian
            return self.restriction.jacobian_matrix(self._theta, basis=basis)

        if self._cached_jacobian is None:
            basis_vectors = self.restriction.tangent_basis(self._theta)
            jac = self.restriction.jacobian_matrix(self._theta, basis=basis_vectors)
            # ``object.__setattr__`` because the dataclass is mutable by
            # default but the fields with init=False default to None and
            # we explicitly want to write through.
            self._cached_jacobian = jac
            self._cached_jacobian_basis = basis_vectors
        return self._cached_jacobian

    # ------------------------------------------------------------------
    # Diagnostics view + backward-compat shims
    # ------------------------------------------------------------------
    @property
    def diagnostics(self) -> Diagnostics:
        """Optimization and numerical-quality diagnostics view.

        See :class:`Diagnostics`.  Each access yields a fresh wrapper
        (cheap); semantically all views of the same ``GMMResult`` are
        equivalent.
        """

        return Diagnostics(_result=self)

    @property
    def optimizer_health(self) -> dict[str, Any]:
        """Deprecated alias for ``result.diagnostics.optimizer_health``."""

        warnings.warn(
            "GMMResult.optimizer_health is deprecated; use "
            "result.diagnostics.optimizer_health instead.  "
            "See the package's diagnostics-vs-inference design split.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.diagnostics.optimizer_health

    def compute_hessian_cond(
        self,
        *,
        ridge_floor: float = 1e-300,
        data_only: bool = False,
        exclude_gauge: bool = False,
    ) -> float:
        """Deprecated alias for ``result.diagnostics.hessian_cond(...)``."""

        warnings.warn(
            "GMMResult.compute_hessian_cond is deprecated; use "
            "result.diagnostics.hessian_cond instead.  "
            "See the package's diagnostics-vs-inference design split.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.diagnostics.hessian_cond(
            ridge_floor=ridge_floor,
            data_only=data_only,
            exclude_gauge=exclude_gauge,
        )

    def _resolve_gauge_dim(self) -> int:
        """Return the gauge nullspace dim of the parameter manifold, or 0."""

        manifold = self._theta.manifold
        if manifold is None:
            return 0
        manifold_data = getattr(manifold, "data", None)
        if manifold_data is None:
            return 0
        return _gauge_dim_of_manifold(manifold_data)

    # ------------------------------------------------------------------
    # Penalty Hessian in the canonical tangent basis (#19 MR1)
    # ------------------------------------------------------------------
    def _penalty_hessian_tangent(self, basis: list[Any]) -> np.ndarray:
        r"""Return :math:`\nabla^2 p(\hat\theta)` in the canonical tangent basis.

        Prefers ``penalty.hessian_tangent(theta_hat, basis)`` when the
        penalty exposes it (e.g. via :class:`CallablePenalty`'s
        ``hessian_tangent`` kwarg).  Otherwise falls back to central
        differences along the basis directions, retracted via the
        manifold's :func:`retract` (or :func:`retraction`).  The
        fallback costs ``O(p^2)`` penalty evaluations; supply an
        analytic Hessian to skip it on larger problems.
        """

        penalty = self.penalty
        if penalty is None:
            return np.zeros((len(basis), len(basis)), dtype=float)

        if hasattr(penalty, "hessian_tangent"):
            H = np.asarray(penalty.hessian_tangent(self._theta, basis), dtype=float)
            return 0.5 * (H + H.T)

        # Central-difference fallback.  Mirrors the retraction-of-basis
        # pattern in :meth:`wald_test`'s FD fallback.
        if hasattr(penalty, "value") and callable(penalty.value):
            pen_fn: Callable[[Any], Any] = penalty.value
        elif callable(penalty):
            pen_fn = penalty
        else:
            raise TypeError(
                "Cannot compute penalty Hessian: penalty must expose "
                "either ``value(theta)`` or be directly callable."
            )

        manifold_wrapper = self._theta.manifold
        if manifold_wrapper is None or manifold_wrapper.data is None:
            raise ValueError(
                "Penalty Hessian fallback requires a manifold with a "
                "retraction; the result's manifold is None."
            )
        retraction_fn = getattr(manifold_wrapper.data, "retraction", None)
        if retraction_fn is None:
            retraction_fn = manifold_wrapper.data.retract

        def _scale(struct: Any, factor: float) -> Any:
            if isinstance(struct, tuple | list):
                return type(struct)(_scale(c, factor) for c in struct)
            return np.asarray(struct) * factor

        def _add(lhs: Any, rhs: Any) -> Any:
            if isinstance(lhs, tuple | list):
                return type(lhs)(
                    _add(lhs_part, rhs_part)
                    for lhs_part, rhs_part in zip(lhs, rhs, strict=False)
                )
            return np.asarray(lhs) + np.asarray(rhs)

        def _eval(combo: list[tuple[int, float]]) -> float:
            tangent_vector: Any = None
            for i, scale in combo:
                term = _scale(basis[i], scale)
                if tangent_vector is None:
                    tangent_vector = term
                else:
                    tangent_vector = _add(tangent_vector, term)
            if tangent_vector is None:
                point = self._theta
            else:
                new_value = retraction_fn(self._theta.value, tangent_vector)
                point = ManifoldPoint(manifold_wrapper, new_value)
            try:
                v = pen_fn(point)
            except TypeError:
                v = pen_fn(point.value)
            return float(np.asarray(v))

        eps = 1e-5
        p = len(basis)
        H = np.zeros((p, p), dtype=float)
        # Diagonal: H_ii = (p(+e_i) - 2 p(0) + p(-e_i)) / eps^2
        p0 = _eval([])
        diag_plus = np.empty(p, dtype=float)
        diag_minus = np.empty(p, dtype=float)
        for i in range(p):
            diag_plus[i] = _eval([(i, eps)])
            diag_minus[i] = _eval([(i, -eps)])
            H[i, i] = (diag_plus[i] - 2.0 * p0 + diag_minus[i]) / (eps * eps)
        # Off-diagonal: H_ij = (p(+e_i+e_j) - p(+e_i) - p(+e_j) + p(0)) / eps^2
        # (forward-difference cross term; symmetrises below).
        for i in range(p):
            for j in range(i + 1, p):
                pij = _eval([(i, eps), (j, eps)])
                H[i, j] = (pij - diag_plus[i] - diag_plus[j] + p0) / (eps * eps)
                H[j, i] = H[i, j]
        return H

    def tangent_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> DataMat:
        r"""Sandwich covariance in the canonical tangent coordinates.

        Bread matrix is :math:`D^\top W D` for the unpenalized case; when
        :attr:`penalty` is set, the bread gains an additive
        :math:`\nabla^2 p(\hat\theta)` term (computed via
        :meth:`_penalty_hessian_tangent`).  The middle matrix
        :math:`D^\top W\,\hat\Omega\,W D` is **data-only**: the penalty
        is deterministic, contributing no variance.

        Estimand caveat under penalization
        ---------------------------------
        The sandwich is valid for asymptotic inference *about
        :math:`\hat\theta_{\text{pen}}`*, which is itself an
        asymptotically biased estimator of :math:`\theta_0`.  Frequentist
        coverage of :math:`\theta_0` is not implied -- bias-aware
        methods (out of scope for #19 MR1) are needed there.  Issue #19
        documents the convention; users wanting an unregularised
        sandwich on the same fit should pass ``weighting=`` explicitly
        and read off-bread from a separate :class:`GMMResult` with
        ``penalty=None``.
        """

        theta_hat = self._theta
        restriction = self.restriction
        if basis is None:
            jac_matrix = self.canonical_jacobian()
            # The basis used in the cache is what ``canonical_jacobian``
            # constructed; resurface it for ``manifold_covariance`` and
            # other downstream consumers that index by basis.
            basis_vectors = self._cached_jacobian_basis
            assert basis_vectors is not None  # set by canonical_jacobian
        else:
            basis_vectors = basis
            jac_matrix = self.canonical_jacobian(basis=basis_vectors)
        weights = weighting or self.weighting

        if weights is None:
            raise ValueError("No weighting strategy available to compute covariance")

        if hasattr(weights, "matrix") and callable(weights.matrix):
            W = weights.matrix(theta_hat)
        elif callable(weights):
            W = weights(theta_hat)
        else:
            W = weights

        W_array = np.asarray(W, dtype=float)
        jac_array = np.asarray(jac_matrix, dtype=float)
        omega_array = np.asarray(restriction.omega_hat(theta_hat), dtype=float)

        from ..utils.numeric import ridge_inverse

        XtWX = jac_array.T @ W_array @ jac_array
        if self.penalty is not None:
            XtWX = XtWX + self._penalty_hessian_tangent(basis_vectors)
        inv_XtWX, ridge = ridge_inverse(XtWX, target_condition=ridge_condition)
        middle = jac_array.T @ W_array @ omega_array @ W_array @ jac_array
        covariance = inv_XtWX @ middle @ inv_XtWX

        if ridge != 0.0:
            covariance = np.asarray(covariance)  # ensure materialised array

        basis_labels = [f"basis[{index}]" for index in range(covariance.shape[0])]
        return DataMat(covariance, index=basis_labels, columns=basis_labels)

    def manifold_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> DataMat:
        """Push forward the tangent covariance to ambient coordinates."""

        restriction = self.restriction
        base_point = self.theta_point
        basis_vectors = (
            basis if basis is not None else restriction.tangent_basis(base_point)
        )
        cov_tangent = self.tangent_covariance(
            weighting=weighting, ridge_condition=ridge_condition, basis=basis_vectors
        )

        columns: list[np.ndarray] = []
        for direction in basis_vectors:
            flat = restriction._array_adapter(direction)
            columns.append(np.asarray(flat, dtype=float).reshape(-1))

        if not columns:
            zero = np.zeros((0, 0), dtype=float)
            return DataMat(zero)

        chart_jacobian = np.column_stack(columns)
        covariance = (
            chart_jacobian @ cov_tangent.to_numpy(dtype=float) @ chart_jacobian.T
        )

        labels = list(restriction.parameter_labels or ())
        if len(labels) != covariance.shape[0]:
            labels = [f"theta[{index}]" for index in range(covariance.shape[0])]

        return DataMat(covariance, index=labels, columns=labels)

    @property
    def theta(self) -> ManifoldPoint:
        """Estimated parameter as a :class:`ManifoldPoint`."""

        if self._theta.formatted is self._theta.value:
            _ = self.theta_labeled
        return self._theta

    @property
    def theta_point(self) -> ManifoldPoint:
        """Explicit alias for the manifold-valued estimate."""

        return self.theta

    @property
    def theta_labeled(self) -> Any:
        """Labelled parameter estimate for user-facing consumption."""

        if self._theta_labeled is None:
            formatted = self._theta.formatted
            if formatted is self._theta.value:
                formatted = self.restriction.format_parameter(self._theta.value)
                self._theta = ManifoldPoint(
                    self._theta.manifold,
                    self._theta.value,
                    formatted=formatted,
                )
            self._theta_labeled = formatted
        return self._theta_labeled

    @property
    def theta_array(self) -> Any:
        """Raw parameter estimate suitable for numerical processing."""

        return self.theta.value

    def ambient_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> DataMat:
        """Backward-compatible alias for :meth:`manifold_covariance`."""

        return self.manifold_covariance(
            weighting=weighting, ridge_condition=ridge_condition, basis=basis
        )

    def as_dict(self) -> Mapping[str, Any]:
        """Return the result as a dictionary for quick inspection."""

        out: dict[str, Any] = {
            "theta": self.theta_labeled,
            "criterion_value": self.criterion_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "weighting": dict(self.weighting_info),
            "optimizer_report": dict(self.optimizer_report),
            "two_step": self.two_step,
        }
        if self.penalty is not None:
            out["data_criterion_value"] = self.data_criterion_value
            out["penalty"] = (
                dict(self.penalty_info) if self.penalty_info is not None else {}
            )
        return out

    def check_inference_validity(self, warn: bool = True) -> Mapping[str, Any]:
        """Deprecated alias for ``result.diagnostics.check_inference_validity(...)``."""

        warnings.warn(
            "GMMResult.check_inference_validity is deprecated; use "
            "result.diagnostics.check_inference_validity instead.  "
            "See the package's diagnostics-vs-inference design split.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.diagnostics.check_inference_validity(warn=warn)

    def wald_test(
        self,
        constraint: Callable[[ManifoldPoint], Any],
        q: int | None = None,
    ) -> WaldTestResult:
        """
        Perform a Wald test for H0: h(theta) = 0.

        Parameters
        ----------
        constraint:
            Function mapping a ManifoldPoint to a vector of size q.
            Returns either a JAX array or NumPy array.
        q:
            Number of constraints (dimension of h(theta)). If None, inferred from output.

        Returns
        -------
        WaldTestResult
            Object containing the Wald statistic, degrees of freedom, and p-value.
        """
        from scipy.stats import chi2

        from ..autodiff import jacobian_operator

        theta_hat = self._theta

        # 1. Evaluate constraint at estimate
        h_val = constraint(theta_hat)
        h_val = np.asarray(h_val, dtype=float).flatten()

        if q is None:
            q = h_val.size

        if q == 0:
            return WaldTestResult(0.0, 0, 1.0)

        # 2. Compute Jacobian of h w.r.t. tangent vector xi at xi=0
        # We leverage JacobianOperator which abstracts JAX/autodiff path
        manifold = theta_hat.manifold
        basis = self.restriction.tangent_basis(theta_hat)
        dim = len(basis)

        try:
            op = jacobian_operator(constraint, theta_hat)
            H_cols = []
            for b in basis:
                # op.matvec computes the directional derivative Dh(theta)[b]
                col = op.matvec(b)
                H_cols.append(np.asarray(col, dtype=float).flatten())
            H = np.column_stack(H_cols)
        except Exception:
            # Fallback to finite differences
            epsilon = 1e-5

            def _scale_structure(struct: Any, factor: float) -> Any:
                if isinstance(struct, tuple | list):
                    return type(struct)(_scale_structure(c, factor) for c in struct)
                return np.asarray(struct) * factor

            def _add_structure(lhs: Any, rhs: Any) -> Any:
                if isinstance(lhs, tuple | list):
                    return type(lhs)(
                        _add_structure(lhs_part, rhs_part)
                        for lhs_part, rhs_part in zip(lhs, rhs, strict=False)
                    )
                return np.asarray(lhs) + np.asarray(rhs)

            def composed_map_numpy(xi: np.ndarray) -> Any:
                tangent_vector = None
                for i, b in enumerate(basis):
                    term = _scale_structure(b, float(xi[i]))
                    if tangent_vector is None:
                        tangent_vector = term
                    else:
                        tangent_vector = _add_structure(tangent_vector, term)

                assert manifold.data is not None  # narrowed for mypy
                retraction_fn = getattr(manifold.data, "retraction", None)
                if retraction_fn is None:
                    retraction_fn = manifold.data.retract

                new_value = retraction_fn(theta_hat.value, tangent_vector)
                new_point = ManifoldPoint(manifold, new_value)
                return constraint(new_point)

            # Central difference
            H_cols = []
            for i in range(dim):
                xi_plus = np.zeros(dim)
                xi_plus[i] = epsilon
                val_plus = np.asarray(
                    composed_map_numpy(xi_plus), dtype=float
                ).flatten()

                xi_minus = np.zeros(dim)
                xi_minus[i] = -epsilon
                val_minus = np.asarray(
                    composed_map_numpy(xi_minus), dtype=float
                ).flatten()

                col = (val_plus - val_minus) / (2 * epsilon)
                H_cols.append(col)

            H = np.column_stack(H_cols)

        # H should be (q, dim). If q=1, jax.jacobian might return (dim,) or (1, dim) depending on output
        if H.ndim == 1 and q == 1:
            H = H.reshape(1, -1)

        # 3. Get Covariance
        Sigma = self.tangent_covariance().to_numpy()

        # 4. Compute W = h' (H Sigma H')^-1 h
        denom = H @ Sigma @ H.T

        try:
            # Use solve for better numerical stability than inv
            if q == 1:
                W = (h_val**2) / denom.item()
            else:
                W = h_val @ np.linalg.solve(denom, h_val)
        except np.linalg.LinAlgError:
            W = np.nan

        # 5. p-value
        if np.ndim(W) == 0:
            W_scalar = float(W)
        else:
            W_scalar = float(np.asarray(W).item())

        p_value = 1.0 - chi2.cdf(W_scalar, df=q)

        return WaldTestResult(W_scalar, int(q), float(p_value))

    def k_statistic(
        self,
        *,
        theta_0: ManifoldPoint | Any | None = None,
        ridge_condition: float = 1e8,
    ) -> KStatisticResult:
        r"""Kleibergen (2005) K-statistic decomposition.

        Decomposes the efficient J-statistic at ``theta_0`` as
        :math:`J(\theta_0) = K(\theta_0) + S(\theta_0)` where:

        - :math:`K` is a score / LM-type statistic, :math:`\chi^2(p)` under
          :math:`H_0\colon \theta = \theta_0` regardless of identification
          strength.
        - :math:`S` captures the overidentifying restrictions,
          :math:`\chi^2(\ell - p)` under :math:`H_0`.

        Parameters
        ----------
        theta_0 : ManifoldPoint or array-like, optional
            The parameter value at which to evaluate the decomposition.
            This is the hypothesised value under :math:`H_0`.  When
            ``None`` (default), the estimator :math:`\hat\theta` is used;
            note that K evaluated at the estimator is typically near zero
            because the first-order condition zeroes the score.

            On a penalised fit (``self.penalty is not None``) ``theta_0``
            **must** be passed explicitly.  ``K(theta_0)`` for a
            user-specified ``theta_0`` is a pure function of
            ``(restriction, theta_0, data)`` -- it does not reference
            the optimiser or the penalty -- so it remains valid under
            penalty with the same asymptotic :math:`\chi^2(p)` reference
            distribution.  ``K`` at the penalised estimator itself
            (``theta_0=None`` defaulting to ``theta_hat_pen``) is a
            different and not-yet-defined object; see #21.
        ridge_condition : float, default 1e8
            Target condition number for matrix inversions via
            :func:`~manifoldgmm.utils.numeric.ridge_inverse`.  On
            severely ill-conditioned ``D'Omega^{-1}D``
            (``cond >> ridge_condition``), ``ridge_inverse`` 's bump loop
            saturates; it bails out under a cap (50 iterations by
            default) and emits a
            :class:`~manifoldgmm._warnings.NumericalWarning`, returning
            a best-effort regularised inverse rather than hanging (the
            historical behaviour reported in #18).  To short-circuit
            the loop on a known ill-conditioned fit, inspect
            :meth:`Diagnostics.hessian_cond` first and pass
            ``ridge_condition`` above the empirical conditioning
            (e.g. ``ridge_condition=1e12`` on a fit with
            ``cond(D'WD) ~ 4e9``).

        Returns
        -------
        KStatisticResult

        References
        ----------
        Kleibergen, F. (2005). "Testing Parameters in GMM Without
        Assuming that They Are Identified." *Econometrica*, 73(4),
        1103--1123.
        """
        if self.penalty is not None and theta_0 is None:
            raise NotImplementedError(
                "k_statistic without an explicit ``theta_0`` defaults to "
                "evaluating at theta_hat; on a penalised fit theta_hat is "
                "theta_hat_pen, and K at theta_hat_pen is not yet defined "
                "(the penalised FOC includes a (1/2) grad(p) term that "
                "shifts the K decomposition's reference distribution).  "
                "Pass ``theta_0`` explicitly to test a specific null "
                "-- K(theta_0) is mathematically identical to the "
                "unpenalised case and works under penalty, since the "
                "formula is a pure function of (restriction, theta_0, data) "
                "and does not reference the optimiser or penalty.  See "
                "issue #21 for the deferred K(theta_hat_pen) derivation, "
                "and #25 for the bootstrap K-statistic that addresses "
                "finite-sample and cluster-aware testing under penalty "
                "without further derivation work."
            )
        from scipy.stats import chi2 as chi2_dist

        from ..utils.numeric import ridge_inverse

        restriction = self.restriction

        # Resolve evaluation point
        if theta_0 is not None:
            if not isinstance(theta_0, ManifoldPoint):
                theta_0 = ManifoldPoint(self._theta.manifold, theta_0)
            eval_point = theta_0
        else:
            eval_point = self._theta

        # 1. Ingredients: g_bar, Omega, D (all in ManifoldGMM sqrt(N) scaling)
        g_bar_vec = np.asarray(restriction.g_bar(eval_point), dtype=float).reshape(-1)
        omega = np.asarray(restriction.omega_hat(eval_point), dtype=float)
        # Reuse the cached Jacobian when evaluating at the estimator;
        # otherwise compute fresh at the hypothesised value.
        if eval_point is self._theta:
            D = self.canonical_jacobian()
            basis = self._cached_jacobian_basis
            assert basis is not None
        else:
            basis = restriction.tangent_basis(eval_point)
            D = restriction.jacobian_matrix(eval_point, basis=basis)

        ell = g_bar_vec.shape[0]
        p = D.shape[1]

        # 2. Omega^{-1} (efficient weighting)
        omega_inv, _ = ridge_inverse(omega, target_condition=ridge_condition)

        # 3. Efficient J = g_bar' Omega^{-1} g_bar
        J_eff = float(g_bar_vec @ omega_inv @ g_bar_vec)

        # 4. K = g_bar' Omega^{-1} D (D' Omega^{-1} D)^{-1} D' Omega^{-1} g_bar
        #    Using the CUE-score vector:  s = (D'W D)^{-1} D'W g_bar
        #    so that  K = s' (D'W D) s
        DtW = D.T @ omega_inv  # (p, ell)
        DtWD = DtW @ D  # (p, p)
        DtWD_inv, _ = ridge_inverse(DtWD, target_condition=ridge_condition)
        DtW_gbar = DtW @ g_bar_vec  # (p,)
        score_vec = DtWD_inv @ DtW_gbar  # (p,)
        K = float(score_vec @ DtWD @ score_vec)

        # 5. S = J - K
        S = max(J_eff - K, 0.0)

        # 6. Degrees of freedom and p-values
        df_K = p
        df_S = max(ell - p, 0)

        p_K = float(1.0 - chi2_dist.cdf(K, df=df_K)) if df_K > 0 else float("nan")
        p_S = float(1.0 - chi2_dist.cdf(S, df=df_S)) if df_S > 0 else float("nan")

        return KStatisticResult(
            K=K,
            S=S,
            J=J_eff,
            df_K=df_K,
            df_S=df_S,
            p_K=p_K,
            p_S=p_S,
        )

    def k_statistic_bootstrap(
        self,
        *,
        theta_0: ManifoldPoint | Any,
        n_replicates: int = 200,
        cluster_index: Any | None = None,
        ridge_condition: float = 1e8,
        rng: Any = None,
    ) -> Any:
        r"""Cluster-wild bootstrap of the Kleibergen K-statistic at ``theta_0``.

        Returns a :class:`~manifoldgmm.econometrics.bootstrap.KStatBootstrapResult`
        carrying observed K, S, J at ``theta_0`` alongside bootstrap
        reference distributions and both percentile (bootstrap) and
        chi^2 (asymptotic) p-values.  See #25 for design discussion.

        Compared to :meth:`k_statistic` (which exposes only the
        asymptotic test), the bootstrap variant is useful when:

        - The cluster count is small enough that the asymptotic
          chi^2 reference is a thin approximation under clustering.
        - ``D' Omega^{-1} D`` is severely ill-conditioned and
          :func:`~manifoldgmm.utils.numeric.ridge_inverse` had to fire
          its cap; the bootstrap implicitly captures the same
          regularisation in the empirical reference distribution.
        - Validation: comparing ``p_K_bootstrap`` and
          ``p_K_asymptotic`` is a cheap sanity check on the asymptotic
          result.

        Penalty independence: like :meth:`k_statistic` with an explicit
        ``theta_0``, the bootstrap is a pure function of
        ``(restriction, theta_0, data)`` and produces identical p-values
        on penalised and unpenalised ``GMMResult`` instances fit on the
        same data.

        Parameters
        ----------
        theta_0:
            Hypothesised parameter value under H0.  Required (no
            default); the bootstrap at the estimator itself
            (``theta_0=None``) would conflate with #21's open
            derivation under penalty.
        n_replicates:
            Number of bootstrap replicates.  Default 200.
        cluster_index:
            Optional override of cluster structure (length-N array of
            cluster labels).  When ``None``, falls back to
            ``self.restriction.clusters``; if that is also ``None``,
            the bootstrap is per-observation iid.
        ridge_condition:
            Target condition number for the data-side
            ``ridge_inverse`` calls.  See :meth:`k_statistic`'s
            docstring for the workaround on severely ill-conditioned
            ``D' Omega^{-1} D``.
        rng:
            ``numpy.random.Generator``, integer seed, or ``None``.

        Returns
        -------
        KStatBootstrapResult
        """

        from .bootstrap import k_statistic_bootstrap_for_result

        return k_statistic_bootstrap_for_result(
            self,
            theta_0=theta_0,
            n_replicates=n_replicates,
            cluster_index=cluster_index,
            ridge_condition=ridge_condition,
            rng=rng,
        )

    def in_asymptotic_region(
        self, point: ManifoldPoint | Any, alpha: float = 0.05
    ) -> bool:
        r"""Test whether ``point`` lies inside the asymptotic confidence region.

        The :math:`(1-\alpha)` confidence region is

        .. math::

            \bigl\{\theta : d^2(\hat\theta, \theta) \le \chi^2_{p,\,1-\alpha}\bigr\}

        where :math:`d^2` is the geodesic Mahalanobis distance (see
        :func:`~manifoldgmm.econometrics.bootstrap.geodesic_mahalanobis_distance`)
        and :math:`p` is the manifold dimension.

        Parameters
        ----------
        point : ManifoldPoint or array-like
            Candidate parameter value.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            ``True`` if ``point`` lies inside the region.
        """

        from scipy.stats import chi2

        from .bootstrap import geodesic_mahalanobis_distance

        d2 = geodesic_mahalanobis_distance(self, point)

        # Manifold dimension
        manifold = self.restriction.manifold
        if manifold is not None and manifold.data is not None:
            p = getattr(manifold.data, "dim", None)
            if callable(p):
                p = p()
        else:
            p = None

        if p is None:
            # Fall back to tangent basis length
            p = len(self.restriction.tangent_basis(self.theta_point))

        cv = float(chi2.ppf(1.0 - alpha, df=p))
        return d2 <= cv


# ----------------------------------------------------------------------
# Diagnostics view (optimisation / numerical-quality)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Diagnostics:
    r"""Optimization and numerical-quality diagnostics view of a :class:`GMMResult`.

    Accessed via ``GMMResult.diagnostics``.  Separates *"what does this fit
    tell me about the optimization itself?"* from *"what statistical claims
    can I make?"* -- the latter live on :class:`GMMResult` directly
    (``tangent_covariance``, ``wald_test``, ``k_statistic``, ...).

    Each ``Diagnostics`` instance is a lightweight wrapper around its
    underlying :class:`GMMResult`; multiple accesses of ``result.diagnostics``
    yield independent (but semantically identical) views.

    Members
    -------
    - :attr:`optimizer_health` (property): condensed view of the optimizer
      trace -- inner-CG cap-hit fractions, gradient-slope tail, etc.
      Targets the MAX_INNER_ITER plateau signature of #10.
    - :meth:`hessian_cond`: condition number of the Gauss-Newton Hessian
      (with optional ``data_only`` and ``exclude_gauge`` modes for
      penalised fits and K>=2 quotient manifolds respectively, per #19
      MR1 and #32).
    - :meth:`check_inference_validity`: ridge-contamination check on
      the CUE weighting.
    """

    _result: GMMResult

    # ------------------------------------------------------------------
    # Optimizer trace summary
    # ------------------------------------------------------------------
    @property
    def optimizer_health(self) -> dict[str, Any]:
        r"""Diagnostics derived from the optimizer trace.

        Reads the telemetry written by
        :class:`~manifoldgmm.optimizers.LoggingTrustRegions` (the default
        optimizer for :meth:`GMM.estimate`) into ``optimizer_report["log"]``
        and condenses it into a handful of headline numbers.  When a
        user supplied their own optimizer instance that does not surface
        these fields, the dict still resolves -- but the cap-hit and
        slope entries are ``None``.

        Returns
        -------
        dict
            Keys:

            ``n_outer_iters``
                Outer iterations executed by the optimizer (from
                ``optimizer_report["iterations"]``).
            ``inner_stop_counts``
                ``dict[str, int]`` of inner-CG stop-reason frequencies
                (e.g., ``{"maximum inner iterations": 14,
                "exceeded trust region": 8}``).
            ``n_inner_cap_hits``
                Outer iterations whose inner CG hit ``MAX_INNER_ITER``.
                Equivalent to
                ``inner_stop_counts.get("maximum inner iterations", 0)``.
            ``inner_cap_hit_frac``
                ``n_inner_cap_hits / (sum of inner_stop_counts)``, or
                ``None`` if no inner trace is available.  Values above
                ~0.5 paired with a non-tolerance ``stopping_criterion``
                indicate the optimizer is plateauing -- consider
                raising ``maxinner``.
            ``tail_grad_slope``
                Least-squares slope of :math:`\log|\nabla|` on the last
                ``tail_window`` per-iter gradient norms.  Near zero on a
                stalled run; strongly negative on healthy convergence.
                ``None`` when fewer than two norms were recorded.
            ``tail_window``
                Window size used for the slope (``min(20, n_norms)``).

        Notes
        -----
        See :meth:`hessian_cond` for a complementary curvature
        diagnostic; together these distinguish "stalled because the
        budget ran out" from "stalled because the geometry is bad".

        Scope: signatures captured and missed
        -------------------------------------
        These fields **capture** the MAX_INNER_ITER plateau signature
        (``inner_cap_hit_frac`` near 1, ``tail_grad_slope`` near 0,
        ``optimizer_report["converged"] is not True``) that issue #10
        was opened against.  The :class:`~manifoldgmm._warnings.ConvergenceWarning`
        emitted by :func:`_maybe_warn_optimizer_health` fires when all
        three signals align.

        These fields do **not** capture *post-convergence runaway*: a
        run that converges cleanly on a tolerance but lands at an
        iterate the user wouldn't want (e.g., the K-Aggregators
        exp-link runaway with ``c_0 = 41.5`` and ``theta(0) = 9.3e18``
        documented in #19's empirical comment).  On those runs every
        field here looks healthy (``inner_cap_hit_frac == 0``,
        ``tail_grad_slope`` strongly negative,
        ``optimizer_report["converged"] is True``) -- the pathology is
        a property of the *result*, not the trajectory.

        For runaway / weak-identification detection, use
        :meth:`hessian_cond` with ``exclude_gauge=True`` on
        K>=2 quotient manifolds per #32.  A large condition number
        paired with a healthy ``optimizer_health`` reading is the
        runaway signature.
        """

        report = self._result.optimizer_report
        n_outer = report.get("iterations")
        log = report.get("log") or {}
        inner_counts = dict(log.get("inner_stop_counts") or {})
        grad_norms: list[float] = list(log.get("gradient_norms") or [])

        total_inner = sum(inner_counts.values())
        n_cap_hits = int(inner_counts.get("maximum inner iterations", 0))
        cap_frac: float | None
        if total_inner > 0:
            cap_frac = n_cap_hits / total_inner
        else:
            cap_frac = None

        tail_slope, tail_window = _tail_log_grad_slope(grad_norms)

        return {
            "n_outer_iters": n_outer,
            "inner_stop_counts": inner_counts,
            "n_inner_cap_hits": n_cap_hits,
            "inner_cap_hit_frac": cap_frac,
            "tail_grad_slope": tail_slope,
            "tail_window": tail_window,
        }

    # ------------------------------------------------------------------
    # Hessian condition number (with gauge / penalty modes)
    # ------------------------------------------------------------------
    def hessian_cond(
        self,
        *,
        ridge_floor: float = 1e-300,
        data_only: bool = False,
        exclude_gauge: bool = False,
    ) -> float:
        r"""Condition number of the Gauss-Newton Hessian at :math:`\hat\theta`.

        For the unpenalized fit, the full criterion Hessian is
        :math:`\nabla^2 J = 2\,(D^\top W D + R)` where :math:`R` involves
        second derivatives of :math:`\bar g_N`.  At the optimum
        :math:`\bar g_N \approx 0`, so :math:`R` contributes only through
        a sum weighted by the zero-residual; the Gauss-Newton piece
        :math:`D^\top W D` dominates.  This matrix is also exactly the
        information matrix driving the sandwich variance in
        :meth:`GMMResult.tangent_covariance`, so callers who already
        trust that SE construction are implicitly trusting this cond
        estimate.

        When the underlying :attr:`GMMResult.penalty` is set, the
        optimizer's effective Hessian gains an additive
        :math:`\nabla^2 p(\hat\theta)` term in the canonical tangent
        basis.  By default this method returns the condition number of
        :math:`D^\top W D + \nabla^2 p`; pass ``data_only=True`` to
        inspect the unregularised :math:`D^\top W D` (e.g. to verify
        that a penalty is rescuing an otherwise ill-conditioned design,
        per #19 MR1 test 4).

        .. warning::

            **Quotient-manifold gauge.**  For manifolds with a non-trivial
            isotropy group (``PSDFixedRank(m, K)`` with ``K >= 2``,
            ``Elliptope``, and other quotient manifolds), the canonical
            tangent basis returned by :meth:`MomentRestriction.tangent_basis`
            includes gauge directions whose entries in ``D'WD`` are
            *exactly zero* by construction.  ``hessian_cond`` with the
            default ``exclude_gauge=False`` reports the condition
            number of that gauge-contaminated matrix -- an honest answer
            to a precise question, but one that saturates near
            ``1/eps_float64`` (~``1e16``) whenever ``K(K-1)/2 > 0`` and
            is therefore **uninformative about identification**.  Pass
            ``exclude_gauge=True`` to mod out the gauge nullspace and
            report the condition number of ``D'WD`` restricted to the
            identified subspace.  Issue #32 is the motivating bug
            report.

        Parameters
        ----------
        ridge_floor : float
            Lower clamp on the smallest absolute eigenvalue in the
            cond denominator; protects against exact singularity.
        data_only : bool, default False
            When True, drop the penalty Hessian term and report
            :math:`\mathrm{cond}(D^\top W D)`.  Bit-identical to the
            pre-#19 return value when the underlying ``penalty`` is
            ``None``.
        exclude_gauge : bool, default False
            When True, mod out the manifold's gauge nullspace before
            computing the condition number.  Detection priority:
            (1) ``manifold.data.gauge_dim`` attribute if exposed;
            (2) ``PSDFixedRank``/``Elliptope`` family via
            :math:`K(K-1)/2`; (3) ``Product`` manifolds via recursion
            over constituents; (4) threshold-based detection on the
            spectrum with a :class:`~manifoldgmm._warnings.NumericalWarning`
            for callers using a manifold the framework doesn't
            recognise.  Bit-identical to ``exclude_gauge=False`` when
            no gauge is detected (e.g. Euclidean parameters,
            ``PSDFixedRank(m, 1)``).

        Returns
        -------
        float
            Ratio of the largest to smallest absolute eigenvalue of
            the chosen Hessian (or its gauge-quotient when
            ``exclude_gauge=True``).  Values above ~``1e8`` paired with
            high ``optimizer_health["inner_cap_hit_frac"]`` indicate a
            poorly-conditioned local geometry and motivate either a
            larger ``maxinner`` or a Hessian ridge.

        Notes
        -----
        Cheap path reuses the cached canonical Jacobian and the
        result's weighting matrix.  The penalty Hessian is taken from
        ``penalty.hessian_tangent(theta, basis)`` when available,
        otherwise computed by central differences along the basis (see
        :meth:`GMMResult._penalty_hessian_tangent`).  Does not include
        the second-derivative :math:`R` correction; a follow-up may add
        a finite-difference :math:`R` if downstream uses demand it.
        """

        result = self._result
        D = result.canonical_jacobian()
        if D.size == 0:
            return float("inf")

        weighting: Any = result.weighting
        if weighting is None:
            raise ValueError(
                "diagnostics.hessian_cond requires a GMMResult carrying a "
                "weighting strategy; got None."
            )
        if hasattr(weighting, "matrix") and callable(weighting.matrix):
            W = np.asarray(weighting.matrix(result._theta), dtype=float)
        elif callable(weighting):
            W = np.asarray(weighting(result._theta), dtype=float)
        else:
            W = np.asarray(weighting, dtype=float)

        H = D.T @ W @ D
        if result.penalty is not None and not data_only:
            basis = result._cached_jacobian_basis
            assert basis is not None  # set by canonical_jacobian()
            H = H + result._penalty_hessian_tangent(basis)
        H = 0.5 * (H + H.T)
        eigs = np.linalg.eigvalsh(H)
        # ``eigvalsh`` returns ascending order; for a PSD H all entries
        # are >= 0 (modulo numerical noise), so ``abs(eigs)`` preserves
        # the ascending order.  We sort defensively so the gauge-drop
        # below is correct even if a tiny negative eigenvalue ended up
        # at the front.
        abs_eigs = np.sort(np.abs(eigs))

        if exclude_gauge and abs_eigs.size > 0:
            gauge_dim = result._resolve_gauge_dim()
            if gauge_dim == 0:
                # Manifold didn't advertise a gauge; fall back to a
                # threshold scan and warn so the caller knows they're
                # outside the framework's recognised manifolds.
                detected = _detect_gauge_dim_by_threshold(abs_eigs)
                if detected > 0:
                    warnings.warn(
                        (
                            "diagnostics.hessian_cond(exclude_gauge=True) "
                            f"detected {detected} near-zero eigenvalue(s) by "
                            "threshold but the manifold did not expose a "
                            "``gauge_dim`` attribute.  Falling back to threshold "
                            "detection; expose ``gauge_dim`` on the manifold for "
                            "a clean diagnostic.  See issue #32."
                        ),
                        NumericalWarning,
                        stacklevel=2,
                    )
                    gauge_dim = detected
            if 0 < gauge_dim < abs_eigs.size:
                abs_eigs = abs_eigs[gauge_dim:]
            elif gauge_dim >= abs_eigs.size:
                # Gauge would consume the whole spectrum; surfaces a
                # malformed manifold or empty identified subspace.
                # Return ``inf`` rather than a fake cond.
                return float("inf")

        return float(abs_eigs[-1] / max(float(abs_eigs[0]), ridge_floor))

    # ------------------------------------------------------------------
    # CUE ridge / inference-validity check
    # ------------------------------------------------------------------
    def check_inference_validity(self, warn: bool = True) -> Mapping[str, Any]:
        """Check whether ridge regularization may distort test statistics.

        When CUE weighting uses ridge regularization, the weighting matrix
        W = (Ω + λI)⁻¹ ≠ Ω⁻¹, which can distort the asymptotic distribution
        of test statistics (J-statistic, Wald tests).

        Parameters
        ----------
        warn : bool, default True
            If True and ridge_ratio > 0.1, emit a ``UserWarning``.

        Returns
        -------
        dict with keys:
            ridge_ratio : float
                Ratio of ridge to smallest eigenvalue of Ω.
                - < 0.01: negligible effect on inference
                - 0.01-0.1: minor effect, standard inference likely OK
                - 0.1-1.0: moderate effect, consider bootstrap
                - > 1.0: substantial effect, standard inference unreliable
            lambda_min : float
                Smallest eigenvalue of Ω (before ridge).
            ridge : float
                Ridge value used.
            inference_warning : str or None
                Warning message if ridge_ratio > 0.1.
        """

        info = self._result.weighting_info
        out = {
            "ridge_ratio": info.get("ridge_ratio", 0.0),
            "lambda_min": info.get("last_lambda_min", None),
            "ridge": info.get("last_ridge", 0.0),
            "inference_warning": info.get("inference_warning", None),
        }

        if warn and out["inference_warning"]:
            warnings.warn(out["inference_warning"], UserWarning, stacklevel=2)

        return out


class GMM:
    """High-level GMM estimator operating on a :class:`MomentRestriction`."""

    def __init__(
        self,
        restriction: MomentRestriction,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        optimizer: type[Optimizer] | Optimizer | None = None,
        initial_point: Any | None = None,
        cue_ridge: float = 0.0,
        cue_target_condition: float | None = None,
        penalty: PenaltyStrategy | Callable[[Any], Any] | None = None,
    ) -> None:
        self._restriction = restriction
        self._cue_ridge = cue_ridge
        self._cue_target_condition = cue_target_condition
        self._weighting = self._coerce_weighting(weighting)
        self._optimizer = optimizer
        self._initial_point = initial_point
        # #19 MR1: parameter-space penalty.  ``None`` is bit-identical
        # to today's behaviour; any other value is coerced into a
        # :class:`CallablePenalty` so downstream code can rely on
        # ``penalty.value(theta)`` and the optional ``hessian_tangent``
        # attribute.
        self._penalty: PenaltyStrategy | None = self._coerce_penalty(penalty)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def moment_restriction(self) -> MomentRestriction:
        return self._restriction

    def g_bar(self, theta: Any) -> Any:
        return self._restriction.g_bar(theta)

    def gN(self, theta: Any) -> Any:
        return self._restriction.gN(theta)

    def omega_hat(self, theta: Any) -> Any:
        return self._restriction.omega_hat(theta)

    def criterion(self, theta: Any) -> float:
        """Optimizer cost: ``g'Wg + penalty(theta)`` (penalty 0 when absent)."""

        weighting = self._weighting
        base = float(self._backend_dot(theta, weighting))
        if self._penalty is None:
            return base
        return base + float(np.asarray(self._penalty_value(theta)))

    def data_criterion(self, theta: Any) -> float:
        """Data-only criterion: ``g_bar' W g_bar`` (penalty stripped).

        Useful for fit diagnostics.  Note this is **not** the
        :math:`\\chi^2`-comparable Hansen J under penalization, since
        :math:`\\hat\\theta_{\\text{pen}}` does not solve the unpenalized
        first-order condition; see issue #19's scope section.
        """

        return float(self._backend_dot(theta, self._weighting))

    @property
    def penalty(self) -> PenaltyStrategy | None:
        """The coerced :class:`PenaltyStrategy`, or ``None`` if absent."""

        return self._penalty

    def _penalty_value(self, theta: Any) -> Any:
        """Evaluate the penalty in whatever backend type it returns.

        Always returns ``0.0`` when no penalty is configured; otherwise
        defers to :meth:`PenaltyStrategy.value`.  ``_coerce_penalty``
        guarantees the strategy exposes ``value`` -- callers do not need
        to handle bare-callable inputs here.
        """

        penalty = self._penalty
        if penalty is None:
            return 0.0
        return penalty.value(theta)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimate(
        self,
        *,
        initial_point: Any | None = None,
        two_step: bool = False,
        weighting_iterations: int | Literal["converge"] = 1,
        weighting_tol: float = 1e-6,
        max_weighting_iterations: int = 25,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        verbose: bool | int | None = None,
    ) -> GMMResult:
        """Run one-step, two-step, or iterated GMM.

        Parameters
        ----------
        initial_point:
            Starting parameter on the manifold.  Falls back to the GMM
            instance's initial point, then to a manifold-aware random draw.
        two_step:
            When ``True`` and ``weighting_iterations == 1`` (default),
            performs the classical two-step procedure (identity weighting
            then ``FixedWeighting(Ω̂(θ̂₁)⁻¹)``).
            Implicit when ``weighting_iterations`` is set to ``> 1`` or
            ``"converge"``.
        weighting_iterations:
            Controls how many reweighting stages follow the initial stage.
            ``1`` (default) reproduces today's behaviour exactly (one-step
            when ``two_step=False``; two-step when ``two_step=True``).  An
            integer ``k > 1`` runs ``k`` reweighting stages after an
            identity-weighted first stage, exposing the *iterated* GMM
            estimator (Hansen, Heaton and Yaron 1996).  ``"converge"``
            iterates until the manifold distance between consecutive
            estimates falls below ``weighting_tol`` or
            ``max_weighting_iterations`` is reached.  Note: this is
            distinct from CUE (see :class:`CUEWeighting`) -- iterated GMM
            holds the weighting fixed within each stage and does not carry
            a ``∂Ω/∂θ`` term in the first-order
            condition.
        weighting_tol:
            Tolerance on the manifold distance between consecutive
            estimates used as the convergence criterion when
            ``weighting_iterations="converge"``.
        max_weighting_iterations:
            Hard cap on the number of reweighting stages when iterating to
            convergence.
        optimizer_kwargs:
            Keyword arguments forwarded to the optimizer.  Forwarding is
            partitioned at the call site (see
            :meth:`_split_optimizer_kwargs`):

            **Forwarded to** ``LoggingTrustRegions.__init__`` (or the
            user-supplied optimizer class), via ``**kwargs`` on pymanopt's
            ``TrustRegions``:

            - ``min_gradient_norm`` (default ``1e-6``): Riemannian gradient
              norm at which the outer loop terminates.  For noisy moment
              objectives a slightly looser ``1e-5`` often saves CG work
              with no statistical cost.
            - ``min_step_size`` (default ``1e-10``): minimum step length
              before declaring stagnation.
            - ``max_iterations`` (default ``1000``): outer trust-region
              iteration cap.
            - ``max_time`` (default ``float('inf')``): wall-clock cap.
            - ``kappa`` / ``theta`` (CG tolerances), ``rho_prime``
              (acceptance threshold), ``use_rand``, ``rho_regularization``:
              pymanopt ``TrustRegions`` knobs; default values rarely need
              tuning.
            - ``verbosity``, ``log_verbosity``: pymanopt base-optimizer
              logging.  Set by ``verbose=`` below when omitted.
            - ``adaptive_maxinner`` (default ``False``),
              ``adaptive_threshold`` / ``adaptive_window`` /
              ``adaptive_ceiling``: opt in to the
              :class:`LoggingTrustRegions` policy that doubles ``maxinner``
              when the inner CG repeatedly hits the cap.  Helpful for
              high-dim manifolds where the default ``maxinner =
              manifold.dim`` is too small.

            **Forwarded to** ``optimizer.run(...)``:

            - ``mininner`` (default ``1``): minimum CG iterations per
              outer step.
            - ``maxinner`` (default ``manifold.dim``): cap on CG
              iterations.
            - ``Delta_bar`` (default ``manifold.typical_dist``): trust-
              region radius upper bound.
            - ``Delta0`` (default ``Delta_bar / 8``): initial trust radius.

            Example::

                gmm.estimate(
                    optimizer_kwargs={
                        "min_gradient_norm": 1e-5,
                        "adaptive_maxinner": True,
                        "maxinner": 50,
                    },
                )

        verbose:
            Convenience flag for setting optimizer ``verbosity``.

        Returns
        -------
        GMMResult
            Result container whose ``weighting_info`` carries iteration
            diagnostics (``iterations``, ``theta_path`` of manifold
            distances, ``converged``, ``tol``).
        """

        theta_start = (
            initial_point if initial_point is not None else self._initial_point
        )
        if theta_start is None:
            theta_start = self._default_initial_point()
        if theta_start is None:
            raise ValueError("Provide an initial_point to start the optimisation.")

        optimizer_kwargs = dict(optimizer_kwargs or {})
        if verbose is not None and "verbosity" not in optimizer_kwargs:
            if isinstance(verbose, bool):
                optimizer_kwargs["verbosity"] = 2 if verbose else 0
            else:
                optimizer_kwargs["verbosity"] = int(verbose)

        # Normalise the iteration spec.  ``weighting_iterations == 1`` keeps
        # the historical two_step semantics (so callers that pass
        # ``two_step=True`` still get exactly today's behaviour); any value
        # greater than 1 (or ``"converge"``) implies iterated GMM with an
        # identity-weighted first stage, regardless of the ``two_step`` flag.
        converge_mode = weighting_iterations == "converge"
        if converge_mode:
            iteration_cap = int(max_weighting_iterations)
            if iteration_cap < 1:
                raise ValueError(
                    "max_weighting_iterations must be at least 1 when "
                    "weighting_iterations='converge'."
                )
        else:
            if not isinstance(weighting_iterations, int):
                raise TypeError(
                    "weighting_iterations must be an int or 'converge'; "
                    f"got {type(weighting_iterations).__name__}."
                )
            if weighting_iterations < 0:
                raise ValueError(
                    "weighting_iterations must be non-negative; "
                    f"got {weighting_iterations}."
                )
            iteration_cap = int(weighting_iterations)

        iterated = converge_mode or iteration_cap > 1
        # Stage 1 uses identity weighting whenever a reweighting is going to
        # happen (matches the conventional iterated-GMM setup).
        first_stage_reweighted = two_step or iterated

        weighting_stage1 = self._weighting
        if first_stage_reweighted:
            num_moments = self._ensure_metadata(theta_start)
            weighting_stage1 = IdentityWeighting(num_moments)

        stage = self._run_stage(theta_start, weighting_stage1, optimizer_kwargs)
        final_weighting: WeightingStrategy = weighting_stage1

        theta_path: list[float] = []
        iterations_run = 0
        converged = True

        # Reweighting loop.  ``iteration_cap`` is the planned number of
        # reweighting stages; ``converge_mode`` lets us stop earlier (or run
        # up to ``iteration_cap``) based on the manifold distance.
        target = iteration_cap if not converge_mode else int(max_weighting_iterations)
        if not first_stage_reweighted:
            target = 0
        # In two_step (default ``weighting_iterations=1``) the historical
        # label is "two_step"; iterated runs use "iterated".
        reweight_label = (
            "two_step" if (target <= 1 and not converge_mode) else "iterated"
        )

        if converge_mode:
            converged = False  # set True when distance < tol

        if target >= 1:
            _, linalg = self._backend_modules()
            for _ in range(target):
                omega = self._to_backend_matrix(
                    self._restriction.omega_hat(stage.theta)
                )
                updated_weighting = FixedWeighting(
                    linalg.inv(omega), label=reweight_label
                )
                prev_theta_point = stage.theta
                stage = self._run_stage(
                    prev_theta_point,
                    updated_weighting,
                    optimizer_kwargs,
                )
                final_weighting = updated_weighting
                iterations_run += 1

                distance = self._theta_distance(prev_theta_point, stage.theta)
                theta_path.append(distance)

                if converge_mode and distance < weighting_tol:
                    converged = True
                    break

        final_stage = stage

        df = self._degrees_of_freedom(final_stage.g_bar, final_stage.theta)
        weighting_info = dict(final_weighting.info())
        weighting_info.setdefault("two_step", first_stage_reweighted)
        weighting_info["iterations"] = iterations_run
        weighting_info["theta_path"] = list(theta_path)
        weighting_info["converged"] = converged
        weighting_info["tol"] = float(weighting_tol)

        data_value = float(self._backend_dot(final_stage.theta, final_weighting))
        if self._penalty is None:
            criterion = data_value
            penalty_info: Mapping[str, Any] | None = None
        else:
            pen_value = float(np.asarray(self._penalty_value(final_stage.theta)))
            criterion = data_value + pen_value
            base_info = (
                self._penalty.info()
                if hasattr(self._penalty, "info") and callable(self._penalty.info)
                else {}
            )
            penalty_info = {
                **dict(base_info),
                "value_at_theta_hat": pen_value,
            }

        result = GMMResult(
            _theta=final_stage.theta,
            criterion_value=criterion,
            degrees_of_freedom=df,
            weighting_info=weighting_info,
            weighting=final_weighting,
            optimizer_report=final_stage.optimizer_report,
            restriction=self._restriction,
            g_bar=final_stage.g_bar,
            two_step=first_stage_reweighted,
            data_criterion_value=data_value,
            penalty=self._penalty,
            penalty_info=penalty_info,
        )
        _maybe_warn_optimizer_health(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_weighting(
        self, weighting: WeightingStrategy | Callable[[Any], Any] | Any | None
    ) -> WeightingStrategy:
        if weighting is None:
            return CUEWeighting(
                self._restriction,
                ridge=self._cue_ridge,
                target_condition=self._cue_target_condition,
            )
        if hasattr(weighting, "matrix") and callable(weighting.matrix):
            return cast(WeightingStrategy, weighting)
        if callable(weighting):
            return CallableWeighting(weighting)
        return FixedWeighting(weighting)

    @staticmethod
    def _coerce_penalty(
        penalty: PenaltyStrategy | Callable[[Any], Any] | None,
    ) -> PenaltyStrategy | None:
        """Normalise ``penalty`` to a :class:`PenaltyStrategy` (or ``None``)."""

        if penalty is None:
            return None
        if hasattr(penalty, "value") and callable(penalty.value):
            return cast(PenaltyStrategy, penalty)
        if callable(penalty):
            return CallablePenalty(penalty)
        raise TypeError(
            "penalty must be None, a callable theta -> float, or an object "
            "exposing a ``value(theta)`` method; got "
            f"{type(penalty).__name__}."
        )

    # Kwargs that belong on TrustRegions.run() rather than __init__().
    # (pymanopt's TrustRegions.run accepts mininner, maxinner, Delta_bar,
    # Delta0; __init__ takes miniter, kappa, theta, rho_prime, use_rand,
    # rho_regularization, plus base Optimizer kwargs.)
    _OPTIMIZER_RUN_KWARGS = frozenset({"mininner", "maxinner", "Delta_bar", "Delta0"})

    @classmethod
    def _split_optimizer_kwargs(
        cls, optimizer_kwargs: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition optimizer_kwargs into (init_kwargs, run_kwargs)."""
        init_kwargs: dict[str, Any] = {}
        run_kwargs: dict[str, Any] = {}
        for key, value in optimizer_kwargs.items():
            if key in cls._OPTIMIZER_RUN_KWARGS:
                run_kwargs[key] = value
            else:
                init_kwargs[key] = value
        return init_kwargs, run_kwargs

    def _resolve_optimizer(self, optimizer_kwargs: Mapping[str, Any]) -> Optimizer:
        base = self._optimizer
        if base is None:
            # Default to LoggingTrustRegions so optimizer_report["log"]
            # carries inner-CG stop counts and per-iter gradient norms.
            # Force log_verbosity >= 1 (unless the caller overrode it)
            # so the base Optimizer initialises the iterations log dict
            # the subclass attaches to.
            kwargs = dict(optimizer_kwargs)
            kwargs.setdefault("log_verbosity", 1)
            return LoggingTrustRegions(**kwargs)
        if isinstance(base, Optimizer):
            if optimizer_kwargs:
                allowed = {"verbosity", "log_verbosity"}
                unexpected = set(optimizer_kwargs) - allowed
                if unexpected:
                    raise ValueError(
                        "optimizer_kwargs are incompatible with a pre-configured "
                        f"optimizer (unexpected keys: {sorted(unexpected)!r})"
                    )
                for key, value in optimizer_kwargs.items():
                    setattr(base, key, value)
            return base
        return base(**optimizer_kwargs)

    def _run_stage(
        self,
        initial_point: Any,
        weighting: WeightingStrategy,
        optimizer_kwargs: Mapping[str, Any],
    ) -> _StageResult:
        cost = self._build_cost(weighting)
        manifold_wrapper = self._restriction.manifold
        if manifold_wrapper is None or manifold_wrapper.data is None:
            raise ValueError("MomentRestriction must define a manifold to run GMM.")
        problem = Problem(cost=cost, manifold=manifold_wrapper.data)
        init_kwargs, run_kwargs = self._split_optimizer_kwargs(dict(optimizer_kwargs))
        optimizer = self._resolve_optimizer(init_kwargs)
        start_value = (
            initial_point.value
            if isinstance(initial_point, ManifoldPoint)
            else initial_point
        )
        result = optimizer.run(problem, initial_point=start_value, **run_kwargs)
        theta_value = result.point
        theta_point = ManifoldPoint(
            manifold_wrapper,
            theta_value,
        )
        g_bar_hat = self._restriction.g_bar(theta_point)
        # Surface the fields pymanopt's ``OptimizerResult`` actually
        # exposes (see ``pymanopt.optimizers.optimizer.OptimizerResult``).
        # The previous report read ``converged`` and ``stopping_reason``
        # via ``getattr`` with a ``None`` fallback; pymanopt defines
        # neither attribute, so both came back silently ``None`` for every
        # fit -- which in turn made ``BootstrapResult.converged`` always
        # ``False`` (bootstrap.py line 328 falls back to ``False`` when
        # the key is None).  ``converged`` is now synthesised from the
        # stopping-criterion string so downstream consumers get a real
        # signal.  ``stopping_criterion`` is the canonical key going
        # forward; we keep the field set tolerant via ``getattr`` so
        # third-party optimizers that omit individual fields still work.
        stopping_criterion = getattr(result, "stopping_criterion", None)
        optimizer_report = {
            "iterations": getattr(result, "iterations", None),
            "stopping_criterion": stopping_criterion,
            "converged": _classify_converged(stopping_criterion),
            "cost": getattr(result, "cost", None),
            "gradient_norm": getattr(result, "gradient_norm", None),
            "step_size": getattr(result, "step_size", None),
            "cost_evaluations": getattr(result, "cost_evaluations", None),
            "time": getattr(result, "time", None),
            "log": getattr(result, "log", None),
        }
        return _StageResult(
            theta=theta_point,
            g_bar=g_bar_hat,
            weighting=weighting,
            optimizer_report=optimizer_report,
        )

    def _theta_distance(
        self, prev_point: ManifoldPoint, cur_point: ManifoldPoint
    ) -> float:
        """Distance between two estimates for the iterated-GMM convergence test.

        Prefers the manifold's intrinsic ``dist(x, y)``; falls back to the
        ambient L2 norm of flattened coordinates when no distance is
        available (or when it raises -- e.g., custom manifolds without a
        Riemannian metric).
        """

        manifold_wrapper = self._restriction.manifold
        manifold_data = (
            getattr(manifold_wrapper, "data", None)
            if manifold_wrapper is not None
            else None
        )
        if manifold_data is not None:
            dist_fn = getattr(manifold_data, "dist", None)
            if callable(dist_fn):
                try:
                    return float(dist_fn(prev_point.value, cur_point.value))
                except Exception:  # pragma: no cover - manifold-specific fallback
                    pass

        flat_prev = np.asarray(
            self._restriction._array_adapter(prev_point.value), dtype=float
        ).reshape(-1)
        flat_cur = np.asarray(
            self._restriction._array_adapter(cur_point.value), dtype=float
        ).reshape(-1)
        return float(np.linalg.norm(flat_cur - flat_prev))

    def _default_initial_point(self) -> Any | None:
        restriction = self._restriction
        manifold_wrapper = restriction.manifold
        if manifold_wrapper is not None:
            try:
                return manifold_wrapper.random_point()
            except AttributeError:
                pass

        param_shape = restriction.parameter_shape
        param_dim = restriction.parameter_dimension
        if param_shape is not None:
            rng = np.random.default_rng()
            noise = rng.normal(scale=1e-3, size=int(np.prod(param_shape)))
            return noise.reshape(param_shape)
        if param_dim is not None:
            rng = np.random.default_rng()
            return rng.normal(scale=1e-3, size=param_dim)
        return None

    def _build_cost(self, weighting: WeightingStrategy) -> Callable[[Any], Any]:
        restriction = self._restriction
        manifold_wrapper: Manifold | None = restriction.manifold
        if manifold_wrapper is None or manifold_wrapper.data is None:
            raise ValueError(
                "MomentRestriction must carry a manifold for optimisation."
            )
        penalty = self._penalty

        def _assemble_theta(blocks: tuple[Any, ...]) -> Any:
            if len(blocks) == 1:
                return blocks[0]
            return blocks

        if restriction._is_jax_backend:
            # pymanopt's JAX backend does not jit the cost (or its
            # derived gradient/Hvp) -- see
            # ``pymanopt.autodiff.backends._jax``.  Wrapping ``cost`` in
            # ``jax.jit`` before ``pymanopt_jax_function`` decorates it
            # means pymanopt's ``jax.grad(cost)`` and ``jax.jvp(jax.grad
            # (cost), ...)`` compose with a jit primitive and inherit
            # the compiled trace, saving a full Python-dispatch pass per
            # cost / gradient / Hvp evaluation in the inner CG.
            #
            # Only wrap when both the weighting and the penalty are
            # ``_jit_safe`` -- theta-dependent strategies (CUE, generic
            # callables) keep Python-side diagnostics that JIT silently
            # drops on retrace.
            jit_safe = getattr(weighting, "_jit_safe", False) and (
                penalty is None or getattr(penalty, "_jit_safe", False)
            )

            def _cost_body(*blocks: Any) -> Any:
                theta = _assemble_theta(blocks)
                base = self._backend_dot(theta, weighting)
                if penalty is None:
                    return base
                pen = self._penalty_value(theta)
                # JAX scalars compose with ``+`` directly; promotion to
                # the moment-restriction dtype happens implicitly.
                return base + pen

            inner = jax.jit(_cost_body) if jit_safe else _cost_body
            cost = pymanopt_jax_function(manifold_wrapper.data)(inner)

        else:

            @pymanopt_numpy_function(manifold_wrapper.data)
            def cost(*blocks: Any) -> Any:
                theta = _assemble_theta(blocks)
                base = float(self._backend_dot(theta, weighting))
                if penalty is None:
                    return base
                pen = float(np.asarray(self._penalty_value(theta)))
                return base + pen

        return cost

    def _backend_dot(self, theta: Any, weighting: WeightingStrategy) -> Any:
        xp, _ = self._backend_modules()
        g_vec = self._to_backend_vector(self._restriction.g_bar(theta))
        W = self._to_backend_matrix(weighting.matrix(theta))
        return xp.dot(g_vec, xp.matmul(W, g_vec))

    def _to_backend_vector(self, value: Any) -> Any:
        xp, _ = self._backend_modules()
        array: Any
        if isinstance(value, jnp.ndarray):
            array = value
        else:
            array = np.asarray(value, dtype=float)
            if xp is not np:
                array = xp.asarray(array)
        array = array.reshape(-1)
        return array

    def _to_backend_matrix(self, value: Any) -> Any:
        xp, _ = self._backend_modules()
        array: Any
        if isinstance(value, jnp.ndarray):
            array = value
        else:
            array = np.asarray(value, dtype=float)
            if xp is not np:
                array = xp.asarray(array)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def _ensure_metadata(self, theta: Any) -> int:
        g_vec = self._to_backend_vector(self._restriction.g_bar(theta))
        return int(np.asarray(g_vec).size)

    def _degrees_of_freedom(self, g_bar_value: Any, theta: Any) -> int:
        num_moments = np.asarray(g_bar_value).reshape(-1).size
        param_dim = self._restriction.parameter_dimension
        if param_dim is None:
            manifold_wrapper = self._restriction.manifold
            manifold_dim = None
            if manifold_wrapper is not None and manifold_wrapper.data is not None:
                manifold_dim = getattr(manifold_wrapper.data, "dim", None)
                if callable(manifold_dim):
                    manifold_dim = manifold_dim()
            if manifold_dim is None:
                theta_sample = theta if theta is not None else self._initial_point
                if theta_sample is None:
                    raise RuntimeError(
                        "MomentRestriction has unknown parameter dimension; provide an initial point."
                    )
                if isinstance(theta_sample, tuple | list):
                    manifold_dim = sum(
                        int(np.asarray(block).size) for block in theta_sample
                    )
                else:
                    manifold_dim = int(np.asarray(theta_sample).size)
            param_dim = int(manifold_dim)
        return max(num_moments - param_dim, 0)

    def _backend_modules(self) -> tuple[Any, Any]:
        xp = getattr(self._restriction, "_xp", np)
        linalg = getattr(self._restriction, "_linalg", np.linalg)
        return xp, linalg


@dataclass
class _StageResult:
    theta: ManifoldPoint
    g_bar: Any
    weighting: WeightingStrategy
    optimizer_report: Mapping[str, Any]
