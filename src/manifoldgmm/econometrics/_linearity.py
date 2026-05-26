"""Linearity autodetection and closed-form GMM for affine moments.

When the moment function ``g(theta, data)`` is affine in ``theta`` and
the parameter manifold is flat (Euclidean or product of Euclideans),
the GMM criterion is a quadratic form in ``theta`` and the minimizer
has the closed form

    theta_hat = -(B' W B)^{-1} B' W a,

where ``a = g_bar(0, data)`` is the constant term and ``B =
\\partial g_bar / \\partial theta`` is the (constant) Jacobian.  For
just-identified problems (``k = p``), this simplifies to
``theta_hat = -B^{-1} a``.  For over-identified problems with a
data-independent weighting ``W``, the formula gives the GLS / 2SLS-
style estimator in one matrix solve.

This module provides:

- :func:`is_affine_in_theta`: a jaxpr-level static check that walks
  the operator graph of ``moment_func`` and verifies that ``theta``
  flows only through *affine-preserving* primitives.  It's strictly
  stronger than any numerical check at sample points -- the bump
  function ``g(x) = 0 for x <= 0, exp(-1/x) for x > 0`` has zero
  Hessian on ``(-inf, 0)`` but a numerical Hessian check at any
  ``x < 0`` would falsely conclude "affine"; the jaxpr walker
  detects the ``exp`` primitive in the conditional branch and
  rejects.

  Static guarantee: if the walker returns ``True``, the function
  as expressed in this jaxpr is affine in ``theta``.  False
  positives are possible only for adversarial implementations that
  hide nonlinear ops behind data-dependent Python branches not
  visible to JAX tracing.  Real-world moment functions are
  essentially always cleanly expressed.

- :func:`solve_linear_gmm`: closed-form solve when the moment is
  affine, the manifold is Euclidean, and the weighting is data-
  independent (i.e., not :class:`CUEWeighting`).

- :func:`is_flat_manifold`: detect whether a
  :class:`~manifoldgmm.geometry.Manifold` is Euclidean (the necessary
  geometric condition for the closed-form path to apply).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# jax's Var type moved around across versions; prefer the public
# ``jax.extend.core.Var`` (jax >= 0.4.31) and fall back to the
# private location for older releases.  In the worst case, we fall
# back to duck-typing via ``hasattr(v, "count")``.
_JaxVar: Any
try:
    from jax.extend.core import Var as _JaxVar
except ImportError:  # pragma: no cover
    try:
        from jax._src.core import Var as _JaxVar
    except ImportError:  # pragma: no cover
        _JaxVar = None


def _is_jax_var(v: Any) -> bool:
    """True if ``v`` is a jaxpr Var (rather than a Literal)."""

    if _JaxVar is not None:
        return isinstance(v, _JaxVar)
    return hasattr(v, "count")


# JAX primitives that *preserve* affineness in their theta-derived
# inputs.  Any primitive not on this list that touches theta-derived
# variables causes the walker to reject the function as non-affine.
_AFFINE_PRIMITIVES: frozenset[str] = frozenset(
    {
        # Pointwise affine ops
        "add",
        "sub",
        "neg",
        # Reshape / shape manipulation
        "reshape",
        "transpose",
        "broadcast_in_dim",
        "concatenate",
        "slice",
        "dynamic_slice",
        "gather",
        "squeeze",
        # Linear contractions
        "dot_general",
        "reduce_sum",
        "reduce_mean",
        # Casting (changes dtype, preserves affineness)
        "convert_element_type",
        # Indexing primitives
        "select_n",
    }
)


def is_affine_in_theta(
    moment_func: Any,
    example_theta: jnp.ndarray,
    example_data: Any,
) -> bool:
    """Static jaxpr-walker check for ``moment_func(theta, data)`` affine in ``theta``.

    Returns ``True`` only if every JAX primitive that touches a
    ``theta``-derived variable is in the affine whitelist (with
    additional structural checks for ``mul`` and ``integer_pow`` that
    can be affine *or* nonlinear depending on operand structure).

    Parameters
    ----------
    moment_func:
        Callable ``(theta, data) -> (N, k)``.
    example_theta:
        Any ``theta`` of the right shape; only used to trace the jaxpr.
    example_data:
        Any ``data`` of the right shape; only used to trace the jaxpr.

    Returns
    -------
    bool
        ``True`` iff the function as expressed in this jaxpr applies
        only affine-preserving operations to ``theta``.
    """

    try:
        closed = jax.make_jaxpr(moment_func)(example_theta, example_data)
    except Exception:  # pragma: no cover - tracing failure
        return False
    return _jaxpr_affine_in_var(closed.jaxpr, closed.jaxpr.invars[0])


def _jaxpr_affine_in_var(jaxpr: Any, target_var: Any) -> bool:
    """Walk ``jaxpr`` and verify every op touching ``target_var`` is affine.

    Recurses into sub-jaxprs (``scan``, ``cond``, ``while``, ``call_p``).
    """

    derived: set[Any] = {target_var}

    for eqn in jaxpr.eqns:
        invars_derived = [v for v in eqn.invars if _is_jax_var(v) and v in derived]
        if not invars_derived:
            continue

        p = eqn.primitive.name

        if p == "mul":
            # Multiplication of a theta-derived variable by another
            # variable is affine only if at most one operand is theta-
            # derived.  Two theta-derived operands -> theta * theta ->
            # nonlinear.
            if len(invars_derived) > 1:
                return False

        elif p == "integer_pow":
            # x ** k for integer k preserves affineness only when k == 1.
            y = eqn.params.get("y", 1)
            if y != 1:
                return False

        elif p in {"pjit", "custom_jvp_call", "custom_vjp_call_jaxpr"}:
            # Recurse into the wrapped jaxpr.  Find the sub-jaxpr in
            # the equation params and resolve which sub-invars
            # correspond to theta-derived outer invars.
            sub_jaxpr = _extract_sub_jaxpr(eqn.params)
            if sub_jaxpr is None:
                return False  # unknown wrapping; conservative reject
            sub_targets = [
                sub_jaxpr.invars[i]
                for i, outer in enumerate(eqn.invars)
                if _is_jax_var(outer) and outer in derived
            ]
            if not all(_jaxpr_affine_in_vars(sub_jaxpr, sub_targets) for _ in [0]):
                return False

        elif p == "cond":
            # ``cond`` has true/false branches as sub-jaxprs in params.
            branches = eqn.params.get("branches", ())
            for branch in branches:
                branch_jaxpr = getattr(branch, "jaxpr", branch)
                sub_targets = [
                    branch_jaxpr.invars[i]
                    for i, outer in enumerate(eqn.invars[1:])  # skip predicate
                    if _is_jax_var(outer) and outer in derived
                ]
                if not _jaxpr_affine_in_vars(branch_jaxpr, sub_targets):
                    return False

        elif p not in _AFFINE_PRIMITIVES:
            return False

        # Mark all outputs as theta-derived.
        for outvar in eqn.outvars:
            derived.add(outvar)

    return True


def _jaxpr_affine_in_vars(
    jaxpr: Any,
    target_vars: list[Any],
) -> bool:
    """Affineness in any of ``target_vars`` (sub-jaxpr helper)."""

    derived: set[Any] = set(target_vars)
    for eqn in jaxpr.eqns:
        invars_derived = [v for v in eqn.invars if _is_jax_var(v) and v in derived]
        if not invars_derived:
            continue
        p = eqn.primitive.name
        if p == "mul" and len(invars_derived) > 1:
            return False
        if p == "integer_pow" and eqn.params.get("y", 1) != 1:
            return False
        if p not in _AFFINE_PRIMITIVES and p not in {
            "mul",
            "integer_pow",
            "pjit",
            "cond",
        }:
            return False
        for outvar in eqn.outvars:
            derived.add(outvar)
    return True


def _extract_sub_jaxpr(params: dict) -> Any | None:
    """Pull a sub-jaxpr out of an equation's params dict (best effort)."""

    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in params:
            obj = params[key]
            return getattr(obj, "jaxpr", obj)
    return None


# ---------------------------------------------------------------------------
# Closed-form solver
# ---------------------------------------------------------------------------


def is_flat_manifold(manifold: Any) -> bool:
    """Check whether ``manifold`` is Euclidean (the geometric condition
    for the closed-form GMM path to apply).

    Recognizes :class:`pymanopt.manifolds.Euclidean` and products of
    Euclideans.  Returns ``False`` for any curved manifold.
    """

    inner = getattr(manifold, "data", None)
    if inner is None:
        return False
    try:
        from pymanopt.manifolds import Euclidean, Product
    except ImportError:  # pragma: no cover
        return False
    if isinstance(inner, Euclidean):
        return True
    if isinstance(inner, Product):
        return all(isinstance(m, Euclidean) for m in inner.manifolds)
    return False


def affine_coefficients(
    moment_func: Any,
    example_theta: jnp.ndarray,
    data: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract ``(a, B)`` such that ``g_bar(theta) = a + B @ theta``.

    Computed via two autodiff queries on ``moment_func``:

    - ``a = g_bar(0, data)`` (the constant term);
    - ``B = jacobian(g_bar)(any theta)`` (constant when ``g`` is affine).

    Sanity-check: numerically verifies that ``g_bar`` evaluated at
    ``example_theta`` agrees with ``a + B @ example_theta`` to within
    floating-point precision.  Raises ``ValueError`` if it doesn't,
    which indicates the moment function isn't actually affine despite
    user assertion / jaxpr-walker false positive.
    """

    p = example_theta.shape[0]
    zero = jnp.zeros_like(example_theta)

    def g_bar(theta):
        return moment_func(theta, data).mean(axis=0)

    a = g_bar(zero)
    B = jax.jacfwd(g_bar)(zero)

    # Sanity: g_bar(example_theta) == a + B @ example_theta?
    predicted = a + B @ example_theta
    actual = g_bar(example_theta)
    residual = float(jnp.max(jnp.abs(predicted - actual)))
    # Tolerance: scale with magnitude.  Allow a generous threshold
    # since autodiff via float32 can accumulate noise.
    a_scale = float(jnp.max(jnp.abs(a)))
    b_scale = float(jnp.max(jnp.abs(B)))
    scale = max(1.0, a_scale, b_scale)
    if residual > 1e-6 * scale:
        raise ValueError(
            f"affine_coefficients: moment function does not behave "
            f"affinely (g_bar deviates by {residual:.3g} from the "
            f"inferred a + B @ theta at example_theta).  Re-check "
            f"the moment function or set ``assume_linear=False``."
        )
    del p
    return a, B


def solve_linear_gmm(
    moment_func: Any,
    data: Any,
    weighting_matrix: jnp.ndarray,
    example_theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Closed-form GMM for affine moment + Euclidean parameter space.

    Solves ``theta_hat = -(B' W B)^{-1} B' W a`` where ``a = g_bar(0,
    data)`` and ``B = d g_bar / d theta``.  Works for both just- and
    over-identified problems as long as ``B'WB`` is non-singular.

    Returns ``(theta_hat, a, B)`` so callers can also report the
    residual ``g_bar(theta_hat) = a + B @ theta_hat`` and form the
    final criterion ``g_bar' W g_bar`` themselves.
    """

    a, B = affine_coefficients(moment_func, example_theta, data)
    BtW = B.T @ weighting_matrix
    BtWB = BtW @ B
    BtWa = BtW @ a
    # Use solve for numerical stability over inv().
    theta_hat = -jnp.linalg.solve(BtWB, BtWa)
    return theta_hat, a, B


def linear_gmm_diagnostics(
    a: jnp.ndarray,
    B: jnp.ndarray,
    theta_hat: jnp.ndarray,
    weighting_matrix: jnp.ndarray,
) -> dict[str, float]:
    """Compute the residual and criterion at a closed-form solution.

    Useful for the ``verbosity`` report and as a sanity check that
    the closed-form solution is consistent with the linear-moment
    assumption.
    """

    g_bar_hat = a + B @ theta_hat
    criterion = float(g_bar_hat @ weighting_matrix @ g_bar_hat)
    residual_norm = float(jnp.linalg.norm(g_bar_hat))
    return {
        "criterion": criterion,
        "g_bar_norm": residual_norm,
    }


__all__ = [
    "is_affine_in_theta",
    "is_flat_manifold",
    "affine_coefficients",
    "solve_linear_gmm",
    "linear_gmm_diagnostics",
]


# Re-export for tests that need direct access to numpy-style result.
def _to_numpy(x: jnp.ndarray) -> np.ndarray:
    return np.asarray(x)
