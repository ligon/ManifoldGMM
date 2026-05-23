"""Numerical utilities shared across the econometrics code."""

from __future__ import annotations

import warnings

import numpy as np

from .._warnings import NumericalWarning


def ridge_inverse(
    matrix: np.ndarray,
    *,
    target_condition: float = 1e8,
    initial_ridge: float | None = None,
    max_iterations: int = 50,
) -> tuple[np.ndarray, float]:
    """Return a stabilized inverse of a symmetric matrix.

    The routine symmetrises ``matrix``, then iteratively adds a ridge
    :math:`\\lambda I` until the estimated condition number falls below
    ``target_condition``.  The final inverse and ridge parameter are
    returned.

    Parameters
    ----------
    matrix:
        Square matrix to invert.  Symmetrised internally.
    target_condition:
        Upper bound on the condition number of the (symmetrically
        ridged) matrix before the inverse is computed.  Default ``1e8``.
    initial_ridge:
        Starting value of the ridge.  When ``None`` (default), starts
        at ``0.0``.  Useful for warm-starting the loop on a previously
        seen matrix.
    max_iterations:
        Hard cap on the number of bump-loop iterations before the
        routine bails out and emits a :class:`~manifoldgmm._warnings.NumericalWarning`.
        Default ``50`` -- well above the 2-3 iterations the formula
        predicts on well-behaved inputs, and well below pathological
        regimes that previously hung indefinitely (#18).  The cap is
        only reached on inputs where no finite ridge brings the
        condition under ``target_condition`` -- in that regime the
        routine returns the inverse it has computed (forcing PD via a
        final ridge bump if needed) with a warning naming the achieved
        condition and final ridge.

    Returns
    -------
    inv, ridge:
        The ridge-regularised inverse and the ridge value that produced
        it.  When the loop terminated normally, ``cond(matrix + ridge*I) <= target_condition``.
        When the cap fired, the inverse is the best-effort result and
        :class:`NumericalWarning` was emitted.

    Notes
    -----
    For pathological matrices (``cond >> target_condition`` paired with
    near-machine-epsilon spectral spread), callers can either
    pre-compute an appropriate ``target_condition`` from
    :meth:`manifoldgmm.GMMResult.compute_hessian_cond` and pass it
    explicitly, or set ``warnings.filterwarnings("error", category=NumericalWarning)``
    to escalate the cap-hit into an exception.  Issue #18 has further
    discussion of the regime where this matters and the workarounds.
    """

    if (
        matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
    ):  # pragma: no cover - sanity
        raise ValueError("matrix must be square")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    sym = 0.5 * (matrix + matrix.T)
    ridge = initial_ridge if initial_ridge is not None else 0.0

    last_max_eig = 0.0
    last_min_eig = 0.0
    last_condition = float("inf")
    augmented = sym  # placeholder; overwritten in the loop
    for _ in range(max_iterations):
        augmented = sym + ridge * np.eye(sym.shape[0], dtype=sym.dtype)
        eigvals = np.linalg.eigvalsh(augmented)
        last_max_eig = float(np.max(eigvals))
        last_min_eig = float(np.min(eigvals))
        if last_min_eig <= 0.0:
            ridge = max(ridge * 2.0, 1e-12 if ridge == 0.0 else ridge)
            last_condition = float("inf")
            continue
        last_condition = last_max_eig / last_min_eig
        if last_condition <= target_condition:
            inv = np.linalg.inv(augmented)
            return inv, ridge
        # Increase ridge to hit the target condition roughly.
        desired_min = last_max_eig / target_condition
        new_ridge = max(ridge, desired_min - last_min_eig)
        # Guard against the formula not strictly increasing ridge in
        # near-singular regimes (the historical hang in #18): when the
        # rule fails to move the ridge, fall back to a multiplicative
        # bump so subsequent iterations make progress before the cap.
        if new_ridge <= ridge:
            new_ridge = ridge * 2.0 if ridge > 0.0 else max(1e-12, last_max_eig * 1e-12)
        ridge = new_ridge

    # Cap reached.  The loop body updated ``ridge`` past the last
    # evaluated ``augmented``; rebuild ``augmented`` from the current
    # ``ridge`` so the returned ``(inv, ridge)`` pair is self-consistent.
    augmented = sym + ridge * np.eye(sym.shape[0], dtype=sym.dtype)
    eigvals = np.linalg.eigvalsh(augmented)
    last_max_eig = float(np.max(eigvals))
    last_min_eig = float(np.min(eigvals))
    last_condition = last_max_eig / last_min_eig if last_min_eig > 0.0 else float("inf")
    # Force a positive-definite augmented matrix (the indefinite branch
    # may have left us with a non-PD slice) before inverting.
    if last_min_eig <= 0.0:
        safety = max(abs(last_min_eig), last_max_eig * 1e-12, 1e-300)
        ridge = ridge + safety * 2.0
        augmented = sym + ridge * np.eye(sym.shape[0], dtype=sym.dtype)
        eigvals = np.linalg.eigvalsh(augmented)
        last_max_eig = float(np.max(eigvals))
        last_min_eig = float(np.min(eigvals))
        last_condition = (
            last_max_eig / last_min_eig if last_min_eig > 0.0 else float("inf")
        )
    inv = np.linalg.inv(augmented)
    warnings.warn(
        (
            f"ridge_inverse hit max_iterations={max_iterations} without "
            f"reducing condition below target_condition={target_condition:.3g}; "
            f"returning best-effort inverse with achieved "
            f"condition={last_condition:.3g}, ridge={ridge:.3g}.  Caller may "
            "pass a larger target_condition to short-circuit the bump loop, or "
            "treat the regime as 'no asymptotic distribution available' (#18)."
        ),
        NumericalWarning,
        stacklevel=2,
    )
    return inv, ridge
