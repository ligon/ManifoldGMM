"""Numerical utilities shared across the econometrics code."""

from __future__ import annotations

import numpy as np


def ridge_inverse(
    matrix: np.ndarray,
    *,
    target_condition: float = 1e8,
    initial_ridge: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return a stabilized inverse of a symmetric matrix.

    The routine symmetrises ``matrix``, then iteratively adds a ridge ``λ I`` until
    the estimated condition number falls below ``target_condition``. The final
    inverse and ridge parameter are returned.
    """

    if (
        matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
    ):  # pragma: no cover - sanity
        raise ValueError("matrix must be square")

    sym = 0.5 * (matrix + matrix.T)
    ridge = initial_ridge if initial_ridge is not None else 0.0

    while True:
        augmented = sym + ridge * np.eye(sym.shape[0], dtype=sym.dtype)
        eigvals = np.linalg.eigvalsh(augmented)
        max_eig = float(np.max(eigvals))
        min_eig = float(np.min(eigvals))
        if min_eig <= 0.0:
            ridge = max(ridge * 2.0, 1e-12 if ridge == 0.0 else ridge)
            continue
        condition = max_eig / min_eig
        if condition <= target_condition:
            inv = np.linalg.inv(augmented)
            return inv, ridge
        # Increase ridge to hit the target condition roughly.
        desired_min = max_eig / target_condition
        ridge = max(ridge, desired_min - min_eig)
