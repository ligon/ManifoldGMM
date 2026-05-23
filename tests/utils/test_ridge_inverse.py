"""Tests for ``manifoldgmm.utils.numeric.ridge_inverse``.

#18 motivated by ``k_statistic`` hanging in a non-terminating ridge-bump
loop on severely ill-conditioned ``D'Omega^{-1}D``.  This module exercises
the loop cap (default ``max_iterations=50``) and the
:class:`NumericalWarning` it emits.  The well-conditioned path is also
covered to ensure the cap is invisible there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from manifoldgmm._warnings import NumericalWarning
from manifoldgmm.utils.numeric import ridge_inverse


# ---------------------------------------------------------------------------
# Well-conditioned: no ridge needed, no warning, identity-like behaviour
# ---------------------------------------------------------------------------
def test_well_conditioned_returns_unridged_inverse() -> None:
    """``cond(matrix) < target_condition``: ridge stays at 0; inverse is exact."""

    matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    expected = np.linalg.inv(matrix)

    with warnings.catch_warnings():
        warnings.simplefilter("error", NumericalWarning)
        inv, ridge = ridge_inverse(matrix, target_condition=1e8)

    assert ridge == 0.0
    assert np.allclose(inv, expected, atol=1e-12)


def test_mildly_ill_conditioned_engages_ridge_without_warning() -> None:
    """``cond > target_condition``: ridge engages, loop converges in a couple iters."""

    # cond ~ 1e6: large but well within the ridge formula's reach.
    matrix = np.diag([1.0, 1e-6])

    with warnings.catch_warnings():
        warnings.simplefilter("error", NumericalWarning)
        inv, ridge = ridge_inverse(matrix, target_condition=1e3)

    augmented = matrix + ridge * np.eye(2)
    cond_after = np.linalg.cond(augmented)
    assert cond_after <= 1e3 * 1.0001  # small slack for floating-point round-trip
    # ``inv`` is the actual inverse of the augmented matrix, not the original.
    assert np.allclose(inv @ augmented, np.eye(2), atol=1e-10)


# ---------------------------------------------------------------------------
# Severely ill-conditioned: cap fires, warning emitted, best-effort inverse
# ---------------------------------------------------------------------------
def test_machine_eps_singular_hits_cap_with_warning() -> None:
    """``cond ~ 1e16`` and small ``target_condition``: cap fires after exhausting bumps.

    We construct a matrix whose smallest eigenvalue is near machine
    epsilon relative to the largest; with ``target_condition=1e2`` the
    ridge formula has no finite ridge that brings the conditioning that
    low (because the spectral spread vs. ridge magnitude saturates), so
    the cap fires.
    """

    matrix = np.diag([1.0, 1e-16])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NumericalWarning)
        inv, ridge = ridge_inverse(matrix, target_condition=1.001, max_iterations=10)

    relevant = [w for w in caught if issubclass(w.category, NumericalWarning)]
    assert len(relevant) == 1, (
        f"Expected exactly one NumericalWarning; got {len(relevant)}: "
        f"{[str(w.message) for w in relevant]}"
    )
    msg = str(relevant[0].message)
    assert "max_iterations=10" in msg
    assert "target_condition" in msg
    assert "best-effort inverse" in msg

    # The returned inverse should be finite and approximately satisfy
    # ``inv @ (matrix + ridge*I) = I`` to numerical tolerance.
    assert np.all(np.isfinite(inv))
    augmented = matrix + ridge * np.eye(2)
    assert np.allclose(inv @ augmented, np.eye(2), atol=1e-8)


def test_indefinite_matrix_terminates_under_cap() -> None:
    """Negative-eigenvalue branch caps cleanly instead of doubling forever.

    Pre-fix, an indefinite input with ``eigvalsh`` returning a slightly
    negative eigenvalue would feed the ``min_eig <= 0`` branch which
    doubles ridge.  If the ridge formula then fails to recover a
    positive minimum eigenvalue under the target conditioning, the loop
    would not terminate.  We don't need to reproduce that exactly --
    we just need to confirm that with a low ``max_iterations`` and an
    indefinite input, the routine terminates with a finite PD inverse
    and a warning.
    """

    matrix = np.diag([1.0, -1e-3, 2.0])  # indefinite

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NumericalWarning)
        inv, ridge = ridge_inverse(matrix, target_condition=1.0001, max_iterations=5)

    # At target_condition ~ 1.0 no finite ridge will hit the target on
    # this design, so the cap should fire.
    relevant = [w for w in caught if issubclass(w.category, NumericalWarning)]
    assert len(relevant) >= 1
    # Final ridge made the matrix PD; inverse is finite and correct.
    assert np.all(np.isfinite(inv))
    augmented = matrix + ridge * np.eye(3)
    eigvals = np.linalg.eigvalsh(augmented)
    assert (
        eigvals.min() > 0.0
    ), f"Augmented matrix should be PD after cap; eigvals={eigvals!r}"
    assert np.allclose(inv @ augmented, np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Short-circuit workaround documented in k_statistic docstring + #18
# ---------------------------------------------------------------------------
def test_short_circuit_with_larger_target_condition_skips_warning() -> None:
    """Passing ``target_condition`` above the actual cond returns immediately.

    This is the workaround the ``k_statistic`` docstring points at:
    when the caller has measured ``compute_hessian_cond()`` and knows
    the matrix is ill-conditioned beyond a useful regulariser, they
    can short-circuit the bump loop by setting a generous target.
    """

    matrix = np.diag([1.0, 1e-12])  # cond ~ 1e12

    with warnings.catch_warnings():
        warnings.simplefilter("error", NumericalWarning)
        inv, ridge = ridge_inverse(matrix, target_condition=1e14)

    # With a generous target, the unridged matrix satisfies the
    # condition criterion on iter 1 and the loop returns the raw
    # inverse with ridge == 0.
    assert ridge == 0.0
    assert np.allclose(inv @ matrix, np.eye(2), atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_rejects_max_iterations_below_one() -> None:
    """``max_iterations < 1`` raises ``ValueError``."""

    with pytest.raises(ValueError, match="max_iterations"):
        ridge_inverse(np.eye(2), max_iterations=0)


def test_backward_compatible_signature() -> None:
    """Existing callers (no ``max_iterations`` kwarg) keep working."""

    matrix = np.array([[3.0, 0.0], [0.0, 1.0]])
    # Same call shape as the existing callsites in econometrics/gmm.py
    inv, ridge = ridge_inverse(matrix, target_condition=1e8)
    assert ridge == 0.0
    assert np.allclose(inv @ matrix, np.eye(2), atol=1e-12)
