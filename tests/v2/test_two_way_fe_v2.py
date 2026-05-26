"""Smoke test for :file:`tests/v2/two_way_fe_v2.py`.

Verifies that the two-way fixed-effects panel example runs end-to-end
and produces statistically sensible output:

- Point estimate ``beta_hat`` lies in a neighbourhood of the truth
  ``(2, -1)`` (loose band; N = 30 with two FE dimensions absorbed
  leaves limited residual signal).
- Just-identified criterion at the optimum is numerically zero.
- Parametric bootstrap mean concentrates near the truth.
- Empirical (cluster-by-unit) bootstrap mean concentrates near the
  point estimate.
- Both bootstrap SEs are non-trivial and roughly agree, demonstrating
  that the sampling-design switch from ``ParametricDGP`` to
  ``EmpiricalDGP`` propagates correctly through the v2 GMM machinery
  without any estimator-side changes.

Behavioral test (tolerance bands sized for MC variation at the
example's panel dimensions), not a pinned-output test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the sibling ``two_way_fe_v2`` module importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

two_way_fe_v2 = pytest.importorskip(
    "two_way_fe_v2",
    reason="tests/v2/two_way_fe_v2.py requires dgp_protocol + jax + pymanopt",
)


def test_two_way_fe_end_to_end_run(capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end smoke: estimate, parametric bootstrap, empirical bootstrap."""

    stats = two_way_fe_v2.run()
    captured = capsys.readouterr()
    assert "FWL-GMM point estimate" in captured.out
    assert "parametric bootstrap" in captured.out
    assert "empirical cluster-by-unit bootstrap" in captured.out

    beta_hat = np.asarray(stats["beta_hat"], dtype=float)
    beta_true = two_way_fe_v2.BETA_TRUE
    # Point estimate within ~0.1 of truth (N=30, T=6, two FE
    # dimensions absorbed; ~150 effective observations).
    assert np.all(
        np.abs(beta_hat - beta_true) < 0.15
    ), f"beta_hat = {beta_hat} too far from truth {beta_true}"

    # Just-identified: criterion at optimum is essentially zero.
    assert stats["criterion_value"] < 1e-10

    # Parametric bootstrap: mean concentrates near the truth (this
    # is the sampling distribution under the *true* DGP).
    pm = np.asarray(stats["bootstrap_parametric_mean"], dtype=float)
    pse = np.asarray(stats["bootstrap_parametric_se"], dtype=float)
    assert np.all(
        np.abs(pm - beta_true) < 0.05
    ), f"parametric bootstrap mean {pm} too far from truth {beta_true}"
    # SEs are non-trivial and modest (we expect O(0.03-0.1) at N=30).
    assert np.all(0.005 < pse) and np.all(
        pse < 0.2
    ), f"parametric bootstrap SE {pse} outside reasonable band"

    # Empirical bootstrap: mean concentrates near the point estimate.
    em = np.asarray(stats["bootstrap_empirical_mean"], dtype=float)
    ese = np.asarray(stats["bootstrap_empirical_se"], dtype=float)
    assert np.all(
        np.abs(em - beta_hat) < 0.1
    ), f"empirical bootstrap mean {em} too far from point estimate {beta_hat}"
    assert np.all(0.005 < ese) and np.all(
        ese < 0.2
    ), f"empirical bootstrap SE {ese} outside reasonable band"

    # The two bootstrap SEs should be roughly the same order of
    # magnitude (one is the sampling distribution under the true
    # DGP; the other is the cluster-by-unit resample of the
    # observed realization).  Allow a 4x ratio either way.
    ratio = pse / ese
    assert np.all((0.25 < ratio) & (ratio < 4.0)), (
        f"parametric vs empirical SE ratio {ratio} suggests the two "
        f"bootstrap paths disagree implausibly -- v2 plumbing bug?"
    )
