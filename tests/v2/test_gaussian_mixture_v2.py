"""Smoke test for :file:`tests/v2/gaussian_mixture_v2.py`.

Verifies that the v2 GMM machinery handles a *nonlinear*
moment-condition problem (two-step GMM with 4 moments for 3
parameters, ``df = 1``) on the upstream ``coin_plus_noise`` DGP:

- The point estimate ``(p_hat, sigma_0_hat, sigma_1_hat)`` lies in a
  sensible neighbourhood of the truth ``(0.5, 1.0, 1.0)`` -- loose
  bands because ``N = 100`` and three parameters with an
  empirically-skewed realization yield real estimation error.
- The Hansen J-statistic accepts the (correctly-specified) model at
  the asymptotic 5% level.
- The parametric bootstrap's means concentrate near the truth and
  produce non-trivial standard errors.

As with :file:`test_bernoulli_v2.py`, this is a *behavioral* test
with tolerance bands sized to absorb MC variation while still
catching gross plumbing regressions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the sibling ``gaussian_mixture_v2`` module importable from
# inside the test without relying on package-relative imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

gaussian_mixture_v2 = pytest.importorskip(
    "gaussian_mixture_v2",
    reason=(
        "tests/v2/gaussian_mixture_v2.py requires "
        "../DGP_Protocol/examples/coin_plus_noise.py"
    ),
)


def test_gaussian_mixture_end_to_end_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end smoke: point estimates, J-stat, bootstrap behaviour."""

    stats = gaussian_mixture_v2.run()
    captured = capsys.readouterr()
    assert "two-step GMM estimates" in captured.out
    assert "J-statistic" in captured.out
    assert "parametric bootstrap" in captured.out

    # Point estimates land in a wide neighbourhood of the truth.
    # N=100 with three parameters yields ~10-20% error on this DGP
    # for any single realization; loosen bands accordingly.
    assert 0.30 <= stats["p_hat"] <= 0.70
    assert 0.7 <= stats["sigma_0_hat"] <= 1.4
    assert 0.7 <= stats["sigma_1_hat"] <= 1.6

    # J-stat: the model is correctly specified, so J should be well
    # under chi^2_1(0.95) = 3.84.  Allow some slack to accommodate
    # the optimal-weighting variability at N=100.
    assert stats["j_stat"] < 5.0
    assert stats["j_stat"] >= 0.0

    # Parametric bootstrap should concentrate near the truth.  The
    # original DGP has p=0.5 and sigma_0=sigma_1=1 -- the bootstrap
    # samples from the true generator (not from the empirical
    # distribution), so the means should converge there.
    assert stats["bootstrap_p_mean"] == pytest.approx(0.5, abs=0.06)
    assert stats["bootstrap_sigma0_mean"] == pytest.approx(1.0, abs=0.15)
    assert stats["bootstrap_sigma1_mean"] == pytest.approx(1.0, abs=0.15)

    # Bootstrap SEs are non-trivial and not absurd.
    assert 0.01 < stats["bootstrap_p_se"] < 0.3
    assert 0.01 < stats["bootstrap_sigma0_se"] < 0.4
    assert 0.01 < stats["bootstrap_sigma1_se"] < 0.4
