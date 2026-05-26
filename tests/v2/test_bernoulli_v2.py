"""Smoke test for the end-to-end :file:`tests/v2/bernoulli_v2.py` demo.

Verifies that the v2 GMM machinery produces statistically sensible
output on the fair-coin Bernoulli example: the point estimate matches
the empirical mean, the bootstrap SE matches the analytic SE to
within Monte Carlo error, and the Wald z lies in the asymptotic
acceptance region for the (true) null ``p = 0.5``.

This is intentionally a *behavioral* test, not a pinned-output test
-- the bootstrap is seeded but the test asserts on intervals so the
v2 inference plumbing can evolve without breaking the test for
trivial reasons.

The smoke test exercises:

- v2 ``GMM(moment_func, dgp, manifold, ...)`` construction on a
  pre-bound ``ParametricDGP``;
- :meth:`GMM.estimate` on a just-identified problem;
- :meth:`GMM.bootstrap` over 500 parametric resamples;
- Bootstrap-vs-analytic SE agreement to within MC tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the sibling ``bernoulli_v2`` module (this directory) importable
# from inside the test without relying on package-relative imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Skip cleanly if the sibling DGP_Protocol/examples isn't reachable
# (e.g., the bernoulli_v2 module fails to import its fair_coin DGP).
bernoulli_v2 = pytest.importorskip(
    "bernoulli_v2",
    reason=(
        "tests/v2/bernoulli_v2.py requires " "../DGP_Protocol/examples/fair_coin.py"
    ),
)


def test_bernoulli_end_to_end_run(capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end smoke: point estimate, bootstrap, Wald test agree."""

    stats = bernoulli_v2.run()
    # Captured stdout is informational; the assertions below carry
    # the semantic guarantees.
    captured = capsys.readouterr()
    assert "p_hat" in captured.out
    assert "bootstrap mean" in captured.out
    assert "Wald test" in captured.out

    # The bound observation has 51 heads out of 100 (seed 2026 in
    # DGP_Protocol/examples/fair_coin.py).  The just-identified GMM
    # estimator equals the empirical mean to optimizer tolerance.
    assert stats["p_hat"] == pytest.approx(0.51, abs=1e-4)

    # Analytic Bernoulli SE: sqrt(p_hat * (1 - p_hat) / N), N = 100.
    assert stats["analytic_se"] == pytest.approx(0.05, abs=5e-4)

    # Bootstrap should converge to the truth (0.5) under the
    # parametric DGP and produce SE close to the analytic value.
    # Tolerances are loose enough to absorb MC variation across
    # numpy / pymanopt / jax versions while still catching gross
    # plumbing breakage.
    assert stats["bootstrap_mean"] == pytest.approx(0.5, abs=0.02)
    assert stats["bootstrap_se"] == pytest.approx(stats["analytic_se"], abs=0.015)

    # Wald z for the fair-coin null: the observed mean of 0.51 lies
    # well within the 95% acceptance region.
    assert abs(stats["wald_z"]) < 1.96
