"""Monte Carlo size and power tests for the J-statistic (overidentification test).

The DGP is an over-identified scalar location model:

    x_i ~ N(μ₀, 1),    i = 1, …, N

with three moment conditions (ℓ = 3) for one parameter (p = 1):

    g₁(μ) = x - μ
    g₂(μ) = x² - (μ² + 1)
    g₃(μ) = x³ - (μ³ + 3μ)

All three moments equal zero at the true μ₀, giving df = ℓ - p = 2.
Under the null, J_N(θ̂) →ᵈ χ²(2).

The *power* test mis-specifies the third moment so the model is
structurally invalid; the J-statistic should diverge and reject frequently.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None


# -- DGP parameters --------------------------------------------------------

N_OBS = 200       # observations per replication
N_REPS = 400      # Monte Carlo replications
ALPHA = 0.05      # nominal significance level
MU_TRUE = 1.0     # true location parameter

# Binomial SE of rejection rate under the null
_SE_NULL = np.sqrt(ALPHA * (1 - ALPHA) / N_REPS)


# -- Moment functions -------------------------------------------------------

def _gi_correct(theta, observation):
    """Three valid moment conditions for a N(μ, 1) location model."""
    mu = theta[0]
    x = observation
    return jnp.array([
        x - mu,
        x ** 2 - (mu ** 2 + 1.0),
        x ** 3 - (mu ** 3 + 3.0 * mu),
    ])


def _gi_misspecified(theta, observation):
    """Third moment deliberately wrong — model is mis-specified."""
    mu = theta[0]
    x = observation
    return jnp.array([
        x - mu,
        x ** 2 - (mu ** 2 + 1.0),
        x ** 3 - (mu ** 3 + 3.0 * mu) - 2.0,   # bias of 2
    ])


# -- Helpers -----------------------------------------------------------------

def _run_j_test_replication(
    rep: int,
    rng: np.random.Generator,
    gi_fn,
) -> dict:
    """Single replication: draw data, estimate, return J-stat and p-value."""
    data = rng.normal(loc=MU_TRUE, scale=1.0, size=N_OBS)

    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_fn,
        data=jnp.array(data, dtype=jnp.float64),
        manifold=manifold,
        backend="jax",
    )

    gmm = GMM(restriction, initial_point=jnp.array([0.0], dtype=jnp.float64))

    try:
        result = gmm.estimate(two_step=True, verbose=0)
    except Exception as exc:
        return {"rep": rep, "error": str(exc)}

    j_stat = result.criterion_value
    df = result.degrees_of_freedom
    p_value = float(1.0 - chi2.cdf(j_stat, df)) if df > 0 else float("nan")

    return {
        "rep": rep,
        "j_stat": j_stat,
        "df": df,
        "p_value": p_value,
        "reject": int(p_value < ALPHA),
    }


# -- Tests -------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_j_statistic_size():
    r"""Under the null the J-stat rejection rate should be close to α.

    With 400 replications the SE of the rejection rate under the null is
    about 0.011, so a ±4 SE band (≈ 0.01 – 0.09) gives a conservative
    acceptance region.
    """
    rng = np.random.default_rng(2026_03_15)

    results = [
        _run_j_test_replication(r, rng, _gi_correct)
        for r in range(N_REPS)
    ]

    # Drop any failed replications
    valid = [r for r in results if "error" not in r]
    assert len(valid) >= 0.95 * N_REPS, (
        f"Too many failures: {N_REPS - len(valid)}/{N_REPS}"
    )

    rejection_rate = np.mean([r["reject"] for r in valid])
    j_stats = np.array([r["j_stat"] for r in valid])
    dfs = np.array([r["df"] for r in valid])

    # All replications should have df = 2 (3 moments - 1 parameter)
    assert np.all(dfs == 2), f"Unexpected degrees of freedom: {np.unique(dfs)}"

    # Check empirical rejection rate is near the nominal level.
    # Allow a ±4 SE band around α.
    tolerance = 4 * _SE_NULL
    assert ALPHA - tolerance <= rejection_rate <= ALPHA + tolerance, (
        f"Rejection rate {rejection_rate:.3f} outside "
        f"[{ALPHA - tolerance:.3f}, {ALPHA + tolerance:.3f}] "
        f"(nominal α = {ALPHA})"
    )

    # Supplementary check: mean of J-stats should be near df = 2
    # (the mean of a χ²(2) distribution is 2).
    mean_j = float(np.mean(j_stats))
    se_mean = float(np.std(j_stats) / np.sqrt(len(j_stats)))
    assert abs(mean_j - 2.0) < 4 * se_mean, (
        f"Mean J-stat {mean_j:.3f} too far from χ²(2) mean of 2.0 "
        f"(SE = {se_mean:.3f})"
    )


@pytest.mark.slow
@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_j_statistic_power():
    r"""Under mis-specification the J-stat should reject most of the time.

    The third moment has a fixed bias of 2, which at N = 200 should give
    high power (rejection rate well above α).
    """
    rng = np.random.default_rng(2026_03_16)

    results = [
        _run_j_test_replication(r, rng, _gi_misspecified)
        for r in range(N_REPS)
    ]

    valid = [r for r in results if "error" not in r]
    assert len(valid) >= 0.95 * N_REPS

    rejection_rate = np.mean([r["reject"] for r in valid])

    # With a bias of 2 in one moment and N = 200, power should be very
    # high.  Use 0.80 as a conservative lower bound.
    assert rejection_rate >= 0.80, (
        f"Power {rejection_rate:.3f} is too low — the J-stat is not "
        f"detecting the mis-specification"
    )
