"""Monte Carlo size and power tests for the Kleibergen (2005) K-statistic.

The Kleibergen decomposition splits the efficient J-statistic into two
independent components:

    J_eff = K + S

where under H0 (correct specification):

    K ~ chi2(p)          p = dim(M)    (manifold dimension)
    S ~ chi2(ell - p)    ell = number of moment conditions

K is a score / LM-type statistic whose distribution does *not* depend on
the concentration parameter (instrument strength).  S captures the
overidentifying restrictions, analogous to the Hansen J-stat but
orthogonal to K.

DGP 1 (tests 1--5): Overidentified Gaussian location model
-----------------------------------------------------------
    x_i ~ N(mu_0, 1),  i = 1, ..., N

    g_1(mu) = x - mu
    g_2(mu) = x^2 - (mu^2 + 1)
    g_3(mu) = x^3 - (mu^3 + 3 mu)

    ell = 3, p = 1  =>  df_K = 1, df_S = 2

DGP 2 (test 6): IV regression with weak instruments
----------------------------------------------------
    z_i ~ N(0, I_3)          (3 instruments)
    x_i = pi * z_{i,1} + v_i
    y_i = theta_0 * x_i + e_i      Corr(e, v) = 0.5

    g_i(theta) = z_i * (y_i - theta * x_i)

    ell = 3, p = 1  =>  df_K = 1, df_S = 2
    pi controls identification strength;
    concentration parameter ~ N * pi^2.

References
----------
Kleibergen, F. (2005). "Testing Parameters in GMM Without Assuming
that They Are Identified." Econometrica, 73(4), 1103--1123.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from joblib import Parallel, delayed
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None

pytestmark = pytest.mark.slow

# -----------------------------------------------------------------------
# DGP parameters
# -----------------------------------------------------------------------

N_OBS = 200  # observations per replication
N_REPS = 400  # Monte Carlo replications
ALPHA = 0.05  # nominal significance level
MU_TRUE = 1.0  # true location parameter
N_JOBS = -2  # joblib parallelism: all cores minus one

# Binomial SE of rejection rate under the null
_SE_NULL = np.sqrt(ALPHA * (1 - ALPHA) / N_REPS)


# -----------------------------------------------------------------------
# Moment functions: Gaussian location model
# -----------------------------------------------------------------------


def _gi_correct(theta, observation):
    """Three valid moment conditions for a N(mu, 1) location model."""
    mu = theta[0]
    x = observation
    return jnp.array(
        [
            x - mu,
            x**2 - (mu**2 + 1.0),
            x**3 - (mu**3 + 3.0 * mu),
        ]
    )


def _gi_misspecified(theta, observation):
    """Third moment biased by 2 --- model is mis-specified."""
    mu = theta[0]
    x = observation
    return jnp.array(
        [
            x - mu,
            x**2 - (mu**2 + 1.0),
            x**3 - (mu**3 + 3.0 * mu) - 2.0,
        ]
    )


# -----------------------------------------------------------------------
# Moment functions: IV regression
# -----------------------------------------------------------------------


def _gi_iv(theta, observation):
    """IV moments g_i(theta) = z_i * (y_i - theta * x_i)."""
    y = observation[0]
    x = observation[1]
    z = observation[2:]
    return z * (y - theta[0] * x)


def _generate_iv_data(
    n: int,
    theta_true: float,
    pi: float,
    rng: np.random.Generator,
    rho: float = 0.5,
) -> np.ndarray:
    """Generate IV data with controllable first-stage strength.

    Returns an (n, 5) array with columns [y, x, z1, z2, z3].
    """
    q = 3
    z = rng.normal(size=(n, q))
    v = rng.normal(size=n)
    e_raw = rng.normal(size=n)
    e = rho * v + np.sqrt(1.0 - rho**2) * e_raw
    x = pi * z[:, 0] + v
    y = theta_true * x + e
    return np.column_stack([y, x, z])


# -----------------------------------------------------------------------
# Replication helpers (each receives a SeedSequence child for
# reproducible, statistically independent parallel streams)
# -----------------------------------------------------------------------


def _run_location_replication(
    rep: int,
    child_seed: np.random.SeedSequence,
    gi_fn,
    *,
    two_step: bool = False,
    theta_0: Any | None = None,
) -> dict:
    """Single MC replication for the Gaussian location DGP."""
    rng = np.random.default_rng(child_seed)
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
        result = gmm.estimate(two_step=two_step, verbose=0)
    except Exception as exc:
        return {"rep": rep, "error": str(exc)}

    k_result = result.k_statistic(theta_0=theta_0)
    return {
        "rep": rep,
        "K": k_result.K,
        "S": k_result.S,
        "J": k_result.J,
        "df_K": k_result.df_K,
        "df_S": k_result.df_S,
        "p_K": k_result.p_K,
        "p_S": k_result.p_S,
        "reject_K": int(k_result.p_K < ALPHA),
        "reject_S": int(k_result.p_S < ALPHA),
    }


def _run_iv_replication(
    rep: int,
    child_seed: np.random.SeedSequence,
    pi: float,
    theta_true: float = 1.0,
    *,
    two_step: bool = True,
    theta_0: Any | None = None,
) -> dict:
    """Single MC replication for the IV DGP."""
    rng = np.random.default_rng(child_seed)
    data = _generate_iv_data(N_OBS, theta_true, pi, rng)

    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(
        gi_jax=_gi_iv,
        data=jnp.array(data, dtype=jnp.float64),
        manifold=manifold,
        backend="jax",
    )
    gmm = GMM(restriction, initial_point=jnp.array([0.0], dtype=jnp.float64))

    try:
        result = gmm.estimate(two_step=two_step, verbose=0)
    except Exception as exc:
        return {"rep": rep, "error": str(exc)}

    k_result = result.k_statistic(theta_0=theta_0)
    return {
        "rep": rep,
        "K": k_result.K,
        "S": k_result.S,
        "J": k_result.J,
        "p_K": k_result.p_K,
        "p_S": k_result.p_S,
        "reject_K": int(k_result.p_K < ALPHA),
    }


# -----------------------------------------------------------------------
# Parallel dispatch helpers
# -----------------------------------------------------------------------


def _run_location_mc(
    n_reps: int,
    seed: int,
    gi_fn,
    *,
    two_step: bool = False,
    theta_0: Any | None = None,
) -> list[dict]:
    """Run ``n_reps`` location replications in parallel via joblib."""
    children = np.random.SeedSequence(seed).spawn(n_reps)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_run_location_replication)(
            r, children[r], gi_fn, two_step=two_step, theta_0=theta_0
        )
        for r in range(n_reps)
    )
    return list(results)


def _run_iv_mc(
    n_reps: int,
    seed: int,
    pi: float,
    *,
    two_step: bool = True,
    theta_0: Any | None = None,
) -> list[dict]:
    """Run ``n_reps`` IV replications in parallel via joblib."""
    children = np.random.SeedSequence(seed).spawn(n_reps)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_run_iv_replication)(
            r, children[r], pi, two_step=two_step, theta_0=theta_0
        )
        for r in range(n_reps)
    )
    return list(results)


# -----------------------------------------------------------------------
# Test 1: Decomposition K + S = J
# -----------------------------------------------------------------------


@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_k_plus_s_equals_j():
    r"""Verify K + S = J_{eff} numerically for every replication.

    The Kleibergen decomposition is exact (up to floating-point
    precision), so we check absolute and relative tolerance.
    """
    results = _run_location_mc(50, seed=42, gi_fn=_gi_correct, two_step=True)
    valid = [r for r in results if "error" not in r]
    assert len(valid) >= 40, f"Too many failures: only {len(valid)}/50 succeeded"

    for r in valid:
        np.testing.assert_allclose(
            r["K"] + r["S"],
            r["J"],
            rtol=1e-6,
            err_msg=f"K + S != J at replication {r['rep']}",
        )


# -----------------------------------------------------------------------
# Test 2: K size under H0
# -----------------------------------------------------------------------


@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_k_statistic_size():
    r"""Under H0: theta = theta_0, K rejection rate should be close to alpha.

    K ~ chi2(p) = chi2(1) for the scalar location model.
    With 400 replications the SE of the rejection rate is about 0.011,
    so a +/-4 SE band (~0.006 -- 0.094) is a conservative acceptance
    region.

    The K-statistic must be evaluated at the hypothesised theta_0 (the
    true parameter), not at the estimator.  At the estimator, the
    first-order condition zeros D'Omega^{-1} g_bar, making K degenerate.
    """
    theta_0 = jnp.array([MU_TRUE], dtype=jnp.float64)
    results = _run_location_mc(
        N_REPS,
        seed=2026_04_03,
        gi_fn=_gi_correct,
        two_step=True,
        theta_0=theta_0,
    )

    valid = [r for r in results if "error" not in r]
    assert (
        len(valid) >= 0.95 * N_REPS
    ), f"Too many failures: {N_REPS - len(valid)}/{N_REPS}"

    # All replications should report df_K = 1
    assert all(r["df_K"] == 1 for r in valid)

    rejection_rate = np.mean([r["reject_K"] for r in valid])
    tolerance = 4 * _SE_NULL
    assert ALPHA - tolerance <= rejection_rate <= ALPHA + tolerance, (
        f"K rejection rate {rejection_rate:.3f} outside "
        f"[{ALPHA - tolerance:.3f}, {ALPHA + tolerance:.3f}]"
    )

    # Supplementary: mean(K) should be near E[chi2(1)] = 1
    k_stats = np.array([r["K"] for r in valid])
    mean_k = float(np.mean(k_stats))
    se_mean = float(np.std(k_stats) / np.sqrt(len(k_stats)))
    assert (
        abs(mean_k - 1.0) < 4 * se_mean
    ), f"Mean K = {mean_k:.3f}, expected 1.0 (SE = {se_mean:.3f})"


# -----------------------------------------------------------------------
# Test 3: S size under H0
# -----------------------------------------------------------------------


@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_s_statistic_size():
    r"""Under H0: theta = theta_0, S rejection rate should be close to alpha.

    S ~ chi2(ell - p) = chi2(2) for the scalar location model with
    three moments.  Evaluated at the true theta_0 so that J ~ chi2(ell)
    and S = J - K ~ chi2(ell - p).
    """
    theta_0 = jnp.array([MU_TRUE], dtype=jnp.float64)
    results = _run_location_mc(
        N_REPS,
        seed=2026_04_04,
        gi_fn=_gi_correct,
        two_step=True,
        theta_0=theta_0,
    )

    valid = [r for r in results if "error" not in r]
    assert len(valid) >= 0.95 * N_REPS

    # All replications should report df_S = 2
    assert all(r["df_S"] == 2 for r in valid)

    rejection_rate = np.mean([r["reject_S"] for r in valid])
    tolerance = 4 * _SE_NULL
    assert ALPHA - tolerance <= rejection_rate <= ALPHA + tolerance, (
        f"S rejection rate {rejection_rate:.3f} outside "
        f"[{ALPHA - tolerance:.3f}, {ALPHA + tolerance:.3f}]"
    )

    # Supplementary: mean(S) should be near E[chi2(2)] = 2
    s_stats = np.array([r["S"] for r in valid])
    mean_s = float(np.mean(s_stats))
    se_mean = float(np.std(s_stats) / np.sqrt(len(s_stats)))
    assert (
        abs(mean_s - 2.0) < 4 * se_mean
    ), f"Mean S = {mean_s:.3f}, expected 2.0 (SE = {se_mean:.3f})"


# -----------------------------------------------------------------------
# Test 4: S power under misspecification
# -----------------------------------------------------------------------


@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_s_statistic_power_misspecification():
    r"""Under mis-specification, S should reject frequently.

    The third moment has a fixed bias of 2.  With N = 200 the
    S-statistic (overidentification complement) should have high power
    to detect the violated restrictions.  Evaluated at the true mu so
    that the mis-specification shows up in S, not in K.
    """
    theta_0 = jnp.array([MU_TRUE], dtype=jnp.float64)
    results = _run_location_mc(
        N_REPS,
        seed=2026_04_06,
        gi_fn=_gi_misspecified,
        two_step=True,
        theta_0=theta_0,
    )

    valid = [r for r in results if "error" not in r]
    assert len(valid) >= 0.95 * N_REPS

    rejection_rate = np.mean([r["reject_S"] for r in valid])
    assert rejection_rate >= 0.80, (
        f"S power {rejection_rate:.3f} is too low --- not detecting "
        "the mis-specification"
    )


# -----------------------------------------------------------------------
# Test 5: K size with weak instruments
# -----------------------------------------------------------------------


@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_k_statistic_size_weak_instruments():
    r"""K has correct size even when instruments are weak.

    With pi = 0.1 and N = 200, the concentration parameter is
    approximately N * pi^2 = 2, which represents weak identification
    (first-stage F ~ 0.7).  The K-statistic should nonetheless
    maintain correct size: K ~ chi2(1) under H0: theta = theta_true
    regardless of instrument strength.  This is the key advantage of
    score-based inference over Wald-type inference.

    The K-statistic is evaluated at the true theta_0, not the
    estimator.
    """
    n_reps = 200
    pi_weak = 0.1  # concentration ~ N * pi^2 = 2
    se_null = np.sqrt(ALPHA * (1 - ALPHA) / n_reps)
    theta_0 = jnp.array([1.0], dtype=jnp.float64)  # true theta

    results = _run_iv_mc(
        n_reps,
        seed=2026_04_07,
        pi=pi_weak,
        theta_0=theta_0,
    )

    valid = [r for r in results if "error" not in r]
    # More lenient convergence threshold for weak-IV setting
    assert (
        len(valid) >= 0.70 * n_reps
    ), f"Too many failures: {n_reps - len(valid)}/{n_reps}"

    rejection_rate = np.mean([r["reject_K"] for r in valid])
    tolerance = 4 * se_null
    assert ALPHA - tolerance <= rejection_rate <= ALPHA + tolerance, (
        f"K rejection rate {rejection_rate:.3f} with weak instruments "
        f"outside [{ALPHA - tolerance:.3f}, {ALPHA + tolerance:.3f}]"
    )


# -----------------------------------------------------------------------
# Test 6: K + S = J also holds in IV setting
# -----------------------------------------------------------------------


@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_k_plus_s_equals_j_iv():
    r"""Verify K + S = J for the IV DGP (strong instruments).

    Uses pi = 1.0 (strong instruments, concentration ~ 200) to ensure
    reliable convergence, then checks the decomposition.
    """
    results = _run_iv_mc(50, seed=2026_04_08, pi=1.0)

    valid = [r for r in results if "error" not in r]
    assert len(valid) >= 40, f"Too many IV failures: only {len(valid)}/50 succeeded"

    for r in valid:
        np.testing.assert_allclose(
            r["K"] + r["S"],
            r["J"],
            rtol=1e-6,
            err_msg=f"K + S != J at IV replication {r['rep']}",
        )
