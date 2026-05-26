"""Two-way fixed-effects panel regression, v2 GMM via FWL projection.

End-to-end demonstration of three v2 design ideas at once:

1. *Panel data fits the (rows = units) convention without extension.*
   Each row of the data array is one unit's full T-vector (or T x p
   block); the GMM framework's iid resampling over rows IS the
   cluster-by-unit bootstrap.  No estimator-side cluster logic needed.

2. *Nuisance parameters (the FE intercepts) are concentrated out via
   Frisch-Waugh-Lovell inside the moment function.*  The optimizer
   sees only ``beta``; the parameter manifold is ``Euclidean(p)``.

3. *The sampling design is an aspect of the DGP, not the estimator.*
   The same ``GMM(moment_func, dgp, ...)`` works whether ``dgp`` is
   a parametric panel simulator (power study) or an
   ``EmpiricalDGP`` over an observed panel (raw-data cluster
   bootstrap); the GMM doesn't know or care.

Model
-----
For unit ``i = 1..N`` and time ``t = 1..T``::

    y_{i,t} = alpha_i + gamma_t + x_{i,t}' beta + eps_{i,t}

with alpha_i ~ N(0, sigma_alpha^2), gamma_t ~ N(0, sigma_gamma^2),
eps ~ N(0, sigma^2) all independent.  Panels can be unbalanced;
missing cells are marked with ``NaN`` in the wide-format data
array.  Truth here: ``beta = (2, -1)``, ``sigma_alpha = sigma_gamma
= 0.5``, ``sigma = 0.3``.

Moment condition
----------------
After Frisch-Waugh-Lovell projection of ``y`` and ``x`` onto the
orthogonal complement of the (unit dummies, time dummies, constant)
nuisance regressors,

    E[ x_tilde_{i,t} (y_tilde_{i,t} - x_tilde_{i,t}' beta) ] = 0,

with the per-unit aggregation summing over the observed time
periods within unit ``i``::

    g_i(beta, X_i) = sum_{t : observed} x_tilde_{i,t}
                     * (y_tilde_{i,t} - x_tilde_{i,t}' beta).

The framework's ``(N, k)``-shaped vectorized moment makes this the
natural unit-of-clustering: each row of ``g(beta, data)`` is one
unit's contribution, summed over time.

FWL for unbalanced panels
-------------------------
The within transformation ``z - z_bar_i - z_bar_t + z_bar`` only
gives the exact FE projection on *balanced* panels.  For unbalanced
panels we apply the general FWL projector ``M_D = I - D (D'D)^-1
D'`` where ``D`` is the (N + T - 1)-column dummy matrix (unit
dummies + time dummies minus one for the absorbed constant), masked
to the observed cells.  The masked least-squares problem is solved
via :func:`jax.numpy.linalg.lstsq`, giving the exact within
estimator on any observation pattern.  This mirrors the spirit of
the ``fwl_regression`` helper in :file:`MetricsMiscellany`: the
projection is the right primitive, and it works without a balanced-
panel assumption.

Run directly::

    python tests/v2/two_way_fe_v2.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import dgp_protocol as dp
import jax
import jax.numpy as jnp
import numpy as np

# Same pymanopt RuntimeWarning as in the other v2 examples: harmless
# trust-region degenerate-step under our scaling.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)

# Ensure ManifoldGMM is importable when run as a script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from manifoldgmm import GMM, Manifold  # noqa: E402
from manifoldgmm.econometrics.gmm import IdentityWeighting  # noqa: E402
from pymanopt.manifolds import Euclidean  # noqa: E402

# --------------------------------------------------------------------------
# Parametric DGP
# --------------------------------------------------------------------------
# Panel dimensions and truth.
N: int = 30
T: int = 6
P: int = 2  # number of x regressors
BETA_TRUE: np.ndarray = np.array([2.0, -1.0])
SIGMA_ALPHA: float = 0.5
SIGMA_GAMMA: float = 0.5
SIGMA_EPS: float = 0.3
# Probability that any given (i, t) cell is observed (1 - missingness).
KEEP_PROB: float = 0.85
# Independent seed for the observation mask -- distinct from the
# DGP's own draw seed so the missingness pattern is fixed.
MASK_SEED: int = 2026


def _draw_mask(rng: np.random.Generator) -> np.ndarray:
    """Draw an (N, T) Bernoulli observation mask, ensuring no fully-missing
    unit or time period (which would make the FE dummy matrix rank-deficient).
    """

    while True:
        mask = rng.binomial(1, KEEP_PROB, size=(N, T)).astype(bool)
        if mask.any(axis=1).all() and mask.any(axis=0).all():
            return mask


# Fix the observation pattern once at module load time; it serves both
# as a structural feature of the parametric DGP (which always returns
# data with this same NaN pattern) and as the mask for the bound
# observation.
_OBS_MASK: np.ndarray = _draw_mask(np.random.default_rng(MASK_SEED))


def two_way_fe_generator(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    """Generate one (N, T, 1+P) panel realization from the two-way FE model.

    Ignores ``shape`` (the panel dimensions are fixed by module
    constants ``N``, ``T``, ``P``).  Returns a numpy array with
    ``NaN`` where the fixed mask ``_OBS_MASK`` says the cell is
    missing.
    """

    del shape  # the generator is dimensioned by module constants
    alpha = rng.normal(0.0, SIGMA_ALPHA, size=N)
    gamma = rng.normal(0.0, SIGMA_GAMMA, size=T)
    x = rng.normal(0.0, 1.0, size=(N, T, P))
    eps = rng.normal(0.0, SIGMA_EPS, size=(N, T))
    y = alpha[:, None] + gamma[None, :] + (x @ BETA_TRUE) + eps
    # Mark missing cells as NaN in both y and x.
    y_masked = np.where(_OBS_MASK, y, np.nan)
    x_masked = np.where(_OBS_MASK[..., None], x, np.nan)
    return np.concatenate([y_masked[..., None], x_masked], axis=-1)


# Bound observation: one draw with a seed distinct from the DGP's
# own draw seed (mirrors the ``OBS_SEED`` convention used in the
# fair_coin and coin_plus_noise upstream examples).
OBS_DRAW_SEED: int = 2027
_OBSERVATION: np.ndarray = two_way_fe_generator(
    np.random.default_rng(OBS_DRAW_SEED), shape=()
)

# The DGP itself: a ParametricDGP that simulates fresh panels from
# the same two-way FE model.  Each ``draw`` produces an (N, T, 1+P)
# array with the same NaN pattern as ``_OBSERVATION``.
panel_dgp = dp.ParametricDGP(
    generator=two_way_fe_generator,
    default_shape=(N, T, 1 + P),
    observation=_OBSERVATION,
    seed=0,
)


# --------------------------------------------------------------------------
# FWL projection (precomputed nuisance design)
# --------------------------------------------------------------------------
# Construct the long-form (NT, N + T - 1) dummy design matrix once.
# The mask is *not* precomputed because under the empirical
# cluster-by-unit bootstrap each replication's mask depends on which
# units were resampled (rows are shuffled wholesale).  Each
# moment-function call therefore recovers the mask from the data's
# NaN pattern and applies a fresh FWL projection via lstsq.
_idx_unit_long = jnp.repeat(jnp.arange(N), T)  # (NT,)
_idx_time_long = jnp.tile(jnp.arange(T), N)  # (NT,)
_D_unit = jax.nn.one_hot(_idx_unit_long, N)  # (NT, N)
_D_time = jax.nn.one_hot(_idx_time_long, T)  # (NT, T)
# Drop one time dummy so the unit dummies span the absorbed constant.
_D = jnp.concatenate([_D_unit, _D_time[:, 1:]], axis=1)  # (NT, N + T - 1)


# --------------------------------------------------------------------------
# Moment function: FWL-projected, per-unit aggregated
# --------------------------------------------------------------------------
def g(beta: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """Two-way FE GMM moment, per-unit aggregated, NaN-aware.

    Parameters
    ----------
    beta:
        Length-``P`` parameter vector.
    data:
        Wide-format panel of shape ``(N, T, 1 + P)``.  Slice
        ``data[..., 0]`` is ``y``; ``data[..., 1:]`` is ``x``.
        ``NaN`` marks missing cells.

    Returns
    -------
    Shape ``(N, P)`` per-unit moment vectors, one row per unit,
    summed over the unit's observed time periods.

    Notes
    -----
    The FWL projection is recomputed each call so this function is
    correct under any bootstrap that may permute the per-row NaN
    patterns (empirical cluster-by-unit in particular).  Cost per
    call is dominated by two SVD-based ``lstsq`` solves of
    ``(NT, N + T - 1)`` size.
    """

    y = data[..., 0]  # (N, T)
    x = data[..., 1:]  # (N, T, P)
    mask = (~jnp.isnan(y)).astype(jnp.float64)  # (N, T): 1 where observed

    # Replace NaN with 0 so the multiplications absorb missing cells
    # without propagating NaN.
    y0 = jnp.nan_to_num(y).reshape(-1)  # (NT,)
    x0 = jnp.nan_to_num(x).reshape(-1, P)  # (NT, P)
    mask_long = mask.reshape(-1)  # (NT,)

    # Apply mask to the design so missing observations contribute 0
    # to the least-squares projection.
    D_m = _D * mask_long[:, None]  # (NT, N + T - 1)
    y_m = y0 * mask_long  # (NT,)
    x_m = x0 * mask_long[:, None]  # (NT, P)

    # FWL projection via SVD-based least squares.  ``lstsq`` handles
    # any rank deficiency from masked rows or empty time/unit
    # categories in the resample.
    coef_y, *_ = jnp.linalg.lstsq(D_m, y_m, rcond=None)
    coef_x, *_ = jnp.linalg.lstsq(D_m, x_m, rcond=None)
    y_tilde = y_m - D_m @ coef_y  # (NT,)
    x_tilde = x_m - D_m @ coef_x  # (NT, P)

    # Per-observation moment: x_tilde * eps_tilde, masked.
    eps_tilde = (y_tilde - x_tilde @ beta) * mask_long  # (NT,)
    m_obs = x_tilde * eps_tilde[:, None]  # (NT, P)

    # Aggregate to per-unit (sum over T inside each unit).  The
    # resulting (N, P) array's row i is one unit's full moment
    # contribution -- the natural cluster-of-clustering unit.
    return m_obs.reshape(N, T, P).sum(axis=1)


# --------------------------------------------------------------------------
# End-to-end run
# --------------------------------------------------------------------------
def run() -> dict[str, float | np.ndarray]:
    """Run the example end-to-end and return summary statistics."""

    print(
        f"panel: N={N} units, T={T} periods, {int(_OBS_MASK.sum())} "
        f"of {N * T} cells observed ({100 * _OBS_MASK.mean():.1f}% balanced)"
    )
    print(
        f"truth: beta={BETA_TRUE},  sigma_alpha={SIGMA_ALPHA},  "
        f"sigma_gamma={SIGMA_GAMMA},  sigma={SIGMA_EPS}"
    )
    print()

    M = Manifold.from_pymanopt(Euclidean(P))
    # IdentityWeighting: just-identified problem (k == p == P), so any
    # positive-definite weighting yields the same point estimate; the
    # default CUEWeighting introduces non-convexity into the criterion
    # surface unnecessarily and can derail the optimizer on noisy
    # bootstrap replications.
    gmm = GMM(
        moment_func=g,
        dgp=panel_dgp,
        manifold=M,
        initial_point=jnp.zeros(P),
        weighting=IdentityWeighting(P),
    )

    # Point estimate.  Just-identified (k = P = 2 moments, p = 2
    # parameters) so two-step and one-step are equivalent and the
    # criterion at the optimum is essentially zero.
    quiet = {"verbosity": 0}
    result = gmm.estimate(optimizer_kwargs=quiet)
    beta_hat = np.asarray(result.theta_array, dtype=float)
    print("FWL-GMM point estimate:")
    print(f"  beta_hat = {beta_hat}     (true {BETA_TRUE})")
    print(f"  criterion at optimum = {result.criterion_value:.4e}")
    print()

    # Parametric bootstrap: each replication draws a fresh panel from
    # the true two-way FE generator (new alpha_i, gamma_t, eps_{i,t},
    # and x_{i,t}, but the same missing-cell pattern) and refits.
    # This is the *parametric* bootstrap of the sampling distribution
    # of beta_hat under the true DGP.
    B = 25
    print(f"parametric bootstrap (fresh panels from the DGP): B={B}, seed=42 ...")
    bs = gmm.bootstrap(B=B, seed=42, optimizer_kwargs=quiet)
    bs_betas = np.stack([np.asarray(t.value, dtype=float) for t in bs.thetas])
    print(f"  bootstrap mean = {bs_betas.mean(axis=0)}")
    print(f"  bootstrap SE   = {bs_betas.std(axis=0, ddof=1)}")
    print()

    # And now the *empirical* bootstrap: take the observed
    # realization, build an EmpiricalDGP whose default ``IIDSampling``
    # resamples rows -- which IS the cluster-by-unit bootstrap, since
    # each row of the wide-format panel is one unit's full T-vector.
    # The GMM, moment function, and bootstrap machinery are
    # bit-identical to the parametric case; only the DGP changes.
    print(
        "empirical cluster-by-unit bootstrap "
        "(EmpiricalDGP over the observed panel): same B and seed ..."
    )
    emp_dgp = dp.EmpiricalDGP(observation=panel_dgp.data, seed=0)
    emp_gmm = GMM(
        moment_func=g,
        dgp=emp_dgp,
        manifold=M,
        initial_point=jnp.zeros(P),
        weighting=IdentityWeighting(P),
    )
    emp_result = emp_gmm.estimate(optimizer_kwargs=quiet)
    emp_beta_hat = np.asarray(emp_result.theta_array, dtype=float)
    # Point estimate is identical (same data, same moment function)
    # up to optimizer tolerance:
    assert np.allclose(beta_hat, emp_beta_hat, atol=1e-6)
    emp_bs = emp_gmm.bootstrap(B=B, seed=42, optimizer_kwargs=quiet)
    emp_bs_betas = np.stack([np.asarray(t.value, dtype=float) for t in emp_bs.thetas])
    print(f"  bootstrap mean = {emp_bs_betas.mean(axis=0)}")
    print(f"  bootstrap SE   = {emp_bs_betas.std(axis=0, ddof=1)}")
    print()

    print(
        "(parametric bootstrap is the sampling distribution under "
        "the *true* DGP; empirical bootstrap is the cluster-by-unit "
        "resample of the observed panel.  Both flow through the same "
        "GMM machinery -- the sampling design lives entirely in the DGP.)"
    )

    return {
        "beta_hat": beta_hat,
        "criterion_value": float(result.criterion_value),
        "bootstrap_parametric_mean": bs_betas.mean(axis=0),
        "bootstrap_parametric_se": bs_betas.std(axis=0, ddof=1),
        "bootstrap_empirical_mean": emp_bs_betas.mean(axis=0),
        "bootstrap_empirical_se": emp_bs_betas.std(axis=0, ddof=1),
    }


if __name__ == "__main__":
    run()
