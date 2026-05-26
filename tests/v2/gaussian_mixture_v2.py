"""Gaussian-mixture v2 GMM example: nonlinear moments via coin_plus_noise.

Generalization of :file:`bernoulli_v2.py` to a two-component Gaussian
mixture with the *means fixed* at 0 and 1 (matching the
``coin_plus_noise`` DGP structure) and *three estimated parameters*::

    Y_i ~ p * N(1, sigma_1^2) + (1 - p) * N(0, sigma_0^2)

The truth under ``../DGP_Protocol/examples/coin_plus_noise.py`` is
``p = 0.5``, ``sigma_0 = sigma_1 = 1`` (Y is a 50/50 mixture of
N(0, 1) and N(1, 1), so E[Y] = 0.5 and Var[Y] = 1.25).

Moment conditions (k = 4, p = 3, ``df = 1`` for the J-statistic)::

    m_1 = E[Y]   = p
    m_2 = E[Y^2] = p (1 + sigma_1^2) + (1 - p) sigma_0^2
    m_3 = E[Y^3] = p (1 + 3 sigma_1^2)
    m_4 = E[Y^4] = p (1 + 6 sigma_1^2 + 3 sigma_1^4)
                  + (1 - p) 3 sigma_0^4

Identification: m_1 pins down ``p``; m_3 pins down ``sigma_1`` given
``p``; m_2 pins down ``sigma_0`` given ``p`` and ``sigma_1``; m_4 is
the over-identifying restriction.  Unlike the Bernoulli example,
the moments are *nonlinear* in ``sigma_0`` and ``sigma_1`` so the
GMM machinery does real work.

Parameterization
----------------
The natural parameter space ``(p, sigma_0, sigma_1) in (0, 1) x R+
x R+`` is reparameterized to unconstrained Euclidean(3)::

    theta = [logit_p, log_v_0, log_v_1]
    p     = sigmoid(theta[0])
    v_j   = exp(theta[j+1])  (= sigma_j^2 -- variance, not std)

We parameterize the *variance* (v_j = sigma_j^2) rather than the
std because the moment equations depend on sigma_j only through
sigma_j^2; parameterizing the variance dodges the (sigma, -sigma)
sign ambiguity that an unconstrained ``Euclidean(3)`` on raw
``sigma_j`` would expose.

Run directly::

    python tests/v2/gaussian_mixture_v2.py

Or load into IPython::

    %run tests/v2/gaussian_mixture_v2.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Suppress pymanopt's harmless trust-region degenerate-step
# RuntimeWarning that fires whenever a step converges in one inner
# iteration (the trust-region tau ratio becomes 0/0).
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)

# The coin_plus_noise example lives in the sibling DGP_Protocol
# repo.  Internally it does ``from examples.fair_coin import ...``,
# so ``DGP_Protocol/`` (the parent of ``examples/``) must be on
# ``sys.path`` for the ``examples`` package to resolve.  We also put
# ``examples/`` itself on the path so the top-level module name
# ``coin_plus_noise`` is importable directly.
#
# Defensive: ManifoldGMM has its own (empty) ``examples`` package at
# its root, which shadows DGP_Protocol's.  Drop any stale ``examples``
# from ``sys.modules`` and insert DGP_Protocol's paths at the head of
# ``sys.path`` so its ``examples`` wins.
_DGP_ROOT = Path(__file__).resolve().parents[3] / "DGP_Protocol"
_DGP_EXAMPLES = _DGP_ROOT / "examples"
for _stale in [m for m in sys.modules if m == "examples" or m.startswith("examples.")]:
    del sys.modules[_stale]
for _path in (_DGP_EXAMPLES, _DGP_ROOT):
    str_path = str(_path)
    # Reinsert at position 0 even if already present, so DGP_Protocol
    # outranks ManifoldGMM-rooted candidates added by pytest.
    while str_path in sys.path:
        sys.path.remove(str_path)
    sys.path.insert(0, str_path)
from coin_plus_noise import dgp as mixture_dgp  # noqa: E402
from manifoldgmm import GMM, Manifold  # noqa: E402
from pymanopt.manifolds import Euclidean  # noqa: E402


# --------------------------------------------------------------------------
# Reparameterization helpers
# --------------------------------------------------------------------------
def _unpack(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map unconstrained ``theta in R^3`` to ``(p, v_0, v_1)``."""

    p = jax.nn.sigmoid(theta[0])
    v_0 = jnp.exp(theta[1])
    v_1 = jnp.exp(theta[2])
    return p, v_0, v_1


# --------------------------------------------------------------------------
# Moment conditions (vectorized over rows of data)
# --------------------------------------------------------------------------
def g(theta: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """Per-row contributions to four moment conditions, shape ``(N, 4)``.

    ``data`` has shape ``(N, 1)``; flatten to ``(N,)`` for elementwise
    power operations.
    """

    p, v_0, v_1 = _unpack(theta)
    # Implied population moments under the mixture model.
    m_1 = p
    m_2 = p * (1.0 + v_1) + (1.0 - p) * v_0
    m_3 = p * (1.0 + 3.0 * v_1)
    m_4 = p * (1.0 + 6.0 * v_1 + 3.0 * v_1**2) + (1.0 - p) * 3.0 * v_0**2

    y = data[:, 0]
    return jnp.stack(
        [
            y - m_1,
            y**2 - m_2,
            y**3 - m_3,
            y**4 - m_4,
        ],
        axis=1,
    )


# --------------------------------------------------------------------------
# End-to-end run
# --------------------------------------------------------------------------
def run() -> dict[str, float]:
    """Run the example end-to-end and return summary statistics.

    Returns a dict with reparameterized estimates plus the recovered
    natural parameters ``p``, ``sigma_0``, ``sigma_1`` and the
    realized empirical moments.  Useful for the smoke test.
    """

    obs = np.asarray(mixture_dgp.data, dtype=float)
    N = obs.shape[0]
    y = obs[:, 0]

    print(f"observed realization:   shape={obs.shape}  " f"dtype={obs.dtype}")
    emp_m = [float((y**k).mean()) for k in (1, 2, 3, 4)]
    print(
        "empirical moments:      "
        f"m1={emp_m[0]:.4f}  m2={emp_m[1]:.4f}  "
        f"m3={emp_m[2]:.4f}  m4={emp_m[3]:.4f}"
    )
    print(
        "model truth:            m1=0.5000  m2=1.5000  "
        "m3=2.0000  m4=6.5000  (p=0.5, sigma_0=sigma_1=1)"
    )
    print()

    M = Manifold.from_pymanopt(Euclidean(3))
    # Initial point: theta = (0, 0, 0) <-> (p=0.5, v_0=v_1=1), the
    # truth under coin_plus_noise.
    theta_0 = jnp.zeros(3)

    gmm = GMM(
        moment_func=g,
        dgp=mixture_dgp,
        manifold=M,
        initial_point=theta_0,
    )

    # Two-step GMM: start with identity weighting, re-weight with the
    # inverse moment covariance at the first-step estimate.
    quiet = {"verbosity": 0}
    result = gmm.estimate(two_step=True, optimizer_kwargs=quiet)
    theta_hat = np.asarray(result.theta_array)
    p_hat, v0_hat, v1_hat = (
        float(1.0 / (1.0 + np.exp(-theta_hat[0]))),
        float(np.exp(theta_hat[1])),
        float(np.exp(theta_hat[2])),
    )
    s0_hat = float(np.sqrt(v0_hat))
    s1_hat = float(np.sqrt(v1_hat))

    print("two-step GMM estimates:")
    print(f"  theta (raw)         = {theta_hat}")
    print(f"  p_hat               = {p_hat:.4f}     (true 0.5)")
    print(f"  sigma_0_hat         = {s0_hat:.4f}     (true 1.0)")
    print(f"  sigma_1_hat         = {s1_hat:.4f}     (true 1.0)")
    print()

    # J-statistic of overidentification.  Under H_0 (model correct)
    # and the optimal weighting, the criterion at the optimum is
    # asymptotically chi^2 with df = k - p = 4 - 3 = 1.
    j_stat = float(result.criterion_value)
    print(f"Hansen J-statistic:     J = {j_stat:.4f}  (df = 1)")
    print(
        "  asymptotic critical chi^2(1) at alpha=0.05 is 3.841; "
        f"{'reject' if j_stat > 3.841 else 'accept'}"
    )
    print()

    # Parametric bootstrap to give a sanity SE on p_hat.  Each
    # replication draws fresh from the original coin_plus_noise DGP,
    # refits.  Small B since each refit is a 3-D nonlinear GMM.
    B = 80
    print(f"parametric bootstrap:   B={B}, seed=42 ...")
    bs = gmm.bootstrap(B=B, seed=42, two_step=True, optimizer_kwargs=quiet)
    p_bs = np.array(
        [float(1.0 / (1.0 + np.exp(-float(np.asarray(t.value)[0])))) for t in bs.thetas]
    )
    s0_bs = np.array(
        [float(np.sqrt(np.exp(float(np.asarray(t.value)[1])))) for t in bs.thetas]
    )
    s1_bs = np.array(
        [float(np.sqrt(np.exp(float(np.asarray(t.value)[2])))) for t in bs.thetas]
    )
    print(f"  p:       mean = {p_bs.mean():.4f}     SE = {p_bs.std(ddof=1):.4f}")
    print(f"  sigma_0: mean = {s0_bs.mean():.4f}     SE = {s0_bs.std(ddof=1):.4f}")
    print(f"  sigma_1: mean = {s1_bs.mean():.4f}     SE = {s1_bs.std(ddof=1):.4f}")

    return {
        "p_hat": p_hat,
        "sigma_0_hat": s0_hat,
        "sigma_1_hat": s1_hat,
        "j_stat": j_stat,
        "bootstrap_p_mean": float(p_bs.mean()),
        "bootstrap_p_se": float(p_bs.std(ddof=1)),
        "bootstrap_sigma0_mean": float(s0_bs.mean()),
        "bootstrap_sigma0_se": float(s0_bs.std(ddof=1)),
        "bootstrap_sigma1_mean": float(s1_bs.mean()),
        "bootstrap_sigma1_se": float(s1_bs.std(ddof=1)),
        "n_obs": N,
    }


if __name__ == "__main__":
    run()
