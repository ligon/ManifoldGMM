"""Bernoulli v2 GMM example: estimate ``p`` from fair-coin draws.

End-to-end demonstration of the v2 ManifoldGMM API (``GMM(moment_func,
dgp, ...)``) consuming the ``fair_coin`` ``ParametricDGP`` from the
sibling DGP_Protocol repo.

The model is ``X_i ~ Bernoulli(p)`` with ``p in [0, 1]``.  The moment
condition

    E[X - p] = 0

is just-identified (one parameter, one moment), so the GMM analog
estimator coincides with the sample mean of the observed
realization.  The example illustrates:

- Constructing a v2 GMM on a DGP that arrives pre-bound to an
  observation (see ``../DGP_Protocol/examples/fair_coin.py``).
- Computing the point estimate via :meth:`GMM.estimate`.
- Running a parametric bootstrap via :meth:`GMM.bootstrap` -- each
  replication draws afresh from the original ``fair_coin``
  generator (i.e., from ``Bernoulli(0.5)``).
- Computing the analytic Wald-style standard error
  ``SE = sqrt(p_hat * (1 - p_hat) / N)`` and a two-sided test of
  ``H_0: p = 0.5``.

Run directly::

    python tests/v2/bernoulli_v2.py

Or load into IPython::

    %run tests/v2/bernoulli_v2.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Suppress pymanopt's trust-region RuntimeWarning that fires when a
# step converges in a single inner iteration (the trust-region
# acceptance ratio degenerates to 0/0).  Harmless for this 1-D linear
# moment condition; otherwise dominates the bootstrap output.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)

# The fair_coin example lives in the sibling DGP_Protocol repo,
# which is not on the import path by default.  This file lives at
# ``tests/v2/bernoulli_v2.py``; resolve up to the parent of the
# ManifoldGMM repo root and then over into ``DGP_Protocol/examples``.
_DGP_EXAMPLES = Path(__file__).resolve().parents[3] / "DGP_Protocol" / "examples"
if str(_DGP_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_DGP_EXAMPLES))
from fair_coin import dgp as fair_coin_dgp  # noqa: E402
from manifoldgmm import GMM, Manifold  # noqa: E402
from pymanopt.manifolds import Euclidean  # noqa: E402


# --------------------------------------------------------------------------
# Moment condition
# --------------------------------------------------------------------------
def g(p, data):
    """Bernoulli location moment ``g(p, X) = X - p``.

    Broadcasts scalar ``p`` over rows of ``data`` of shape ``(N, 1)``.
    Returns shape ``(N, 1)`` of per-row contributions; the GMM forms
    ``g_bar(p) = (1/N) sum_i g_i(p)`` internally.
    """

    return data - p


# --------------------------------------------------------------------------
# End-to-end run
# --------------------------------------------------------------------------
def run() -> dict[str, float]:
    """Run the example end-to-end and return summary statistics.

    Returns a dict with ``p_hat``, ``analytic_se``, ``bootstrap_mean``,
    ``bootstrap_se``, and ``wald_z``.  Useful for tests.
    """

    obs = np.asarray(fair_coin_dgp.data)
    N = obs.shape[0]
    print(f"observed realization:   shape={obs.shape}  dtype={obs.dtype}")
    print(f"empirical mean:         {float(obs.mean()):.4f}  (analog estimator)")
    print()

    # Manifold: p as an unconstrained Euclidean coordinate.  The
    # moment-condition root lies in (0, 1) so the bound is non-binding
    # for non-degenerate Bernoulli data.
    M = Manifold.from_pymanopt(Euclidean(1))

    gmm = GMM(
        moment_func=g,
        dgp=fair_coin_dgp,
        manifold=M,
        initial_point=jnp.array([0.5]),
    )

    # Point estimate.  ``verbosity=0`` silences the per-iteration
    # optimizer chatter (uninformative for a 1-d just-identified
    # problem; would dominate the bootstrap output otherwise).
    quiet = {"verbosity": 0}
    result = gmm.estimate(optimizer_kwargs=quiet)
    p_hat = float(np.asarray(result.theta_array)[0])
    print(f"GMM v2 estimate:        p_hat = {p_hat:.6f}")
    print("  (just-identified -> matches the empirical mean to " "optimizer tolerance)")
    print()

    # Analytic SE for the Bernoulli mean.
    analytic_se = float(np.sqrt(p_hat * (1.0 - p_hat) / N))
    print(
        f"analytic SE:            {analytic_se:.4f}  "
        f"(sqrt(p_hat * (1 - p_hat) / N), N = {N})"
    )

    # Parametric bootstrap: each replication draws from the original
    # fair_coin generator (Bernoulli(0.5)), refits, returns p_b.
    B = 500
    print(f"parametric bootstrap:   B={B}, seed=42 ...")
    bs = gmm.bootstrap(B=B, seed=42, optimizer_kwargs=quiet)
    p_bs = np.array([float(np.asarray(t.value)[0]) for t in bs.thetas])
    bootstrap_mean = float(p_bs.mean())
    bootstrap_se = float(p_bs.std(ddof=1))
    print(f"  bootstrap mean:       {bootstrap_mean:.4f}")
    print(f"  bootstrap SE:         {bootstrap_se:.4f}")
    print()

    # Wald-style two-sided test of H_0: p = 0.5.  Under the null,
    # z ~ N(0, 1) asymptotically.
    z = (p_hat - 0.5) / analytic_se
    print("Wald test of H_0: p = 0.5")
    print(f"  z = (p_hat - 0.5) / SE = {z:+.3f}")
    verdict = "reject" if abs(z) > 1.96 else "accept"
    print(
        f"  two-sided, alpha=0.05: {verdict}  (|z| {'>' if abs(z) > 1.96 else '<='} 1.96)"
    )

    return {
        "p_hat": p_hat,
        "analytic_se": analytic_se,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_se": bootstrap_se,
        "wald_z": z,
    }


if __name__ == "__main__":
    run()
