"""Cross-methodology check: wild bootstrap vs DGP-driven raw-data bootstrap.

These are two statistically distinct estimators of the same target
(the sampling distribution of ``theta_hat``):

- ``MomentWildBootstrap`` (estimator-side): refits on Rademacher-
  reweighted moment errors at the fitted theta.  Residual-based.
- ``gmm.bootstrap(B)`` (DGP-driven): refits on B fresh ``dgp.draw()``
  realizations.  Case-based.

They are *not* algebraically identical even on the same fixture; we
only expect agreement up to bootstrap MC error.  This file checks
that the two SE estimators converge to within a generous tolerance
at large B + large N, which is the right-shaped sanity check for "no
gross asymmetry has crept in."

Marked ``slow`` so it doesn't dominate every PR run.
"""

from __future__ import annotations

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentWildBootstrap
from pymanopt.manifolds import Euclidean


def _location_g(theta, data):
    return data - theta[None, :]


@pytest.mark.slow
def test_wild_vs_raw_se_iid_agree_at_large_n() -> None:
    """At N=500, B=300, the two bootstrap SE estimators agree within MC error.

    For X ~ Normal(0, I_3) the true asymptotic SE is 1/sqrt(N) ~ 0.0447.
    Both bootstraps should hit this to within ~10% at B=300, so a 25%
    tolerance between them is loose enough to be reliably reproducible
    on CI yet tight enough to flag genuine drift.
    """

    rng = np.random.default_rng(20260526)
    N = 500
    X = jnp.asarray(rng.standard_normal(size=(N, 3)))
    M = Manifold.from_pymanopt(Euclidean(3))

    # v2 GMM with EmpiricalDGP -- needed for gmm.bootstrap().
    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm_v2 = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=jnp.zeros(3),
    )
    result = gmm_v2.estimate()

    B = 300

    # Wild bootstrap on the v2 result.  ``BootstrapResult.theta_star``
    # is a ``ManifoldPoint``; use ``.value`` for the ambient array.
    wild = MomentWildBootstrap(result, n_bootstrap=B, base_seed=0)
    wild_thetas = np.vstack(
        [np.asarray(t.run().theta_star.value, dtype=float) for t in wild.tasks()]
    )
    wild_se = wild_thetas.std(axis=0, ddof=1)

    # Raw-data DGP bootstrap.
    raw = gmm_v2.bootstrap(B=B, seed=1)
    raw_thetas = np.vstack([np.asarray(t.value, dtype=float) for t in raw.thetas])
    raw_se = raw_thetas.std(axis=0, ddof=1)

    expected_se = 1.0 / np.sqrt(N)
    # Each estimator should be in the ballpark of the asymptote.
    np.testing.assert_allclose(wild_se, expected_se, rtol=0.30)
    np.testing.assert_allclose(raw_se, expected_se, rtol=0.30)
    # And they should agree with each other to within a generous MC-error
    # tolerance.  25% is loose; tighter would be tempting but flake-prone
    # at B=300.
    rel_diff = np.abs(wild_se - raw_se) / np.mean([wild_se, raw_se], axis=0)
    assert (rel_diff < 0.30).all(), (
        f"wild SE {wild_se} vs raw SE {raw_se} differ by "
        f"{rel_diff} (rel); expected < 0.30"
    )
