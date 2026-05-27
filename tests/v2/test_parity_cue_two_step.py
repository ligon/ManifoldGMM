"""v1 vs v2 parity: CUE, two-step, and iterated GMM weighting.

The v2 GMM gained ``assume_linear`` / ``detect_linear`` knobs that
enable a closed-form two-step short-circuit for linear moments
(``test_linearity.py`` exercises the v2 side alone).  This file
checks that with ``detect_linear=False`` -- forcing v2 onto the same
iterative path as v1 -- the numerical results match v1 byte-for-byte
(within optimizer tolerance).

Fixture: overidentified 1-d Normal(theta, 1) sample with two moments,
so the weighting matrix actually matters.
"""

from __future__ import annotations

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean


def _two_moment_g(theta, data):
    eps = data[:, 0] - theta[0]
    return jnp.stack([eps, eps**2 - 1.0], axis=-1)


def _make_sample(seed: int = 17):
    rng = np.random.default_rng(seed)
    n = 250
    x = rng.standard_normal(size=n) + 0.3
    return jnp.asarray(x.reshape(-1, 1))


def _M1():
    return Manifold.from_pymanopt(Euclidean(1))


def _build_v1():
    X = _make_sample()
    M = _M1()
    r1 = MomentRestriction(g=_two_moment_g, data=X, manifold=M, backend="jax")
    return GMM(r1, initial_point=jnp.array([0.0])), X


def _build_v2(*, detect_linear: bool = False):
    X = _make_sample()
    M = _M1()
    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    return (
        GMM(
            moment_func=_two_moment_g,
            dgp=dgp,
            manifold=M,
            backend="jax",
            initial_point=jnp.array([0.0]),
            detect_linear=detect_linear,
        ),
        X,
    )


def test_two_step_v1_v2_match() -> None:
    """``estimate(two_step=True)`` on overidentified problem matches v1 vs v2."""

    gmm_v1, _ = _build_v1()
    res_v1 = gmm_v1.estimate(two_step=True)

    gmm_v2, _ = _build_v2(detect_linear=False)
    res_v2 = gmm_v2.estimate(two_step=True)

    np.testing.assert_allclose(
        np.asarray(res_v1.theta_array, dtype=float),
        np.asarray(res_v2.theta_array, dtype=float),
        atol=1e-7,
    )


def test_iterated_v1_v2_match() -> None:
    """``estimate(weighting_iterations=3)`` matches v1 vs v2 on overidentified."""

    gmm_v1, _ = _build_v1()
    res_v1 = gmm_v1.estimate(weighting_iterations=3)

    gmm_v2, _ = _build_v2(detect_linear=False)
    res_v2 = gmm_v2.estimate(weighting_iterations=3)

    np.testing.assert_allclose(
        np.asarray(res_v1.theta_array, dtype=float),
        np.asarray(res_v2.theta_array, dtype=float),
        atol=1e-7,
    )


def test_cue_ridge_v1_v2_match() -> None:
    """``cue_ridge`` regularised CUE matches v1 vs v2 on overidentified.

    Uses ``cue_ridge=1e-4`` -- a small fixed ridge -- to avoid the
    adaptive-ridge basin-shifting warned about in ``cue_basin-shift-doc``.
    """

    X = _make_sample()
    M = _M1()

    r1 = MomentRestriction(g=_two_moment_g, data=X, manifold=M, backend="jax")
    gmm_v1 = GMM(r1, initial_point=jnp.array([0.0]), cue_ridge=1e-4)
    res_v1 = gmm_v1.estimate()

    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm_v2 = GMM(
        moment_func=_two_moment_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=jnp.array([0.0]),
        cue_ridge=1e-4,
        detect_linear=False,
    )
    res_v2 = gmm_v2.estimate()

    np.testing.assert_allclose(
        np.asarray(res_v1.theta_array, dtype=float),
        np.asarray(res_v2.theta_array, dtype=float),
        atol=1e-7,
    )
