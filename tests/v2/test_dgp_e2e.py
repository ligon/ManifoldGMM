"""End-to-end test for the v2 (DGP-based) GMM construction path.

Covers:

- Point estimate agreement between the v1 (``restriction=``) and v2
  (``moment_func=``, ``dgp=``) construction routes on a small
  Gaussian location problem.
- v2 mutual exclusivity: passing both ``restriction=`` and
  ``moment_func=`` raises; passing neither raises.
- ``gmm.bootstrap(B)`` runs on a v2-constructed GMM, returns ``B``
  refit theta values within a sensible distance of the empirical
  mean.
- ``gmm.bootstrap(B)`` on a v1-constructed GMM raises ``RuntimeError``
  (no DGP is attached).

These tests use the standard JAX backend so the GMM optimizer can
autodiff through the moment function.  The DGP carries the data as
a JAX array (``EmpiricalDGP(observation=jnp.array(...))``); ``data``
flows through unchanged.

See ``docs/design/v2_dgp.org`` for the full v2 design.
"""

from __future__ import annotations

import dgp_protocol as dp
import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Euclidean


def _location_g(theta, data):
    """Vectorized location moment ``g(theta, X) = X - theta``.

    JAX-compatible; broadcasts ``theta`` of shape ``(p,)`` over rows
    of ``data`` of shape ``(N, p)``.  Returns shape ``(N, p)``.
    """

    return data - theta[None, :]


def _build_dataset(seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(size=(200, 3)))


def _build_manifold() -> Manifold:
    return Manifold.from_pymanopt(Euclidean(3))


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------
def test_v2_requires_both_moment_func_and_dgp() -> None:
    """Passing only ``moment_func=`` (or only ``dgp=``) raises."""

    dgp = dp.EmpiricalDGP(observation=_build_dataset())
    with pytest.raises(TypeError, match="missing argument"):
        GMM(moment_func=_location_g)
    with pytest.raises(TypeError, match="missing argument"):
        GMM(dgp=dgp)


def test_rejects_both_restriction_and_v2_kwargs() -> None:
    """Passing both v1 and v2 inputs raises."""

    X = _build_dataset()
    M = _build_manifold()
    r = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    dgp = dp.EmpiricalDGP(observation=X)
    with pytest.raises(TypeError, match="either"):
        GMM(restriction=r, moment_func=_location_g, dgp=dgp)


def test_rejects_non_dgp_object_in_dgp_kwarg() -> None:
    """``dgp=`` must satisfy the DataGeneratingProcess Protocol."""

    with pytest.raises(TypeError, match="DataGeneratingProcess"):
        GMM(moment_func=_location_g, dgp="not a dgp")


def test_rejects_dgp_with_no_bound_data() -> None:
    """A pure-parametric DGP with ``data is None`` raises a clear error.

    Mirrors the fair_coin example in ``DGP_Protocol/examples/``: the
    DGP is constructed with ``observation=None`` (no realization
    yet).  The v2 GMM needs a bound realization for the point
    estimate, so we refuse with a hint pointing at
    ``dgp.with_data(dgp.draw())``.
    """

    def gen(rng, shape):
        return rng.standard_normal(shape)

    pure_param = dp.ParametricDGP(generator=gen, default_shape=(50, 3), seed=0)
    assert pure_param.data is None  # sanity for the test premise

    with pytest.raises(ValueError, match="with_data"):
        GMM(
            moment_func=_location_g,
            dgp=pure_param,
            manifold=_build_manifold(),
            initial_point=jnp.zeros(3),
        )


def test_accepts_pure_parametric_dgp_after_with_data() -> None:
    """The same DGP, bound to a draw, constructs cleanly."""

    def gen(rng, shape):
        return rng.standard_normal(shape)

    pure_param = dp.ParametricDGP(generator=gen, default_shape=(50, 3), seed=0)
    bound = pure_param.with_data(pure_param.draw())
    assert bound.data is not None

    gmm = GMM(
        moment_func=_location_g,
        dgp=bound,
        manifold=_build_manifold(),
        initial_point=jnp.zeros(3),
    )
    # And the estimate runs.
    result = gmm.estimate()
    assert result.theta_array is not None


def test_rejects_v2_only_kwargs_with_restriction() -> None:
    """``manifold=`` / ``backend=`` are v2-only; v1 callers must use MR."""

    X = _build_dataset()
    M = _build_manifold()
    r = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    with pytest.raises(TypeError, match="v2-only"):
        GMM(restriction=r, manifold=M)
    with pytest.raises(TypeError, match="v2-only"):
        GMM(restriction=r, backend="numpy")


# ---------------------------------------------------------------------------
# Point estimate equivalence: v1 vs v2
# ---------------------------------------------------------------------------
def test_v1_and_v2_point_estimates_agree_on_iid_data() -> None:
    """v2-constructed GMM produces the same theta_hat as v1 on iid data."""

    X = _build_dataset(seed=0)
    M = _build_manifold()
    theta_0 = jnp.zeros(3)

    # v1 path: data bound in the MomentRestriction.
    r1 = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    gmm_v1 = GMM(r1, initial_point=theta_0)
    result_v1 = gmm_v1.estimate()

    # v2 path: data via an EmpiricalDGP.
    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm_v2 = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        backend="jax",
        initial_point=theta_0,
    )
    result_v2 = gmm_v2.estimate()

    np.testing.assert_allclose(
        np.asarray(result_v1.theta_array, dtype=float),
        np.asarray(result_v2.theta_array, dtype=float),
        atol=1e-8,
    )


def test_v2_dgp_property_exposes_dgp() -> None:
    """The v2 GMM exposes ``gmm.dgp``; v1 GMM returns ``None``."""

    X = _build_dataset()
    M = _build_manifold()
    dgp = dp.EmpiricalDGP(observation=X, seed=0)

    gmm_v2 = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(3),
    )
    assert gmm_v2.dgp is dgp

    r = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    gmm_v1 = GMM(r, initial_point=jnp.zeros(3))
    assert gmm_v1.dgp is None


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_runs_and_returns_b_replications() -> None:
    """``gmm.bootstrap(B)`` returns a BootstrapResult holding B refits."""

    X = _build_dataset(seed=0)
    M = _build_manifold()
    dgp = dp.EmpiricalDGP(observation=X, seed=0)
    gmm = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(3),
    )

    B = 4
    bs = gmm.bootstrap(B=B, seed=42)
    assert len(bs.thetas) == B
    assert bs.base is gmm
    # Each refit lies within a small multiple of the bootstrap SE of
    # the empirical mean.
    X_np = np.asarray(X, dtype=float)
    emp_mean = X_np.mean(axis=0)
    emp_se = X_np.std(axis=0, ddof=1) / np.sqrt(X_np.shape[0])
    for theta_b in bs.thetas:
        arr = np.asarray(theta_b.value, dtype=float)
        assert np.all(np.abs(arr - emp_mean) < 6.0 * emp_se), (
            f"bootstrap refit theta={arr} too far from empirical "
            f"mean={emp_mean} (se={emp_se})"
        )


def test_bootstrap_is_reproducible_under_seed() -> None:
    """Two bootstrap calls with the same seed produce identical refits."""

    X = _build_dataset(seed=0)
    M = _build_manifold()

    def _build_gmm() -> GMM:
        # Fresh DGP each time so the DGP's own RNG is identical.
        return GMM(
            moment_func=_location_g,
            dgp=dp.EmpiricalDGP(observation=X, seed=0),
            manifold=M,
            initial_point=jnp.zeros(3),
        )

    bs_a = _build_gmm().bootstrap(B=3, seed=11)
    bs_b = _build_gmm().bootstrap(B=3, seed=11)
    for ta, tb in zip(bs_a.thetas, bs_b.thetas, strict=True):
        np.testing.assert_array_equal(
            np.asarray(ta.value, dtype=float),
            np.asarray(tb.value, dtype=float),
        )


def test_bootstrap_raises_on_v1_constructed_gmm() -> None:
    """A GMM built with only ``restriction=`` has no DGP; bootstrap raises."""

    X = _build_dataset()
    M = _build_manifold()
    r = MomentRestriction(g=_location_g, data=X, manifold=M, backend="jax")
    gmm = GMM(r, initial_point=jnp.zeros(3))
    with pytest.raises(RuntimeError, match="v2"):
        gmm.bootstrap(B=2)


def test_bootstrap_validates_B() -> None:
    """B must be positive."""

    M = _build_manifold()
    dgp = dp.EmpiricalDGP(observation=_build_dataset())
    gmm = GMM(
        moment_func=_location_g,
        dgp=dgp,
        manifold=M,
        initial_point=jnp.zeros(3),
    )
    with pytest.raises(ValueError, match="B must be >= 1"):
        gmm.bootstrap(B=0)
