from __future__ import annotations

import numpy as np
import pytest

from manifoldgmm import GMM, Manifold, MomentRestriction

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from jax.flatten_util import ravel_pytree  # noqa: E402

try:  # pragma: no cover - optional dependency
    from pymanopt.manifolds import Sphere
except ImportError:  # pragma: no cover
    Sphere = None  # type: ignore[assignment]


ROT90 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)


def gi_jax(theta: jnp.ndarray, observation: jnp.ndarray) -> jnp.ndarray:
    theta_perp = ROT90 @ theta
    return jnp.array([jnp.dot(theta_perp, observation)], dtype=jnp.float64)


def jacobian_dense(operator, basis) -> np.ndarray:
    columns: list[np.ndarray] = []
    for direction in basis:
        image = operator.matvec(direction)
        flat, _ = ravel_pytree(image)
        columns.append(np.asarray(flat, dtype=float))
    return np.vstack(columns).T


@pytest.mark.skipif(Sphere is None, reason="pymanopt is required for circle manifold")
def test_circle_mean_inference_matches_sandwich():
    angles_deg = jnp.array([20.0, 35.0, 38.0, 42.0, 55.0, 60.0, 28.0], dtype=jnp.float64)
    angles_rad = jnp.deg2rad(angles_deg)
    observations = jnp.stack([jnp.cos(angles_rad), jnp.sin(angles_rad)], axis=1)
    observations = observations / jnp.linalg.norm(observations, axis=1, keepdims=True)

    manifold = Manifold.from_pymanopt(Sphere(2))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=observations,
        manifold=manifold,
        backend="jax",
    )

    estimator = GMM(
        restriction,
        initial_point=jnp.array([1.0, 0.0], dtype=jnp.float64),
    )
    result = estimator.estimate()
    theta_hat = result.theta

    jac = restriction.jacobian(theta_hat)
    basis = restriction.tangent_basis(theta_hat)
    assert len(basis) == 1

    D = jacobian_dense(jac, basis)
    assert D.shape == (1, 1)

    S = np.asarray(restriction.omega_hat(theta_hat), dtype=float)
    W = np.linalg.inv(S)
    covariance = float((np.linalg.inv(D.T @ W @ D) @ (D.T @ W @ S @ W @ D) @ np.linalg.inv(D.T @ W @ D)).squeeze())
    assert covariance > 0.0

    mean_vector = np.asarray(observations, dtype=float).mean(axis=0)
    R_bar = np.linalg.norm(mean_vector)
    classical_variance = 2.0 * (1.0 - R_bar)
    assert classical_variance > 0.0
    assert covariance < 10.0 * classical_variance + 1e-8
