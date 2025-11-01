"""Circle mean GMM example (tangled from documentation)."""

from __future__ import annotations

import datamat as dm
import jax.numpy as jnp
import numpy as np
from datamat import DataMat, DataVec
from jax.scipy.special import ndtri
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Sphere

ROT90 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)


def gi_jax(theta: jnp.ndarray, observation: jnp.ndarray) -> jnp.ndarray:
    """Single sine moment expressed in tangent coordinates."""

    theta_perp = ROT90 @ theta
    return jnp.array([jnp.dot(theta_perp, observation)], dtype=jnp.float64)


def gaussian_quantile(confidence: float = 0.95) -> float:
    """Two-sided Gaussian critical value for a confidence level."""

    upper_tail = 0.5 + 0.5 * confidence
    return float(jnp.asarray(ndtri(upper_tail)))

# "True" population mean angle
mu_0 = np.pi/2

# Draw 256 angles, keep them in a labelled DataMat, and build unit vectors.
angles = DataVec.random(256, rng=2025, name="phi", idxnames="obs") + mu_0

observations = dm.concat(
        {'x': np.cos(angles), 'y': np.sin(angles)},
        axis=1,
        levelnames=True,
)

m = MomentRestriction(
        gi_jax=gi_jax,
        parameter_labels=('x','y'),
        data=observations,
        manifold=Manifold.from_pymanopt(Sphere(2)),
        backend="jax",
)

gmm = GMM(m)

result = gmm.estimate()
estimate = result.theta

tangent_cov = result.tangent_covariance()
ambient_cov = result.manifold_covariance()
standard_error = DataVec(
        np.sqrt(ambient_cov.dg().values),
        index=ambient_cov.index,
)

alpha = 0.95
z = gaussian_quantile(alpha)
confidence_interval = DataMat(
        np.vstack(
            [
                estimate.values - z * standard_error.values,
                estimate.values + z * standard_error.values,
            ]
        ),
        index=["lower", "upper"],
        columns=estimate.index,
)

print("Estimate\n", estimate)
print("Standard error\n", standard_error)
print(f"{int(alpha * 100)}% confidence interval\n", confidence_interval)
