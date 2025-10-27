from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root (where the tangled example lives) is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

gaussian_example = pytest.importorskip("examples.gaussian_covariance")


def test_gaussian_example_estimation_produces_psd_covariance():
    _observations, observations_array, mu_true, sigma_true = (
        gaussian_example.build_dataset(seed=7)
    )
    result, restriction = gaussian_example.run_estimation(observations_array)
    mu_hat, sigma_hat = result.point

    observations_np = np.asarray(observations_array, dtype=float)
    sample_mean = observations_np.mean(axis=0)
    centered = observations_np - sample_mean
    sample_cov = centered.T @ centered / observations_np.shape[0]

    np.testing.assert_allclose(np.asarray(mu_hat), sample_mean, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sigma_hat), sample_cov, atol=1e-6)

    np.testing.assert_allclose(
        np.asarray(mu_hat),
        np.asarray(mu_true),
        atol=2e-1,
    )
    np.testing.assert_allclose(
        np.asarray(sigma_hat),
        np.asarray(sigma_true),
        atol=2e-1,
    )

    eigenvalues = np.linalg.eigvalsh(np.asarray(sigma_hat))
    assert np.all(eigenvalues >= -1e-8)

    residual_norm = float(jnp.linalg.norm(restriction.g_bar(result.point)))
    assert residual_norm < 1e-3
