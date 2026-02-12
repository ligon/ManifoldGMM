"""Tests for the geodesic_mahalanobis_distance public function.

Uses Euclidean(1) mean-estimation fixtures for speed.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, ManifoldPoint, MomentRestriction
from manifoldgmm.econometrics.bootstrap import (
    MomentWildBootstrap,
    geodesic_mahalanobis_distance,
)
from pymanopt.manifolds import Euclidean as PymanoptEuclidean


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _euclidean1_result(data=None):
    """Build a Euclidean(1) mean-estimation GMM result."""

    if data is None:
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def gi_jax(theta: Any, obs: Any) -> Any:
        return obs - theta[0]

    manifold = Manifold.from_pymanopt(PymanoptEuclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax",
    )
    result = GMM(restriction, initial_point=jnp.array([0.0])).estimate()
    return result


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestGeodesicMahalanobisDistance:
    """Tests for the extracted geodesic_mahalanobis_distance function."""

    def test_distance_at_estimate_is_zero(self) -> None:
        result = _euclidean1_result()
        d2 = geodesic_mahalanobis_distance(result, result.theta_point)
        assert d2 == pytest.approx(0.0, abs=1e-8)

    def test_distance_positive_away_from_estimate(self) -> None:
        result = _euclidean1_result()
        manifold = result.restriction.manifold
        far_point = ManifoldPoint(manifold, jnp.array([100.0]))
        d2 = geodesic_mahalanobis_distance(result, far_point)
        assert d2 > 0.0

    def test_accepts_raw_array(self) -> None:
        """Should auto-wrap array-like into ManifoldPoint."""
        result = _euclidean1_result()
        d2 = geodesic_mahalanobis_distance(result, jnp.array([100.0]))
        assert d2 > 0.0

    def test_custom_covariance(self) -> None:
        result = _euclidean1_result()
        # Larger covariance -> smaller distance
        cov_large = np.array([[100.0]])
        d2_large = geodesic_mahalanobis_distance(
            result, jnp.array([10.0]), covariance=cov_large,
        )
        d2_default = geodesic_mahalanobis_distance(
            result, jnp.array([10.0]),
        )
        # With larger covariance, distance should be smaller
        assert d2_large < d2_default

    def test_refactored_bootstrap_distances_unchanged(self) -> None:
        """Regression: refactoring should not change bootstrap distance values."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _euclidean1_result(data)

        boot = MomentWildBootstrap(
            result, n_bootstrap=20, base_seed=42, weight_scheme="rademacher",
        )
        boot.run_sequential()
        d2 = boot.geodesic_distances()

        # Independently compute each distance using the public function
        for i, br in enumerate(boot._results):
            d2_manual = geodesic_mahalanobis_distance(result, br.theta_star)
            assert d2[i] == pytest.approx(d2_manual, rel=1e-10)
