"""Tests for GMMResult.in_asymptotic_region().

Uses Euclidean(1) mean-estimation fixtures for speed.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, Manifold, ManifoldPoint, MomentRestriction
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

class TestInAsymptoticRegion:
    """Tests for GMMResult.in_asymptotic_region()."""

    def test_estimate_in_own_region(self) -> None:
        """The estimate theta_hat should always be inside its own region."""
        result = _euclidean1_result()
        assert result.in_asymptotic_region(result.theta_point, alpha=0.05)

    def test_estimate_in_own_region_strict_alpha(self) -> None:
        """Even at very strict alpha, theta_hat is inside (d^2 = 0)."""
        result = _euclidean1_result()
        assert result.in_asymptotic_region(result.theta_point, alpha=0.001)

    def test_distant_point_outside_region(self) -> None:
        """A point far from the estimate should be outside the region."""
        result = _euclidean1_result()
        manifold = result.restriction.manifold
        far_point = ManifoldPoint(manifold, jnp.array([1000.0]))
        assert not result.in_asymptotic_region(far_point, alpha=0.05)

    def test_accepts_raw_array(self) -> None:
        """Should accept raw array (auto-wraps to ManifoldPoint)."""
        result = _euclidean1_result()
        # theta_hat should be near 3.0 (mean of [1,2,3,4,5])
        # A point right at the estimate should be inside
        assert result.in_asymptotic_region(result.theta_array, alpha=0.05)

    def test_larger_alpha_wider_region(self) -> None:
        """A larger alpha means a smaller critical value (narrower region)."""
        result = _euclidean1_result()
        # A moderately far point might be in the 1% region but not the 50% region
        manifold = result.restriction.manifold
        # Find a point that's borderline
        theta_hat_val = float(result.theta_array[0])
        # With n=5, SE is modest so small offsets can be outside at large alpha
        moderate_point = ManifoldPoint(manifold, jnp.array([theta_hat_val + 0.5]))
        in_005 = result.in_asymptotic_region(moderate_point, alpha=0.05)
        in_050 = result.in_asymptotic_region(moderate_point, alpha=0.50)
        # If inside at alpha=0.50, must also be inside at alpha=0.05
        if in_050:
            assert in_005
