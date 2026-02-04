"""Tests for the Wald test functionality."""

import pytest
import numpy as np
import jax.numpy as jnp
from manifoldgmm.econometrics.gmm import WaldTestResult, GMM, GMMResult, MomentRestriction
from manifoldgmm import Manifold, ManifoldPoint
from pymanopt.manifolds import Euclidean

def test_wald_test_result_initialization():
    """Test that WaldTestResult can be initialized with correct attributes."""
    result = WaldTestResult(statistic=5.0, degrees_of_freedom=2, p_value=0.05)
    
    assert result.statistic == 5.0
    assert result.degrees_of_freedom == 2
    assert result.p_value == 0.05

def test_wald_test_method_exists_and_runs():
    """Test that GMMResult has a wald_test method that returns a WaldTestResult."""
    # Setup a simple GMMResult
    data = jnp.array([1.0, 2.0, 3.0])
    def gi_jax(theta, observation):
        return observation - theta[0]
        
    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax,
        data=data,
        manifold=manifold,
        backend="jax",
        parameter_labels=["theta"],
    )
    
    # Fake a GMMResult
    theta_hat = ManifoldPoint(manifold, jnp.array([2.0]))
    result = GMMResult(
        _theta=theta_hat,
        criterion_value=0.0,
        degrees_of_freedom=2,
        weighting_info={},
        weighting=np.eye(1),
        optimizer_report={},
        restriction=restriction,
        g_bar=jnp.zeros(1),
        two_step=False
    )
    
    # Define a constraint function h(theta) = theta - 2 = 0
    def constraint_func(theta_point):
        return theta_point.value - 2.0
        
    # This should fail if wald_test is not implemented
    wald_result = result.wald_test(constraint_func, q=1)
    
    assert isinstance(wald_result, WaldTestResult)
    # Since theta_hat = 2.0 and constraint checks theta - 2 = 0, statistic should be 0
    assert np.isclose(wald_result.statistic, 0.0)
    assert wald_result.degrees_of_freedom == 1
    assert np.isclose(wald_result.p_value, 1.0)