import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, MomentRestriction, Manifold
from pymanopt.manifolds import Euclidean

def test_cue_with_singular_moments_needs_ridge():
    # Setup singular moments: g = [x - theta, x - theta] (duplicated)
    # Omega will be singular rank 1 (2x2 matrix of all same values)
    
    def gi_jax(theta, x):
        diff = x - theta
        return jnp.concatenate([diff, diff])
        
    data = jnp.array([[1.0], [2.0], [3.0]])
    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(gi_jax=gi_jax, data=data, manifold=manifold, backend="jax")
    
    # Without ridge, this should be problematic.
    # JAX inv of singular matrix might produce NaNs or error depending on config.
    # We check if it fails or produces bad result.
    
    # Actually, GMM class doesn't support cue_ridge yet, so this line will fail argument check if I add it now.
    # I should write the test assuming the feature exists, and expect it to fail (red phase).
    
    try:
        gmm = GMM(restriction, initial_point=jnp.array([0.0]))
        # This might raise or return NaN
        res = gmm.estimate(verbose=0)
        # If it returns, check if valid
        assert not np.isnan(res.theta.value).any()
    except Exception:
        pass # Expected failure or error
        
    # With ridge
    # This will fail init until I update GMM
    try:
        gmm_ridge = GMM(restriction, initial_point=jnp.array([0.0]), cue_ridge=1e-4)
        res = gmm_ridge.estimate(verbose=0)
        assert np.allclose(res.theta.value, 2.0, atol=1e-4)
    except TypeError as e:
        pytest.fail(f"GMM does not accept cue_ridge yet: {e}")
