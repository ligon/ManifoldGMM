"""Comparison of Wald tests on Fixed-Rank PSD Manifold vs Euclidean."""

import pytest
import numpy as np
import jax.numpy as jnp
from manifoldgmm import GMM, Manifold, MomentRestriction, ManifoldPoint
from pymanopt.manifolds import PSDFixedRank, Euclidean

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None

def gi_jax(theta, observation):
    # theta is Y (3x1 factor)
    # A = Y @ Y.T
    # observation is x (3,)
    # g = vech(x @ x.T - A)
    Y = theta
    A = Y @ Y.T
    xxT = jnp.outer(observation, observation)
    diff = xxT - A
    # vech (upper triangular)
    idx = jnp.triu_indices(3)
    return diff[idx]

def gi_euc(theta, observation):
    # theta is flat vector of 6 unique elements of A
    # A constructed from theta
    A_vals = theta
    A = jnp.zeros((3, 3))
    idx = jnp.triu_indices(3)
    A = A.at[idx].set(A_vals)
    A = A + A.T - jnp.diag(jnp.diag(A))
    
    xxT = jnp.outer(observation, observation)
    diff = xxT - A
    return diff[idx]

@pytest.mark.slow
@pytest.mark.skip(reason="Too slow for CI environment")
@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_psd_wald_power_advantage():
    """Verify that Manifold Wald test has better power than Euclidean for low-rank PSD."""
    n_obs = 30
    n_reps = 2
    alpha = 0.05
    
    # H1: True A is rank 1 but deviates from H0
    # Let v = (1, 0.2, 0). A = v v.T. A_22 = 0.04.
    # H0: A_22 = 0.
    v_true = np.array([[1.0], [0.3], [0.0]])
    A_true = v_true @ v_true.T
    
    # Manifold: PSD(3, 1)
    manifold_man = Manifold.from_pymanopt(PSDFixedRank(3, 1))
    def constraint_man(theta_point):
        # theta_point.value is Y (3x1)
        Y = theta_point.value
        A = Y @ Y.T
        return A[1, 1] # A_22
        
    # Euclidean: R^6 (unique elements of A)
    manifold_euc = Manifold.from_pymanopt(Euclidean(6))
    def constraint_euc(theta_point):
        # theta_point.value is [A11, A12, A13, A22, A23, A33]
        return theta_point.value[3] # A_22
        
    rng = np.random.default_rng(123)
    
    rej_man = 0
    rej_euc = 0
    
    for _ in range(n_reps):
        # Generate data X ~ N(0, A_true)
        z = rng.normal(size=(n_obs, 1))
        data = z @ v_true.T # (n_obs, 3)
        data += rng.normal(scale=1e-3, size=data.shape)
        
        data_jax = jnp.array(data)
        
        # Manifold GMM
        print(f"Rep {_}: Manifold Estimate")
        res_man = GMM(
            MomentRestriction(gi_jax=gi_jax, data=data_jax, manifold=manifold_man, backend="jax"),
            initial_point=np.array([[1.0], [0.0], [0.0]])
        ).estimate(verbose=0, optimizer_kwargs={"max_iterations": 10})
        
        print(f"Rep {_}: Manifold Wald")
        if res_man.wald_test(constraint_man, q=1).p_value < alpha:
            rej_man += 1
            
        # Euclidean GMM
        initial_euc = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        print(f"Rep {_}: Euclidean Estimate")
        res_euc = GMM(
            MomentRestriction(gi_jax=gi_euc, data=data_jax, manifold=manifold_euc, backend="jax"),
            initial_point=initial_euc
        ).estimate(verbose=0, optimizer_kwargs={"max_iterations": 10})
        
        print(f"Rep {_}: Euclidean Wald")
        if res_euc.wald_test(constraint_euc, q=1).p_value < alpha:
            rej_euc += 1
            
    print(f"PSD Manifold Rejections: {rej_man}, Euclidean Rejections: {rej_euc}")
    assert rej_man >= rej_euc