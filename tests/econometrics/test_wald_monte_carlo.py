"""Monte Carlo tests for Wald statistic on Circle manifold."""

import pytest
import numpy as np
import jax.numpy as jnp
from manifoldgmm import GMM, Manifold, MomentRestriction, ManifoldPoint
from pymanopt.manifolds import Sphere

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None

# Circle manifold (S^1 embedded in R^2)
ROT90 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)

def gi_jax(theta, observation):
    # E[theta_perp . x] = 0 means x is concentrated around theta
    theta_perp = ROT90 @ theta
    return jnp.array([jnp.dot(theta_perp, observation)], dtype=jnp.float64)

@pytest.mark.slow
@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_wald_size_on_circle():
    """Verify size of Wald test under H0: theta_y = 0."""
    n_reps = 100 # Keep low for CI
    n_obs = 100
    alpha_level = 0.05
    
    # H0: theta = (1, 0)
    # Constraint h(theta) = theta[1] = 0
    def constraint(theta_point):
        return theta_point.value[1]
    
    rejections = 0
    
    manifold = Manifold.from_pymanopt(Sphere(2))
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_reps):
        # Generate data around (1,0)
        # Use projected Gaussian for simplicity
        data_raw = rng.normal(loc=[1.0, 0.0], scale=0.5, size=(n_obs, 2))
        # Normalize to circle
        norms = np.linalg.norm(data_raw, axis=1, keepdims=True)
        data = data_raw / norms
        
        restriction = MomentRestriction(
            gi_jax=gi_jax,
            data=jnp.array(data),
            manifold=manifold,
            backend="jax"
        )
        
        gmm = GMM(restriction, initial_point=jnp.array([1.0, 0.0]))
        result = gmm.estimate(verbose=0)
        
        wald = result.wald_test(constraint, q=1)
        if wald.p_value < alpha_level:
            rejections += 1
            
    rejection_rate = rejections / n_reps
    # With 100 reps, SE is sqrt(0.05*0.95/100) approx 0.02
    # So 0.05 +/- 0.08 is a safe wide range (0.0 to 0.13)
    assert 0.0 <= rejection_rate <= 0.15, f"Rejection rate {rejection_rate} is too far from nominal {alpha_level}"

@pytest.mark.slow
@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_wald_power_on_circle():
    """Verify power of Wald test under H1: theta_y != 0."""
    n_reps = 20
    n_obs = 100
    alpha_level = 0.05
    
    # H1: theta rotated by 0.5 radians approx 30 degrees
    angle = 0.5
    true_mean = np.array([np.cos(angle), np.sin(angle)])
    # Constraint h(theta) = theta[1] = 0 is violated since sin(0.5) != 0
    
    def constraint(theta_point):
        return theta_point.value[1]
    
    rejections = 0
    
    manifold = Manifold.from_pymanopt(Sphere(2))
    rng = np.random.default_rng(2025)
    
    for _ in range(n_reps):
        data_raw = rng.normal(loc=true_mean, scale=0.5, size=(n_obs, 2))
        norms = np.linalg.norm(data_raw, axis=1, keepdims=True)
        data = data_raw / norms
        
        restriction = MomentRestriction(
            gi_jax=gi_jax,
            data=jnp.array(data),
            manifold=manifold,
            backend="jax"
        )
        
        gmm = GMM(restriction, initial_point=jnp.array([1.0, 0.0]))
        result = gmm.estimate(verbose=0)
        
        wald = result.wald_test(constraint, q=1)
        if wald.p_value < alpha_level:
            rejections += 1
            
    rejection_rate = rejections / n_reps
    assert rejection_rate >= 0.8, f"Power {rejection_rate} is too low"

@pytest.mark.slow
@pytest.mark.skipif(chi2 is None, reason="scipy is required for p-values")
def test_power_comparison_logic():
    """Verify that Manifold test has higher power than Euclidean test."""
    n_reps = 20
    n_obs = 100
    alpha = 0.05
    
    # Choose an angle where power difference is likely visible
    # Small angle: 0.15 rad.
    angle = 0.15
    mu_true = np.array([np.cos(angle), np.sin(angle)])
    
    # Manifold
    manifold = Manifold.from_pymanopt(Sphere(2))
    def constraint(theta_point):
        return theta_point.value[1]
        
    # Euclidean
    from pymanopt.manifolds import Euclidean
    manifold_euc = Manifold.from_pymanopt(Euclidean(2))
    def gi_euc(theta, observation):
        return observation - theta
        
    rng = np.random.default_rng(999)
    
    rej_man = 0
    rej_euc = 0
    
    for _ in range(n_reps):
        data_raw = rng.normal(loc=mu_true, scale=0.5, size=(n_obs, 2))
        data_raw /= np.linalg.norm(data_raw, axis=1, keepdims=True)
        data_jax = jnp.array(data_raw)
        
        # Manifold
        gmm_man = GMM(
            MomentRestriction(gi_jax=gi_jax, data=data_jax, manifold=manifold, backend="jax"),
            initial_point=jnp.array([1.0, 0.0])
        )
        res_man = gmm_man.estimate(verbose=0)
        if res_man.wald_test(constraint, q=1).p_value < alpha:
            rej_man += 1
            
        # Euclidean
        gmm_euc = GMM(
            MomentRestriction(gi_jax=gi_euc, data=data_jax, manifold=manifold_euc, backend="jax"),
            initial_point=jnp.array([1.0, 0.0])
        )
        res_euc = gmm_euc.estimate(verbose=0)
        if res_euc.wald_test(constraint, q=1).p_value < alpha:
            rej_euc += 1
            
    print(f"Manifold Rejections: {rej_man}, Euclidean Rejections: {rej_euc}")
    assert rej_man >= rej_euc
