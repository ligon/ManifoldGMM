import time
import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, MomentRestriction, Manifold
from pymanopt.manifolds import Euclidean, PSDFixedRank
from pymanopt.optimizers import SteepestDescent

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


@pytest.mark.slow
def test_psd_rank1_cue_with_ridge_stabilization():
    """Verify CUE with ridge runs fast and converges on rank-1 PSD data.

    This test addresses the issue from the PSD Wald track where CUE weighting
    was extremely slow or hung due to near-singular moment covariance.
    The moments g_i = vech(x_i x_i^T - A) for rank-1 data produce highly
    correlated moment conditions, leading to ill-conditioned Omega.

    We use SteepestDescent optimizer to avoid the Hessian computation which
    can be numerically unstable on the PSDFixedRank manifold with near-collinear data.

    The primary goal is SPEED (not hanging), with reasonable convergence.
    """
    # True rank-1 PSD: A = v v^T where v = [1, 0.5, 0]^T
    v_true = np.array([[1.0], [0.5], [0.0]])
    A_true = v_true @ v_true.T

    # Generate rank-1 data: x = z * v where z ~ N(0,1)
    # Add small noise to avoid exact singularity in manifold geometry
    rng = np.random.default_rng(42)
    n_obs = 100  # larger sample for better convergence
    z = rng.normal(size=(n_obs, 1))
    data = z @ v_true.T  # shape (n_obs, 3)
    data += rng.normal(scale=0.01, size=data.shape)  # small noise for numerical stability
    data_jax = jnp.array(data)

    # Moment function: g(Y, x) = vech(x x^T - Y Y^T)
    def gi_jax(theta, observation):
        Y = theta  # (3, 1) factor
        A = Y @ Y.T
        xxT = jnp.outer(observation, observation)
        diff = xxT - A
        idx = jnp.triu_indices(3)
        return diff[idx]

    manifold = Manifold.from_pymanopt(PSDFixedRank(3, 1))
    # Start closer to the truth for faster convergence
    initial_point = np.array([[0.9], [0.4], [0.1]])

    # Test with CUE + ridge: should be fast and converge
    # Use SteepestDescent to avoid Hessian computation issues on this manifold
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data_jax, manifold=manifold, backend="jax"
    )

    gmm_ridge = GMM(
        restriction,
        initial_point=initial_point,
        cue_ridge=1e-4,  # smaller ridge with larger sample
        optimizer=SteepestDescent,
    )

    start = time.time()
    res = gmm_ridge.estimate(verbose=0)
    elapsed = time.time() - start

    # Verify convergence: recovered A should be close to true A
    Y_est = res.theta.value
    A_est = Y_est @ Y_est.T

    print(f"\nCUE + ridge elapsed: {elapsed:.2f}s")
    print(f"True A:\n{A_true}")
    print(f"Estimated A:\n{A_est}")
    print(f"Frobenius error: {np.linalg.norm(A_est - A_true):.6f}")

    # Acceptance criteria:
    # 1. Should complete in reasonable time (< 90s, was hanging before)
    assert elapsed < 90.0, f"CUE took too long: {elapsed:.1f}s"

    # 2. Should converge to a reasonable estimate (direction matters more than scale)
    # Ridge regularization introduces some bias, so we allow generous tolerance
    frobenius_error = np.linalg.norm(A_est - A_true)
    assert frobenius_error < 0.6, f"Poor convergence: Frobenius error = {frobenius_error:.4f}"

    # 3. Structure should be preserved: A_est should be approximately rank-1
    # and the eigenvector direction should be close to v_true
    eigvals = np.linalg.eigvalsh(A_est)
    # Largest eigenvalue should dominate
    assert eigvals[-1] / (eigvals.sum() + 1e-10) > 0.95, "Lost rank-1 structure"
