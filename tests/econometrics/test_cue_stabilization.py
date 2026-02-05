import time
import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm import GMM, MomentRestriction, Manifold
from pymanopt.manifolds import Euclidean, PSDFixedRank

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

    With sufficient ridge regularization (0.1), TrustRegions optimizer works
    and is much faster than SteepestDescent (~1s vs ~60s).

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

    # Test with CUE + ridge: moderate ridge (0.1) stabilizes both:
    # 1. The moment covariance inversion in CUE
    # 2. JAX autodiff through the inverse for Hessian computation
    # This allows TrustRegions to work (default optimizer, much faster)
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data_jax, manifold=manifold, backend="jax"
    )

    gmm_ridge = GMM(
        restriction,
        initial_point=initial_point,
        cue_ridge=0.1,  # moderate ridge for Hessian stability
    )

    start = time.time()
    res = gmm_ridge.estimate(verbose=0)
    elapsed = time.time() - start

    # Verify convergence: recovered A should be close to true A
    Y_est = res.theta.value
    A_est = Y_est @ Y_est.T

    print(f"\nCUE + ridge (TrustRegions) elapsed: {elapsed:.2f}s")
    print(f"True A:\n{A_true}")
    print(f"Estimated A:\n{A_est}")
    print(f"Frobenius error: {np.linalg.norm(A_est - A_true):.6f}")

    # Acceptance criteria:
    # 1. Should complete fast (< 30s with TrustRegions, was hanging before)
    assert elapsed < 30.0, f"CUE took too long: {elapsed:.1f}s"

    # 2. Should converge to a reasonable estimate (direction matters more than scale)
    # Ridge regularization introduces some bias, so we allow generous tolerance
    frobenius_error = np.linalg.norm(A_est - A_true)
    assert frobenius_error < 0.6, f"Poor convergence: Frobenius error = {frobenius_error:.4f}"

    # 3. Structure should be preserved: A_est should be approximately rank-1
    # and the eigenvector direction should be close to v_true
    eigvals = np.linalg.eigvalsh(A_est)
    # Largest eigenvalue should dominate
    assert eigvals[-1] / (eigvals.sum() + 1e-10) > 0.95, "Lost rank-1 structure"


def test_cue_adaptive_ridge_with_target_condition():
    """Test that cue_target_condition enables adaptive ridge selection.

    When target_condition is set, CUEWeighting computes eigenvalues of Ω(θ)
    at each evaluation and adjusts ridge to keep cond(Ω + ridge·I) ≤ target.
    """
    # Singular moments: duplicated => Omega is rank-deficient
    def gi_jax(theta, x):
        diff = x - theta
        return jnp.concatenate([diff, diff])

    data = jnp.array([[1.0], [2.0], [3.0]])
    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )

    # With adaptive ridge via target_condition
    gmm = GMM(
        restriction,
        initial_point=jnp.array([0.0]),
        cue_target_condition=1e8,  # Let it adapt
    )
    res = gmm.estimate(verbose=0)

    # Should converge to mean = 2.0
    assert np.allclose(res.theta.value, 2.0, atol=0.1)

    # Check that adaptive ridge was used
    info = res.weighting_info
    assert info["target_condition"] == 1e8
    # last_ridge should be > 0 since moments are singular
    assert info["last_ridge"] > 0, "Expected adaptive ridge to kick in"


@pytest.mark.slow
def test_psd_rank1_cue_adaptive_ridge():
    """Verify adaptive ridge works on PSD rank-1 problem with TrustRegions.

    This is the recommended approach: set target_condition and let the
    algorithm determine the appropriate ridge at each step, rather than
    manually tuning a fixed ridge value.
    """
    # True rank-1 PSD: A = v v^T where v = [1, 0.5, 0]^T
    v_true = np.array([[1.0], [0.5], [0.0]])
    A_true = v_true @ v_true.T

    rng = np.random.default_rng(42)
    n_obs = 100
    z = rng.normal(size=(n_obs, 1))
    data = z @ v_true.T
    data += rng.normal(scale=0.01, size=data.shape)
    data_jax = jnp.array(data)

    def gi_jax(theta, observation):
        Y = theta
        A = Y @ Y.T
        xxT = jnp.outer(observation, observation)
        diff = xxT - A
        idx = jnp.triu_indices(3)
        return diff[idx]

    manifold = Manifold.from_pymanopt(PSDFixedRank(3, 1))
    initial_point = np.array([[0.9], [0.4], [0.1]])

    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data_jax, manifold=manifold, backend="jax"
    )

    # Adaptive ridge: target_condition controls when ridge kicks in
    gmm = GMM(
        restriction,
        initial_point=initial_point,
        cue_target_condition=1e6,  # Adaptive ridge when cond > 1e6
    )

    start = time.time()
    res = gmm.estimate(verbose=0)
    elapsed = time.time() - start

    Y_est = res.theta.value
    A_est = Y_est @ Y_est.T

    print(f"\nCUE + adaptive ridge elapsed: {elapsed:.2f}s")
    print(f"Frobenius error: {np.linalg.norm(A_est - A_true):.6f}")
    print(f"Last condition: {res.weighting_info['last_condition']:.2e}")
    print(f"Last ridge: {res.weighting_info['last_ridge']:.2e}")

    # Should complete in reasonable time
    assert elapsed < 30.0, f"CUE took too long: {elapsed:.1f}s"

    # Should converge reasonably
    frobenius_error = np.linalg.norm(A_est - A_true)
    assert frobenius_error < 0.6, f"Poor convergence: {frobenius_error:.4f}"

    # Adaptive ridge should have been used (condition was > target)
    assert res.weighting_info["last_ridge"] > 0


def test_cue_inference_validity_diagnostic():
    """Test that check_inference_validity reports ridge ratio correctly."""
    # Setup with singular moments to force large ridge
    def gi_jax(theta, x):
        diff = x - theta
        return jnp.concatenate([diff, diff])  # Duplicated => singular Ω

    data = jnp.array([[1.0], [2.0], [3.0]])
    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )

    # Large ridge relative to eigenvalues
    gmm = GMM(restriction, initial_point=jnp.array([0.0]), cue_ridge=1.0)
    res = gmm.estimate(verbose=0)

    # Check inference validity
    validity = res.check_inference_validity(warn=False)

    assert "ridge_ratio" in validity
    assert "lambda_min" in validity
    assert "ridge" in validity
    assert "inference_warning" in validity

    # With such a large ridge on near-singular Ω, ratio should be high
    print(f"\nRidge ratio: {validity['ridge_ratio']:.2f}")
    print(f"Lambda min: {validity['lambda_min']:.2e}")
    print(f"Ridge: {validity['ridge']:.2e}")
    print(f"Warning: {validity['inference_warning']}")

    # The ridge (1.0) should be significant relative to eigenvalues
    # For near-singular Ω, ridge_ratio should be high (>= 0.1 triggers warning)
    assert validity["ridge_ratio"] >= 0.1, "Expected significant ridge ratio for singular moments"
    assert validity["inference_warning"] is not None, "Expected warning for significant ridge"


def test_cue_inference_validity_no_warning_when_small_ridge():
    """Test that no warning is issued when ridge is small relative to eigenvalues."""
    # Well-conditioned problem
    def gi_jax(theta, x):
        return x - theta  # Simple, well-conditioned

    data = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    manifold = Manifold.from_pymanopt(Euclidean(1))
    restriction = MomentRestriction(
        gi_jax=gi_jax, data=data, manifold=manifold, backend="jax"
    )

    # Small ridge
    gmm = GMM(restriction, initial_point=jnp.array([0.0]), cue_ridge=1e-10)
    res = gmm.estimate(verbose=0)

    validity = res.check_inference_validity(warn=False)

    print(f"\nRidge ratio: {validity['ridge_ratio']:.2e}")
    print(f"Warning: {validity['inference_warning']}")

    # Small ridge on well-conditioned problem => small ratio, no warning
    assert validity["ridge_ratio"] < 0.1, "Expected small ridge ratio"
    assert validity["inference_warning"] is None, "Expected no warning for small ridge"
