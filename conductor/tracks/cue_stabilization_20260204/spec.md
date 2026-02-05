# Specification: Stabilize CUE Weighting for Near-Singular Moments

## Goal
Improve the numerical stability and performance of `CUEWeighting` (Continuously Updated Estimator) when the moment covariance matrix $\hat{\Omega}(\theta)$ is near-singular or ill-conditioned.

## Background
In the "Compare Wald tests on Fixed-Rank PSD Manifold" track, we observed that GMM estimation using `CUEWeighting` was extremely slow (or hung) when estimating a rank-1 PSD matrix.
The moment condition $g_i(\theta) = \text{vech}(x_i x_i^\top - \theta)$ for rank-1 data involves highly correlated moments, leading to a near-singular $\hat{\Omega}$.
`CUEWeighting` computes $\hat{\Omega}^{-1}$ at every step. If $\hat{\Omega}$ is ill-conditioned, standard inversion (`linalg.inv`) is numerically unstable and might cause JAX to take a long time (e.g., in iterative refinement or NaN checks) or produce garbage that confuses the optimizer.

## Requirements
1.  **Ridge Regularization:** Implement a ridge-regularized inverse for `CUEWeighting`.
    $W = (\hat{\Omega} + \lambda I)^{-1}$ or similar robust inversion.
2.  **Configuration:** Allow the user to specify the ridge parameter $\lambda$ or an automatic selection strategy.
3.  **Performance:** Ensure that the stabilized weighting computes efficiently in JAX.

## Design
- Modify `CUEWeighting` class in `src/manifoldgmm/econometrics/gmm.py`.
- Add `ridge` parameter to `CUEWeighting.__init__`.
- Use `ridge_inverse` utility (already exists in `src/manifoldgmm/utils/numeric.py`?) or implement a JAX-friendly version.
