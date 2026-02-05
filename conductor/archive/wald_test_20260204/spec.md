# Specification: Wald Test for Manifold-Valued Parameters

## Goal
Implement a basic Wald test for hypothesis testing on parameters that reside on a Riemannian manifold, specifically within the `ManifoldGMM` framework.

## Background
In standard GMM, the Wald test for $H_0: h(\theta) = 0$ is based on the asymptotic normality of the estimator $\hat{\theta}$. When $\theta$ is on a manifold $M$, the test must account for the geometry. A simple approach is to perform the test in the tangent space $T_{\hat{\theta}}M$ or using a constraint function $h: M \to \mathbb{R}^k$ that is differentiable.

## Requirements
- Define a function `wald_test(result, constraint_func, constraint_value)` where `result` is a `GMMResult`.
- The `constraint_func` should accept a `ManifoldPoint`.
- Calculate the Wald statistic using the estimated tangent space covariance from `GMMResult`.
- Support computing a p-value based on the $\chi^2$ distribution.

## API Design
- `GMMResult.wald_test(h, q=0)`
- Returns a `WaldTestResult` object containing the statistic, degrees of freedom, and p-value.

