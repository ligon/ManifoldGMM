# Specification: Investigate and Optimize Wald Test Performance

## Goal
Identify the root causes of the significant performance overhead in `GMMResult.wald_test` and implement optimizations to make it suitable for large-scale Monte Carlo simulations.

## Bottleneck Hypotheses
1.  **JAX Recompilation:** `wald_test` defines a new `composed_map` closure every time it is called, forcing JAX to recompile the Jacobian function.
2.  **`tangent_basis` Construction:** The generic `ambient_basis` generator in `MomentRestriction` might be inefficient for higher-dimensional manifolds (like $R^6$).
3.  **Numerical Differentiation Fallback:** If JAX fails (e.g., due to pymanopt's non-JAX code), the fallback loop over basis vectors might be slow.
4.  **`ManifoldPoint` Overhead:** Frequent construction of `ManifoldPoint` objects during tracing or loops might add significant latency.

## Optimization Strategies
- **Leverage `JacobianOperator`:** Use the existing `JacobianOperator` class which abstracts JAX differentiation and might be more efficient if used correctly.
- **Cache Compiled Functions:** Restructure `wald_test` to use JIT-compiled functions where possible, or avoid closing over varying parameters.
- **Optimize Basis Generation:** Implement more direct tangent basis construction for common manifolds (Euclidean, Sphere).
- **Parallelize simulations:** While not an optimization of `wald_test` itself, it makes Monte Carlo feasible. (Note: dependent on `joblib` availability).
