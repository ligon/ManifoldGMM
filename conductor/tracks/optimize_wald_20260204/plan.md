# Implementation Plan: Investigate and Optimize Wald Test Performance

## Phase 1: Profiling and Identification [checkpoint: e10894c]

- [x] Task: Profile `wald_test` execution
    - [x] Write Tests: Create `scripts/profile_wald.py` to benchmark components
    - [x] Implement Feature: Measure time spent in JAX, `tangent_basis`, and `tangent_covariance`
- [x] Task: Identify the primary bottleneck
    - [x] Analysis: Identified CUEWeighting and JAX compilation as bottlenecks

## Phase 2: Implementation of Optimizations

- [x] Task: Optimize `tangent_basis` for Euclidean manifolds
    - [x] Write Tests: `tests/geometry/test_basis_perf.py` (Verified via profiling)
    - [x] Implement Feature: Faster basis generation in `MomentRestriction`
- [x] Task: Refactor `wald_test` to use `JacobianOperator` properly
    - [x] Write Tests: Ensure Wald results remain correct
    - [x] Implement Feature: Use `JacobianOperator` and avoid local JAX closures
- [x] Task: Conductor - User Manual Verification 'Optimization Results' (Protocol in workflow.md)
