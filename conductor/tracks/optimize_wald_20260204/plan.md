# Implementation Plan: Investigate and Optimize Wald Test Performance

## Phase 1: Profiling and Identification

- [ ] Task: Profile `wald_test` execution
    - [ ] Write Tests: Create `scripts/profile_wald.py` to benchmark components
    - [ ] Implement Feature: Measure time spent in JAX, `tangent_basis`, and `tangent_covariance`
- [ ] Task: Identify the primary bottleneck
    - [ ] Analysis: Determine if recompilation or `tangent_basis` is the main culprit

## Phase 2: Implementation of Optimizations

- [ ] Task: Optimize `tangent_basis` for Euclidean manifolds
    - [ ] Write Tests: `tests/geometry/test_basis_perf.py`
    - [ ] Implement Feature: Faster basis generation in `MomentRestriction`
- [ ] Task: Refactor `wald_test` to use `JacobianOperator` properly
    - [ ] Write Tests: Ensure Wald results remain correct
    - [ ] Implement Feature: Use `JacobianOperator` and avoid local JAX closures
- [ ] Task: Conductor - User Manual Verification 'Optimization Results' (Protocol in workflow.md)
