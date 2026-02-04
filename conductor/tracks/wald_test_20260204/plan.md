# Implementation Plan: Wald Test for Manifold-Valued Parameters

## Phase 1: Foundation and Types [checkpoint: d89d2a9]

- [x] Task: Define `WaldTestResult` data class b82f936
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation and Types' (Protocol in workflow.md)

## Phase 2: Implementation and Calculation [checkpoint: c8e7034]

- [x] Task: Implement `wald_test` method in `GMMResult` ee852cf
- [x] Task: Verify statistic with a simple identity constraint dd9c658
- [x] Task: Implement Monte Carlo tests for Wald statistic distribution on Circle manifold b91a9b5
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation and Calculation' (Protocol in workflow.md)

## Phase 3: Integration and Examples [checkpoint: d86b869]

- [x] Task: Add Wald test example to documentation 3ed4130
    - [x] Write Tests: Ensure the example code runs correctly
    - [x] Implement Feature: Create `docs/examples/wald_test_example.org`
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration and Examples' (Protocol in workflow.md)

## Phase 4: Extended Analysis

- [x] Task: Add size and power simulation analysis to documentation 89293aa
    - [ ] Write Tests: Verify the simulation code logic (can reuse existing test logic)
    - [x] Implement Feature: Extend `docs/examples/wald_test_example.org` with Monte Carlo simulation blocks
- [ ] Task: Compare Manifold vs Euclidean power curves
    - [ ] Implement Feature: Add power curve generation code to `docs/examples/wald_test_example.org`
    - [ ] Analysis: Discuss the intuition (shrinkage/bias) and results
- [~] Task: Conductor - User Manual Verification 'Phase 4: Extended Analysis' (Protocol in workflow.md)
