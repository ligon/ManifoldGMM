# Implementation Plan: Wald Test Comparison on Fixed-Rank PSD Manifold

## Phase 1: Simulation and Comparison [checkpoint: e10894c]

- [x] Task: Implement Monte Carlo simulation for PSD Manifold vs Euclidean e10894c
    - [x] Write Tests: Create `tests/econometrics/test_psd_wald_comparison.py`
    - [~] Implement Feature: Compare size and power curves (basic test only)
- [~] Task: Document results in a new example
    - [x] Implement Feature: Create `docs/examples/psd_fixed_rank_inference.org` with plots
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Simulation and Comparison' (Protocol in workflow.md)

## Phase 2: Factor Model Interpretation (added)

- [x] Task: Reframe comparison as factor extraction problem
    - [x] Analysis: Identified that Euclidean estimates Cov(x) = vv' + σ²I while Manifold extracts vv'
    - [x] Analysis: Euclidean "size inflation" is actually correct behavior for wrong target
    - [x] Implement Feature: Updated org doc with full factor model exposition
- [x] Task: Add power curve simulation to org doc
    - [x] Implement Feature: Monte Carlo simulation varying effect size v₁
    - [x] Implement Feature: Matplotlib plotting code for power curves
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Factor Model Interpretation'
