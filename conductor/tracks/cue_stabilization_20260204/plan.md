# Implementation Plan: Stabilize CUE Weighting for Near-Singular Moments

## Phase 1: Implementation

- [ ] Task: Check existing `ridge_inverse` utility
    - [ ] Analysis: Verify if `src/manifoldgmm/utils/numeric.py` has a JAX-compatible `ridge_inverse`.
- [ ] Task: Update `CUEWeighting` to support ridge regularization
    - [ ] Write Tests: `tests/econometrics/test_cue_stabilization.py` comparing singular CUE with ridge CUE.
    - [ ] Implement Feature: Add `ridge` arg to `CUEWeighting` and use robust inversion.
- [ ] Task: Expose ridge configuration in `GMM` class
    - [ ] Implement Feature: Update `GMM.__init__` or `estimate` to allow passing ridge to default CUE.

## Phase 2: Verification

- [ ] Task: Verify fix with PSD rank-1 simulation
    - [ ] Write Tests: Re-run the problematic simulation from the previous track (but as a new test) using the stabilized CUE.
    - [ ] Analysis: Confirm it runs fast and converges.
- [ ] Task: Conductor - User Manual Verification 'Stability Results' (Protocol in workflow.md)
