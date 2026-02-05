# Implementation Plan: Stabilize CUE Weighting for Near-Singular Moments

## Phase 1: Implementation [checkpoint: 7c7c7c3]

- [x] Task: Check existing `ridge_inverse` utility
- [x] Task: Update `CUEWeighting` to support ridge regularization 7c7c7c3
    - [x] Write Tests: `tests/econometrics/test_cue_stabilization.py` comparing singular CUE with ridge CUE.
    - [x] Implement Feature: Add `ridge` arg to `CUEWeighting` and use robust inversion.
- [x] Task: Expose ridge configuration in `GMM` class 7c7c7c3
    - [x] Implement Feature: Update `GMM.__init__` with `cue_ridge` parameter.

## Phase 2: Verification

- [x] Task: Verify fix with PSD rank-1 simulation
    - [x] Write Tests: Added `test_psd_rank1_cue_with_ridge_stabilization` to `tests/econometrics/test_cue_stabilization.py`
    - [x] Analysis: CUE + ridge (0.1) with TrustRegions completes in ~8s (vs hanging before), converges with Frobenius error ~0.50
- [ ] Task: Conductor - User Manual Verification 'Stability Results' (Protocol in workflow.md)
