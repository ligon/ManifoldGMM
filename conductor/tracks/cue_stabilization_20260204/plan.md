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

## Phase 3: Adaptive Ridge

- [x] Task: Implement adaptive ridge via `target_condition` parameter
    - [x] Implement Feature: `CUEWeighting` computes eigenvalues of Ω(θ) at each step,
          adjusts ridge to keep cond(Ω + ridge·I) ≤ target_condition
    - [x] Implement Feature: `GMM` exposes `cue_target_condition` parameter
    - [x] Write Tests: `test_cue_adaptive_ridge_with_target_condition`, `test_psd_rank1_cue_adaptive_ridge`
    - [x] Optimization: NumPy path uses `cond()` first, only computes eigenvalues when needed;
          JAX path computes eigenvalues directly (required for tracing)

## Phase 4: Inference Validity Diagnostic

- [x] Task: Implement diagnostic for ridge-induced distortion of test statistics
    - [x] Analysis: Ridge regularization W = (Ω + λI)⁻¹ distorts J-statistic and Wald tests
          when λ is comparable to λ_min(Ω)
    - [x] Implement Feature: `CUEWeighting.info()` computes `ridge_ratio = λ/λ_min` and
          issues warnings when ratio > 0.1 (mild) or ratio > 1.0 (severe)
    - [x] Implement Feature: `GMMResult.check_inference_validity()` method exposes diagnostic
    - [x] Write Tests: `test_cue_inference_validity_diagnostic`, `test_cue_inference_validity_no_warning_when_small_ridge`
    - [x] Bug fix: Handle λ_min ≈ 0 case (set ridge_ratio = ∞ when λ_min < 1e-14)
