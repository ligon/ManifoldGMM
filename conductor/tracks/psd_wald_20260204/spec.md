# Specification: Wald Test Comparison on Fixed-Rank PSD Manifold

## Goal
Demonstrate the advantages of manifold-constrained inference on the Fixed-Rank Positive Semi-Definite (PSD) manifold compared to naive Euclidean inference.

## Problem Description
Estimate a $3 \times 3$ PSD matrix of rank 1 ($k=1$) using GMM.
- Manifold: $\mathcal{S}_+(3, 1)$ represented by factor $Y \in \mathbb{R}^{3 \times 1}$ such that $A = YY^\top$.
- Hypothesis $H_0$: A specific linear constraint on the elements of $A$ (e.g., $A_{11} = A_{22}$). 

## Comparison
1. **Manifold Wald Test:** Respects rank-1 constraint during estimation and inference.
2. **Euclidean Wald Test:** Ignores rank/definiteness, estimates all 9 elements (or 6 unique) independently.

## Expectations
- Manifold test should have higher power because it focuses the search and variance estimation on the valid parameter space.
- Euclidean test might have poor size or power due to the "leakage" of variance into invalid (higher-rank or non-PSD) dimensions.

