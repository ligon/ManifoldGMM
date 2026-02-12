"""Tests for the monte_carlo simulation runner."""

from __future__ import annotations

import numpy as np
import pytest
from manifoldgmm.econometrics.simulation import monte_carlo


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestMonteCarlo:
    """Tests for the monte_carlo() function."""

    @staticmethod
    def _coin_flip(rep: int, rng: np.random.Generator) -> dict:
        """Trivial replication function for testing."""
        return {"rep": rep, "heads": int(rng.integers(0, 2))}

    def test_serial_correct_count(self) -> None:
        results = monte_carlo(self._coin_flip, 20, seed=42, progress=False)
        assert len(results) == 20

    def test_records_contain_expected_keys(self) -> None:
        results = monte_carlo(self._coin_flip, 5, seed=0, progress=False)
        for r in results:
            assert "rep" in r
            assert "heads" in r

    def test_reproducible_with_seed(self) -> None:
        r1 = monte_carlo(self._coin_flip, 10, seed=42, progress=False)
        r2 = monte_carlo(self._coin_flip, 10, seed=42, progress=False)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        r1 = monte_carlo(self._coin_flip, 50, seed=0, progress=False)
        r2 = monte_carlo(self._coin_flip, 50, seed=999, progress=False)
        # At least some results should differ (extremely high probability)
        heads1 = [r["heads"] for r in r1]
        heads2 = [r["heads"] for r in r2]
        assert heads1 != heads2

    def test_parallel_matches_serial(self) -> None:
        """n_jobs=2 should produce identical results to n_jobs=1 (same seeds)."""
        r_serial = monte_carlo(self._coin_flip, 20, seed=42, n_jobs=1, progress=False)
        r_parallel = monte_carlo(self._coin_flip, 20, seed=42, n_jobs=2, progress=False)
        assert r_serial == r_parallel

    def test_error_handling(self) -> None:
        """Errors in individual replications should be captured, not propagated."""

        def failing_fn(rep: int, rng: np.random.Generator) -> dict:
            if rep == 2:
                raise ValueError("intentional failure")
            return {"rep": rep, "value": float(rng.random())}

        results = monte_carlo(failing_fn, 5, seed=0, progress=False)
        assert len(results) == 5
        # Rep 2 should have an error
        assert "error" in results[2]
        assert "intentional failure" in results[2]["error"]
        # Other reps should succeed
        for i in [0, 1, 3, 4]:
            assert "error" not in results[i]
            assert "rep" in results[i]

    def test_zero_reps(self) -> None:
        results = monte_carlo(self._coin_flip, 0, seed=42, progress=False)
        assert results == []
