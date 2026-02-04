"""Tests for the Wald test functionality."""

import pytest
from manifoldgmm.econometrics.gmm import WaldTestResult

def test_wald_test_result_initialization():
    """Test that WaldTestResult can be initialized with correct attributes."""
    result = WaldTestResult(statistic=5.0, degrees_of_freedom=2, p_value=0.05)
    
    assert result.statistic == 5.0
    assert result.degrees_of_freedom == 2
    assert result.p_value == 0.05
