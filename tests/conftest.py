"""
Pytest configuration and fixtures.
"""

import pytest
import torch


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def device():
    """Get device for testing."""
    return "cpu"
