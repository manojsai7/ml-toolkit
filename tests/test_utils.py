"""
Tests for utility functions
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_toolkit.utils import (
    setup_logging, timer, create_directory_structure,
    ProgressTracker, get_random_state
)


def test_timer_decorator():
    """Test timer decorator."""
    @timer
    def sample_function():
        return sum(range(1000))
    
    result = sample_function()
    assert result == 499500


def test_create_directory_structure():
    """Test creating directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_directory_structure(tmpdir)
        
        assert (Path(tmpdir) / "data" / "raw").exists()
        assert (Path(tmpdir) / "data" / "processed").exists()
        assert (Path(tmpdir) / "models").exists()
        assert (Path(tmpdir) / "results").exists()


def test_progress_tracker():
    """Test progress tracker."""
    tracker = ProgressTracker(total=10, description="Test")
    
    tracker.update(5)
    assert tracker.current == 5
    
    tracker.update(5)
    assert tracker.current == 10
    
    tracker.close()


def test_get_random_state():
    """Test getting random state."""
    assert get_random_state() == 42
    assert get_random_state(100) == 100
    assert get_random_state(None) == 42
