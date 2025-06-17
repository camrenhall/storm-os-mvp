#!/usr/bin/env python3
"""
Pytest fixtures and helpers for flood classification tests
"""

import pytest
import numpy as np
import time
import psutil
import os
import sys
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Setup test environment
from test_utils import setup_test_environment

# Ensure test fixtures exist
try:
    setup_test_environment()
except Exception as e:
    print(f"Warning: Could not setup test fixtures: {e}")

# Add parent directories to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
pipeline_dir = parent_dir / "pipeline"

# Add paths for imports
sys.path.insert(0, str(parent_dir))
if pipeline_dir.exists():
    sys.path.insert(0, str(pipeline_dir))

# Import modules under test
try:
    from flash_ingest import FlashIngest
    from ffw_client import NWSFlashFloodWarnings
    from exposure import initialize_population_grid, homes, is_initialized
    from flood_classifier import FloodClassifier, ClassificationConfig, ProcessingMode
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def exposure_grid():
    """Initialize population exposure grid once per session"""
    
    # Setup test fixtures
    try:
        fixtures_dir = setup_test_environment()
    except Exception as e:
        pytest.skip(f"Could not setup test fixtures: {e}")
    
    if not is_initialized():
        try:
            # Try with test fixture path
            fixtures_path = Path(__file__).parent / "fixtures"
            parquet_file = fixtures_path / "pixel_exposure_conus.parquet"
            
            if parquet_file.exists():
                # Pass the actual test fixture path
                initialize_population_grid(str(parquet_file))
            else:
                pytest.skip("Test exposure grid not available")
                
        except Exception as e:
            pytest.skip(f"Population grid initialization failed: {e}")
    
    return True


@pytest.fixture
def standard_config():
    """Standard test configuration for classifier"""
    return ClassificationConfig(
        processing_mode=ProcessingMode.PRODUCTION,
        use_fixed_thresholds=True,
        enable_detailed_logging=False,  # Reduce test noise
        enable_ffw_enhancement=True,
        min_flood_area_pixels=9,
        min_valid_pixels=1000  # Lower for tests
    )


@pytest.fixture
def conus_grid_shape():
    """Standard CONUS grid shape"""
    return (3500, 7000)  # nj, ni


def make_synthetic_grid(shape: Tuple[int, int], value_dict: Dict[str, Any]) -> np.ndarray:
    """
    Helper to create synthetic streamflow grids with specified values at locations
    
    Args:
        shape: (nj, ni) grid dimensions
        value_dict: {(row, col): value} or {'background': value, 'land_mask_prob': prob}
    """
    nj, ni = shape
    grid = np.full((nj, ni), -9999.0, dtype=np.float32)
    
    # Handle background and land mask
    if 'background' in value_dict:
        background = value_dict['background']
        land_prob = value_dict.get('land_mask_prob', 0.3)
        
        # FIXED: Create realistic land coverage that won't trigger dead data fallback
        # Use current random state (tests should set seed before calling)
        land_mask = np.random.random((nj, ni)) < land_prob
        
        if land_mask.any():
            # Add some variability to background to make it more realistic
            land_pixels = land_mask.sum()
            background_values = np.random.exponential(background, land_pixels)
            grid[land_mask] = background_values
    
    # Set specific values (these override background)
    for key, value in value_dict.items():
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if 0 <= row < nj and 0 <= col < ni:
                grid[row, col] = value
    
    return grid


def make_qpe_grid(shape: Tuple[int, int], rain_locations: Dict[Tuple[int, int], float]) -> np.ndarray:
    """Create synthetic QPE (rainfall) grid"""
    nj, ni = shape
    qpe_grid = np.zeros((nj, ni), dtype=np.float32)
    
    for (row, col), rain_value in rain_locations.items():
        if 0 <= row < nj and 0 <= col < ni:
            qpe_grid[row, col] = rain_value
    
    return qpe_grid


def clock_and_peak_mem(fn, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Measure function execution time and peak memory usage
    
    Returns:
        (result, elapsed_seconds, peak_memory_mb)
    """
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start_time
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = final_memory - initial_memory
    
    return result, elapsed, peak_memory


@pytest.fixture
def synthetic_flood_events():
    """Create standard synthetic flood events for testing"""
    return [
        # (row, col, streamflow, qpe, ffw, homes, expected_tier)
        (1000, 2000, 1.5, 20.0, False, 100, None),      # Below threshold
        (1200, 2200, 2.5, 30.0, False, 200, 'moderate'), # Moderate
        (1400, 2400, 6.0, 60.0, True, 600, 'high'),      # High + QPE + FFW + exposure
        (1600, 2600, 12.0, 75.0, True, 800, 'critical'), # Critical + all bonuses
        (1800, 2800, 25.0, 100.0, True, 1000, 'critical'), # Extreme (test clamping)
    ]


class NetworkMixin:
    """Mixin for tests requiring network access"""
    
    @staticmethod
    def skip_if_no_network():
        """Skip test if network unavailable"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except (socket.error, OSError):
            pytest.skip("Network unavailable")


# Parametrized test data
THRESHOLD_TEST_CASES = [
    # (flow_value, expected_tier, description)
    (0.5, None, "below_moderate"),
    (1.9, None, "just_below_moderate"), 
    (2.1, "moderate", "just_above_moderate"),
    (4.9, "moderate", "just_below_high"),
    (5.1, "high", "just_above_high"),
    (9.9, "high", "just_below_critical"),
    (10.1, "critical", "just_above_critical"),
    (15.0, "critical", "max_streamflow_component"),
    (25.0, "critical", "extreme_flow"),
]

SCORING_TEST_CASES = [
    # (streamflow, qpe, ffw, homes, expected_score, description)
    (0.0, 0.0, False, 0, 0, "zero_everything"),
    (3.0, 0.0, False, 0, 10, "streamflow_only_moderate"), # 3/15*50 = 10
    (6.0, 0.0, False, 0, 20, "streamflow_only_high"),     # 6/15*50 = 20
    (12.0, 0.0, False, 0, 40, "streamflow_only_critical"), # 12/15*50 = 40
    (15.0, 0.0, False, 0, 50, "streamflow_max"),           # 15/15*50 = 50
    (12.0, 60.0, False, 0, 55, "streamflow_plus_qpe"),     # 40 + 15
    (12.0, 0.0, True, 0, 65, "streamflow_plus_ffw"),       # 40 + 25
    (12.0, 0.0, False, 600, 50, "streamflow_plus_exposure"), # 40 + 10
    (12.0, 60.0, True, 600, 90, "all_bonuses_no_clamp"),    # FIXED: 40+15+25+10 = 90 (no clamping needed)
    (18.0, 60.0, True, 600, 100, "all_bonuses_clamped"),    # NEW: 60+15+25+10 = 110 → clamped to 100
    (20.0, 80.0, True, 1000, 100, "extreme_clamped"),       # NEW: 50+15+25+10 = 100 (at max streamflow)
]