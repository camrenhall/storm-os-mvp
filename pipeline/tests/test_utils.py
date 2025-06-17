#!/usr/bin/env python3
"""
Test utilities for creating fixtures and stubs
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_stub_exposure_grid(fixtures_dir: Path) -> str:
    """
    Create minimal exposure grid for testing to eliminate warnings
    """
    fixtures_dir.mkdir(exist_ok=True)
    
    # Create minimal realistic test data
    # Cover a few test areas where our tests place events
    test_data = []
    
    # Add data around our common test coordinates (1000, 2000) and (1500, 2500)
    test_areas = [
        (1000, 2000, 150),  # Moderate population
        (1001, 2000, 200),
        (1000, 2001, 100),
        (1500, 2500, 600),  # High population (triggers exposure bonus)
        (1501, 2500, 800),
        (1500, 2501, 550),
        (2000, 3000, 300),  # Another test area
        (2001, 3000, 250),
    ]
    
    for row, col, homes in test_areas:
        test_data.append({
            'row': row,
            'col': col, 
            'roof_cnt': homes
        })
    
    # Add some random populated areas for realism
    np.random.seed(42)  # Deterministic
    for _ in range(100):
        test_data.append({
            'row': np.random.randint(500, 2500),
            'col': np.random.randint(1000, 4000),
            'roof_cnt': np.random.randint(0, 500)
        })
    
    df = pd.DataFrame(test_data)
    
    parquet_path = fixtures_dir / "pixel_exposure_conus.parquet"
    df.to_parquet(parquet_path, index=False)
    
    print(f"Created stub exposure grid: {parquet_path} with {len(df)} populated pixels")
    return str(parquet_path)


def setup_test_environment():
    """Set up test environment with required fixtures"""
    current_dir = Path(__file__).parent
    fixtures_dir = current_dir / "fixtures"
    
    # Create exposure grid if it doesn't exist
    parquet_path = fixtures_dir / "pixel_exposure_conus.parquet"
    if not parquet_path.exists():
        create_stub_exposure_grid(fixtures_dir)
    
    return str(fixtures_dir)