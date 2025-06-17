# Create this as a standalone script: tests/create_test_fixtures.py
#!/usr/bin/env python3
"""
Create test fixtures for the test suite
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_minimal_exposure_grid():
    """Create minimal exposure grid that satisfies test requirements"""
    
    # Test data around our common test coordinates
    test_data = []
    
    # Key test locations from our tests
    test_locations = [
        # (row, col, homes) - covering test event locations
        (1000, 2000, 150),   # Moderate population
        (999, 1999, 100),    # Adjacent pixels for 3x3 blocks
        (999, 2000, 120),
        (999, 2001, 140),
        (1000, 1999, 110),
        (1000, 2001, 130),
        (1001, 1999, 105),
        (1001, 2000, 160),
        (1001, 2001, 135),
        
        (1500, 2500, 600),   # High population (triggers exposure bonus ≥500)
        (1499, 2499, 550),   # Adjacent pixels
        (1499, 2500, 580),
        (1499, 2501, 620),
        (1500, 2499, 590),
        (1500, 2501, 610),
        (1501, 2499, 570),
        (1501, 2500, 650),
        (1501, 2501, 630),
    ]
    
    for row, col, homes in test_locations:
        test_data.append({
            'row': row,
            'col': col,
            'roof_cnt': homes
        })
    
    # Add some scattered population for realism
    np.random.seed(42)  # Deterministic
    for _ in range(200):
        test_data.append({
            'row': np.random.randint(500, 2500),
            'col': np.random.randint(1000, 5000),
            'roof_cnt': np.random.randint(1, 300)
        })
    
    df = pd.DataFrame(test_data)
    
    # Create fixtures directory
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    # Save parquet file
    parquet_path = fixtures_dir / "pixel_exposure_conus.parquet"
    df.to_parquet(parquet_path, index=False)
    
    print(f"✓ Created test exposure grid: {parquet_path}")
    print(f"  Total populated pixels: {len(df)}")
    print(f"  Total homes: {df['roof_cnt'].sum():,}")
    print(f"  Coverage around test locations: ✓")
    
    return str(parquet_path)

if __name__ == "__main__":
    create_minimal_exposure_grid()