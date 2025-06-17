#!/usr/bin/env python3
"""
G6 - Performance & resource usage
Tests lookup performance and classification latency
"""

import sys
from pathlib import Path

# Add parent directories to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
pipeline_dir = parent_dir / "pipeline"

sys.path.insert(0, str(parent_dir))
if pipeline_dir.exists():
    sys.path.insert(0, str(pipeline_dir))

import pytest
import numpy as np
from datetime import datetime
import time

from conftest import clock_and_peak_mem, make_synthetic_grid

try:
    from flood_classifier import FloodClassifier
    from exposure import homes
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class TestPerformance:
    """Test performance benchmarks"""
    
    @pytest.mark.slow
    def test_lookup_performance_50k(self, exposure_grid):
        """Test 17: Population grid lookup performance"""
        
        n_samples = 50000
        
        # Generate random valid coordinates
        nj, ni = 3500, 7000
        rows = np.random.randint(0, nj, n_samples)
        cols = np.random.randint(0, ni, n_samples)
        
        # Benchmark lookups
        def do_lookups():
            total_homes = 0
            for row, col in zip(rows, cols):
                total_homes += homes(row, col)
            return total_homes
        
        result, elapsed, _ = clock_and_peak_mem(do_lookups)
        
        # Performance requirements
        max_time_ms = 50.0  # 50ms for 50k lookups
        actual_time_ms = elapsed * 1000
        
        assert actual_time_ms < max_time_ms, \
            f"Lookup performance too slow: {actual_time_ms:.1f}ms > {max_time_ms}ms for {n_samples} lookups"
        
        lookups_per_second = n_samples / elapsed
        assert lookups_per_second >= 1_000_000, \
            f"Lookup rate {lookups_per_second:,.0f} < 1M lookups/sec"
        
        print(f"✓ Lookup performance: {n_samples:,} lookups in {actual_time_ms:.1f}ms ({lookups_per_second:,.0f} lookups/sec)")
    
    @pytest.mark.slow
    def test_classify_conus_latency(self, standard_config, conus_grid_shape, exposure_grid):
        """Test 18: Full CONUS classification latency"""
        
        # GAP G-3 FIX: Deterministic seed
        np.random.seed(44)
        
        classifier = FloodClassifier(standard_config)
        
        # Create realistic test data
        nj, ni = conus_grid_shape
        
        def create_test_data():
            # Create background
            unit_streamflow = make_synthetic_grid(
                conus_grid_shape,
                {
                    'background': 0.02,
                    'land_mask_prob': 0.35  # 35% land coverage
                }
            )
            
            # Add some flood events
            np.random.seed(42)  # Reproducible
            n_events = 20
            for i in range(n_events):
                center_row = np.random.randint(100, nj-100)
                center_col = np.random.randint(100, ni-100)
                intensity = np.random.choice([2.5, 6.0, 12.0])  # Mix of severities
                
                # Create 5x5 event
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if 0 <= center_row+dr < nj and 0 <= center_col+dc < ni:
                            unit_streamflow[center_row+dr, center_col+dc] = intensity
            
            return unit_streamflow
        
        # Benchmark classification
        def run_classification():
            unit_streamflow = create_test_data()
            result = classifier.classify(unit_streamflow, datetime.now())
            
            if result is not None:
                events = classifier.extract_flood_events(result, unit_streamflow)
                total_events = sum(len(event_list) for event_list in events.values())
                return result, events, total_events
            return None, {}, 0
        
        result_data, elapsed, peak_memory = clock_and_peak_mem(run_classification)
        result, events, total_events = result_data
        
        # Performance requirements
        max_time_sec = 5.0  # 5 seconds
        max_memory_mb = 600.0  # 600 MB peak
        
        assert elapsed < max_time_sec, \
            f"Classification too slow: {elapsed:.2f}s > {max_time_sec}s"
        
        assert peak_memory < max_memory_mb, \
            f"Memory usage too high: {peak_memory:.1f}MB > {max_memory_mb}MB"
        
        assert result is not None, "Classification failed"
        assert total_events >= 0, "Event extraction failed"
        
        print(f"✓ CONUS classification: {elapsed:.2f}s, {peak_memory:.1f}MB peak, {total_events} events")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])