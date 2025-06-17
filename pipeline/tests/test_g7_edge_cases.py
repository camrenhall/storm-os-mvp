#!/usr/bin/env python3
"""
G7 - Negative & edge cases
Tests robustness with extreme inputs
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

from conftest import make_synthetic_grid

try:
    from flood_classifier import FloodClassifier
except ImportError:
    pytest.skip("flood_classifier not available", allow_module_level=True)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_all_zero_grid(self, standard_config, conus_grid_shape):
        """Test 19: All-zero grid handling"""
        
        # GAP G-3 FIX: Deterministic seed
        np.random.seed(45)
        
        classifier = FloodClassifier(standard_config)
        
        # Create grid with all zeros (valid land but no flow)
        nj, ni = conus_grid_shape
        unit_streamflow = np.full((nj, ni), -9999.0, dtype=np.float32)
        
        # Add land areas with exactly zero flow
        land_mask = np.random.random((nj, ni)) > 0.7  # 30% land
        unit_streamflow[land_mask] = 0.0
        
        # Should not crash
        result = classifier.classify(unit_streamflow, datetime.now())
        
        if result is not None:
            # Should produce no flood pixels
            total_flood = result.critical_count + result.high_count + result.moderate_count
            assert total_flood == 0, f"All-zero grid should produce no floods, got {total_flood}"
            
            # Should extract no events
            events = classifier.extract_flood_events(result, unit_streamflow)
            total_events = sum(len(event_list) for event_list in events.values())
            assert total_events == 0, f"All-zero grid should produce no events, got {total_events}"
        
        print(f"✓ All-zero grid: handled gracefully, no false positives")
    
    def test_nan_values_cleaned(self, standard_config, conus_grid_shape):
        """Test 20: NaN values handled gracefully"""
        
        # GAP G-3 FIX: Deterministic seed
        np.random.seed(46)
        
        classifier = FloodClassifier(standard_config)
        
        # Create grid with NaN pollution
        unit_streamflow = make_synthetic_grid(
            conus_grid_shape,
            {
                'background': 0.02,
                'land_mask_prob': 0.3,
                (1000, 2000): 8.0  # Valid flood event
            }
        )
        
        # Make it 3x3 valid event
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow[1000 + dr, 2000 + dc] = 8.0
        
        # Inject NaN values throughout
        nj, ni = conus_grid_shape
        n_nans = 10000
        nan_rows = np.random.randint(0, nj, n_nans)
        nan_cols = np.random.randint(0, ni, n_nans)
        unit_streamflow[nan_rows, nan_cols] = np.nan
        
        # Also inject inf values
        n_infs = 1000
        inf_rows = np.random.randint(0, nj, n_infs)
        inf_cols = np.random.randint(0, ni, n_infs)
        unit_streamflow[inf_rows, inf_cols] = np.inf
        
        # Should not crash and should still detect valid events
        result = classifier.classify(unit_streamflow, datetime.now())
        assert result is not None, "Classification with NaNs failed"
        
        # Should still detect the valid flood event
        total_flood = result.critical_count + result.high_count + result.moderate_count
        assert total_flood > 0, "NaN pollution prevented detection of valid events"
        
        events = classifier.extract_flood_events(result, unit_streamflow)
        total_events = sum(len(event_list) for event_list in events.values())
        assert total_events > 0, "Event extraction failed with NaN pollution"
        
        print(f"✓ NaN/inf handling: {n_nans} NaNs, {n_infs} infs injected, still detected {total_events} events")
    
    def test_extreme_flow_value(self, standard_config, conus_grid_shape):
        """Test 21: Extreme flow values don't cause overflow"""
        
        classifier = FloodClassifier(standard_config)
        
        # Test extreme but realistic values
        extreme_flows = [50.0, 100.0, 500.0, 1000.0]
        
        for extreme_flow in extreme_flows:
            # Create grid with extreme value
            unit_streamflow = make_synthetic_grid(
                conus_grid_shape,
                {
                    'background': 0.02,
                    'land_mask_prob': 0.3
                }
            )
            
            # Create 3x3 extreme event
            center_row, center_col = 1000, 2000
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    unit_streamflow[center_row + dr, center_col + dc] = extreme_flow
            
            # Should not crash
            result = classifier.classify(unit_streamflow, datetime.now())
            assert result is not None, f"Classification failed with extreme flow {extreme_flow}"
            
            # Should produce at least one critical event
            assert result.critical_count > 0, f"Extreme flow {extreme_flow} should produce critical events"
            
            # Test event scoring doesn't overflow
            events = classifier.extract_flood_events(result, unit_streamflow)
            all_events = []
            for severity, event_list in events.items():
                all_events.extend(event_list)
            
            assert len(all_events) > 0, f"No events extracted for extreme flow {extreme_flow}"
            
            for event in all_events:
                score = event['event_score']
                assert isinstance(score, int), f"Score should be int, got {type(score)}"
                assert 0 <= score <= 100, f"Score {score} outside valid range for flow {extreme_flow}"
                assert not np.isnan(score) and not np.isinf(score), f"Score {score} is NaN/inf for flow {extreme_flow}"
            
            max_score = max(e['event_score'] for e in all_events)
            assert max_score <= 100, f"Extreme flow {extreme_flow} produced score {max_score} > 100"
        
        print(f"✓ Extreme values: tested flows up to {max(extreme_flows)}, all scores ≤ 100")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])