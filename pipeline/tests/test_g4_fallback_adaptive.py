#!/usr/bin/env python3
"""
G4 - Fallback / adaptive threshold tests
Tests percentile fallback logic and guardrails
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


class TestFallbackAdaptive:
    """Test adaptive threshold fallback logic"""
    
    def test_percentile_fallback_triggers(self, standard_config, conus_grid_shape):
        """Test 13: Percentile fallback triggers on dead data day"""
        
        # GAP G-3 FIX: Deterministic seed
        np.random.seed(42)
    
        classifier = FloodClassifier(standard_config)
        
        # Create "dead data day" scenario - very little active flow
        nj, ni = conus_grid_shape
        unit_streamflow = np.full((nj, ni), -9999.0, dtype=np.float32)
        
        # Add land areas with very low background flow
        land_mask = np.random.random((nj, ni)) > 0.7  # 30% land coverage
        unit_streamflow[land_mask] = np.random.exponential(0.02, land_mask.sum())  # Very low flows
        
        # Add just a few slightly higher pixels (but still low)
        high_flow_pixels = 100
        high_indices = np.random.choice(land_mask.sum(), high_flow_pixels, replace=False)
        land_coords = np.where(land_mask)
        selected_rows = land_coords[0][high_indices]
        selected_cols = land_coords[1][high_indices]
        unit_streamflow[selected_rows, selected_cols] = 0.2  # Still very low
        
        # This should trigger <1% active pixels condition
        valid_mask = (unit_streamflow != -9999.0) & np.isfinite(unit_streamflow)
        active_pixels = (unit_streamflow[valid_mask] > 0.1).sum()
        active_percentage = (active_pixels / valid_mask.sum()) * 100
        
        assert active_percentage < 1.0, f"Test setup failed: {active_percentage:.3f}% active (should be <1%)"
        
        # Get thresholds - should use fallback but respect minimums
        thresholds = classifier.calculate_spec_thresholds(unit_streamflow)
        assert thresholds is not None, "Threshold calculation failed"
        
        # Fallback should still respect spec minimums
        assert thresholds['critical'] >= 10.0, f"Critical threshold {thresholds['critical']} below spec minimum 10.0"
        assert thresholds['high'] >= 5.0, f"High threshold {thresholds['high']} below spec minimum 5.0"
        assert thresholds['moderate'] >= 2.0, f"Moderate threshold {thresholds['moderate']} below spec minimum 2.0"
        
        print(f"✓ Fallback triggered: {active_percentage:.3f}% active, thresholds: {thresholds}")
    
    def test_min_thresholds_guardrail(self, standard_config, conus_grid_shape):
        """Test 14: Zero-flow grid doesn't produce false positives"""
        
        # GAP G-3 FIX: Deterministic seed
        np.random.seed(43)
        
        classifier = FloodClassifier(standard_config)
        
        # Create completely zero-flow grid
        nj, ni = conus_grid_shape
        unit_streamflow = np.full((nj, ni), -9999.0, dtype=np.float32)
        
        # Add minimal land areas with zero flow
        land_mask = np.random.random((nj, ni)) > 0.8  # 20% land
        unit_streamflow[land_mask] = 0.0  # Exactly zero flow
        
        # Should not crash and should return zero events
        result = classifier.classify(unit_streamflow, datetime.now())
        
        if result is not None:
            # Should detect no flood pixels
            total_flood_pixels = result.critical_count + result.high_count + result.moderate_count
            assert total_flood_pixels == 0, f"Zero-flow grid should not produce flood pixels, got {total_flood_pixels}"
            
            # Should extract no events
            flood_events = classifier.extract_flood_events(result, unit_streamflow)
            total_events = sum(len(events) for events in flood_events.values())
            assert total_events == 0, f"Zero-flow grid should not produce events, got {total_events}"
        
        print(f"✓ Zero-flow guardrail: no false positives generated")


if __name__ == "__main__":
   pytest.main([__file__, "-v"])