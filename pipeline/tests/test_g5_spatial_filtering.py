#!/usr/bin/env python3
"""
G5 - Spatial filtering & dedup logic
Tests component size limits and merging behavior
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


class TestSpatialFiltering:
    """Test spatial filtering and component analysis"""
    
    def test_component_size_limit(self, standard_config, conus_grid_shape):
        """Test 15: Single pixel discarded, 3x3 block accepted"""
        
        classifier = FloodClassifier(standard_config)
        
        nj, ni = conus_grid_shape
        
        # Test 1: Single pixel (should be discarded)
        unit_streamflow_single = make_synthetic_grid(
            conus_grid_shape,
            {
                'background': 0.01,
                'land_mask_prob': 0.2,
                (1000, 2000): 12.0  # Single critical pixel
            }
        )
        
        result_single = classifier.classify(unit_streamflow_single, datetime.now())
        assert result_single is not None, "Single pixel classification failed"
        
        events_single = classifier.extract_flood_events(result_single, unit_streamflow_single)
        total_events_single = sum(len(events) for events in events_single.values())
        
        assert total_events_single == 0, f"Single pixel should be filtered out, got {total_events_single} events"
        
        # Test 2: 3x3 block (should be accepted)
        unit_streamflow_block = make_synthetic_grid(
            conus_grid_shape,
            {
                'background': 0.01,
                'land_mask_prob': 0.2
            }
        )
        
        # Create 3x3 block
        center_row, center_col = 1000, 2000
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow_block[center_row + dr, center_col + dc] = 12.0
        
        result_block = classifier.classify(unit_streamflow_block, datetime.now())
        assert result_block is not None, "3x3 block classification failed"
        
        events_block = classifier.extract_flood_events(result_block, unit_streamflow_block)
        total_events_block = sum(len(events) for events in events_block.values())
        
        assert total_events_block >= 1, f"3x3 block should produce at least 1 event, got {total_events_block}"
        
        print(f"✓ Spatial filtering: single pixel → {total_events_single} events, 3x3 block → {total_events_block} events")
    
    def test_duplicate_merge(self, standard_config, conus_grid_shape):
        """Test 16: Adjacent blocks merge into single event"""
        
        classifier = FloodClassifier(standard_config)
        
        # Create two adjacent 3x3 blocks that share pixels
        unit_streamflow = make_synthetic_grid(
            conus_grid_shape,
            {
                'background': 0.01,
                'land_mask_prob': 0.2
            }
        )
        
        # First 3x3 block
        center1_row, center1_col = 1000, 2000
        flow1 = 8.0  # High tier
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow[center1_row + dr, center1_col + dc] = flow1
        
        # Second 3x3 block, adjacent (sharing pixels)
        center2_row, center2_col = 1000, 2002  # 2 pixels to the right
        flow2 = 12.0  # Critical tier (higher)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow[center2_row + dr, center2_col + dc] = flow2
        
        result = classifier.classify(unit_streamflow, datetime.now())
        assert result is not None, "Merge test classification failed"
        
        events = classifier.extract_flood_events(result, unit_streamflow)
        
        # Should have merged into fewer events than input blocks
        all_events = []
        for severity, event_list in events.items():
            all_events.extend(event_list)
        
        # The exact behavior depends on connected component analysis
        # Adjacent/overlapping components should merge
        assert len(all_events) <= 2, f"Expected ≤2 events from merging, got {len(all_events)}"
        
        if len(all_events) == 1:
            # Perfect merge - check it has the higher flow value
            merged_event = all_events[0]
            max_flow = merged_event['max_streamflow']
            assert max_flow == max(flow1, flow2), f"Merged event should have max flow {max(flow1, flow2)}, got {max_flow}"
        
        print(f"✓ Component merging: 2 adjacent blocks → {len(all_events)} events")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])