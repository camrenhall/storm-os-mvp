#!/usr/bin/env python3
"""
G3 - Metadata integrity
Tests event metadata fields and consistency
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
from datetime import datetime, timezone

from conftest import make_synthetic_grid, synthetic_flood_events

try:
    from flood_classifier import FloodClassifier
except ImportError:
    pytest.skip("flood_classifier not available", allow_module_level=True)


class TestMetadataIntegrity:
    """Test event metadata completeness and consistency"""
    
    def test_metadata_fields_exist(self, standard_config, conus_grid_shape, synthetic_flood_events):
        """Test 10: All required metadata fields present"""
        
        classifier = FloodClassifier(standard_config)
        
        # Create test grid with multiple events
        nj, ni = conus_grid_shape
        unit_streamflow = make_synthetic_grid(conus_grid_shape, {'background': 0.01, 'land_mask_prob': 0.2})
        
        # Add test events (3x3 each to pass spatial filter)
        for row, col, flow, qpe, ffw, homes, tier in synthetic_flood_events:
            if flow >= 2.0:  # Only events above threshold
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        unit_streamflow[row + dr, col + dc] = flow
        
        # Classify and extract events
        result = classifier.classify(unit_streamflow, datetime.now())
        assert result is not None, "Classification failed"
        
        flood_events = classifier.extract_flood_events(result, unit_streamflow)
        
        # Collect all events
        all_events = []
        for severity, events in flood_events.items():
            all_events.extend(events)
        
        assert len(all_events) > 0, "No events extracted for metadata testing"
        
        # Required fields per spec
        required_fields = {
            'segment_id': str,
            'first_seen': str,
            'ttl_minutes': int,
            'event_score': int,
            'home_estimate': int  # Note: 'homes' → 'home_estimate' in implementation
        }
        
        for i, event in enumerate(all_events):
            for field, expected_type in required_fields.items():
                assert field in event, f"Event {i} missing required field '{field}'"
                
                value = event[field]
                assert isinstance(value, expected_type), \
                    f"Event {i} field '{field}' has wrong type: expected {expected_type}, got {type(value)}"
                
                # Additional validations
                if field == 'event_score':
                    assert 0 <= value <= 100, f"Event {i} score {value} outside valid range [0,100]"
                elif field == 'ttl_minutes':
                    assert value > 0, f"Event {i} TTL {value} should be positive"
                elif field == 'home_estimate':
                    assert value >= 0, f"Event {i} home estimate {value} should be non-negative"
                elif field == 'segment_id':
                    assert '_' in value, f"Event {i} segment_id '{value}' should contain underscore"
        
        print(f"✓ Metadata validation: {len(all_events)} events, all required fields present")
    
    def test_segment_id_uniqueness(self, standard_config, conus_grid_shape):
        """Test 11: segment_id consistency across runs"""
        
        classifier = FloodClassifier(standard_config)
        
        # Create deterministic test grid
        np.random.seed(42)  # Fixed seed for reproducibility
        
        unit_streamflow = make_synthetic_grid(
            conus_grid_shape,
            {
                'background': 0.01,
                'land_mask_prob': 0.2,
                (1000, 2000): 6.0,  # High tier event
                (1200, 2200): 12.0, # Critical tier event
            }
        )
        
        # Make events 3x3
        for center_row, center_col in [(1000, 2000), (1200, 2200)]:
            flow = unit_streamflow[center_row, center_col]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    unit_streamflow[center_row + dr, center_col + dc] = flow
        
        # Run classifier twice with same data
        valid_time = datetime.now()
        
        result1 = classifier.classify(unit_streamflow, valid_time)
        events1 = classifier.extract_flood_events(result1, unit_streamflow)
        
        result2 = classifier.classify(unit_streamflow, valid_time)
        events2 = classifier.extract_flood_events(result2, unit_streamflow)
        
        # Collect segment_ids from both runs
        ids1 = set()
        ids2 = set()
        
        for severity in ['critical', 'high', 'moderate']:
            for event in events1.get(severity, []):
                ids1.add(event['segment_id'])
            for event in events2.get(severity, []):
                ids2.add(event['segment_id'])
        
        assert len(ids1) > 0, "No segment IDs found in first run"
        assert len(ids2) > 0, "No segment IDs found in second run"
        assert ids1 == ids2, f"Segment IDs not consistent: run1={ids1}, run2={ids2}"
        
        print(f"✓ Segment ID consistency: {len(ids1)} unique IDs, consistent across runs")
    
    def test_ttl_minutes_default(self, standard_config, conus_grid_shape):
        """Test 12: TTL minutes set to spec default (180)"""
        
        classifier = FloodClassifier(standard_config)
        
        # Create test event
        unit_streamflow = make_synthetic_grid(
            conus_grid_shape,
            {
                'background': 0.01,
                'land_mask_prob': 0.2,
                (1000, 2000): 8.0,  # High event
            }
        )
        
        # Make it 3x3
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow[1000 + dr, 2000 + dc] = 8.0
        
        result = classifier.classify(unit_streamflow, datetime.now())
        flood_events = classifier.extract_flood_events(result, unit_streamflow)
        
        # Check all events have correct TTL
        all_events = []
        for severity, events in flood_events.items():
            all_events.extend(events)
        
        assert len(all_events) > 0, "No events found for TTL testing"
        
        expected_ttl = 180  # Per spec
        for event in all_events:
            actual_ttl = event['ttl_minutes']
            assert actual_ttl == expected_ttl, \
                f"Event TTL {actual_ttl} != expected {expected_ttl}"
        
        print(f"✓ TTL validation: {len(all_events)} events, all have TTL={expected_ttl}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])