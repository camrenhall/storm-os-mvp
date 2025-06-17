#!/usr/bin/env python3
"""
Phase 1 Completion Test - Validates spec-compliant implementation
"""

import numpy as np
import pytest
from datetime import datetime
from flood_classifier import FloodClassifier, ClassificationConfig, ProcessingMode

def test_phase1_completion():
        """TEST 4: Verify Phase 1 completion criteria per spec"""
        
        # Create spec-compliant configuration
        config = ClassificationConfig(
            processing_mode=ProcessingMode.PRODUCTION,
            use_fixed_thresholds=True,
            enable_detailed_logging=True,
            enable_ffw_enhancement=True
        )
        
        classifier = FloodClassifier(config)
        
        # Create test data with sufficient active streamflow to avoid dead data fallback
        nj, ni = 3500, 7000
        unit_streamflow = np.full((nj, ni), -9999.0, dtype=np.float32)
        
        # Add land areas with higher background flow to exceed 1% threshold
        land_mask = np.random.random((nj, ni)) > 0.3  # More land coverage
        # Use higher background flows - many pixels > 0.1 m³/s/km²
        background_flows = np.random.lognormal(mean=-1.0, sigma=1.5, size=land_mask.sum())
        background_flows = np.clip(background_flows, 0.0, 2.0)  # Clip to reasonable range
        unit_streamflow[land_mask] = background_flows
        
        # Add many additional active pixels to ensure >1% are above 0.1 m³/s/km²
        # This prevents dead data day fallback
        n_active_needed = int(0.02 * nj * ni)  # 2% of grid to be safe
        active_rows = np.random.randint(0, nj, n_active_needed)
        active_cols = np.random.randint(0, ni, n_active_needed)
        
        for i in range(n_active_needed):
            row, col = active_rows[i], active_cols[i]
            if unit_streamflow[row, col] == -9999.0:  # Only set if not already set
                unit_streamflow[row, col] = np.random.uniform(0.15, 1.0)  # Above 0.1 threshold
        
        # NOW add test events at exact spec thresholds
        test_events = [
            (1000, 2000, 2.1),   # moderate threshold
            (1500, 2500, 5.2),   # high threshold  
            (2000, 3000, 10.5),  # critical threshold
            (2500, 3500, 15.0)   # high intensity
        ]
        
        for row, col, intensity in test_events:
            # Create 5x5 flood area to ensure detection
            unit_streamflow[row-2:row+3, col-2:col+3] = intensity
        
        # Verify we have enough active data
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        active_pixels = (unit_streamflow[valid_mask] > 0.1).sum()
        active_percentage = (active_pixels / valid_mask.sum()) * 100
        
        print(f"Created test data with {len(test_events)} flood events")
        print(f"Active pixels: {active_pixels:,} ({active_percentage:.2f}% - should be >1% to avoid fallback)")
        
        assert active_percentage > 1.0, f"Test data has insufficient active pixels: {active_percentage:.2f}% (need >1%)"
        
        # Test classification
        result = classifier.classify(unit_streamflow, datetime.now())
        
        assert result is not None, "Classification failed"
        assert result.normalization_method == "Fixed_Spec_Thresholds", f"Wrong normalization method: {result.normalization_method}"
        
        # Verify thresholds are spec-compliant (should now use fixed values)
        assert result.critical_threshold_value == 10.0, f"Critical threshold should be 10.0, got {result.critical_threshold_value}"
        assert result.high_threshold_value == 5.0, f"High threshold should be 5.0, got {result.high_threshold_value}"
        assert result.moderate_threshold_value == 2.0, f"Moderate threshold should be 2.0, got {result.moderate_threshold_value}"
        
        # Verify we detected the expected events
        total_flood_pixels = result.critical_count + result.high_count + result.moderate_count
        assert total_flood_pixels > 0, "No flood pixels detected"
        
        print(f"Detected {total_flood_pixels} flood pixels")
        print(f"Critical: {result.critical_count}, High: {result.high_count}, Moderate: {result.moderate_count}")
        
        # Test event extraction
        flood_events = classifier.extract_flood_events(result, unit_streamflow)
        
        # Collect all events
        all_events = []
        for severity, events in flood_events.items():
            all_events.extend(events)
        
        assert len(all_events) > 0, "No flood events extracted"
        
        # PHASE 1 CRITICAL VALIDATION: All events must have event_score > 0
        zero_score_events = [e for e in all_events if e.get('event_score', 0) <= 0]
        assert len(zero_score_events) == 0, f"PHASE 1 FAILURE: {len(zero_score_events)} events have zero event_score"
        
        # PHASE 1 CRITICAL VALIDATION: All events must have home_estimate
        missing_homes_events = [e for e in all_events if e.get('home_estimate') is None]
        assert len(missing_homes_events) == 0, f"PHASE 1 FAILURE: {len(missing_homes_events)} events missing home_estimate"
        
        # Verify scoring formula components for a few events
        for i, event in enumerate(all_events[:3]):  # Check top 3 events
            streamflow = event['max_streamflow']
            homes = event['home_estimate']
            ffw = event.get('ffw_confirmed', False)
            score = event['event_score']
            
            # Calculate expected score using exact spec formula
            expected_streamflow_pts = min(int(streamflow / 0.3 * 10), 50)
            expected_ffw_pts = 25 if ffw else 0
            expected_qpe_pts = 0  # Phase 2
            expected_exposure_pts = 10 if homes >= 500 else 0
            expected_score = expected_streamflow_pts + expected_ffw_pts + expected_qpe_pts + expected_exposure_pts
            
            assert score == expected_score, f"Score mismatch for event {i+1}: got {score}, expected {expected_score} (streamflow_pts={expected_streamflow_pts}, ffw_pts={expected_ffw_pts}, exposure_pts={expected_exposure_pts})"
            
            print(f"Event {i+1} validation: streamflow={streamflow:.1f}, homes={homes}, ffw={ffw}, score={score}")
        
        print(f"✅ PHASE 1 COMPLETION TEST PASSED")
        print(f"  Total events: {len(all_events)}")
        print(f"  All events have event_score > 0: ✓")
        print(f"  All events have home_estimate: ✓") 
        print(f"  Scoring    formula verified: ✓")
        print(f"  Fixed thresholds implemented: ✓")
        print(f"  Thresholds used: {result.critical_threshold_value}/{result.high_threshold_value}/{result.moderate_threshold_value} m³/s/km²")

if __name__ == "__main__":
    test_phase1_completion()