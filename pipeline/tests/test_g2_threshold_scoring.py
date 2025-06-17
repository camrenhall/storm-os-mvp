#!/usr/bin/env python3
"""
G2 - Threshold & scoring correctness
Tests fixed 2/5/10 thresholds and exact scoring formula
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

from conftest import make_synthetic_grid, make_qpe_grid, THRESHOLD_TEST_CASES, SCORING_TEST_CASES

try:
    from flood_classifier import FloodClassifier
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    pytest.skip("flood_classifier not available", allow_module_level=True)


class TestThresholdScoring:
    """Test threshold classification and event scoring"""
    
    @pytest.mark.parametrize("flow,expected_tier,description", THRESHOLD_TEST_CASES, 
                       ids=[case[2] for case in THRESHOLD_TEST_CASES])
    def test_fixed_thresholds_hit_correct_tier(self, standard_config, conus_grid_shape, 
                                            flow, expected_tier, description):
        """Test 5: Fixed thresholds classify correctly"""
        
        # GAP G-3 FIX: Deterministic seed
        np.random.seed(42)
        
        classifier = FloodClassifier(standard_config)
        
        # Create grid with adequate background to avoid dead data fallback
        nj, ni = conus_grid_shape
        unit_streamflow = make_synthetic_grid(
            conus_grid_shape, 
            {
                'background': 0.05,  # Higher background to avoid dead data detection
                'land_mask_prob': 0.35  # More land coverage
            }
        )
        
        # FIXED: Create events that respect spatial filtering requirements
        center_row, center_col = 1000, 2000
        
        if expected_tier is not None:
            # Create 3x3 block (≥9 pixels) to survive spatial filtering
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    unit_streamflow[center_row + dr, center_col + dc] = flow
        else:
            # For events that should be filtered out, create single pixel
            unit_streamflow[center_row, center_col] = flow
        
        # Classify
        result = classifier.classify(unit_streamflow, datetime.now())
        assert result is not None, f"Classification failed for {description}"
        
        # Verify we're using fixed thresholds (not fallback)
        if expected_tier is not None:
            assert result.normalization_method == "Fixed_Spec_Thresholds", \
                f"Should use fixed thresholds for {description}, got {result.normalization_method}"
        
        # Check tier assignment
        if expected_tier is None:
            total_pixels = result.critical_count + result.high_count + result.moderate_count
            assert total_pixels == 0, f"Expected no pixels classified for {description}, got {total_pixels}"
        elif expected_tier == "moderate":
            assert result.moderate_count >= 1, f"Expected moderate pixels for {description}"
        elif expected_tier == "high":
            assert result.high_count >= 1, f"Expected high pixels for {description}"
        elif expected_tier == "critical":
            assert result.critical_count >= 1, f"Expected critical pixels for {description}"
        
        print(f"✓ {description}: flow={flow:.1f} → {expected_tier or 'none'}")
    
    @pytest.mark.parametrize("streamflow,qpe,ffw,homes,expected_score,description", SCORING_TEST_CASES,
                       ids=[case[5] for case in SCORING_TEST_CASES])
    def test_event_score_components(self, standard_config, conus_grid_shape,
                                   streamflow, qpe, ffw, homes, expected_score, description):
        """Test 6: Event scoring formula components"""
        
        classifier = FloodClassifier(standard_config)
        
        # Create minimal grid with one significant event (3x3 to pass spatial filter)
        nj, ni = conus_grid_shape
        center_row, center_col = 1000, 2000
        
        unit_streamflow = make_synthetic_grid(conus_grid_shape, {'background': 0.01, 'land_mask_prob': 0.2})
        qpe_grid = np.zeros((nj, ni), dtype=np.float32)
        
        # Create 3x3 event
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow[center_row + dr, center_col + dc] = streamflow
                qpe_grid[center_row + dr, center_col + dc] = qpe
        
        # Mock FFW mask
        ffw_mask = np.zeros((nj, ni), dtype=np.uint8)
        if ffw:
            ffw_mask[center_row-1:center_row+2, center_col-1:center_col+2] = 1
        
        # Mock home estimate (would need to inject into exposure system)
        # For now, test the scoring formula directly
        calculated_score = classifier.calculate_event_score(streamflow, ffw, qpe, homes)
        
        assert calculated_score == expected_score, \
            f"Score mismatch for {description}: expected {expected_score}, got {calculated_score}"
        
        print(f"✓ {description}: score={calculated_score}")
    
    def test_score_clamped_to_100(self, standard_config):
        """Test 7: Scores are clamped to 100 maximum"""
        
        classifier = FloodClassifier(standard_config)
        
        # Test extreme values that would exceed 100
        extreme_cases = [
            (25.0, 80.0, True, 1000),  # Very high streamflow + all bonuses
            (15.0, 100.0, True, 2000), # Max streamflow + extreme bonuses
        ]
        
        for streamflow, qpe, ffw, homes in extreme_cases:
            score = classifier.calculate_event_score(streamflow, ffw, qpe, homes)
            assert score <= 100, f"Score {score} exceeds 100 for inputs: flow={streamflow}, qpe={qpe}, ffw={ffw}, homes={homes}"
            
            # Calculate what unclamped score would be
            streamflow_pts = min(int(streamflow / 15.0 * 50), 50)
            ffw_pts = 25 if ffw else 0
            qpe_pts = 15 if qpe >= 50.0 else 0
            exposure_pts = 10 if homes >= 500 else 0
            unclamped = streamflow_pts + ffw_pts + qpe_pts + exposure_pts
            
            if unclamped > 100:
                assert score == 100, f"Score should be clamped to 100, got {score}"
            
            print(f"✓ Clamping test: unclamped={unclamped} → clamped={score}")
    
    def test_no_false_ffw_bonus(self, standard_config, conus_grid_shape):
        """Test 8: FFW bonus only applied when in FFW area"""
        
        classifier = FloodClassifier(standard_config)
        
        # Test event scoring with and without FFW
        streamflow = 12.0
        qpe = 55.0
        homes = 600
        
        score_with_ffw = classifier.calculate_event_score(streamflow, True, qpe, homes)
        score_without_ffw = classifier.calculate_event_score(streamflow, False, qpe, homes)
        
        expected_difference = 25  # FFW bonus
        actual_difference = score_with_ffw - score_without_ffw
        
        assert actual_difference == expected_difference, \
            f"FFW bonus should be exactly 25 points, got {actual_difference}"
        
        print(f"✓ FFW bonus: with={score_with_ffw}, without={score_without_ffw}, diff={actual_difference}")
    
    def test_exposure_cut(self, standard_config):
        """Test 9: Exposure bonus only for homes >= 500"""
        
        classifier = FloodClassifier(standard_config)
        
        streamflow = 12.0
        qpe = 55.0
        ffw = False
        
        score_high_exposure = classifier.calculate_event_score(streamflow, ffw, qpe, 600)  # >= 500
        score_low_exposure = classifier.calculate_event_score(streamflow, ffw, qpe, 400)   # < 500
        
        expected_difference = 10  # Exposure bonus
        actual_difference = score_high_exposure - score_low_exposure
        
        assert actual_difference == expected_difference, \
            f"Exposure bonus should be exactly 10 points, got {actual_difference}"
        
        print(f"✓ Exposure bonus: high={score_high_exposure}, low={score_low_exposure}, diff={actual_difference}")
        
    def test_ffw_qpe_bonus_pipeline(self, standard_config, conus_grid_shape):
        """GAP G-2: End-to-end FFW/QPE bonus through full pipeline"""
        
        classifier = FloodClassifier(standard_config)
        
        nj, ni = conus_grid_shape
        
        # Set deterministic seed
        np.random.seed(47)
        
        # Create streamflow grid with 3x3 high-tier event
        unit_streamflow = make_synthetic_grid(conus_grid_shape, {
            'background': 0.05,  # Higher to avoid dead data fallback
            'land_mask_prob': 0.4
        })
        
        center_row, center_col = 1000, 2000
        base_flow = 8.0  # High tier (contributes ~27 points: 8/15*50 ≈ 27)
        
        # Create 3x3 event
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                unit_streamflow[center_row + dr, center_col + dc] = base_flow
        
        # Create QPE grid with high rainfall at same location
        qpe_grid = np.zeros((nj, ni), dtype=np.float32)
        qpe_value = 60.0  # ≥50mm → +15 points
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                qpe_grid[center_row + dr, center_col + dc] = qpe_value
        
        # Create FFW mask covering the same area
        ffw_polygons = None
        if GEOPANDAS_AVAILABLE:
            from shapely.geometry import box
            import geopandas as gpd
            
            # Convert grid coordinates to lat/lon for FFW polygon
            center_lon, center_lat = classifier.grid_to_lonlat(center_row, center_col)
            
            # Create polygon covering the event area
            buffer = 0.05  # degrees
            ffw_polygon = box(center_lon - buffer, center_lat - buffer, 
                            center_lon + buffer, center_lat + buffer)
            
            ffw_polygons = gpd.GeoDataFrame(
                {'id': ['TEST_FFW']}, 
                geometry=[ffw_polygon], 
                crs='EPSG:4326'
            )
        
        # Run full classification pipeline with QPE
        result = classifier.classify(unit_streamflow, datetime.now(), 
                                    ffw_polygons=ffw_polygons, 
                                    qpe_1h_grid=qpe_grid)
        
        assert result is not None, "Classification failed in pipeline test"
        assert result.high_count > 0, "Expected high-tier events"
        
        # Extract events with QPE data
        flood_events = classifier.extract_flood_events(result, unit_streamflow, qpe_grid)
        
        # Find our test event
        all_events = []
        for severity, events in flood_events.items():
            all_events.extend(events)
        
        assert len(all_events) > 0, "No events extracted in pipeline test"
        
        # Find event closest to our test location
        test_event = None
        min_distance = float('inf')
        
        for event in all_events:
            event_row, event_col = event['centroid_grid']
            distance = np.sqrt((event_row - center_row)**2 + (event_col - center_col)**2)
            if distance < min_distance:
                min_distance = distance
                test_event = event
        
        assert test_event is not None, "Could not find test event"
        assert min_distance < 5.0, f"Test event too far from expected location: {min_distance:.1f} pixels"
        
        # Verify scoring components were applied
        event_score = test_event['event_score']
        streamflow_pts = int(base_flow / 15.0 * 50)  # ~27 points
        expected_qpe_bonus = 15  # QPE ≥50mm
        expected_ffw_bonus = 25 if ffw_polygons is not None else 0  # FFW coverage
        
        min_expected_score = streamflow_pts + expected_qpe_bonus + expected_ffw_bonus
        
        assert event_score >= min_expected_score, \
            f"Event score {event_score} < expected minimum {min_expected_score} " \
            f"(streamflow={streamflow_pts} + qpe={expected_qpe_bonus} + ffw={expected_ffw_bonus})"
        
        # FIXED: Verify QPE data is properly captured
        assert 'qpe_1h' in test_event, "QPE field missing from event"
        actual_qpe = test_event['qpe_1h']
        assert actual_qpe >= 50.0, f"QPE value {actual_qpe} should be ≥50.0 for bonus"
        
        # FIXED: Verify FFW data is properly captured
        assert 'ffw_confirmed' in test_event, "FFW field missing from event"
        if ffw_polygons is not None:
            assert test_event['ffw_confirmed'] == True, "FFW confirmation should be True"
        
        print(f"✓ End-to-end pipeline: score={event_score}, streamflow={streamflow_pts}pts, "
            f"qpe={expected_qpe_bonus}pts, ffw={expected_ffw_bonus}pts")
        print(f"  QPE value captured: {actual_qpe:.1f}mm/h")
        print(f"  FFW confirmed: {test_event['ffw_confirmed']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])