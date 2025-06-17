#!/usr/bin/env python3
"""
G1 - Fixtures & decoding (real-data smoke tests)
Tests MRMS data ingestion and FFW/exposure integration
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
import asyncio
from pathlib import Path

from conftest import NetworkMixin, clock_and_peak_mem

try:
    from flash_ingest import FlashIngest
    from ffw_client import NWSFlashFloodWarnings
    from exposure import homes, get_population_stats
    import geopandas as gpd
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class TestFixturesDecoding(NetworkMixin):
    """Real-data smoke tests for ingestion components"""
    
    @pytest.mark.slow
    def test_download_unitstreamflow_latest(self):
        """Test 1: Download and decode latest UnitStreamflow GRIB2"""
        self.skip_if_no_network()
        
        async def fetch_test():
            async with FlashIngest() as ingest:
                result = await ingest.fetch_latest()
                return result
        
        result = asyncio.run(fetch_test())
        
        assert result is not None, "Failed to fetch UnitStreamflow data"
        assert 'unit_streamflow' in result, "Missing unit_streamflow array"
        
        unit_streamflow = result['unit_streamflow']
        assert unit_streamflow.shape == (3500, 7000), f"Wrong shape: {unit_streamflow.shape}"
        assert unit_streamflow.dtype in [np.float32, np.float64], f"Wrong dtype: {unit_streamflow.dtype}"
        
        # Check for reasonable data
        valid_mask = (unit_streamflow != -9999.0) & np.isfinite(unit_streamflow)
        valid_data = unit_streamflow[valid_mask]
        
        assert len(valid_data) > 100000, f"Too few valid pixels: {len(valid_data)}"
        assert valid_data.max() > 0.5, f"Suspiciously low max value: {valid_data.max()}"
        
        print(f"✓ UnitStreamflow: {unit_streamflow.shape}, valid={len(valid_data):,}, range=[{valid_data.min():.3f}, {valid_data.max():.3f}]")
    
    @pytest.mark.slow  
    def test_download_qpe_latest(self):
        """Test 2: Download and decode latest QPE GRIB2"""
        self.skip_if_no_network()
        
        async def fetch_qpe():
            # Import the new QPE ingest module
            from qpe_ingest import QPEIngest
            
            async with QPEIngest() as ingest:
                result = await ingest.fetch_latest()
                return result
        
        result = asyncio.run(fetch_qpe())
        
        assert result is not None, "Failed to fetch QPE data"
        assert 'qpe_1h' in result, "Missing qpe_1h array"
        
        qpe_1h = result['qpe_1h']
        assert qpe_1h.shape == (3500, 7000), f"Wrong QPE shape: {qpe_1h.shape}"
        assert qpe_1h.dtype == np.float32, f"Wrong QPE dtype: {qpe_1h.dtype}"
        
        # Check for reasonable data
        valid_mask = (qpe_1h >= 0) & np.isfinite(qpe_1h)
        valid_data = qpe_1h[valid_mask]
        
        assert len(valid_data) > 100000, f"Too few valid QPE pixels: {len(valid_data)}"
        
        # Check that at least 1% of pixels have ≥1mm rain
        rain_pixels = (valid_data >= 1.0).sum()
        rain_percentage = (rain_pixels / len(valid_data)) * 100
        
        assert rain_percentage >= 1.0, f"QPE rain coverage too low: {rain_percentage:.1f}% (expected ≥1%)"
        
        print(f"✓ QPE: {qpe_1h.shape}, valid={len(valid_data):,}, rain_coverage={rain_percentage:.1f}%, range=[{valid_data.min():.1f}, {valid_data.max():.1f}]mm")
    
    @pytest.mark.slow
    def test_ffw_rasterise(self, standard_config):
        """Test 3: Fetch FFW and rasterize to grid"""
        self.skip_if_no_network()
        
        async def fetch_ffw():
            async with NWSFlashFloodWarnings() as ffw_client:
                warnings_gdf = await ffw_client.get_active_warnings()
                return warnings_gdf
        
        warnings_gdf = asyncio.run(fetch_ffw())
        
        if warnings_gdf is None or len(warnings_gdf) == 0:
            pytest.skip("No active FFW warnings available for testing")
        
        assert isinstance(warnings_gdf, gpd.GeoDataFrame), "FFW should return GeoDataFrame"
        assert len(warnings_gdf) >= 1, "Should have at least one warning"
        assert 'geometry' in warnings_gdf.columns, "Missing geometry column"
        
        # Test rasterization
        from flood_classifier import FloodClassifier
        from conftest import standard_config
        
        classifier = FloodClassifier(standard_config)
        ffw_mask = classifier.rasterize_ffw_polygons(warnings_gdf)
        
        assert ffw_mask.shape == (3500, 7000), f"Wrong mask shape: {ffw_mask.shape}"
        assert ffw_mask.dtype == np.uint8, f"Wrong mask dtype: {ffw_mask.dtype}"
        assert ffw_mask.sum() >= 1, "FFW mask should have at least 1 pixel set"
        
        print(f"✓ FFW: {len(warnings_gdf)} warnings, {ffw_mask.sum():,} pixels rasterized")
    
    def test_population_grid_integrity(self, exposure_grid):
        """Test 4: Population grid loading and integrity checks"""
        
        # Test basic functionality
        test_homes = homes(1000, 2000)
        assert isinstance(test_homes, int), f"homes() should return int, got {type(test_homes)}"
        assert test_homes >= 0, f"Home count should be non-negative: {test_homes}"
        
        # Test known desert/water pixel (approximate)
        desert_homes = homes(500, 1000)  # Somewhere in Nevada/Utah desert
        water_homes = homes(100, 100)    # Somewhere in ocean
        
        # Desert/water should have low home counts
        assert desert_homes < 50, f"Desert pixel should have few homes: {desert_homes}"
        assert water_homes == 0, f"Water pixel should have zero homes: {water_homes}"
        
        # Test populated area
        urban_homes = homes(1500, 2500)  # Somewhere in populated area
        
        # Test grid statistics
        stats = get_population_stats()
        total_homes = stats['total_homes']
        
        # US has approximately 140M housing units
        expected_min = 138_000_000  # 140M - 2%
        expected_max = 142_000_000  # 140M + 2% 
        
        assert expected_min <= total_homes <= expected_max, \
            f"Total homes {total_homes:,} outside expected range [{expected_min:,}, {expected_max:,}]"
        
        print(f"✓ Population grid: {total_homes:,} total homes, test lookups work")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])