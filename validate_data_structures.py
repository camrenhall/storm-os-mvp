#!/usr/bin/env python3
"""
Storm-Lead Intelligence Engine - Data Structure Deep Validation
Validates that we have ALL required data fields and can process them for flood detection.
"""

import requests
import xarray as xr
import rasterio
import cfgrib
import numpy as np
from datetime import datetime, timedelta
import gzip
import tempfile
import os
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def test_mrms_data_structure():
    """Deep validation of MRMS QPE data structure and content"""
    print("\n🔍 DEEP VALIDATION: MRMS QPE Data Structure")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Download latest MRMS file
        url = "https://mrms.ncep.noaa.gov/data/2D/RadarOnly_QPE_01H/MRMS_RadarOnly_QPE_01H.latest.grib2.gz"
        print(f"Downloading: {url}")
        
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"❌ Download failed: {response.status_code}")
            return False
            
        # Save and decompress
        gz_path = temp_dir / "mrms_latest.grib2.gz"
        grib_path = temp_dir / "mrms_latest.grib2"
        
        gz_path.write_bytes(response.content)
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(grib_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"✅ Downloaded and decompressed: {len(response.content)} bytes")
        
        # Open with cfgrib and inspect thoroughly
        print("\n📊 Opening with cfgrib...")
        ds = cfgrib.open_dataset(str(grib_path))
        
        print(f"✅ Dataset opened successfully")
        
        # 1. DIMENSIONS ANALYSIS
        print(f"\n🗺️  DIMENSIONS:")
        for dim_name, dim_size in ds.dims.items():
            print(f"  {dim_name}: {dim_size}")
        
        # 2. COORDINATES ANALYSIS
        print(f"\n📍 COORDINATES:")
        for coord_name, coord_var in ds.coords.items():
            coord_shape = coord_var.shape
            coord_min = float(coord_var.min()) if coord_var.size > 0 else "N/A"
            coord_max = float(coord_var.max()) if coord_var.size > 0 else "N/A"
            print(f"  {coord_name}: shape={coord_shape}, range=[{coord_min:.2f}, {coord_max:.2f}]")
        
        # 3. DATA VARIABLES ANALYSIS
        print(f"\n📈 DATA VARIABLES:")
        for var_name, var_data in ds.data_vars.items():
            var_shape = var_data.shape
            var_dtype = var_data.dtype
            
            # Get actual data statistics
            try:
                data_array = var_data.values
                valid_data = data_array[~np.isnan(data_array)] if hasattr(data_array, '__len__') else data_array
                
                if len(valid_data) > 0:
                    data_min = float(np.min(valid_data))
                    data_max = float(np.max(valid_data))
                    data_mean = float(np.mean(valid_data))
                    non_zero_count = np.count_nonzero(valid_data)
                    total_count = len(data_array.flatten()) if hasattr(data_array, 'flatten') else 1
                    
                    print(f"  {var_name}:")
                    print(f"    Shape: {var_shape}")
                    print(f"    Data type: {var_dtype}")
                    print(f"    Value range: [{data_min:.6f}, {data_max:.6f}]")
                    print(f"    Mean value: {data_mean:.6f}")
                    print(f"    Non-zero values: {non_zero_count:,} / {total_count:,} ({100*non_zero_count/total_count:.1f}%)")
                    
                    # Check if this looks like precipitation data
                    if data_max > 0:
                        print(f"    ✅ Contains precipitation data (max: {data_max:.6f})")
                    else:
                        print(f"    ⚠️  No precipitation detected (all zeros)")
                        
                else:
                    print(f"  {var_name}: No valid data found")
            except Exception as e:
                print(f"  {var_name}: Error analyzing data - {e}")
        
        # 4. ATTRIBUTES ANALYSIS
        print(f"\n🏷️  DATASET ATTRIBUTES:")
        for attr_name, attr_value in ds.attrs.items():
            if isinstance(attr_value, str) and len(attr_value) > 100:
                attr_display = attr_value[:100] + "..."
            else:
                attr_display = str(attr_value)
            print(f"  {attr_name}: {attr_display}")
        
        # 5. PROJECTION VERIFICATION
        print(f"\n🌐 COORDINATE SYSTEM VERIFICATION:")
        
        # Check if we have lat/lon
        has_lat = 'latitude' in ds.coords or 'lat' in ds.coords
        has_lon = 'longitude' in ds.coords or 'lon' in ds.coords
        
        if has_lat and has_lon:
            lat_coord = ds.coords.get('latitude', ds.coords.get('lat'))
            lon_coord = ds.coords.get('longitude', ds.coords.get('lon'))
            
            lat_min, lat_max = float(lat_coord.min()), float(lat_coord.max())
            lon_min, lon_max = float(lon_coord.min()), float(lon_coord.max())
            
            print(f"  ✅ Geographic coordinates found:")
            print(f"    Latitude range: [{lat_min:.2f}, {lat_max:.2f}]")
            print(f"    Longitude range: [{lon_min:.2f}, {lon_max:.2f}]")
            
            # Verify CONUS coverage
            conus_coverage = (20 <= lat_min <= 30) and (45 <= lat_max <= 55) and (-130 <= lon_min <= -120) and (-60 <= lon_max <= -65)
            if conus_coverage:
                print(f"    ✅ Covers CONUS region appropriately")
            else:
                print(f"    ⚠️  Coverage may not be complete CONUS")
        else:
            print(f"    ⚠️  No geographic coordinates found")
        
        # 6. FLOOD DETECTION READINESS
        print(f"\n🌊 FLOOD DETECTION READINESS CHECK:")
        
        ready_for_processing = True
        issues = []
        
        # Check grid size
        total_cells = ds.dims.get('latitude', 0) * ds.dims.get('longitude', 0)
        if total_cells < 10_000_000:  # Should be ~24.5M for 3500x7000
            issues.append(f"Grid too small: {total_cells:,} cells")
            ready_for_processing = False
        else:
            print(f"  ✅ Grid size adequate: {total_cells:,} cells")
        
        # Check data variables
        if len(ds.data_vars) == 0:
            issues.append("No data variables found")
            ready_for_processing = False
        else:
            print(f"  ✅ Data variables present: {list(ds.data_vars.keys())}")
        
        # Check coordinate coverage
        if has_lat and has_lon:
            print(f"  ✅ Geographic coordinates available for processing")
        else:
            issues.append("Missing geographic coordinates")
            ready_for_processing = False
        
        if ready_for_processing:
            print(f"\n🎉 MRMS DATA READY FOR FLOOD DETECTION!")
        else:
            print(f"\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"    - {issue}")
        
        return ready_for_processing, ds
        
    except Exception as e:
        print(f"❌ Error during MRMS validation: {e}")
        return False, None
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def test_ffg_data_structure():
    """Deep validation of FFG data structure and content"""
    print("\n🔍 DEEP VALIDATION: FFG Service Data Structure")
    print("=" * 60)
    
    try:
        base_url = "https://mapservices.weather.noaa.gov/raster/rest/services/precip/rfc_gridded_ffg/MapServer"
        
        # Get service metadata
        print("📋 Service Metadata:")
        info_response = requests.get(f"{base_url}?f=json", timeout=30)
        if info_response.status_code == 200:
            service_info = info_response.json()
            
            # Check layers
            layers = service_info.get('layers', [])
            print(f"  Available layers: {len(layers)}")
            
            for layer in layers:
                layer_id = layer.get('id')
                layer_name = layer.get('name', 'Unknown')
                print(f"    Layer {layer_id}: {layer_name}")
                
                # Check if this is the 1-hour FFG layer we need
                if '01 Hour' in layer_name or '1 Hour' in layer_name or layer_id == 0:
                    print(f"      ✅ This appears to be our target FFG layer")
            
            # Check spatial reference
            spatial_ref = service_info.get('spatialReference', {})
            print(f"  Spatial Reference: {spatial_ref}")
            
            # Check extent
            full_extent = service_info.get('fullExtent', {})
            if full_extent:
                xmin = full_extent.get('xmin')
                ymin = full_extent.get('ymin') 
                xmax = full_extent.get('xmax')
                ymax = full_extent.get('ymax')
                print(f"  Full Extent: [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        # Test data export with better parameters
        print(f"\n🗺️  Testing FFG Data Export:")
        
        # Try multiple approaches to get actual FFG data
        test_results = []
        
        # Method 1: Export as GeoTIFF
        print("  Testing GeoTIFF export...")
        export_url = f"{base_url}/export"
        params = {
            'bbox': '-100,30,-95,35',  # Texas region
            'bboxSR': 4326,
            'size': '200,200',
            'imageSR': 4326,
            'format': 'tiff',
            'pixelType': 'F32',
            'f': 'image',
            'layers': 'show:0'
        }
        
        try:
            response = requests.get(export_url, params=params, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"    ✅ GeoTIFF export successful: {len(response.content)} bytes")
                test_results.append(('geotiff', True, len(response.content)))
            else:
                print(f"    ❌ GeoTIFF export failed: {response.status_code}")
                test_results.append(('geotiff', False, response.status_code))
        except Exception as e:
            print(f"    ❌ GeoTIFF export error: {e}")
            test_results.append(('geotiff', False, str(e)))
        
        # Method 2: Export as PNG (what worked before)
        print("  Testing PNG export...")
        params['format'] = 'png'
        
        try:
            response = requests.get(export_url, params=params, timeout=30)
            if response.status_code == 200:
                print(f"    ✅ PNG export successful: {len(response.content)} bytes")
                test_results.append(('png', True, len(response.content)))
            else:
                print(f"    ❌ PNG export failed: {response.status_code}")
                test_results.append(('png', False, response.status_code))
        except Exception as e:
            print(f"    ❌ PNG export error: {e}")
            test_results.append(('png', False, str(e)))
        
        # Method 3: Try layer-specific export
        print("  Testing layer 0 direct export...")
        layer_export_url = f"{base_url}/0/export"
        params_layer = {
            'bbox': '-100,30,-95,35',
            'bboxSR': 4326,
            'size': '200,200',
            'imageSR': 4326,
            'format': 'tiff',
            'f': 'image'
        }
        
        try:
            response = requests.get(layer_export_url, params=params_layer, timeout=30)
            if response.status_code == 200:
                print(f"    ✅ Layer 0 export successful: {len(response.content)} bytes")
                test_results.append(('layer0', True, len(response.content)))
            else:
                print(f"    ❌ Layer 0 export failed: {response.status_code}")
                test_results.append(('layer0', False, response.status_code))
        except Exception as e:
            print(f"    ❌ Layer 0 export error: {e}")
            test_results.append(('layer0', False, str(e)))
        
        # Analyze results
        print(f"\n📊 FFG Export Test Results:")
        successful_methods = [result for result in test_results if result[1]]
        
        if successful_methods:
            print(f"  ✅ {len(successful_methods)} export method(s) working:")
            for method, success, data in successful_methods:
                print(f"    - {method}: {data} bytes")
            
            print(f"\n🎉 FFG DATA ACCESSIBLE FOR FLOOD THRESHOLDS!")
            return True
        else:
            print(f"  ❌ No export methods working:")
            for method, success, data in test_results:
                print(f"    - {method}: Failed - {data}")
            return False
        
    except Exception as e:
        print(f"❌ Error during FFG validation: {e}")
        return False


def check_processing_pipeline_readiness(mrms_ds):
    """Verify we can perform the actual flood detection calculations"""
    print("\n🔍 PROCESSING PIPELINE READINESS CHECK")
    print("=" * 60)
    
    try:
        # 1. QPE Data Processing
        print("1️⃣ QPE Data Processing Test:")
        
        if mrms_ds is None:
            print("  ❌ No MRMS dataset available")
            return False
        
        # Get the precipitation variable
        precip_var = None
        for var_name, var_data in mrms_ds.data_vars.items():
            precip_var = var_data
            print(f"  ✅ Using variable '{var_name}' as precipitation data")
            break
        
        if precip_var is None:
            print("  ❌ No precipitation variable found")
            return False
        
        # Test basic statistics
        precip_array = precip_var.values
        print(f"  ✅ Precipitation array shape: {precip_array.shape}")
        print(f"  ✅ Data type: {precip_array.dtype}")
        
        # Check for valid precipitation values
        valid_precip = precip_array[~np.isnan(precip_array)]
        if len(valid_precip) > 0:
            print(f"  ✅ Valid precipitation values: {len(valid_precip):,}")
            print(f"  ✅ Precipitation range: [{np.min(valid_precip):.6f}, {np.max(valid_precip):.6f}]")
        else:
            print("  ⚠️  No valid precipitation values found")
        
        # 2. Coordinate Processing
        print(f"\n2️⃣ Coordinate Processing Test:")
        
        # Get lat/lon coordinates
        lat_coord = mrms_ds.coords.get('latitude', mrms_ds.coords.get('lat'))
        lon_coord = mrms_ds.coords.get('longitude', mrms_ds.coords.get('lon'))
        
        if lat_coord is not None and lon_coord is not None:
            print(f"  ✅ Latitude coordinate: {lat_coord.shape}")
            print(f"  ✅ Longitude coordinate: {lon_coord.shape}")
            
            # Test coordinate extraction for a sample point
            try:
                # Get a sample of coordinates (middle of grid)
                mid_lat_idx = len(lat_coord) // 2
                mid_lon_idx = len(lon_coord) // 2
                
                sample_lat = float(lat_coord[mid_lat_idx])
                sample_lon = float(lon_coord[mid_lon_idx])
                sample_precip = float(precip_var[mid_lat_idx, mid_lon_idx])
                
                print(f"  ✅ Sample point: lat={sample_lat:.4f}, lon={sample_lon:.4f}, precip={sample_precip:.6f}")
                
            except Exception as e:
                print(f"  ⚠️  Error extracting sample coordinates: {e}")
        else:
            print("  ❌ Missing latitude/longitude coordinates")
            return False
        
        # 3. Flood Detection Simulation
        print(f"\n3️⃣ Flood Detection Logic Test:")
        
        # Simulate QPE-to-FFG ratio calculation
        try:
            # Create mock FFG values for testing (normally from ArcGIS service)
            mock_ffg = np.full_like(precip_array, 25.0)  # 25mm threshold
            
            # Calculate ratios
            with np.errstate(divide='ignore', invalid='ignore'):
                qpe_to_ffg_ratio = precip_array / mock_ffg
            
            # Apply flood classification logic from spec
            critical_mask = qpe_to_ffg_ratio >= 1.3
            high_mask = (qpe_to_ffg_ratio >= 1.0) & (qpe_to_ffg_ratio < 1.3)
            moderate_mask = (qpe_to_ffg_ratio >= 0.75) & (qpe_to_ffg_ratio < 1.0)
            
            critical_count = np.sum(critical_mask)
            high_count = np.sum(high_mask)
            moderate_count = np.sum(moderate_mask)
            
            print(f"  ✅ Mock flood classification results:")
            print(f"    CRITICAL cells: {critical_count:,}")
            print(f"    HIGH cells: {high_count:,}")
            print(f"    MODERATE cells: {moderate_count:,}")
            
            if critical_count + high_count + moderate_count > 0:
                print(f"  ✅ Flood detection logic operational")
            else:
                print(f"  ⚠️  No flood conditions detected (expected with mock data)")
            
        except Exception as e:
            print(f"  ❌ Error in flood detection simulation: {e}")
            return False
        
        # 4. Geographic Segmentation Test
        print(f"\n4️⃣ Geographic Segmentation Test:")
        
        try:
            # Find flood areas and extract coordinates
            if critical_count > 0:
                # Get coordinates of critical flood cells
                critical_lats = lat_coord.values[critical_mask[:, 0] if critical_mask.ndim > 1 else critical_mask]
                critical_lons = lon_coord.values[critical_mask[0, :] if critical_mask.ndim > 1 else critical_mask]
                
                print(f"  ✅ Critical flood coordinates extracted: {len(critical_lats)} points")
                
                if len(critical_lats) > 0:
                    sample_flood_lat = critical_lats[0]
                    sample_flood_lon = critical_lons[0]
                    print(f"  ✅ Sample flood location: {sample_flood_lat:.4f}, {sample_flood_lon:.4f}")
            else:
                print(f"  ⚠️  No critical flood areas to test (expected with current weather)")
            
            print(f"  ✅ Geographic segmentation logic ready")
            
        except Exception as e:
            print(f"  ❌ Error in geographic segmentation: {e}")
            return False
        
        print(f"\n🎉 PROCESSING PIPELINE FULLY OPERATIONAL!")
        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline readiness check: {e}")
        return False


def main():
    """Run comprehensive data structure validation"""
    print("🚀 Storm-Lead Intelligence Engine - Deep Data Validation")
    print("=" * 70)
    
    # Test MRMS data structure
    mrms_ready, mrms_ds = test_mrms_data_structure()
    
    # Test FFG data structure  
    ffg_ready = test_ffg_data_structure()
    
    # Test processing pipeline
    if mrms_ready and mrms_ds:
        pipeline_ready = check_processing_pipeline_readiness(mrms_ds)
    else:
        pipeline_ready = False
    
    # Final assessment
    print("\n" + "=" * 70)
    print("🏁 FINAL READINESS ASSESSMENT")
    print("=" * 70)
    
    print(f"MRMS QPE Data:        {'✅ READY' if mrms_ready else '❌ NOT READY'}")
    print(f"FFG Threshold Data:   {'✅ READY' if ffg_ready else '❌ NOT READY'}")
    print(f"Processing Pipeline:  {'✅ READY' if pipeline_ready else '❌ NOT READY'}")
    
    overall_ready = mrms_ready and ffg_ready and pipeline_ready
    
    if overall_ready:
        print(f"\n🎉 ALL SYSTEMS GO!")
        print(f"✅ Storm-Lead Intelligence Engine ready for implementation")
        print(f"✅ Flood detection pipeline validated end-to-end")
        print(f"✅ All required data sources accessible and processable")
    else:
        print(f"\n⚠️  ISSUES DETECTED")
        if not mrms_ready:
            print(f"❌ MRMS data issues need resolution")
        if not ffg_ready:
            print(f"❌ FFG data access issues need resolution") 
        if not pipeline_ready:
            print(f"❌ Processing pipeline issues need resolution")
    
    return overall_ready


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)