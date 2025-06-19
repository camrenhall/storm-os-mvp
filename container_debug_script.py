#!/usr/bin/env python3
"""
Container-level debugging script for geopandas issues
Add this to your container startup to diagnose the problem
"""

import sys
import os
import subprocess

def debug_geopandas():
    print("=== GEOPANDAS DEBUG ANALYSIS ===")
    
    # Check if geopandas package is installed
    try:
        import pkg_resources
        geopandas_dist = pkg_resources.get_distribution('geopandas')
        print(f"✅ GeoPandas package installed: {geopandas_dist.version}")
    except pkg_resources.DistributionNotFound:
        print("❌ GeoPandas package not found in pip list")
        return
    
    # Check individual dependency imports
    dependencies = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('shapely', 'shapely'),
        ('fiona', 'fiona'),
        ('pyproj', 'pyproj'),
        ('rasterio', 'rasterio')
    ]
    
    print("\n--- Dependency Check ---")
    failed_deps = []
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed_deps.append(name)
    
    # Try to import geopandas with detailed error
    print("\n--- GeoPandas Import Test ---")
    try:
        import geopandas as gpd
        print(f"✅ GeoPandas import successful: {gpd.__version__}")
        
        # Test basic functionality
        from shapely.geometry import Point
        import pandas as pd
        
        test_gdf = gpd.GeoDataFrame({'col1': [1, 2]}, 
                                   geometry=[Point(1, 1), Point(2, 2)])
        print(f"✅ GeoPandas basic functionality: OK")
        
    except ImportError as e:
        print(f"❌ GeoPandas import failed: {e}")
        
        # Get more detailed error
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    
    # Check system libraries
    print("\n--- System Libraries Check ---")
    libs_to_check = [
        'libgdal.so',
        'libproj.so', 
        'libgeos_c.so'
    ]
    
    for lib in libs_to_check:
        try:
            result = subprocess.run(['find', '/usr', '-name', lib, '-type', 'f'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print(f"✅ {lib}: Found at {result.stdout.strip().split()[0]}")
            else:
                print(f"❌ {lib}: Not found")
        except Exception as e:
            print(f"❌ {lib}: Check failed - {e}")
    
    # Check GDAL environment
    print("\n--- GDAL Environment ---")
    gdal_vars = ['GDAL_DATA', 'PROJ_LIB', 'GDAL_CONFIG']
    for var in gdal_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Try alternative imports
    print("\n--- Alternative Import Paths ---")
    try:
        import fiona
        print(f"✅ Fiona works: {fiona.__version__}")
    except Exception as e:
        print(f"❌ Fiona failed: {e}")
        
    try:
        import rasterio
        print(f"✅ Rasterio works: {rasterio.__version__}")
    except Exception as e:
        print(f"❌ Rasterio failed: {e}")
    
    print("\n=== DEBUG COMPLETE ===")
    
    if failed_deps:
        print(f"\n🔧 RECOMMENDATION: Fix these dependencies: {', '.join(failed_deps)}")
    
    return len(failed_deps) == 0

if __name__ == "__main__":
    debug_geopandas()