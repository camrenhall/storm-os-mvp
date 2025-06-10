#!/usr/bin/env python3
"""
Storm-Lead Intelligence Engine - API Validation Script
Validates all required data sources before main implementation.

Based on research:
- MRMS data: https://mrms.ncep.noaa.gov/data/2D/ (from search results)
- FFG service: https://mapservices.weather.noaa.gov/raster/rest/services/precip/rfc_gridded_ffg/MapServer
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

class APIValidator:
    def __init__(self):
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Using temp directory: {self.temp_dir}")
        
    def log_result(self, test_name: str, success: bool, details: str, data: Any = None):
        """Log test results"""
        self.results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        
    def validate_mrms_qpe(self) -> bool:
        """Validate NOAA MRMS QPE RadarOnly_QPE_01H data source"""
        print("\n🌧️  Testing MRMS QPE RadarOnly_QPE_01H...")
        
        try:
            # Use the .latest.grib2.gz file (most reliable method)
            base_url = "https://mrms.ncep.noaa.gov/data/2D/RadarOnly_QPE_01H"
            latest_url = f"{base_url}/MRMS_RadarOnly_QPE_01H.latest.grib2.gz"
            
            print(f"Fetching latest file: {latest_url}")
            
            response = requests.get(latest_url, timeout=30)
            if response.status_code != 200:
                self.log_result("MRMS_QPE", False, f"Latest file download failed: {response.status_code}")
                return False

            # Save & decompress
            gz_path = self.temp_dir / "MRMS_RadarOnly_QPE_01H.latest.grib2.gz"
            grib_path = self.temp_dir / "MRMS_RadarOnly_QPE_01H.latest.grib2"
            
            gz_path.write_bytes(response.content)
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(grib_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            # Open with cfgrib
            ds = cfgrib.open_dataset(str(grib_path))

            # Validate structure
            dims = dict(ds.dims)
            y_dim = dims.get('latitude', dims.get('y', 0))
            x_dim = dims.get('longitude', dims.get('x', 0))
            data_vars = list(ds.data_vars.keys())
            
            # Check if dimensions are correct for CONUS coverage
            # MRMS CONUS grid is approximately 3500×7000
            dims_ok = (3000 <= y_dim <= 4000) and (6000 <= x_dim <= 8000)
            
            if dims_ok and data_vars:
                file_size = len(response.content)
                details = f"Successfully downloaded MRMS file: {y_dim}×{x_dim}, vars: {data_vars}"
                
                self.log_result("MRMS_QPE", True, details, {
                    'filename': 'MRMS_RadarOnly_QPE_01H.latest.grib2.gz',
                    'dimensions': dims,
                    'data_variables': data_vars,
                    'file_size': file_size,
                    'url': latest_url,
                    'grid_size': f"{y_dim}x{x_dim}"
                })
                return True
            else:
                self.log_result("MRMS_QPE", False, f"Invalid dimensions: expected ~3500×7000, got {y_dim}×{x_dim}")
                return False
            
        except Exception as e:
            self.log_result("MRMS_QPE", False, f"Unexpected error: {str(e)}")
            return False
    
    def validate_ffg_data(self) -> bool:
        """Validate Flash Flood Guidance ArcGIS service"""
        print("\n🌊 Testing FFG ArcGIS service...")
        
        try:
            # Based on research, this is the correct service URL
            base_url = "https://mapservices.weather.noaa.gov/raster/rest/services/precip/rfc_gridded_ffg/MapServer"
            
            # Test 1: Get service info
            print(f"Testing service info: {base_url}")
            info_response = requests.get(f"{base_url}?f=json", timeout=30)
            print(f"Service info response: {info_response.status_code}")
            
            if info_response.status_code == 200:
                service_info = info_response.json()
                print(f"Service info keys: {list(service_info.keys())}")
                if 'layers' in service_info:
                    print(f"Available layers: {len(service_info['layers'])}")
                    for layer in service_info['layers'][:3]:  # Show first 3
                        print(f"  Layer {layer.get('id')}: {layer.get('name')}")
            
            # Test 2: Try to identify the correct layer for 1-hour FFG
            # From research, we need to find the layer for 1-hour guidance
            layers_to_try = [0, 1, 2, 3, 4, 5]  # Try first few layers
            
            for layer_id in layers_to_try:
                print(f"\nTesting layer {layer_id} export...")
                
                export_url = f"{base_url}/{layer_id}/export"
                
                # Small test area in Texas (known flood region)
                params = {
                    'bbox': '-98,30,-97,31',  # Austin area
                    'bboxSR': 4326,
                    'size': '100,100',
                    'imageSR': 4326,
                    'format': 'png',
                    'f': 'image'
                }
                
                try:
                    export_response = requests.get(export_url, params=params, timeout=30)
                    print(f"Layer {layer_id} export: {export_response.status_code}")
                    
                    if export_response.status_code == 200:
                        # Check if it's actually image data
                        content_type = export_response.headers.get('content-type', '')
                        file_size = len(export_response.content)
                        
                        print(f"Content-Type: {content_type}, Size: {file_size} bytes")
                        
                        if file_size > 100 and 'image' in content_type:  # Reasonable image
                            details = f"Layer {layer_id} export successful: {file_size} bytes, {content_type}"
                            
                            self.log_result("FFG_DATA", True, details, {
                                'layer_id': layer_id,
                                'export_url': export_url,
                                'content_type': content_type,
                                'file_size': file_size,
                                'test_params': params
                            })
                            return True
                except Exception as e:
                    print(f"Layer {layer_id} failed: {e}")
                    continue
            
            # Test 3: Try alternative export approaches
            print("\nTrying alternative export methods...")
            
            # Try without layer specification
            export_url = f"{base_url}/export"
            params = {
                'bbox': '-98,30,-97,31',
                'bboxSR': 4326,
                'size': '100,100',
                'imageSR': 4326,
                'format': 'png',
                'f': 'image',
                'layers': 'show:0'  # Explicitly show layer 0
            }
            
            try:
                export_response = requests.get(export_url, params=params, timeout=30)
                print(f"Alternative export: {export_response.status_code}")
                
                if export_response.status_code == 200:
                    file_size = len(export_response.content)
                    if file_size > 100:
                        details = f"Alternative export successful: {file_size} bytes"
                        self.log_result("FFG_DATA", True, details, {
                            'method': 'alternative_export',
                            'file_size': file_size
                        })
                        return True
            except Exception as e:
                print(f"Alternative export failed: {e}")
            
            self.log_result("FFG_DATA", False, "No successful export method found", {
                'service_accessible': info_response.status_code == 200 if 'info_response' in locals() else False,
                'layers_tested': layers_to_try
            })
            return False
            
        except Exception as e:
            self.log_result("FFG_DATA", False, f"Unexpected error: {str(e)}")
            return False
    
    def validate_nws_alerts(self) -> bool:
        """Validate NWS Alerts API"""
        print("\n⚠️  Testing NWS Alerts API...")
        
        try:
            url = "https://api.weather.gov/alerts/active"
            params = {
                'event': 'Flash Flood Warning'
            }
            
            headers = {
                'User-Agent': 'StormLeadEngine/1.0 (test@example.com)'  # NWS requires User-Agent
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])
                
                # API is working correctly even with 0 alerts
                details = f"API reachable, {len(features)} active Flash Flood Warnings"
                
                # Show API structure info
                api_info = {
                    'alert_count': len(features),
                    'api_working': True,
                    'response_size': len(response.content)
                }
                
                if features:
                    # Sample first alert structure
                    first_alert = features[0]
                    api_info.update({
                        'sample_properties': list(first_alert.get('properties', {}).keys()),
                        'has_geometry': 'geometry' in first_alert,
                        'geometry_type': first_alert.get('geometry', {}).get('type') if first_alert.get('geometry') else None
                    })
                
                self.log_result("NWS_ALERTS", True, details, api_info)
                return True
            else:
                self.log_result("NWS_ALERTS", False, f"API request failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("NWS_ALERTS", False, f"Error: {str(e)}")
            return False
    
    def validate_geocoding(self) -> bool:
        """Validate reverse geocoding (using a free service for testing)"""
        print("\n🗺️  Testing reverse geocoding...")
        
        try:
            # Test with a known location (Austin, TX city center)
            lat, lon = 30.2672, -97.7431
            
            # Using OpenStreetMap Nominatim for testing (free, no API key needed)
            url = f"https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json'
            }
            
            headers = {
                'User-Agent': 'StormLeadEngine/1.0'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                required_fields = ['display_name', 'address']
                has_required = all(field in data for field in required_fields)
                
                if has_required:
                    address = data.get('address', {})
                    address_components = list(address.keys())
                    
                    details = f"Address: {data.get('display_name', '')[:100]}..."
                    
                    self.log_result("GEOCODING", True, details, {
                        'display_name': data.get('display_name'),
                        'address_components': address_components,
                        'test_coordinates': (lat, lon),
                        'service': 'OpenStreetMap Nominatim'
                    })
                    return True
                else:
                    self.log_result("GEOCODING", False, f"Missing required fields: {required_fields}")
                    return False
            else:
                self.log_result("GEOCODING", False, f"Request failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("GEOCODING", False, f"Error: {str(e)}")
            return False
    
    def test_projection_alignment(self) -> bool:
        """Test reprojection between FFG (EPSG:4326) and MRMS grids"""
        print("\n🔄 Testing projection alignment...")
        
        try:
            # Sample point in Texas (known flood area)
            test_lat, test_lon = 30.0, -97.0
            
            print(f"Test point: {test_lat}, {test_lon}")
            
            from pyproj import Transformer
            
            # MRMS uses polar stereographic projection
            mrms_proj = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +datum=WGS84"
            
            # Transform from WGS84 to MRMS projection
            transformer_to_mrms = Transformer.from_crs("EPSG:4326", mrms_proj)
            x, y = transformer_to_mrms.transform(test_lat, test_lon)
            
            # Transform back to verify
            transformer_from_mrms = Transformer.from_crs(mrms_proj, "EPSG:4326")
            lat_back, lon_back = transformer_from_mrms.transform(x, y)
            
            # Check round-trip accuracy
            lat_diff = abs(test_lat - lat_back)
            lon_diff = abs(test_lon - lon_back)
            
            accuracy_ok = lat_diff < 0.001 and lon_diff < 0.001
            
            details = f"Round-trip error: lat={lat_diff:.6f}, lon={lon_diff:.6f}"
            
            if accuracy_ok:
                self.log_result("PROJECTION", True, details, {
                    'original': (test_lat, test_lon),
                    'mrms_coords': (x, y),
                    'back_transformed': (lat_back, lon_back),
                    'errors': (lat_diff, lon_diff),
                    'mrms_projection': mrms_proj
                })
                return True
            else:
                self.log_result("PROJECTION", False, f"Poor accuracy: {details}")
                return False
                
        except ImportError:
            self.log_result("PROJECTION", False, "pyproj not available for projection testing")
            return False
        except Exception as e:
            self.log_result("PROJECTION", False, f"Error: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate final validation report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'details': self.results,
            'ready_for_implementation': passed_tests >= 4,  # Core APIs working
            'recommendations': self._get_recommendations()
        }
        
        return report
    
    def _get_recommendations(self) -> Dict[str, str]:
        """Get recommendations based on test results"""
        recommendations = {}
        
        for test_name, result in self.results.items():
            if not result['success']:
                if test_name == "MRMS_QPE":
                    recommendations[test_name] = (
                        "MRMS data access may require authentication or the URL pattern has changed. "
                        "Consider contacting NCEP at idp-support@noaa.gov for access requirements."
                    )
                elif test_name == "FFG_DATA":
                    recommendations[test_name] = (
                        "ArcGIS FFG service may require specific layer identification. "
                        "Try examining layer metadata or contact NOAA GIS team."
                    )
                elif test_name == "NWS_ALERTS":
                    recommendations[test_name] = (
                        "NWS API may be temporarily unavailable. Check api.weather.gov status."
                    )
        
        return recommendations

def main():
    """Run all API validations"""
    print("🚀 Starting Storm-Lead Intelligence Engine API Validation")
    print("=" * 60)
    
    validator = APIValidator()
    
    try:
        # Run all validations with detailed logging
        validator.validate_mrms_qpe()
        validator.validate_ffg_data()
        validator.validate_nws_alerts()
        validator.validate_geocoding()
        validator.test_projection_alignment()
        
        # Generate report
        report = validator.generate_report()
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = report['summary']
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        if report['ready_for_implementation']:
            print("\n🎉 READY FOR IMPLEMENTATION!")
            print("Core APIs are validated and working correctly.")
        else:
            print("\n⚠️  NOT READY - Fix failing APIs before proceeding")
            print("\n📋 RECOMMENDATIONS:")
            for test, rec in report['recommendations'].items():
                print(f"\n{test}: {rec}")
        
        # Save detailed report
        report_path = Path("api_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return report['ready_for_implementation']
        
    finally:
        validator.cleanup()

if __name__ == "__main__":
    # Install required packages
    required_packages = [
        "requests", "xarray", "rasterio", "cfgrib", "numpy", 
        "pyproj"
    ]
    
    print("📦 Required packages for this script:")
    for pkg in required_packages:
        print(f"  pip install {pkg}")
    print()
    
    success = main()
    exit(0 if success else 1)