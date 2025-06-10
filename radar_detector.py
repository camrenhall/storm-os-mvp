import os
import sys
import requests
import numpy as np
import geopandas as gpd
from datetime import datetime, timezone
import tempfile
import json
from typing import Dict, List, Optional, Tuple
import logging
import gzip
import subprocess
from collections import Counter

# Fix ECCODES warnings at the root by disabling problematic features
os.environ['ECCODES_GRIB_WRITE_ON_FAIL'] = '0'
os.environ['ECCODES_GRIB_LARGE_CONSTANT_FIELDS'] = '1'
os.environ['ECCODES_DEBUG'] = '0'
os.environ['ECCODES_LOG_STREAM'] = 'stdout'

# Redirect eccodes stderr to null to suppress C-level warnings
try:
    # Create null device for suppressing C library output
    DEVNULL = open(os.devnull, 'w')
    # This won't catch all C-level output but helps with some
except:
    DEVNULL = None

import warnings
import xarray as xr

# Import cfgrib with proper warning suppression
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import cfgrib

from shapely.geometry import shape, Point, Polygon, MultiPoint
from shapely.ops import transform
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all cfgrib/xarray warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
warnings.filterwarnings("ignore", category=UserWarning, module="cfgrib")
warnings.filterwarnings("ignore", message=".*decode_timedelta.*")

class RadarDetector:
    """
    Enhanced NOAA MRMS radar intelligence for severe storm detection.
    Identifies severe storm areas with comprehensive logging and filtering.
    """
    
    def __init__(self):
        self.base_url = "https://mrms.ncep.noaa.gov/data/2D/ReflectivityAtLowestAltitude/"
        self.dbz_threshold = 45.0
        self.min_storm_area_km2 = 25.0
        self.max_storms_output = 50
        
        # Alert types for boosting
        self.guaranteed_events = [
            'Tornado Warning',
            'Flash Flood Warning', 
            'Severe Thunderstorm Warning'
        ]
        
        # Storm filtering criteria
        self.filtering_criteria = {
            'min_reflectivity_dbz': self.dbz_threshold,
            'min_area_km2': self.min_storm_area_km2,
            'min_pixels': 10,  # Minimum pixel count
            'max_intensity_score': 1.0
        }
    
    def get_latest_radar_data(self) -> Optional[str]:
        """Download the latest NOAA MRMS Base Reflectivity GRIB2 file."""
        try:
            latest_file_url = f"{self.base_url}MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2.gz')
            logger.info(f"📡 Downloading MRMS data: {latest_file_url}")
            
            response = requests.get(latest_file_url, timeout=120, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            temp_file.close()
            
            file_size = os.path.getsize(temp_file.name)
            if file_size < 100000:
                logger.error(f"❌ Downloaded file too small: {file_size:,} bytes")
                os.unlink(temp_file.name)
                return None
                
            logger.info(f"✅ Successfully downloaded MRMS data: {file_size:,} bytes")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"❌ Failed to download MRMS data: {e}")
            return None
    
    def process_radar_data(self, file_path: str) -> Optional[Tuple]:
        """Process GRIB2 radar data with proper error handling."""
        try:
            # Extract gzip file
            if file_path.endswith('.gz'):
                logger.info("📦 Extracting gzip compressed GRIB2 file")
                
                extracted_file = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
                with gzip.open(file_path, 'rb') as f_in:
                    with open(extracted_file.name, 'wb') as f_out:
                        f_out.write(f_in.read())
                extracted_file.close()
                
                grib_file_path = extracted_file.name
                os.unlink(file_path)
            else:
                grib_file_path = file_path
            
            # Read GRIB2 file with maximum warning suppression
            logger.info("🔍 Reading GRIB2 file (suppressing eccodes warnings)")
            
            # Temporarily redirect stderr to suppress eccodes C-level warnings
            original_stderr = sys.stderr
            if DEVNULL:
                sys.stderr = DEVNULL
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Try multiple approaches to open the file
                    datasets = None
                    try:
                        # Method 1: With backend kwargs
                        datasets = cfgrib.open_datasets(
                            grib_file_path,
                            backend_kwargs={
                                'decode_timedelta': False,
                                'errors': 'ignore',
                                'time_dims': ['time', 'step'],
                            }
                        )
                    except:
                        try:
                            # Method 2: Simple approach
                            datasets = cfgrib.open_datasets(grib_file_path)
                        except:
                            # Method 3: Using xarray directly
                            datasets = [xr.open_dataset(grib_file_path, engine='cfgrib')]
            finally:
                # Restore stderr
                sys.stderr = original_stderr
            
            if not datasets:
                logger.error("❌ No datasets found in GRIB2 file")
                return None
            
            # Find reflectivity data
            reflectivity_ds = None
            reflectivity_var = None
            
            for ds in datasets:
                var_names = list(ds.data_vars.keys())
                logger.debug(f"🔍 Dataset variables: {var_names}")
                
                # Enhanced variable detection
                for var in var_names:
                    try:
                        test_data = ds[var].values
                        
                        # Validate data characteristics
                        if hasattr(test_data, 'shape') and len(test_data.shape) >= 2:
                            # Squeeze to 2D
                            while test_data.ndim > 2:
                                test_data = test_data.squeeze()
                            
                            if test_data.ndim == 2:
                                test_min, test_max = np.nanmin(test_data), np.nanmax(test_data)
                                valid_range = -50 <= test_min <= 20 and 20 <= test_max <= 100
                                
                                if valid_range:
                                    reflectivity_ds = ds
                                    reflectivity_var = var
                                    logger.info(f"✅ Found reflectivity data: '{var}' (range: {test_min:.1f} to {test_max:.1f} dBZ)")
                                    break
                                else:
                                    logger.debug(f"⚠️  Variable '{var}' has invalid range: {test_min:.1f} to {test_max:.1f}")
                    except Exception as e:
                        logger.debug(f"⚠️  Could not validate variable '{var}': {e}")
                        continue
                
                if reflectivity_ds is not None:
                    break
            
            if reflectivity_ds is None:
                logger.warning("⚠️  No valid reflectivity variable found, using first available")
                reflectivity_ds = datasets[0]
                reflectivity_var = list(reflectivity_ds.data_vars.keys())[0]
            
            # Extract and process reflectivity data
            reflectivity_data = reflectivity_ds[reflectivity_var]
            reflectivity = reflectivity_data.values
            
            # Reduce to 2D
            original_shape = reflectivity.shape
            while reflectivity.ndim > 2:
                reflectivity = reflectivity.squeeze()
            
            if reflectivity.ndim != 2:
                logger.error(f"❌ Cannot reduce data to 2D: {original_shape} -> {reflectivity.shape}")
                return None
            
            # Clean data
            original_range = (np.nanmin(reflectivity), np.nanmax(reflectivity))
            reflectivity = np.where(reflectivity < -900, np.nan, reflectivity)
            reflectivity = np.where(reflectivity > 100, np.nan, reflectivity)
            cleaned_range = (np.nanmin(reflectivity), np.nanmax(reflectivity))
            
            logger.info(f"📊 Data shape: {reflectivity.shape}")
            logger.info(f"📊 Data range: {original_range[0]:.1f} to {original_range[1]:.1f} dBZ (raw)")
            logger.info(f"📊 Data range: {cleaned_range[0]:.1f} to {cleaned_range[1]:.1f} dBZ (cleaned)")
            
            # Build coordinate system
            try:
                if 'latitude' in reflectivity_ds.coords and 'longitude' in reflectivity_ds.coords:
                    lats = reflectivity_ds.latitude.values
                    lons = reflectivity_ds.longitude.values
                    logger.info("🗺️  Using explicit lat/lon coordinates")
                elif 'y' in reflectivity_ds.coords and 'x' in reflectivity_ds.coords:
                    y_coords = reflectivity_ds.y.values  
                    x_coords = reflectivity_ds.x.values
                    logger.info("🗺️  Using projected coordinates")
                    
                    # Create meshgrid
                    lons, lats = np.meshgrid(x_coords, y_coords)
                else:
                    raise ValueError("No coordinate information found")
                    
            except Exception as e:
                logger.warning(f"⚠️  Coordinate extraction failed: {e}")
                logger.info("🗺️  Generating approximate CONUS coordinates")
                
                rows, cols = reflectivity.shape
                lats_1d = np.linspace(54.99, 20.01, rows)  # North to South
                lons_1d = np.linspace(-129.99, -60.01, cols)  # West to East
                lons, lats = np.meshgrid(lons_1d, lats_1d)
            
            # Ensure coordinate arrays match data shape
            if lats.shape != reflectivity.shape:
                logger.warning(f"⚠️  Coordinate mismatch: data {reflectivity.shape}, coords {lats.shape}")
                if lats.ndim == 1 and lons.ndim == 1:
                    lons, lats = np.meshgrid(lons, lats)
                    logger.info("🔧 Created coordinate meshgrid")
                else:
                    rows, cols = reflectivity.shape
                    lats_1d = np.linspace(54.99, 20.01, rows)
                    lons_1d = np.linspace(-129.99, -60.01, cols)  
                    lons, lats = np.meshgrid(lons_1d, lats_1d)
                    logger.warning("🔧 Regenerated coordinates to match data")
            
            # Build transform info
            lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
            lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
            
            # Convert longitude from 0-360 to -180-180 if needed
            if lon_min > 180:
                lon_min -= 360
            if lon_max > 180:
                lon_max -= 360
            
            y_res = (lat_max - lat_min) / reflectivity.shape[0]
            x_res = (lon_max - lon_min) / reflectivity.shape[1]
            
            transform_info = {
                'x_min': lon_min,
                'y_max': lat_max,
                'x_res': x_res,
                'y_res': -abs(y_res),
                'shape': reflectivity.shape,
                'bounds': {
                    'north': lat_max,
                    'south': lat_min,
                    'east': lon_max,
                    'west': lon_min
                }
            }
            
            logger.info(f"🗺️  Geographic bounds: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}° to {lon_max:.2f}°")
            
            # Cleanup
            if grib_file_path != file_path:
                try:
                    os.unlink(grib_file_path)
                except:
                    pass
            
            return reflectivity, transform_info, 'EPSG:4326'
            
        except Exception as e:
            logger.error(f"❌ Failed to process GRIB2 data: {e}")
            return None
    
    def identify_storm_areas(self, reflectivity: np.ndarray, transform_info: Dict, crs_info: str) -> Tuple[List[Dict], Dict]:
        """Identify storm areas with detailed filtering statistics."""
        try:
            logger.info(f"🔍 Identifying storm areas (threshold: {self.dbz_threshold} dBZ)")
            
            # Create binary mask
            severe_mask = reflectivity >= self.dbz_threshold
            severe_pixels = np.sum(severe_mask)
            total_pixels = reflectivity.size
            valid_pixels = np.sum(~np.isnan(reflectivity))
            
            logger.info(f"📊 Pixel analysis:")
            logger.info(f"   • Total pixels: {total_pixels:,}")
            logger.info(f"   • Valid pixels: {valid_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")
            logger.info(f"   • Severe pixels: {severe_pixels:,} ({100*severe_pixels/valid_pixels:.1f}% of valid)")
            
            # Remove isolated pixels
            severe_mask = ndimage.binary_opening(severe_mask, structure=np.ones((3,3)))
            cleaned_pixels = np.sum(severe_mask)
            
            logger.info(f"   • After noise removal: {cleaned_pixels:,} pixels ({severe_pixels-cleaned_pixels:,} removed)")
            
            # Label connected components
            labeled_storms, num_storms = ndimage.label(severe_mask)
            logger.info(f"🌪️  Found {num_storms} potential storm areas")
            
            # Analyze each storm
            storms = []
            filtering_stats = {
                'total_detected': num_storms,
                'filtered_by_size': 0,
                'filtered_by_area': 0,
                'filtered_by_pixels': 0,
                'selected': 0,
                'criteria': self.filtering_criteria.copy()
            }
            
            for storm_label in range(1, num_storms + 1):
                storm_pixels = labeled_storms == storm_label
                pixel_count = np.sum(storm_pixels)
                
                # Filter by minimum pixels
                if pixel_count < self.filtering_criteria['min_pixels']:
                    filtering_stats['filtered_by_pixels'] += 1
                    continue
                
                # Calculate storm statistics
                storm_reflectivity = reflectivity[storm_pixels]
                max_dbz = float(np.nanmax(storm_reflectivity))
                mean_dbz = float(np.nanmean(storm_reflectivity))
                
                # Convert to geographic polygon
                try:
                    storm_rows, storm_cols = np.where(storm_pixels)
                    
                    if len(storm_rows) < 3:
                        filtering_stats['filtered_by_size'] += 1
                        continue
                    
                    # Convert to geographic coordinates
                    geo_points = []
                    for row, col in zip(storm_rows, storm_cols):
                        lon = transform_info['x_min'] + col * transform_info['x_res']
                        lat = transform_info['y_max'] + row * transform_info['y_res']
                        geo_points.append((lon, lat))
                    
                    # Create convex hull
                    if len(geo_points) >= 3:
                        multi_point = MultiPoint(geo_points)
                        storm_geom = multi_point.convex_hull
                    else:
                        filtering_stats['filtered_by_size'] += 1
                        continue
                    
                    # Calculate area
                    area_degrees_sq = storm_geom.area
                    area_km2 = area_degrees_sq * 111.32 * 111.32
                    
                    # Filter by minimum area
                    if area_km2 < self.filtering_criteria['min_area_km2']:
                        filtering_stats['filtered_by_area'] += 1
                        continue
                    
                    # Calculate intensity score
                    intensity_score = min((max_dbz / 70.0) * (area_km2 / 1000.0), 1.0)
                    
                    # Create storm record
                    centroid = storm_geom.centroid
                    storm = {
                        "id": f"storm_{len(storms)+1:03d}",
                        "geometry": storm_geom,
                        "max_reflectivity_dbz": max_dbz,
                        "mean_reflectivity_dbz": mean_dbz,
                        "area_sq_km": float(area_km2),
                        "pixel_count": int(pixel_count),
                        "storm_intensity_score": float(intensity_score),
                        "storm_center": [float(centroid.x), float(centroid.y)],
                        "detection_timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    storms.append(storm)
                    filtering_stats['selected'] += 1
                    
                except Exception as e:
                    logger.debug(f"Failed to process storm {storm_label}: {e}")
                    filtering_stats['filtered_by_size'] += 1
                    continue
            
            # Sort by intensity score
            storms.sort(key=lambda x: x["storm_intensity_score"], reverse=True)
            
            # Limit output
            if len(storms) > self.max_storms_output:
                original_count = len(storms)
                storms = storms[:self.max_storms_output]
                logger.info(f"📊 Limited output to top {self.max_storms_output} storms (from {original_count})")
            
            # Log filtering results
            logger.info(f"📊 Storm filtering results:")
            logger.info(f"   • Total detected: {filtering_stats['total_detected']}")
            logger.info(f"   • Filtered by pixels (<{self.filtering_criteria['min_pixels']}): {filtering_stats['filtered_by_pixels']}")
            logger.info(f"   • Filtered by size (<3 points): {filtering_stats['filtered_by_size']}")
            logger.info(f"   • Filtered by area (<{self.filtering_criteria['min_area_km2']} km²): {filtering_stats['filtered_by_area']}")
            logger.info(f"   • Selected for output: {filtering_stats['selected']}")
            
            if storms:
                intensity_scores = [s["storm_intensity_score"] for s in storms]
                logger.info(f"📊 Intensity score range: {min(intensity_scores):.3f} to {max(intensity_scores):.3f}")
            
            return storms, filtering_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to identify storm areas: {e}")
            return [], {}
    
    def get_nws_alerts(self) -> Tuple[Optional[gpd.GeoDataFrame], Dict]:
        """Fetch NWS alerts with detailed breakdown."""
        try:
            url = "https://api.weather.gov/alerts/active"
            headers = {
                'Accept': 'application/geo+json',
                'User-Agent': 'Storm-Lead-Intelligence-Engine/1.0'
            }
            
            logger.info("📡 Fetching active NWS alerts")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Count all alerts by type
            all_alerts = data.get('features', [])
            all_alert_counts = Counter(f['properties'].get('event', 'Unknown') for f in all_alerts)
            
            # Define comprehensive storm-related events
            # All of these will be correlated with radar-detected storms
            storm_events = [
                # Tornado alerts
                'Tornado Warning',
                'Tornado Watch',
                # Thunderstorm alerts  
                'Severe Thunderstorm Warning',
                'Severe Thunderstorm Watch',
                # Flood alerts (ALL types)
                'Flash Flood Warning',
                'Flash Flood Watch', 
                'Flood Warning',
                'Flood Advisory',
                'Flood Watch',
                'Urban and Small Stream Flood Advisory',
                'River Flood Warning',
                'River Flood Advisory',
                # Other precipitation-related
                'Heavy Rain',
                'Excessive Heat Warning'  # Can correlate with storm intensity
            ]
            
            relevant_features = [
                feature for feature in all_alerts
                if feature['properties'].get('event') in storm_events
            ]
            
            # Count relevant alerts by category
            relevant_counts = Counter(f['properties'].get('event', 'Unknown') for f in relevant_features)
            
            # Categorize alerts for better reporting
            flood_related = ['Flash Flood Warning', 'Flash Flood Watch', 'Flood Warning', 'Flood Advisory', 
                           'Flood Watch', 'Urban and Small Stream Flood Advisory', 'River Flood Warning', 'River Flood Advisory']
            tornado_related = ['Tornado Warning', 'Tornado Watch']
            thunderstorm_related = ['Severe Thunderstorm Warning', 'Severe Thunderstorm Watch']
            
            flood_count = sum(relevant_counts.get(alert, 0) for alert in flood_related)
            tornado_count = sum(relevant_counts.get(alert, 0) for alert in tornado_related)
            thunderstorm_count = sum(relevant_counts.get(alert, 0) for alert in thunderstorm_related)
            
            logger.info(f"📊 NWS Alert summary:")
            logger.info(f"   • Total active alerts: {len(all_alerts)}")
            logger.info(f"   • Storm-related alerts: {len(relevant_features)}")
            logger.info(f"     - Flood-related: {flood_count}")
            logger.info(f"     - Tornado-related: {tornado_count}")  
            logger.info(f"     - Thunderstorm-related: {thunderstorm_count}")
            
            if relevant_counts:
                logger.info("📊 Detailed storm-related alert breakdown:")
                # Group by category for clearer reporting
                for category, alerts in [
                    ("Flood Alerts", flood_related),
                    ("Tornado Alerts", tornado_related), 
                    ("Thunderstorm Alerts", thunderstorm_related)
                ]:
                    category_alerts = {alert: relevant_counts.get(alert, 0) for alert in alerts if relevant_counts.get(alert, 0) > 0}
                    if category_alerts:
                        logger.info(f"   {category}:")
                        for alert_type, count in category_alerts.items():
                            logger.info(f"     • {alert_type}: {count}")
            
            if relevant_features:
                gdf = gpd.GeoDataFrame.from_features(relevant_features)
                current_time = datetime.now(timezone.utc)
                
                # Filter expired alerts
                original_count = len(gdf)
                if 'expires' in gdf.columns:
                    gdf['expires_dt'] = gpd.pd.to_datetime(gdf['expires'], utc=True)
                    gdf = gdf[gdf['expires_dt'] > current_time]
                
                expired_count = original_count - len(gdf)
                if expired_count > 0:
                    logger.info(f"   • Expired alerts removed: {expired_count}")
                
                alert_stats = {
                    'total_alerts': len(all_alerts),
                    'storm_related': len(relevant_features),
                    'active_storm_alerts': len(gdf),
                    'expired_removed': expired_count,
                    'breakdown': dict(relevant_counts),
                    'by_category': {
                        'flood_related': flood_count,
                        'tornado_related': tornado_count,
                        'thunderstorm_related': thunderstorm_count
                    }
                }
                
                logger.info(f"📍 All {len(gdf)} active storm alerts will be spatially correlated with radar storms")
                
                return gdf, alert_stats
            
            return None, {
                'total_alerts': len(all_alerts),
                'storm_related': 0,
                'active_storm_alerts': 0,
                'expired_removed': 0,
                'breakdown': {},
                'by_category': {
                    'flood_related': 0,
                    'tornado_related': 0, 
                    'thunderstorm_related': 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch NWS alerts: {e}")
            return None, {}
    
    def enhance_storms_with_alerts(self, storms: List[Dict], alerts_gdf: Optional[gpd.GeoDataFrame]) -> Tuple[List[Dict], Dict]:
        """Enhance storms with alert information and provide detailed statistics."""
        
        enhancement_stats = {
            'storms_with_alerts': 0,
            'storms_boosted': 0,
            'alert_intersections': 0,
            'storms_by_alert_type': {},
            'boost_applications': {},
            'spatial_correlations': {
                'point_contained': 0,
                'geometry_intersects': 0,
                'both_methods': 0
            }
        }
        
        if alerts_gdf is None or len(alerts_gdf) == 0:
            logger.info("⚠️  No active alerts to correlate with radar storms")
            for storm in storms:
                self._add_default_alert_fields(storm)
            return storms, enhancement_stats
        
        try:
            logger.info(f"🔗 Spatially correlating {len(storms)} radar storms with {len(alerts_gdf)} NWS alerts")
            logger.info(f"🗺️  Correlation methods: point containment + geometry intersection")
            
            enhanced_storms = []
            
            for storm in storms:
                storm_geom = storm["geometry"]
                storm_point = Point(storm["storm_center"])
                intersecting_alerts = []
                
                # Check intersections with each alert
                for idx, alert_row in alerts_gdf.iterrows():
                    try:
                        alert_geom = alert_row.geometry
                        alert_type = alert_row.get('event', 'Unknown')
                        
                        if alert_geom is None or alert_geom.is_empty:
                            continue
                        
                        # Test both spatial correlation methods
                        point_contained = False
                        geom_intersects = False
                        
                        try:
                            point_contained = alert_geom.contains(storm_point)
                            if point_contained:
                                enhancement_stats['spatial_correlations']['point_contained'] += 1
                        except:
                            pass
                            
                        try:
                            geom_intersects = alert_geom.intersects(storm_geom)
                            if geom_intersects:
                                enhancement_stats['spatial_correlations']['geometry_intersects'] += 1
                        except:
                            pass
                        
                        if point_contained and geom_intersects:
                            enhancement_stats['spatial_correlations']['both_methods'] += 1
                        
                        if point_contained or geom_intersects:
                            enhancement_stats['alert_intersections'] += 1
                            
                            correlation_method = []
                            if point_contained:
                                correlation_method.append("center-contained")
                            if geom_intersects:
                                correlation_method.append("geometry-overlap")
                            
                            alert_info = {
                                "event": alert_type,
                                "urgency": alert_row.get('urgency', 'Unknown'),
                                "severity": alert_row.get('severity', 'Unknown'),
                                "certainty": alert_row.get('certainty', 'Unknown'),
                                "headline": alert_row.get('headline', ''),
                                "expires": alert_row.get('expires', ''),
                                "point_contained": point_contained,
                                "geometry_intersects": geom_intersects,
                                "correlation_method": correlation_method
                            }
                            intersecting_alerts.append(alert_info)
                            
                            # Count storms by alert type
                            if alert_type not in enhancement_stats['storms_by_alert_type']:
                                enhancement_stats['storms_by_alert_type'][alert_type] = set()
                            enhancement_stats['storms_by_alert_type'][alert_type].add(storm['id'])
                            
                    except Exception as e:
                        logger.debug(f"Error processing alert {idx}: {e}")
                        continue
                
                # Apply enhancements
                if intersecting_alerts:
                    enhancement_stats['storms_with_alerts'] += 1
                    
                    storm["nws_alerts_present"] = True
                    storm["alert_types"] = [alert["event"] for alert in intersecting_alerts]
                    storm["alert_details"] = intersecting_alerts
                    storm["contained_in_alert"] = any(alert["point_contained"] for alert in intersecting_alerts)
                    
                    # Apply boost (prioritize high-impact flood and severe weather alerts)
                    original_score = storm["storm_intensity_score"]
                    alert_boost = 1.5 if any(alert["event"] in self.guaranteed_events for alert in intersecting_alerts) else 1.2
                    storm["storm_intensity_score"] = min(original_score * alert_boost, 1.0)
                    storm["alert_boosted"] = True
                    storm["alert_boost_factor"] = alert_boost
                    
                    enhancement_stats['storms_boosted'] += 1
                    boost_type = 'guaranteed' if alert_boost == 1.5 else 'standard'
                    enhancement_stats['boost_applications'][boost_type] = enhancement_stats['boost_applications'].get(boost_type, 0) + 1
                    
                    # Log correlation details
                    alert_types_str = ', '.join(set(alert["event"] for alert in intersecting_alerts))
                    correlation_methods = set()
                    for alert in intersecting_alerts:
                        correlation_methods.update(alert["correlation_method"])
                    
                    logger.info(f"🚨 {storm['id']} ← correlated with {len(intersecting_alerts)} alerts: {alert_types_str}")
                    logger.info(f"   └── Spatial correlation: {', '.join(correlation_methods)} (boost: {alert_boost:.1f}x)")
                else:
                    self._add_default_alert_fields(storm)
                
                enhanced_storms.append(storm)
            
            # Re-sort by intensity score
            enhanced_storms.sort(key=lambda x: x["storm_intensity_score"], reverse=True)
            
            # Convert sets to counts for logging
            for alert_type in enhancement_stats['storms_by_alert_type']:
                enhancement_stats['storms_by_alert_type'][alert_type] = len(enhancement_stats['storms_by_alert_type'][alert_type])
            
            # Log comprehensive enhancement results
            logger.info(f"📊 Spatial correlation results:")
            logger.info(f"   • Total alert-storm intersections: {enhancement_stats['alert_intersections']}")
            logger.info(f"   • Point containment matches: {enhancement_stats['spatial_correlations']['point_contained']}")
            logger.info(f"   • Geometry intersection matches: {enhancement_stats['spatial_correlations']['geometry_intersects']}")
            logger.info(f"   • Both methods matched: {enhancement_stats['spatial_correlations']['both_methods']}")
            logger.info(f"   • Storms with alerts: {enhancement_stats['storms_with_alerts']}/{len(storms)}")
            logger.info(f"   • Storms boosted: {enhancement_stats['storms_boosted']}")
            
            if enhancement_stats['storms_by_alert_type']:
                logger.info("📊 Storms correlated by alert type:")
                # Group by category for clearer reporting
                flood_alerts = ['Flash Flood Warning', 'Flood Warning', 'Flood Advisory', 'Flash Flood Watch', 'Flood Watch']
                tornado_alerts = ['Tornado Warning', 'Tornado Watch']
                thunderstorm_alerts = ['Severe Thunderstorm Warning', 'Severe Thunderstorm Watch']
                
                for category, alert_list in [
                    ("Flood-related", flood_alerts),
                    ("Tornado-related", tornado_alerts),
                    ("Thunderstorm-related", thunderstorm_alerts)
                ]:
                    category_found = False
                    for alert_type in alert_list:
                        if alert_type in enhancement_stats['storms_by_alert_type']:
                            if not category_found:
                                logger.info(f"   {category}:")
                                category_found = True
                            count = enhancement_stats['storms_by_alert_type'][alert_type]
                            logger.info(f"     • {alert_type}: {count} storm{'s' if count != 1 else ''}")
                
                # Show any other alert types not in the categories above
                other_alerts = {k: v for k, v in enhancement_stats['storms_by_alert_type'].items() 
                              if k not in flood_alerts + tornado_alerts + thunderstorm_alerts}
                if other_alerts:
                    logger.info(f"   Other alerts:")
                    for alert_type, count in other_alerts.items():
                        logger.info(f"     • {alert_type}: {count} storm{'s' if count != 1 else ''}")
            
            return enhanced_storms, enhancement_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance storms with alerts: {e}")
            for storm in storms:
                self._add_default_alert_fields(storm)
            return storms, enhancement_stats
    
    def _add_default_alert_fields(self, storm: Dict):
        """Add default alert fields to storm."""
        storm["nws_alerts_present"] = False
        storm["alert_types"] = []
        storm["alert_details"] = []
        storm["contained_in_alert"] = False
        storm["alert_boosted"] = False
        storm["alert_boost_factor"] = 1.0
    
    def detect_severe_storms(self) -> Dict:
        """Main detection pipeline with comprehensive logging."""
        start_time = datetime.now()
        logger.info("🚀 Starting NOAA MRMS severe storm detection pipeline")
        
        try:
            # Step 1: Download radar data
            radar_file = self.get_latest_radar_data()
            if not radar_file:
                return self._create_empty_response("Failed to download radar data")
            
            # Step 2: Process radar data
            radar_result = self.process_radar_data(radar_file)
            if radar_result is None:
                return self._create_empty_response("Failed to process radar data")
            
            reflectivity, transform_info, crs_info = radar_result
            
            # Step 3: Identify storm areas
            storms, filtering_stats = self.identify_storm_areas(reflectivity, transform_info, crs_info)
            if not storms:
                return self._create_empty_response("No severe storms detected")
            
            # Step 4: Fetch NWS alerts
            alerts_gdf, alert_stats = self.get_nws_alerts()
            
            # Step 5: Enhance storms with alerts
            enhanced_storms, enhancement_stats = self.enhance_storms_with_alerts(storms, alerts_gdf)
            
            # Step 6: Format output
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create GeoJSON features
            features = []
            for storm in enhanced_storms:
                feature = {
                    "type": "Feature",
                    "id": storm["id"],
                    "geometry": storm["geometry"].__geo_interface__,
                    "properties": {
                        key: value for key, value in storm.items() 
                        if key not in ["id", "geometry"]
                    }
                }
                features.append(feature)
            
            # Build comprehensive metadata
            result = {
                "type": "FeatureCollection",
                "metadata": {
                    "total_storm_areas": len(enhanced_storms),
                    "radar_timestamp": datetime.now(timezone.utc).isoformat(),
                    "data_source": "NOAA_MRMS",
                    "max_reflectivity_dbz": max(s["max_reflectivity_dbz"] for s in enhanced_storms) if enhanced_storms else 0,
                    "processing_time_seconds": round(processing_time, 2),
                    "alerts_integrated": alerts_gdf is not None and len(alerts_gdf) > 0,
                    "storms_with_alerts": enhancement_stats.get('storms_with_alerts', 0),
                    "filtering_criteria": self.filtering_criteria,
                    "filtering_stats": filtering_stats,
                    "alert_stats": alert_stats,
                    "enhancement_stats": enhancement_stats
                },
                "features": features
            }
            
            logger.info(f"✅ Storm detection completed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Final results: {len(enhanced_storms)} severe storm areas detected")
            
            # Cleanup
            try:
                os.unlink(radar_file)
            except:
                pass
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Storm detection pipeline failed: {e}")
            return self._create_empty_response(f"Pipeline error: {str(e)}")
    
    def _create_empty_response(self, error_message: str) -> Dict:
        """Create empty response with error information."""
        return {
            "type": "FeatureCollection",
            "metadata": {
                "total_storm_areas": 0,
                "radar_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_source": "NOAA_MRMS",
                "max_reflectivity_dbz": 0,
                "processing_time_seconds": 0,
                "error": error_message,
                "alerts_integrated": False,
                "storms_with_alerts": 0,
                "filtering_criteria": self.filtering_criteria,
                "filtering_stats": {},
                "alert_stats": {},
                "enhancement_stats": {}
            },
            "features": []
        }

# CLI Implementation
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    def setup_logging(verbose: bool = False):
        """Configure logging with proper eccodes suppression."""
        
        # Final attempt to suppress eccodes - redirect at process level
        if not verbose:
            try:
                # Try to suppress eccodes at the environment level
                os.environ['ECCODES_GRIB_WRITE_ON_FAIL'] = '0'
                os.environ['ECCODES_DEBUG'] = '0'
                os.environ['ECCODES_LOG_STREAM'] = '/dev/null'
                
                # For some systems, try disabling the problematic time parsing
                os.environ['ECCODES_GRIB_TIME_STRICT'] = '0'
            except:
                pass
        
        level = logging.DEBUG if verbose else logging.INFO
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers = [console_handler]
        
        # Suppress third-party loggers
        if not verbose:
            for logger_name in ['cfgrib', 'eccodes', 'xarray', 'urllib3']:
                logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    def print_detection_summary(result: Dict):
        """Print enhanced detection summary with clear explanations."""
        metadata = result.get('metadata', {})
        features = result.get('features', [])
        
        print("\n" + "="*80)
        print("🌩️  NOAA MRMS SEVERE STORM DETECTION SUMMARY")
        print("="*80)
        
        # Processing Summary
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   • Radar Timestamp:     {metadata.get('radar_timestamp', 'N/A')}")
        print(f"   • Processing Time:     {metadata.get('processing_time_seconds', 0):.2f} seconds")
        print(f"   • Data Source:         {metadata.get('data_source', 'N/A')}")
        print(f"   • Max Reflectivity:    {metadata.get('max_reflectivity_dbz', 0):.1f} dBZ")
        
        # Filtering Results
        filtering_stats = metadata.get('filtering_stats', {})
        if filtering_stats:
            print(f"\n🔍 STORM FILTERING PROCESS:")
            print(f"   • Potential storms detected:     {filtering_stats.get('total_detected', 0)}")
            print(f"   • Filtered by pixel count:       {filtering_stats.get('filtered_by_pixels', 0)} (< {filtering_stats.get('criteria', {}).get('min_pixels', 0)} pixels)")
            print(f"   • Filtered by geometry:          {filtering_stats.get('filtered_by_size', 0)} (< 3 boundary points)")
            print(f"   • Filtered by area:              {filtering_stats.get('filtered_by_area', 0)} (< {filtering_stats.get('criteria', {}).get('min_area_km2', 0)} km²)")
            print(f"   • Selected for analysis:         {filtering_stats.get('selected', 0)}")
        
        # Alert Summary
        alert_stats = metadata.get('alert_stats', {})
        if alert_stats:
            print(f"\n🚨 NWS ALERT ANALYSIS:")
            print(f"   • Total active alerts:           {alert_stats.get('total_alerts', 0)}")
            print(f"   • Storm-related alerts:          {alert_stats.get('storm_related', 0)}")
            print(f"   • Active storm alerts:           {alert_stats.get('active_storm_alerts', 0)}")
            
            breakdown = alert_stats.get('breakdown', {})
            if breakdown:
                print(f"   • Alert type breakdown:")
                for alert_type, count in breakdown.items():
                    print(f"     - {alert_type}: {count}")
        
        # Enhancement Results  
        enhancement_stats = metadata.get('enhancement_stats', {})
        total_storms = metadata.get('total_storm_areas', 0)
        storms_with_alerts = enhancement_stats.get('storms_with_alerts', 0)
        
        print(f"\n⛈️  STORM-ALERT CORRELATION:")
        print(f"   • Final storms identified:       {total_storms}")
        print(f"   • Storms with active alerts:     {storms_with_alerts}")
        print(f"   • Alert-storm intersections:     {enhancement_stats.get('alert_intersections', 0)}")
        print(f"   • Storms boosted by alerts:      {enhancement_stats.get('storms_boosted', 0)}")
        
        if total_storms == 0:
            print(f"\n   ✅ No severe storms detected at this time")
            error_msg = metadata.get('error')
            if error_msg:
                print(f"   ⚠️  Error: {error_msg}")
            return
        
        # Top Storms Table
        print(f"\n🌪️  TOP SEVERE STORMS:")
        print(f"   {'ID':<12} {'dBZ':<6} {'Area':<8} {'Score':<6} {'Boost':<6} {'Alert Types':<30} {'Location':<20}")
        print(f"   {'-'*12} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*30} {'-'*20}")
        
        for i, feature in enumerate(features[:10]):  # Show top 10
            props = feature.get('properties', {})
            storm_id = feature.get('id', f'storm_{i+1}')
            max_dbz = props.get('max_reflectivity_dbz', 0)
            area = props.get('area_sq_km', 0)
            score = props.get('storm_intensity_score', 0)
            boost = props.get('alert_boost_factor', 1.0)
            alert_types = props.get('alert_types', [])
            center = props.get('storm_center', [0, 0])
            
            # Format alert types
            alerts_str = ', '.join(alert_types[:1]) if alert_types else 'None'
            if len(alert_types) > 1:
                alerts_str += f' +{len(alert_types)-1}'
            alerts_str = alerts_str[:29]  # Truncate if too long
            
            # Format location
            location = f"{center[1]:.2f}°N, {abs(center[0]):.2f}°W"
            
            boost_str = f"{boost:.1f}x" if boost > 1.0 else "None"
            
            print(f"   {storm_id:<12} {max_dbz:<6.1f} {area:<8.1f} {score:<6.3f} {boost_str:<6} {alerts_str:<30} {location:<20}")
        
        if len(features) > 10:
            print(f"   ... and {len(features) - 10} more storms")
        
        # Alert Impact Summary
        storms_by_alert = enhancement_stats.get('storms_by_alert_type', {})
        if storms_by_alert:
            print(f"\n🚨 ALERT IMPACT SUMMARY:")
            print(f"   (This shows how many storms were affected by each alert type)")
            for alert_type, count in storms_by_alert.items():
                print(f"   • {count} storm{'s' if count != 1 else ''} affected by {alert_type}")
        
        print("\n" + "="*80)
    
    def save_json_output(result: Dict, output_file: Path, pretty: bool = True):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                if pretty:
                    json.dump(result, f, indent=2, default=str)
                else:
                    json.dump(result, f, default=str)
            
            file_size = output_file.stat().st_size
            print(f"\n💾 Results saved to: {output_file} ({file_size:,} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save JSON output: {e}")
            sys.exit(1)
    
    # CLI argument parsing
    parser = argparse.ArgumentParser(
        description="🌩️ Enhanced NOAA MRMS Severe Storm Detection System",
        epilog="Detects severe weather storms with comprehensive filtering and alert correlation.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('storm_detection_results.json'),
        help='Output JSON file path (default: storm_detection_results.json)'
    )
    
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Save JSON in compact format (no pretty printing)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose debug logging (shows eccodes warnings)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip console summary output'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=45.0,
        help='Reflectivity threshold in dBZ (default: 45.0)'
    )
    
    parser.add_argument(
        '--min-area',
        type=float,
        default=25.0,
        help='Minimum storm area in km² (default: 25.0)'
    )
    
    parser.add_argument(
        '--max-storms',
        type=int,
        default=50,
        help='Maximum number of storms to return (default: 50)'
    )
    
    parser.add_argument(
        '--min-pixels',
        type=int,
        default=10,
        help='Minimum pixel count for storm detection (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize detector with custom parameters
    detector = RadarDetector()
    detector.dbz_threshold = args.threshold
    detector.min_storm_area_km2 = args.min_area
    detector.max_storms_output = args.max_storms
    detector.filtering_criteria['min_pixels'] = args.min_pixels
    detector.filtering_criteria['min_reflectivity_dbz'] = args.threshold
    detector.filtering_criteria['min_area_km2'] = args.min_area
    
    try:
        print("🌩️ Starting Enhanced NOAA MRMS Severe Storm Detection...")
        print(f"   • Reflectivity threshold: {args.threshold} dBZ")
        print(f"   • Minimum storm area: {args.min_area} km²")
        print(f"   • Minimum pixels: {args.min_pixels}")
        print(f"   • Maximum storms: {args.max_storms}")
        if args.verbose:
            print("   • Verbose mode: ECCODES warnings will be shown")
        
        # Run detection
        storm_data = detector.detect_severe_storms()
        
        # Display summary
        if not args.no_summary:
            print_detection_summary(storm_data)
        
        # Save JSON output
        save_json_output(storm_data, args.output, pretty=not args.compact)
        
        # Exit codes
        total_storms = storm_data.get('metadata', {}).get('total_storm_areas', 0)
        storms_with_alerts = storm_data.get('metadata', {}).get('storms_with_alerts', 0)
        
        if storm_data.get('metadata', {}).get('error'):
            print(f"\n❌ Detection completed with errors")
            sys.exit(2)
        elif storms_with_alerts > 0:
            print(f"\n🚨 SEVERE WEATHER ALERT: {storms_with_alerts} storms have active NWS alerts!")
            sys.exit(1)
        elif total_storms > 0:
            print(f"\n⚠️  {total_storms} severe storms detected (no active alerts)")
            sys.exit(0)
        else:
            print(f"\n✅ No severe storms detected")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Detection interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)