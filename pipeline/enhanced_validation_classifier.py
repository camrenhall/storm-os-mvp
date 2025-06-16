#!/usr/bin/env python3
"""
Enhanced Validation FLASH Classification Engine
Comprehensive quality checks and validation to ensure data integrity
Leverages full 20-25 minute processing window for maximum data quality
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from scipy import ndimage
from dataclasses import dataclass, field
import concurrent.futures
from functools import partial
import threading
import time
from enum import Enum
import json
from pathlib import Path

# Import population grid module
try:
    from exposure import homes, initialize_population_grid, is_initialized, GRID_SHAPE
    EXPOSURE_AVAILABLE = True
except ImportError:
    EXPOSURE_AVAILABLE = False
    print("Warning: exposure module not available, home estimates disabled")

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from rasterio import features
    import matplotlib.pyplot as plt
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available, FFW integration disabled")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    DEVELOPMENT = "development"  # Fast processing for iteration (2-5 minutes)
    PRODUCTION = "production"    # Comprehensive validation (10-25 minutes)
    VALIDATION = "validation"    # Maximum quality checks (15-25 minutes)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for quality assurance"""
    
    # Connected component validation
    total_components_found: int = 0
    components_merged: int = 0
    components_split: int = 0
    components_filtered: int = 0
    
    # Spatial validation
    max_flood_area_km2: float = 0.0
    min_flood_area_km2: float = 0.0
    mean_flood_area_km2: float = 0.0
    suspicious_large_floods: int = 0
    suspicious_small_floods: int = 0
    
    # Intensity validation
    max_streamflow_found: float = 0.0
    min_streamflow_found: float = 0.0
    intensity_coherence_score: float = 0.0
    gradient_anomalies: int = 0
    
    # FFW validation
    ffw_coverage_percentage: float = 0.0
    floods_outside_ffw: int = 0
    ffw_areas_without_floods: int = 0
    
    # Cross-method validation
    grid_vs_components_centroid_diff: List[float] = field(default_factory=list)
    method_agreement_score: float = 0.0
    
    # Quality flags
    data_quality_excellent: bool = False
    data_quality_good: bool = False
    data_quality_concerns: List[str] = field(default_factory=list)
    
    # Processing time breakdown
    classification_time: float = 0.0
    validation_time: float = 0.0
    extraction_time: float = 0.0
    total_time: float = 0.0


@dataclass
class ClassificationConfig:
    # CHANGED: Remove FFW filtering requirement  
    require_ffw_intersection: bool = False  # Was True
    
    # NEW: Enable FFW as enhancement instead
    enable_ffw_enhancement: bool = True
    ffw_quality_multiplier: float = 1.5
    
    # NEW: Proper flood threshold settings
    use_extreme_percentiles: bool = True
    min_critical_threshold_m3s_km2: float = 15.0
    min_high_threshold_m3s_km2: float = 8.0  
    min_moderate_threshold_m3s_km2: float = 3.0
    
    # NEW: Anti-consolidation settings
    max_event_size_km2: float = 50.0
    enable_size_limiting: bool = True
    
    # EXISTING: Keep all current validation settings
    processing_mode: ProcessingMode = ProcessingMode.PRODUCTION
    enable_detailed_logging: bool = True
    
    # Pure science-based thresholds (updated values)
    percentile_level: float = 99.5  # Use higher percentile
    critical_threshold: float = 1.0   # Will be overridden by physical minimums
    high_threshold: float = 0.7       # Will be overridden by physical minimums  
    moderate_threshold: float = 0.5   # Will be overridden by physical minimums
    
    # Enhanced validation settings
    enable_cross_method_validation: bool = True
    enable_spatial_validation: bool = True
    enable_intensity_validation: bool = True
    enable_hydrological_validation: bool = True
    
    # NEW: Adaptive threshold settings
    enable_adaptive_thresholds: bool = True
    min_detection_rate_percent: float = 0.001  # Minimum 0.001% detection rate
    major_event_threshold: float = 20.0  # m³/s/km² - threshold for "major event" mode
    moderate_event_threshold: float = 5.0  # m³/s/km² - threshold for "moderate event" mode
    
    # Quality control thresholds (updated for realistic flooding)
    max_reasonable_flood_area_km2: float = 1000.0  # Was 10000.0 - more realistic
    min_reasonable_flood_area_km2: float = 0.25    # Was 1.0 - allow smaller events
    max_reasonable_streamflow: float = 100.0
    max_intensity_gradient: float = 10.0
    
    # Spatial filtering
    min_flood_area_pixels: int = 2
    min_valid_pixels: int = 50000
    
    # Performance settings
    max_workers: int = 4
    enable_detailed_logging: bool = True
    save_validation_plots: bool = False
    validation_output_dir: str = "./validation_output"


@dataclass 
class ClassificationResult:
    """Enhanced classification results with comprehensive validation"""
    
    # Core classification arrays
    critical_mask: np.ndarray
    high_mask: np.ndarray  
    moderate_mask: np.ndarray
    
    # Metadata
    valid_time: datetime
    total_pixels: int
    valid_pixels: int
    
    # Threshold values
    p98_value: float
    critical_threshold_value: float
    high_threshold_value: float
    moderate_threshold_value: float
    
    # Statistics
    critical_count: int
    high_count: int
    moderate_count: int
    
    # Processing metadata
    processing_mode: str
    normalization_method: str
    
    # Enhanced validation results
    validation_metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    
    # Optional fields
    ffw_mask: Optional[np.ndarray] = None
    ffw_intersection_applied: bool = False
    ffw_boosted_pixels: int = 0
    ffw_verified_pixels: int = 0
    quality_degraded: bool = False


class OptimizedFlashClassifier:
    """
    Enhanced FLASH Classification Engine with Comprehensive Validation
    Leverages full processing time (20-25 minutes) for maximum data quality
    """
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig()
        self.last_p98 = None
        
        # Setup validation output directory
        self.validation_dir = Path(self.config.validation_output_dir)
        self.validation_dir.mkdir(exist_ok=True)
        
        # Pre-compute grid definition
        self._setup_grid_coordinates()
        
        # Initialize population grid if available
        self._initialize_exposure_data()
        
        # Performance and validation tracking
        self.processing_stats = {
            'last_classification_time': 0.0,
            'last_extraction_time': 0.0,
            'last_validation_time': 0.0,
            'last_hotspot_count': 0
        }
        
        # Score computation warning (one-time per run)
        self._score_warning_logged = False
    
    def _initialize_exposure_data(self):
        """Initialize population exposure data if available"""
        if EXPOSURE_AVAILABLE:
            try:
                if not is_initialized():
                    logger.info("Initializing population exposure grid...")
                    initialize_population_grid()
                    logger.info("✅ Population exposure grid ready")
                else:
                    logger.info("✅ Population exposure grid already initialized")
                self._exposure_enabled = True
            except Exception as e:
                logger.error(f"❌ Population exposure initialization failed: {e}")
                self._exposure_enabled = False
        else:
            logger.warning("⚠️  Population exposure module not available")
            self._exposure_enabled = False
        
    def _setup_grid_coordinates(self):
        """Pre-compute coordinate transformation arrays"""
        # Use GRID_SHAPE from exposure module if available, otherwise fallback
        if EXPOSURE_AVAILABLE:
            self.nj, self.ni = GRID_SHAPE
            logger.info(f"Using exposure module grid dimensions: {self.ni}×{self.nj}")
        else:
            # Fallback to original hardcoded values
            self.nj, self.ni = 3500, 7000
            logger.warning(f"Using fallback grid dimensions: {self.ni}×{self.nj}")
        
        # CONUS grid bounds (MRMS specification)
        self.west, self.east = -130.0, -60.0
        self.south, self.north = 20.0, 55.0
        
        # Resolution
        self.lon_res = (self.east - self.west) / self.ni
        self.lat_res = (self.north - self.south) / self.nj
        
        # Pre-compute coordinate arrays
        self.col_to_lon = self.west + (np.arange(self.ni) + 0.5) * self.lon_res
        self.row_to_lat = self.north - (np.arange(self.nj) + 0.5) * self.lat_res
        
        # Grid rasterization transform
        self.affine_transform = [
            self.lon_res, 0.0, self.west, 0.0, -self.lat_res, self.north
        ]
        
        logger.debug("Grid coordinates and validation framework initialized")
        
    def lonlat_to_grid(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert longitude/latitude to grid indices"""
        col = int((lon - self.west) / self.lon_res)
        row = int((self.north - lat) / self.lat_res)
        return row, col

    def grid_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to longitude/latitude coordinates"""
        lon = self.west + (col + 0.5) * self.lon_res
        lat = self.north - (row + 0.5) * self.lat_res
        return lon, lat
        
    def grid_to_lonlat_vectorized(self, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized coordinate transformation"""
        lons = self.col_to_lon[cols]
        lats = self.row_to_lat[rows]
        return lons, lats

    def rasterize_ffw_polygons(self, ffw_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Rasterize FFW polygons with validation"""
        if not GEOPANDAS_AVAILABLE:
            logger.warning("geopandas not available, FFW integration disabled")
            return np.zeros((self.nj, self.ni), dtype=np.uint8)
            
        try:
            if len(ffw_gdf) == 0:
                logger.info("No active FFW polygons to rasterize")
                return np.zeros((self.nj, self.ni), dtype=np.uint8)
            
            # Validate FFW data quality
            if self.config.enable_detailed_logging:
                logger.info("Validating FFW polygon quality...")
                valid_polygons = 0
                total_area = 0.0
                
                for idx, row in ffw_gdf.iterrows():
                    if row.geometry is not None and row.geometry.is_valid:
                        valid_polygons += 1
                        if hasattr(row, 'area_km2') and row.area_km2:
                            total_area += row.area_km2
                
                logger.info(f"  Valid FFW polygons: {valid_polygons}/{len(ffw_gdf)}")
                logger.info(f"  Total FFW area: {total_area:.0f} km²")
            
            # Ensure correct CRS
            if ffw_gdf.crs != 'EPSG:4326':
                ffw_gdf = ffw_gdf.to_crs('EPSG:4326')
            
            # Create geometry pairs for rasterization
            geom_value_pairs = [(geom, 1) for geom in ffw_gdf.geometry 
                               if geom is not None and geom.is_valid]
            
            if not geom_value_pairs:
                logger.warning("No valid FFW geometries found")
                return np.zeros((self.nj, self.ni), dtype=np.uint8)
            
            # Rasterize with validation
            start_time = time.time()
            rasterized = features.rasterize(
                geom_value_pairs,
                out_shape=(self.nj, self.ni),
                transform=self.affine_transform,
                fill=0,
                dtype=np.uint8
            )
            rasterize_time = time.time() - start_time
            
            ffw_pixel_count = (rasterized == 1).sum()
            logger.info(f"✓ Rasterized {len(geom_value_pairs)} FFW polygons to {ffw_pixel_count:,} pixels in {rasterize_time:.2f}s")
            
            # Validation: Check rasterization quality
            if self.config.enable_spatial_validation:
                expected_pixels = sum(geom.area for geom in ffw_gdf.geometry) / (self.lon_res * self.lat_res)
                pixel_ratio = ffw_pixel_count / expected_pixels if expected_pixels > 0 else 0
                
                if pixel_ratio < 0.5 or pixel_ratio > 2.0:
                    logger.warning(f"FFW rasterization may have issues: {pixel_ratio:.2f} pixel ratio")
            
            return rasterized
            
        except Exception as e:
            logger.error(f"FFW polygon rasterization failed: {e}")
            return np.zeros((self.nj, self.ni), dtype=np.uint8)
        
    def _extract_fast_wrapper(self, mask, unit_streamflow, severity, valid_time, normalization_method, ffw_mask):
        """Wrapper for fast extraction - ignores ffw_mask parameter"""
        return self._extract_hotspots_fast(mask, unit_streamflow, severity, valid_time, normalization_method)

    def _extract_precise_with_ffw(self, mask, unit_streamflow, severity, valid_time, normalization_method, ffw_mask):
        """Wrapper for precise extraction with ffw_mask"""
        return self._extract_hotspots_precise(mask, unit_streamflow, severity, valid_time, normalization_method, ffw_mask)

    def calculate_adaptive_thresholds(self, unit_streamflow: np.ndarray) -> Optional[Dict[str, float]]:
        """Calculate adaptive flood thresholds that scale with current conditions"""
        
        # Get valid data
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        valid_data = unit_streamflow[valid_mask]
        
        if len(valid_data) < self.config.min_valid_pixels:
            if self.last_p98 is not None:
                # Use cached values with adaptive scaling
                p99_9 = self.last_p98 * 1.5
                p99_5 = self.last_p98 * 1.2
                p99_0 = self.last_p98 * 1.0
            else:
                logger.error("No cached data available for threshold calculation")
                return None
        else:
            # Calculate percentiles for adaptive thresholds
            p99_9 = np.percentile(valid_data, 99.9)
            p99_5 = np.percentile(valid_data, 99.5)
            p99_0 = np.percentile(valid_data, 99.0)
            p98 = np.percentile(valid_data, 98.0)
            self.last_p98 = p98
            
            # Log distribution for validation
            if self.config.enable_detailed_logging:
                p50 = np.percentile(valid_data, 50)
                p90 = np.percentile(valid_data, 90)
                
                logger.info(f"Streamflow distribution analysis:")
                logger.info(f"  P50: {p50:.3f}, P90: {p90:.3f}, P98: {p98:.3f}")
                logger.info(f"  P99.0: {p99_0:.3f}, P99.5: {p99_5:.3f}, P99.9: {p99_9:.3f}")
                logger.info(f"  Max: {valid_data.max():.3f} m³/s/km²")
        
        # ADAPTIVE APPROACH: Scale with conditions but maintain reasonable minimums
        data_max = valid_data.max() if len(valid_data) > 0 else 0
        
        # During major events (high max values), use higher thresholds
        # During normal conditions, use lower but still meaningful thresholds
        if data_max > 20.0:  # Major flood event conditions
            # Use higher thresholds similar to original approach
            thresholds = {
                'p999': p99_9,
                'critical': max(p99_9, 10.0),    # High threshold for major events
                'high': max(p99_5, 5.0),         # Medium threshold  
                'moderate': max(p99_0, 2.0)      # Lower threshold
            }
            logger.info("Major event conditions detected - using elevated thresholds")
        elif data_max > 5.0:  # Moderate event conditions  
            thresholds = {
                'p999': p99_9,
               'critical': max(p99_9, 3.0),     # Moderate thresholds
               'high': max(p99_5, 1.5),
               'moderate': max(p99_0, 0.8)
           }
           logger.info("Moderate event conditions detected - using intermediate thresholds")
       else:  # Normal/quiet conditions
           # Use percentile-based with low minimums to catch meaningful flows
           thresholds = {
               'p999': p99_9,
               'critical': max(p99_9, 1.0),     # Lower thresholds for quiet periods
               'high': max(p99_5, 0.5),
               'moderate': max(p99_0, 0.2)
           }
           logger.info("Normal conditions detected - using adaptive low thresholds")
       
       # Validation: Ensure minimum detection rate
       estimated_pixels = 0
       for severity, threshold in [('moderate', thresholds['moderate']), 
                               ('high', thresholds['high']), 
                               ('critical', thresholds['critical'])]:
           pixels_above = (valid_data >= threshold).sum()
           estimated_pixels += pixels_above
       
       estimated_rate = (estimated_pixels / len(valid_data)) * 100 if len(valid_data) > 0 else 0
       
       # If still too restrictive, lower thresholds further
       if estimated_rate < 0.001:  # Less than 0.001% detection
           logger.warning(f"Estimated detection rate {estimated_rate:.6f}% very low, applying minimum thresholds")
           thresholds = {
                'p999': p99_9,
                'critical': max(p99_9, 0.5),
                'high': max(p99_5, 0.3), 
                'moderate': max(p99_0, 0.1)
            }
        
        logger.info(f"ADAPTIVE Flood Thresholds (Condition-Responsive):")
        logger.info(f"  Critical: ≥{thresholds['critical']:.3f} m³/s/km²")
        logger.info(f"  High: ≥{thresholds['high']:.3f} m³/s/km²")
        logger.info(f"  Moderate: ≥{thresholds['moderate']:.3f} m³/s/km²")
        logger.info(f"  Estimated detection rate: {estimated_rate:.6f}%")
        
        return thresholds
    
    def apply_flood_classification_no_filter(self, unit_streamflow: np.ndarray, 
                                        thresholds: Dict[str, float],
                                        ffw_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """Apply flood classification WITHOUT FFW filtering - detect floods everywhere"""
        
        # Create base valid data mask
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        
        # Apply proper flood thresholds (no FFW filtering)
        critical_mask = valid_mask & (unit_streamflow >= thresholds['critical'])
        high_mask = valid_mask & (unit_streamflow >= thresholds['high']) & (~critical_mask)
        moderate_mask = valid_mask & (unit_streamflow >= thresholds['moderate']) & (~critical_mask) & (~high_mask)
        
        # Calculate FFW statistics for enhancement (not filtering)
        ffw_stats = {'enhanced_critical': 0, 'enhanced_high': 0, 'total_in_ffw': 0}
        
        if ffw_mask is not None and ffw_mask.any():
            ffw_areas = (ffw_mask == 1)
            
            # Count floods within FFW areas (for quality enhancement later)
            ffw_stats['enhanced_critical'] = (critical_mask & ffw_areas).sum()
            ffw_stats['enhanced_high'] = (high_mask & ffw_areas).sum() 
            ffw_stats['total_in_ffw'] = ((critical_mask | high_mask | moderate_mask) & ffw_areas).sum()
            
            logger.info(f"FFW Enhancement Statistics (NOT filtering):")
            logger.info(f"  Critical floods in FFW areas: {ffw_stats['enhanced_critical']:,}")
            logger.info(f"  High floods in FFW areas: {ffw_stats['enhanced_high']:,}")
            logger.info(f"  Total floods in FFW areas: {ffw_stats['total_in_ffw']:,}")
        
        # Log total detections (should be much lower with proper thresholds)
        total_floods = critical_mask.sum() + high_mask.sum() + moderate_mask.sum()
        total_pixels = unit_streamflow.size
        detection_rate = (total_floods / total_pixels) * 100
        
        logger.info(f"Flood Detection Results (No FFW Filtering):")
        logger.info(f"  Critical floods: {critical_mask.sum():,} pixels")
        logger.info(f"  High floods: {high_mask.sum():,} pixels") 
        logger.info(f"  Moderate floods: {moderate_mask.sum():,} pixels")
        logger.info(f"  Total flood pixels: {total_floods:,}")
        logger.info(f"  Detection rate: {detection_rate:.3f}% of grid")
        
        # Validation: Warn if detection rate is still too high
        if detection_rate > 1.0:
            logger.warning(f"Detection rate {detection_rate:.3f}% may still be too high - check thresholds")
        elif detection_rate < 0.001:
            logger.warning(f"Detection rate {detection_rate:.3f}% may be too low - no floods detected")
        else:
            logger.info(f"✅ Detection rate {detection_rate:.3f}% appears reasonable for actual flooding")
        
        return critical_mask, high_mask, moderate_mask, ffw_stats

    def validate_connected_components(self, mask: np.ndarray, severity: str) -> Dict[str, Any]:
        """Comprehensive validation of connected component analysis"""
        
        validation_results = {
            'total_components': 0,
            'component_sizes': [],
            'largest_component': 0,
            'smallest_component': 0,
            'mean_component_size': 0.0,
            'suspicious_components': 0,
            'spatial_coherence_score': 0.0
        }
        
        if not mask.any():
            return validation_results
        
        try:
            start_time = time.time()
            
            # Perform connected component analysis with detailed tracking
            labeled_array, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return validation_results
            
            # Analyze each component
            component_sizes = []
            suspicious_count = 0
            
            for label in range(1, num_features + 1):
                component_mask = (labeled_array == label)
                size = component_mask.sum()
                component_sizes.append(size)
                
                # Check for suspicious components
                area_km2 = size * 1.0  # 1 km² per pixel
                if area_km2 > self.config.max_reasonable_flood_area_km2:
                    suspicious_count += 1
                    logger.warning(f"Suspicious large {severity} component: {area_km2:.0f} km²")
                elif area_km2 < self.config.min_reasonable_flood_area_km2:
                    suspicious_count += 1
            
            # Calculate statistics
            validation_results.update({
                'total_components': num_features,
                'component_sizes': component_sizes,
                'largest_component': max(component_sizes) if component_sizes else 0,
                'smallest_component': min(component_sizes) if component_sizes else 0,
                'mean_component_size': np.mean(component_sizes) if component_sizes else 0.0,
                'suspicious_components': suspicious_count
            })
            
            analysis_time = time.time() - start_time
            
            if self.config.enable_detailed_logging:
                logger.info(f"Connected components validation for {severity}:")
                logger.info(f"  Total components found: {num_features}")
                logger.info(f"  Size range: {min(component_sizes) if component_sizes else 0} - {max(component_sizes) if component_sizes else 0} pixels")
                logger.info(f"  Mean size: {np.mean(component_sizes):.1f} pixels" if component_sizes else "  No components")
                logger.info(f"  Suspicious components: {suspicious_count}")
                logger.info(f"  Analysis time: {analysis_time:.3f}s")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Connected component validation failed: {e}")
            return validation_results

    def cross_validate_extraction_methods(self, mask: np.ndarray, unit_streamflow: np.ndarray, 
                                        severity: str, valid_time: datetime) -> Dict[str, Any]:
        """Cross-validate grid-based vs connected component methods"""
        
        validation_results = {
            'grid_method_events': 0,
            'component_method_events': 0,
            'centroid_differences': [],
            'method_agreement_score': 0.0,
            'concerning_differences': []
        }
        
        if not mask.any():
            return validation_results
        
        try:
            logger.info(f"Cross-validating extraction methods for {severity}...")
            
            # Method 1: Grid-based (fast) - doesn't need ffw_mask
            grid_events = self._extract_hotspots_fast(mask, unit_streamflow, severity, valid_time, "validation")

            # Method 2: Connected components (precise) - needs ffw_mask
            component_events = self._extract_hotspots_precise(mask, unit_streamflow, severity, valid_time, "validation", None)
            
            validation_results['grid_method_events'] = len(grid_events)
            validation_results['component_method_events'] = len(component_events)
            
            # Compare centroids if both methods found events
            if grid_events and component_events:
                # For each component event, find closest grid event
                centroid_diffs = []
                
                for comp_event in component_events:
                    comp_lon, comp_lat = comp_event['longitude'], comp_event['latitude']
                    
                    # Find closest grid event
                    min_distance = float('inf')
                    for grid_event in grid_events:
                        grid_lon, grid_lat = grid_event['longitude'], grid_event['latitude']
                        distance = np.sqrt((comp_lon - grid_lon)**2 + (comp_lat - grid_lat)**2)
                        min_distance = min(min_distance, distance)
                    
                    centroid_diffs.append(min_distance)
                
                validation_results['centroid_differences'] = centroid_diffs
                validation_results['method_agreement_score'] = 1.0 / (1.0 + np.mean(centroid_diffs)) if centroid_diffs else 0.0
                
                # Check for concerning differences
                concerning_threshold = 0.1  # 0.1 degree (~11 km)
                concerning_diffs = [d for d in centroid_diffs if d > concerning_threshold]
                validation_results['concerning_differences'] = concerning_diffs
                
                if concerning_diffs:
                    logger.warning(f"Cross-validation found {len(concerning_diffs)} concerning centroid differences for {severity}")
                
                logger.info(f"Cross-validation results for {severity}:")
                logger.info(f"  Grid method: {len(grid_events)} events")
                logger.info(f"  Component method: {len(component_events)} events")
                logger.info(f"  Mean centroid difference: {np.mean(centroid_diffs):.4f} degrees")
                logger.info(f"  Method agreement score: {validation_results['method_agreement_score']:.3f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed for {severity}: {e}")
            return validation_results

    def validate_intensity_coherence(self, mask: np.ndarray, unit_streamflow: np.ndarray, 
                                severity: str) -> Dict[str, Any]:
        """
        ENHANCED: Validate streamflow intensity coherence with improved robustness
        Includes additional checks for the quality calculation bug
        """
        
        validation_results = {
            'coherence_score': 0.0,
            'gradient_anomalies': 0,
            'intensity_range': (0.0, 0.0),
            'spatial_correlation': 0.0,
            'quality_calculation_issues': 0  # NEW: Track quality calculation problems
        }
        
        if not mask.any():
            return validation_results
        
        try:
            # Get flood pixel coordinates and values
            flood_coords = np.where(mask)
            flood_values = unit_streamflow[flood_coords]
            
            if len(flood_values) < 10:  # Need minimum pixels for validation
                return validation_results
            
            # Calculate intensity statistics
            intensity_range = (float(flood_values.min()), float(flood_values.max()))
            intensity_std = float(flood_values.std())
            intensity_mean = float(flood_values.mean())
            
            # Check for extreme gradients between adjacent pixels
            gradient_anomalies = 0
            rows, cols = flood_coords
            
            for i in range(len(rows)):
                current_value = unit_streamflow[rows[i], cols[i]]
                
                # Check 4-connected neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = rows[i] + dr, cols[i] + dc
                    
                    if (0 <= nr < self.nj and 0 <= nc < self.ni and 
                        mask[nr, nc]):  # Neighbor is also a flood pixel
                        
                        neighbor_value = unit_streamflow[nr, nc]
                        gradient = abs(current_value - neighbor_value)
                        
                        if gradient > self.config.max_intensity_gradient:
                            gradient_anomalies += 1
            
            # ENHANCED: Test the quality calculation components
            quality_calculation_issues = 0
            
            # Test intensity uniformity calculation that was causing the bug
            if intensity_mean > 0:
                test_uniformity = max(0.0, 1.0 - (intensity_std / intensity_mean))
                
                # Check for problematic cases that could cause zero quality scores
                if test_uniformity == 0.0 and intensity_std >= intensity_mean:
                    quality_calculation_issues += 1
                    logger.debug(f"Detected high variability case: std={intensity_std:.3f} >= mean={intensity_mean:.3f}")
            
            # Calculate coherence score with improved robustness
            if intensity_mean > 0.001:  # Prevent division by very small numbers
                coherence_score = 1.0 / (1.0 + intensity_std / intensity_mean)
            else:
                coherence_score = 0.1  # Conservative fallback for very low flows
            
            validation_results.update({
                'coherence_score': coherence_score,
                'gradient_anomalies': gradient_anomalies,
                'intensity_range': intensity_range,
                'spatial_correlation': coherence_score,  # Simplified for now
                'quality_calculation_issues': quality_calculation_issues
            })
            
            if self.config.enable_detailed_logging:
                logger.info(f"Intensity validation for {severity}:")
                logger.info(f"  Intensity range: {intensity_range[0]:.3f} - {intensity_range[1]:.3f} m³/s/km²")
                logger.info(f"  Coherence score: {coherence_score:.3f}")
                logger.info(f"  Gradient anomalies: {gradient_anomalies}")
                
                if quality_calculation_issues > 0:
                    logger.warning(f"  Quality calculation issues detected: {quality_calculation_issues}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Intensity validation failed for {severity}: {e}")
            return validation_results

    def comprehensive_validation(self, result: ClassificationResult, 
                                unit_streamflow: np.ndarray,
                                ffw_mask: Optional[np.ndarray] = None) -> ValidationMetrics:
        """Comprehensive validation of classification results"""
        
        logger.info("Starting comprehensive validation...")
        validation_start = time.time()
        
        metrics = ValidationMetrics()
        
        try:
            # 1. Connected Components Validation
            
            # Test the flood events that will be extracted to ensure no zero quality scores
            if hasattr(self, '_test_quality_calculations'):
                quality_issues = self._test_quality_calculations(result, unit_streamflow)
                if quality_issues > 0:
                    metrics.data_quality_concerns.append(f"Found {quality_issues} potential quality score calculation issues")
                
            if self.config.enable_spatial_validation:
                logger.info("Validating connected components...")
                
                for severity, mask in [('critical', result.critical_mask), 
                                        ('high', result.high_mask), 
                                        ('moderate', result.moderate_mask)]:
                    if mask.any():
                        comp_validation = self.validate_connected_components(mask, severity)
                        metrics.total_components_found += comp_validation['total_components']
                        metrics.suspicious_large_floods += comp_validation['suspicious_components']
                        
                        if comp_validation['largest_component'] > metrics.max_flood_area_km2:
                            metrics.max_flood_area_km2 = comp_validation['largest_component']
                        
                        if comp_validation['smallest_component'] < metrics.min_flood_area_km2 or metrics.min_flood_area_km2 == 0:
                            metrics.min_flood_area_km2 = comp_validation['smallest_component']
            
            # 2. Cross-Method Validation
            if self.config.enable_cross_method_validation:
                logger.info("Cross-validating extraction methods...")
                
                all_centroid_diffs = []
                for severity, mask in [('critical', result.critical_mask), ('high', result.high_mask)]:
                    if mask.any():
                        cross_val = self.cross_validate_extraction_methods(
                            mask, unit_streamflow, severity, result.valid_time
                        )
                        all_centroid_diffs.extend(cross_val['centroid_differences'])
                
                metrics.grid_vs_components_centroid_diff = all_centroid_diffs
                metrics.method_agreement_score = (1.0 / (1.0 + np.mean(all_centroid_diffs)) 
                                                if all_centroid_diffs else 1.0)
            
            # 3. Intensity Coherence Validation
            if self.config.enable_intensity_validation:
                logger.info("Validating intensity coherence...")
                
                total_gradient_anomalies = 0
                coherence_scores = []
                
                for severity, mask in [('critical', result.critical_mask), 
                                        ('high', result.high_mask), 
                                        ('moderate', result.moderate_mask)]:
                    if mask.any():
                        intensity_val = self.validate_intensity_coherence(mask, unit_streamflow, severity)
                        total_gradient_anomalies += intensity_val['gradient_anomalies']
                        coherence_scores.append(intensity_val['coherence_score'])
                        
                        if intensity_val['intensity_range'][1] > metrics.max_streamflow_found:
                            metrics.max_streamflow_found = intensity_val['intensity_range'][1]
                        
                        if (intensity_val['intensity_range'][0] < metrics.min_streamflow_found or 
                            metrics.min_streamflow_found == 0):
                            metrics.min_streamflow_found = intensity_val['intensity_range'][0]
                
                metrics.gradient_anomalies = total_gradient_anomalies
                metrics.intensity_coherence_score = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # 4. FFW Coverage Validation
            if ffw_mask is not None and ffw_mask.any():
                logger.info("Validating FFW coverage...")
                
                ffw_pixels = (ffw_mask == 1).sum()
                total_flood_pixels = (result.critical_mask | result.high_mask | result.moderate_mask).sum()
                
                if total_flood_pixels > 0:
                    # Check what percentage of floods are in FFW areas
                    floods_in_ffw = ((result.critical_mask | result.high_mask | result.moderate_mask) & 
                                    (ffw_mask == 1)).sum()
                    metrics.ffw_coverage_percentage = (floods_in_ffw / total_flood_pixels * 100)
                    
                    # Check for floods outside FFW areas (shouldn't happen with FFW-first)
                    floods_outside = total_flood_pixels - floods_in_ffw
                    metrics.floods_outside_ffw = floods_outside
                    
                    if floods_outside > 0:
                        logger.warning(f"Found {floods_outside} flood pixels outside FFW areas")
                
                # Check FFW areas without detected floods
                ffw_with_floods = ((ffw_mask == 1) & 
                                    (result.critical_mask | result.high_mask | result.moderate_mask)).sum()
                ffw_without_floods = ffw_pixels - ffw_with_floods
                metrics.ffw_areas_without_floods = ffw_without_floods
                
                logger.info(f"FFW validation results:")
                logger.info(f"  FFW coverage: {metrics.ffw_coverage_percentage:.1f}%")
                logger.info(f"  FFW areas without floods: {ffw_without_floods:,} pixels")
            
            # 5. Overall Quality Assessment
            quality_concerns = []
            
            # Check for data quality issues
            if metrics.suspicious_large_floods > 0:
                quality_concerns.append(f"Found {metrics.suspicious_large_floods} suspiciously large floods")
            
            if metrics.gradient_anomalies > 100:
                quality_concerns.append(f"High number of intensity gradient anomalies: {metrics.gradient_anomalies}")
            
            if metrics.method_agreement_score < 0.7:
                quality_concerns.append(f"Poor agreement between extraction methods: {metrics.method_agreement_score:.2f}")
            
            if metrics.ffw_coverage_percentage < 95 and result.ffw_intersection_applied:
                quality_concerns.append(f"Low FFW coverage despite FFW-first filtering: {metrics.ffw_coverage_percentage:.1f}%")
            
            metrics.data_quality_concerns = quality_concerns
            
            # Set quality flags
            if len(quality_concerns) == 0:
                metrics.data_quality_excellent = True
                logger.info("✅ Data quality assessment: EXCELLENT")
            elif len(quality_concerns) <= 2:
                metrics.data_quality_good = True
                logger.info("✅ Data quality assessment: GOOD")
            else:
                logger.warning("⚠️  Data quality assessment: CONCERNS IDENTIFIED")
                for concern in quality_concerns:
                    logger.warning(f"  - {concern}")
            
            # Timing
            validation_time = time.time() - validation_start
            metrics.validation_time = validation_time
            
            logger.info(f"✓ Comprehensive validation completed in {validation_time:.2f}s")
            
            logger.info("Final quality score verification...")
        
            # Extract a sample of flood events to test quality calculations
            sample_events = []
            for severity, mask in [('critical', result.critical_mask), ('high', result.high_mask)]:
                if mask.any():
                    sample_events.extend(self._extract_hotspots_precise(
                mask, unit_streamflow, severity, result.valid_time, result.normalization_method, result.ffw_mask
            )[:3])  # Test top 3 events per severity
            
            zero_quality_count = sum(1 for event in sample_events if event.get('quality_rank', 0) <= 0.001)
            
            if zero_quality_count > 0:
                error_msg = f"CRITICAL BUG: {zero_quality_count} events have zero quality scores"
                logger.error(error_msg)
                metrics.data_quality_concerns.append(error_msg)
                metrics.data_quality_excellent = False
                metrics.data_quality_good = False
            else:
                logger.info(f"✅ Quality score verification passed: {len(sample_events)} events tested")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return metrics

    def apply_spatial_filtering(self, mask: np.ndarray) -> np.ndarray:
        """Apply spatial filtering with validation"""
        if not mask.any():
            return mask
        
        try:
            start_time = time.time()
            
            # Label connected components
            labeled_array, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return mask
            
            # Count pixels in each component
            component_sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
            
            # Keep components above minimum size
            large_components = np.where(component_sizes >= self.config.min_flood_area_pixels)[0] + 1
            
            # Create filtered mask
            filtered_mask = np.isin(labeled_array, large_components)
            
            removed_count = mask.sum() - filtered_mask.sum()
            removed_components = num_features - len(large_components)
            
            filter_time = time.time() - start_time
            
            if self.config.enable_detailed_logging and removed_count > 0:
                logger.info(f"Spatial filtering results:")
                logger.info(f"  Original components: {num_features}")
                logger.info(f"  Filtered components: {len(large_components)}")
                logger.info(f"  Removed pixels: {removed_count:,}")
                logger.info(f"  Filter time: {filter_time:.3f}s")
            
            return filtered_mask
            
        except Exception as e:
            logger.warning(f"Spatial filtering failed: {e}")
            return mask
    
    def classify(self, unit_streamflow: np.ndarray, valid_time: datetime,
                ffw_polygons: Optional[gpd.GeoDataFrame] = None) -> Optional[ClassificationResult]:
        """
        Main classification method with proper thresholds and no FFW filtering
        """
        try:
            overall_start = time.time()
            
            # Input validation
            if unit_streamflow is None or unit_streamflow.size == 0:
                logger.error("Invalid input: empty or None unit_streamflow array")
                return None
            
            logger.info(f"Starting {self.config.processing_mode.value} flood classification for {valid_time}")
            logger.info(f"FIXED APPROACH: Proper thresholds, no FFW filtering, FFW enhancement only")
            
            # Step 1: Rasterize FFW polygons (for enhancement, not filtering)
            ffw_mask = None
            if ffw_polygons is not None and len(ffw_polygons) > 0:
                logger.info(f"Processing {len(ffw_polygons)} active Flash Flood Warnings for enhancement...")
                ffw_mask = self.rasterize_ffw_polygons(ffw_polygons)
            else:
                logger.info("No FFW polygons provided - enhancement disabled but detection continues")
                ffw_mask = np.zeros((self.nj, self.ni), dtype=np.uint8)
            
            # Step 2: Calculate PROPER science thresholds with physical minimums
            threshold_start = time.time()
            thresholds = self.calculate_adaptive_thresholds(unit_streamflow)
            if not thresholds:
                return None
            threshold_time = time.time() - threshold_start
            
            # Step 3: Apply flood classification WITHOUT FFW filtering
            classification_start = time.time()
            critical_mask, high_mask, moderate_mask, ffw_stats = self.apply_flood_classification_no_filter(
                unit_streamflow, thresholds, ffw_mask
            )
            classification_time = time.time() - classification_start
            
            # Step 4: Apply anti-consolidation (limit event sizes)
            if self.config.enable_size_limiting:
                logger.info("Applying anti-consolidation to prevent mega-events...")
                critical_mask = self.segment_floods_with_size_limits(critical_mask, self.config.max_event_size_km2)
                high_mask = self.segment_floods_with_size_limits(high_mask, self.config.max_event_size_km2)
                moderate_mask = self.segment_floods_with_size_limits(moderate_mask, self.config.max_event_size_km2)
            
            # Step 5: Spatial filtering based on processing mode
            filtering_start = time.time()
            if self.config.processing_mode in [ProcessingMode.PRODUCTION, ProcessingMode.VALIDATION]:
                logger.info("Applying spatial filtering...")
                critical_mask = self.apply_spatial_filtering(critical_mask)
                high_mask = self.apply_spatial_filtering(high_mask) 
                moderate_mask = self.apply_spatial_filtering(moderate_mask)
            filtering_time = time.time() - filtering_start
            
            # Step 6: Calculate statistics
            total_pixels = unit_streamflow.size
            valid_mask = (unit_streamflow != -9999.0) & np.isfinite(unit_streamflow)
            valid_pixels = valid_mask.sum()
            
            critical_count = critical_mask.sum()
            high_count = high_mask.sum()
            moderate_count = moderate_mask.sum()
            
            # Determine quality status
            quality_degraded = len(unit_streamflow[valid_mask]) < self.config.min_valid_pixels
            
            # Step 7: Create result object
            result = ClassificationResult(
                critical_mask=critical_mask,
                high_mask=high_mask,
                moderate_mask=moderate_mask,
                ffw_mask=ffw_mask,
                valid_time=valid_time,
                total_pixels=int(total_pixels),
                valid_pixels=int(valid_pixels),
                p98_value=thresholds.get('p999', thresholds.get('critical', 15.0)),  # Updated
                critical_threshold_value=thresholds['critical'],
                high_threshold_value=thresholds['high'], 
                moderate_threshold_value=thresholds['moderate'],
                critical_count=int(critical_count),
                high_count=int(high_count),
                moderate_count=int(moderate_count),
                processing_mode=self.config.processing_mode.value,
                normalization_method="Physical_Minimums_Applied",  # Updated
                ffw_intersection_applied=False,  # Changed - no filtering
                ffw_boosted_pixels=ffw_stats.get('total_in_ffw', 0),  # Enhanced events
                ffw_verified_pixels=0,  # No longer applicable
                quality_degraded=quality_degraded
            )
            
            # Step 8: Comprehensive Validation (if enabled)
            if self.config.processing_mode in [ProcessingMode.PRODUCTION, ProcessingMode.VALIDATION]:
                logger.info("Running comprehensive validation suite...")
                validation_metrics = self.comprehensive_validation(result, unit_streamflow, ffw_mask)
                result.validation_metrics = validation_metrics
                
                # Update timing in validation metrics
                validation_metrics.classification_time = classification_time
                validation_metrics.total_time = time.time() - overall_start
            
            # Coordinate accuracy validation
            if self.config.processing_mode == ProcessingMode.VALIDATION:
                logger.info("Validating coordinate accuracy...")
                coord_error = self.validate_coordinate_accuracy()
                if hasattr(result, 'validation_metrics'):
                    result.validation_metrics.coordinate_accuracy_km = coord_error
            
            # Performance tracking
            total_time = time.time() - overall_start
            self.processing_stats['last_classification_time'] = total_time
            
            # Results summary
            total_flood_pixels = critical_count + high_count + moderate_count
            logger.info(f"✅ FIXED Classification complete in {total_time:.2f}s:")
            logger.info(f"  Mode: {self.config.processing_mode.value}")
            logger.info(f"  Critical: {critical_count:,} pixels ({critical_count/total_pixels*100:.3f}%)")
            logger.info(f"  High: {high_count:,} pixels ({high_count/total_pixels*100:.3f}%)")
            logger.info(f"  Moderate: {moderate_count:,} pixels ({moderate_count/total_pixels*100:.3f}%)")
            logger.info(f"  Total flood pixels: {total_flood_pixels:,} ({total_flood_pixels/total_pixels*100:.3f}%)")
            logger.info(f"  Processing breakdown:")
            logger.info(f"    Thresholds: {threshold_time:.2f}s")
            logger.info(f"    Classification: {classification_time:.2f}s")
            logger.info(f"    Spatial filtering: {filtering_time:.2f}s")
            
            # Success validation
            detection_rate = (total_flood_pixels / total_pixels) * 100
            if detection_rate > 1.0:
                logger.warning(f"⚠️  Detection rate {detection_rate:.3f}% still high - may need threshold adjustment")
            elif detection_rate < 0.001:
                logger.warning(f"⚠️  Detection rate {detection_rate:.3f}% very low - may be too restrictive")
            else:
                logger.info(f"✅ Detection rate {detection_rate:.3f}% appears realistic for actual flooding")
            
            if ffw_stats.get('total_in_ffw', 0) > 0:
                logger.info(f"  FFW enhancement: {ffw_stats['total_in_ffw']:,} flood pixels in warning areas")
            
            if hasattr(result, 'validation_metrics') and result.validation_metrics:
                vm = result.validation_metrics
                logger.info(f"  Validation summary:")
                logger.info(f"    Total components: {vm.total_components_found}")
                logger.info(f"    Quality: {'EXCELLENT' if vm.data_quality_excellent else 'GOOD' if vm.data_quality_good else 'CONCERNS'}")
                logger.info(f"    Validation time: {vm.validation_time:.2f}s")
            
            if quality_degraded:
                logger.warning("⚠️ Quality degraded due to insufficient valid pixels")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def validate_coordinate_accuracy(self) -> float:
        """Validate coordinate accuracy against known landmarks"""
        
        test_locations = [
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
            {"name": "Denver", "lat": 39.7392, "lon": -104.9903}, 
            {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880},
            {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
            {"name": "Miami", "lat": 25.7617, "lon": -80.1918}
        ]
        
        max_error_km = 0.0
        errors = []
        
        for location in test_locations:
            try:
                # Convert known coordinates to grid indices
                row, col = self.lonlat_to_grid(location["lon"], location["lat"])
                
                # Check if within grid bounds
                if not (0 <= row < self.nj and 0 <= col < self.ni):
                    logger.warning(f"{location['name']} outside grid bounds: ({row}, {col})")
                    continue
                
                # Convert back to coordinates
                calc_lon, calc_lat = self.grid_to_lonlat(row, col)
                
                # Calculate error using Haversine formula
                error_km = self._calculate_distance_km(
                    location["lat"], location["lon"], 
                    calc_lat, calc_lon
                )
                
                errors.append(error_km)
                max_error_km = max(max_error_km, error_km)
                
                logger.info(f"Coordinate validation - {location['name']}: {error_km:.2f}km error")
                
            except Exception as e:
                logger.error(f"Coordinate validation failed for {location['name']}: {e}")
        
        if errors:
            mean_error = np.mean(errors)
            logger.info(f"Coordinate accuracy summary:")
            logger.info(f"  Mean error: {mean_error:.2f}km")
            logger.info(f"  Max error: {max_error_km:.2f}km")
            
            if max_error_km > 10.0:
                logger.error(f"❌ Coordinate accuracy POOR: {max_error_km:.2f}km max error")
            elif max_error_km > 5.0:
                logger.warning(f"⚠️  Coordinate accuracy MODERATE: {max_error_km:.2f}km max error")
            else:
                logger.info(f"✅ Coordinate accuracy GOOD: {max_error_km:.2f}km max error")
        
        return max_error_km

    def _calculate_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        from math import radians, sin, cos, sqrt, asin
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in kilometers
        earth_radius_km = 6371.0
        
        return earth_radius_km * c
        
    def segment_floods_with_size_limits(self, mask: np.ndarray, max_event_size_km2: float = 50.0) -> np.ndarray:
        """Prevent over-consolidation by limiting connected component sizes"""
        
        if not mask.any():
            return mask
        
        try:
            # Initial connected components
            labeled_array, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return mask
            
            # Find oversized components
            component_sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
            oversized_components = np.where(component_sizes > max_event_size_km2)[0] + 1
            
            if len(oversized_components) > 0:
                logger.info(f"Found {len(oversized_components)} oversized flood components (>{max_event_size_km2} km²)")
                
                # Break oversized components into sub-regions
                for comp_id in oversized_components:
                    comp_mask = (labeled_array == comp_id)
                    
                    # Use watershed or grid-based segmentation to break up large areas
                    sub_components = self._break_large_component(comp_mask, max_event_size_km2)
                    
                    # Replace original component with sub-components
                    labeled_array[comp_mask] = 0  # Clear original
                    for i, sub_comp in enumerate(sub_components):
                        new_label = num_features + i + 1
                        labeled_array[sub_comp] = new_label
                    
                    num_features += len(sub_components)
            
            # Recreate mask from size-limited components
            size_limited_mask = labeled_array > 0
            
            logger.info(f"Component size limiting results:")
            logger.info(f"  Original components: {(labeled_array > 0).sum():,} pixels")
            logger.info(f"  Size-limited components: {size_limited_mask.sum():,} pixels")
            
            return size_limited_mask
            
        except Exception as e:
            logger.error(f"Component size limiting failed: {e}")
            return mask

    def _break_large_component(self, component_mask: np.ndarray, max_size_km2: float) -> List[np.ndarray]:
        """Break a large connected component into smaller sub-regions"""
        
        # Simple grid-based approach: divide into roughly equal sub-regions
        coords = np.where(component_mask)
        min_row, max_row = coords[0].min(), coords[0].max()
        min_col, max_col = coords[1].min(), coords[1].max()
        
        # Calculate grid divisions needed
        total_area = component_mask.sum()
        num_divisions = int(np.ceil(total_area / max_size_km2))
        grid_div = int(np.ceil(np.sqrt(num_divisions)))
        
        row_step = (max_row - min_row) // grid_div + 1
        col_step = (max_col - min_col) // grid_div + 1
        
        sub_components = []
        for i in range(grid_div):
            for j in range(grid_div):
                sub_row_start = min_row + i * row_step
                sub_row_end = min_row + (i + 1) * row_step
                sub_col_start = min_col + j * col_step  
                sub_col_end = min_col + (j + 1) * col_step
                
                # Create sub-region mask
                sub_mask = np.zeros_like(component_mask)
                sub_mask[sub_row_start:sub_row_end, sub_col_start:sub_col_end] = component_mask[sub_row_start:sub_row_end, sub_col_start:sub_col_end]
                
                if sub_mask.any():
                    sub_components.append(sub_mask)
        
        return sub_components
    
    def _extract_hotspots_precise(self, mask: np.ndarray, unit_streamflow: np.ndarray, 
                            severity: str, valid_time: datetime, 
                            normalization_method: str, ffw_mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        FIXED: Precise hotspot extraction with corrected quality ranking calculation
        BUG FIX: Ensures quality_rank never becomes 0.0 for valid flood events
        """
        if not mask.any():
            return []
        
        try:
            start_time = time.time()
            
            # Label connected components with detailed logging
            labeled_array, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return []
            
            if self.config.enable_detailed_logging:
                logger.info(f"Precise extraction for {severity}: analyzing {num_features} connected components")
            
            hotspots = []
            
            # Process each connected flood event
            for label in range(1, num_features + 1):
                flood_event_mask = (labeled_array == label)
                pixel_count = flood_event_mask.sum()
                
                # Get meteorological data for this flood event
                flood_streamflow_values = unit_streamflow[flood_event_mask]
                
                # Validate flood event quality
                if len(flood_streamflow_values) < self.config.min_flood_area_pixels:
                    continue
                
                # Calculate spatial properties
                coords = np.where(flood_event_mask)
                centroid_row = coords[0].mean()
                centroid_col = coords[1].mean()
                
                # Calculate flood extent metrics
                row_extent = coords[0].max() - coords[0].min() + 1
                col_extent = coords[1].max() - coords[1].min() + 1
                flood_aspect_ratio = max(row_extent, col_extent) / min(row_extent, col_extent)
                
                # Convert to geographic coordinates
                lons, lats = self.grid_to_lonlat_vectorized(
                    np.array([int(centroid_row)]), 
                    np.array([int(centroid_col)])
                )
                
                # Enhanced meteorological analysis
                max_streamflow = float(flood_streamflow_values.max())
                mean_streamflow = float(flood_streamflow_values.mean())
                min_streamflow = float(flood_streamflow_values.min())
                streamflow_std = float(flood_streamflow_values.std())
                
                # CRITICAL BUG FIX: Correct intensity uniformity calculation
                # This was causing division by zero and negative values
                if mean_streamflow > 0.001:  # Prevent division by very small numbers
                    intensity_uniformity = max(0.0, 1.0 - (streamflow_std / mean_streamflow))
                else:
                    # For very low mean flows, base uniformity on coefficient of variation
                    intensity_uniformity = 0.1  # Conservative fallback
                
                # Calculate meteorological score (significance based on intensity and area)
                meteorological_score = max_streamflow * np.sqrt(pixel_count)
                
                # CRITICAL BUG FIX: Correct quality ranking calculation
                # OLD BUGGY CODE: quality_rank = meteorological_score * intensity_uniformity
                # This caused quality_rank = 0 when intensity_uniformity = 0
                
                # NEW FIXED CODE: Always use minimum threshold to prevent zero scores
                quality_rank = meteorological_score * max(0.1, intensity_uniformity)
                
                # Additional validation: Ensure quality_rank is never zero for valid floods
                if quality_rank <= 0.001 and max_streamflow > 0:
                    # Fallback quality based purely on meteorological significance
                    quality_rank = meteorological_score * 0.1
                    logger.debug(f"Applied fallback quality ranking for event {label}: {quality_rank:.2f}")
                
                # Spatial coherence (compactness)
                theoretical_area = np.pi * (np.sqrt(pixel_count / np.pi))**2
                compactness = min(1.0, pixel_count / theoretical_area) if theoretical_area > 0 else 0.0
                
                # Get home estimate for this location if available
                home_estimate = 0
                if self._exposure_enabled:
                    try:
                        center_row = int(centroid_row)
                        center_col = int(centroid_col)
                        if 0 <= center_row < self.nj and 0 <= center_col < self.ni:
                            home_estimate = homes(center_row, center_col)
                    except Exception as e:
                        logger.debug(f"Home estimate lookup failed for event {label}: {e}")
                        home_estimate = 0
                
                # Create comprehensive hotspot record
                hotspot = {
                    'flood_event_id': int(label),
                    'severity': severity,
                    'pixel_count': int(pixel_count),
                    'area_km2': float(pixel_count * 1.0),  # 1km² per pixel
                    'centroid_grid': (float(centroid_row), float(centroid_col)),
                    'longitude': float(lons[0]),
                    'latitude': float(lats[0]),
                    'valid_time': valid_time,
                    'classification_method': normalization_method,
                    
                    # NEW: Required metadata fields
                    'segment_id': f"{int(centroid_row)}_{int(centroid_col)}",
                    'first_seen': valid_time.isoformat() + "Z",
                    'ttl_minutes': 180,
                    'home_estimate': int(home_estimate),
                    'event_score': 0.0,  # Placeholder for scoring system
                    
                    # ENHANCED METEOROLOGICAL DATA
                    'max_streamflow': max_streamflow,
                    'mean_streamflow': mean_streamflow,
                    'min_streamflow': min_streamflow,
                    'streamflow_std': streamflow_std,
                    'intensity_range': max_streamflow - min_streamflow,
                    
                    # FIXED QUALITY METRICS
                    'meteorological_score': float(meteorological_score),
                    'intensity_uniformity': float(intensity_uniformity),
                    'spatial_compactness': float(compactness),
                    'aspect_ratio': float(flood_aspect_ratio),
                    
                    # SPATIAL METRICS
                    'extent_rows': int(row_extent),
                    'extent_cols': int(col_extent),
                    'bounding_box_area': int(row_extent * col_extent),
                    'fill_ratio': float(pixel_count / (row_extent * col_extent)),
                    
                    # RANKING DATA - FIXED CALCULATION
                    'intensity_rank': float(max_streamflow),
                    'size_rank': float(pixel_count),
                    'quality_rank': float(quality_rank)  # NOW GUARANTEED NON-ZERO
                }
                
                # FFW Enhancement (instead of filtering)
                ffw_multiplier = 1.0
                if ffw_mask is not None and ffw_mask.any():
                    # Check if event centroid is in FFW area
                    centroid_row_int = int(centroid_row)
                    centroid_col_int = int(centroid_col)
                    
                    if (0 <= centroid_row_int < self.nj and 0 <= centroid_col_int < self.ni and
                        ffw_mask[centroid_row_int, centroid_col_int] == 1):
                        ffw_multiplier = 1.5  # 50% quality boost for FFW confirmation
                        ffw_confirmed = True
                    else:
                        ffw_confirmed = False
                else:
                    ffw_confirmed = False

                # Apply FFW enhancement to quality score
                enhanced_quality_rank = quality_rank * ffw_multiplier

                # Add to hotspot record
                hotspot.update({
                    'ffw_confirmed': ffw_confirmed,
                    'ffw_multiplier': ffw_multiplier,
                    'base_quality_rank': float(quality_rank),
                    'enhanced_quality_rank': float(enhanced_quality_rank),
                    'quality_rank': float(enhanced_quality_rank)  # Use enhanced as primary
                })
                
                hotspots.append(hotspot)
            
            # Sort by comprehensive quality ranking (now meaningful)
            hotspots.sort(key=lambda x: x['quality_rank'], reverse=True)
            
            extraction_time = time.time() - start_time
            
            if self.config.enable_detailed_logging:
                logger.info(f"Precise extraction completed for {severity}:")
                logger.info(f"  Components processed: {num_features}")
                logger.info(f"  Valid events extracted: {len(hotspots)}")
                logger.info(f"  Processing time: {extraction_time:.3f}s")
                
                if hotspots:
                    top_event = hotspots[0]
                    logger.info(f"  Top event: {top_event['max_streamflow']:.3f} m³/s/km², "
                            f"{top_event['pixel_count']} pixels, quality={top_event['quality_rank']:.1f}")
                    logger.info(f"  Top event home estimate: {top_event['home_estimate']:,} homes")
                    
                    # Validation: Check that all events have non-zero quality scores
                    zero_quality_events = [e for e in hotspots if e['quality_rank'] <= 0.001]
                    if zero_quality_events:
                        logger.error(f"BUG ALERT: {len(zero_quality_events)} events still have zero quality scores!")
                    else:
                        logger.info(f"  ✅ Quality score fix verified: All {len(hotspots)} events have non-zero quality")
            
            # Log scoring warning once per run
            if not self._score_warning_logged and hotspots:
                logger.warning("⚠️  Event scores not yet computed - scoring system not implemented")
                self._score_warning_logged = True
            
            return hotspots
            
        except Exception as e:
            logger.error(f"Precise hotspot extraction failed for {severity}: {e}")
            return []
    
    def _extract_hotspots_fast(self, mask: np.ndarray, unit_streamflow: np.ndarray,
                            severity: str, valid_time: datetime, 
                            normalization_method: str) -> List[Dict[str, Any]]:
        """Fast hotspot extraction with basic validation"""
        if not mask.any():
            return []
        
        try:
            start_time = time.time()
            
            # Get all flood pixel coordinates
            flood_coords = np.where(mask)
            total_pixels = len(flood_coords[0])
            
            if total_pixels == 0:
                return []
            
            # Grid-based clustering for speed
            rows = flood_coords[0]
            cols = flood_coords[1]
            
            cluster_size = 5
            cluster_rows = rows // cluster_size
            cluster_cols = cols // cluster_size
            
            cluster_ids = cluster_rows * (self.ni // cluster_size + 1) + cluster_cols
            unique_clusters = np.unique(cluster_ids)
            
            hotspots = []
            
            for cluster_id in unique_clusters:
                cluster_mask = (cluster_ids == cluster_id)
                cluster_rows_sel = rows[cluster_mask]
                cluster_cols_sel = cols[cluster_mask]
                
                pixel_count = len(cluster_rows_sel)
                
                if pixel_count < self.config.min_flood_area_pixels:
                    continue
                
                # Calculate centroid
                centroid_row = cluster_rows_sel.mean()
                centroid_col = cluster_cols_sel.mean()
                
                # Get streamflow values for this cluster
                cluster_streamflow = unit_streamflow[cluster_rows_sel, cluster_cols_sel]
                
                # Convert to geographic coordinates
                lons, lats = self.grid_to_lonlat_vectorized(
                    np.array([int(centroid_row)]), 
                    np.array([int(centroid_col)])
                )
                
                # ADD: Get home estimate if available
                home_estimate = 0
                if self._exposure_enabled:
                    try:
                        center_row = int(centroid_row)
                        center_col = int(centroid_col)
                        if 0 <= center_row < self.nj and 0 <= center_col < self.ni:
                            home_estimate = homes(center_row, center_col)
                    except Exception as e:
                        home_estimate = 0
                
                hotspot = {
                    'flood_event_id': int(cluster_id),
                    'severity': severity,
                    'pixel_count': int(pixel_count),
                    'area_km2': float(pixel_count * 1.0),
                    'centroid_grid': (float(centroid_row), float(centroid_col)),
                    'longitude': float(lons[0]),
                    'latitude': float(lats[0]),
                    'valid_time': valid_time,
                    'classification_method': normalization_method,
                    'max_streamflow': float(cluster_streamflow.max()),
                    'mean_streamflow': float(cluster_streamflow.mean()),
                    'meteorological_score': float(cluster_streamflow.max() * np.sqrt(pixel_count)),
                    'processing_mode': 'fast_development',
                    
                    # ADD: Required metadata fields that were missing
                    'segment_id': f"{int(centroid_row)}_{int(centroid_col)}",
                    'first_seen': valid_time.isoformat() + "Z",
                    'ttl_minutes': 180,
                    'home_estimate': int(home_estimate),
                    'event_score': 0.0  # Placeholder for scoring system
                }
                
                hotspots.append(hotspot)
            
            # Sort by meteorological score
            hotspots.sort(key=lambda x: x['meteorological_score'], reverse=True)
            
            extraction_time = time.time() - start_time
            
            if self.config.enable_detailed_logging:
                logger.info(f"Fast extraction for {severity}: {len(hotspots)} events in {extraction_time:.3f}s")
            
            # Log scoring warning once per run
            if not self._score_warning_logged and hotspots:
                logger.warning("⚠️  Event scores not yet computed - scoring system not implemented")
                self._score_warning_logged = True
            
            return hotspots
            
        except Exception as e:
            logger.error(f"Fast hotspot extraction failed for {severity}: {e}")
            return []
            
        except Exception as e:
            logger.error(f"Fast hotspot extraction failed for {severity}: {e}")
            return []
    
    def extract_flood_events(self, result: ClassificationResult, 
                            unit_streamflow: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract flood events with mode-dependent processing and validation
        """
        logger.info(f"Extracting flood events using {self.config.processing_mode.value} method...")
        start_time = time.time()
        
        # Store ffw_mask for precise extraction
        self._current_ffw_mask = result.ffw_mask

        # Choose extraction method based on processing mode
        if self.config.processing_mode == ProcessingMode.DEVELOPMENT:
            extract_func = self._extract_fast_wrapper
            logger.info("Using FAST extraction for development iteration")
        else:
            extract_func = self._extract_precise_with_ffw
            logger.info("Using PRECISE extraction with comprehensive validation")
        
        # Prepare tasks for parallel execution
        tasks = [
            ('critical', result.critical_mask),
            ('high', result.high_mask),
            ('moderate', result.moderate_mask)
        ]
        
        # Extract flood events in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_severity = {
                executor.submit(
                    extract_func,
                    mask, unit_streamflow, severity, result.valid_time, result.normalization_method, 
                    result.ffw_mask if extract_func == self._extract_precise_with_ffw else None
                ): severity
                for severity, mask in tasks
            }
            
            # Collect results
            all_flood_events = {}
            for future in concurrent.futures.as_completed(future_to_severity):
                severity = future_to_severity[future]
                try:
                    events = future.result(timeout=600)  # 10 minute timeout per severity
                    all_flood_events[severity] = events
                except Exception as e:
                    logger.error(f"Failed to extract {severity} flood events: {e}")
                    all_flood_events[severity] = []
        
        # Calculate timing and statistics
        processing_time = time.time() - start_time
        total_events = sum(len(events) for events in all_flood_events.values())
        
        self.processing_stats['last_extraction_time'] = processing_time
        self.processing_stats['last_hotspot_count'] = total_events
        
        # Update validation metrics if available
        if hasattr(result, 'validation_metrics') and result.validation_metrics:
            result.validation_metrics.extraction_time = processing_time
        
        logger.info(f"✓ Extracted {total_events} flood events in {processing_time:.2f}s")
        logger.info(f"  Critical events: {len(all_flood_events.get('critical', []))}")
        logger.info(f"  High events: {len(all_flood_events.get('high', []))}")
        logger.info(f"  Moderate events: {len(all_flood_events.get('moderate', []))}")
        
        # Performance assessment
        if processing_time > 300:  # 5 minutes
            logger.warning(f"⚠️  Flood event extraction took {processing_time:.1f}s - using available time for quality")
        
        return all_flood_events
    
    def save_validation_report(self, result: ClassificationResult, flood_events: Dict, 
                                timestamp: str) -> str:
        """Save comprehensive validation report"""
        
        try:
            report_file = self.validation_dir / f"validation_report_{timestamp}.json"
            
            # Compile comprehensive validation data
            validation_report = {
                'metadata': {
                    'timestamp': timestamp,
                    'processing_mode': result.processing_mode,
                    'valid_time': result.valid_time.isoformat(),
                    'classification_method': result.normalization_method,
                    'ffw_intersection_applied': result.ffw_intersection_applied,
                    'exposure_enabled': self._exposure_enabled
                },
                'classification_results': {
                    'total_pixels': result.total_pixels,
                    'valid_pixels': result.valid_pixels,
                    'critical_pixels': result.critical_count,
                    'high_pixels': result.high_count,
                    'moderate_pixels': result.moderate_count,
                    'ffw_boosted_pixels': result.ffw_boosted_pixels,
                    'ffw_verified_pixels': result.ffw_verified_pixels
                },
                'thresholds': {
                    'p98_value': result.p98_value,
                    'critical_threshold': result.critical_threshold_value,
                    'high_threshold': result.high_threshold_value,
                    'moderate_threshold': result.moderate_threshold_value
                },
                'flood_events_summary': {
                    'critical_events': len(flood_events.get('critical', [])),
                    'high_events': len(flood_events.get('high', [])),
                    'moderate_events': len(flood_events.get('moderate', [])),
                    'total_events': sum(len(events) for events in flood_events.values())
                }
            }
            
            # Add validation metrics if available
            if hasattr(result, 'validation_metrics') and result.validation_metrics:
                vm = result.validation_metrics
                validation_report['validation_metrics'] = {
                    'total_components_found': vm.total_components_found,
                    'max_flood_area_km2': vm.max_flood_area_km2,
                    'min_flood_area_km2': vm.min_flood_area_km2,
                    'suspicious_large_floods': vm.suspicious_large_floods,
                    'max_streamflow_found': vm.max_streamflow_found,
                    'intensity_coherence_score': vm.intensity_coherence_score,
                    'gradient_anomalies': vm.gradient_anomalies,
                    'ffw_coverage_percentage': vm.ffw_coverage_percentage,
                    'method_agreement_score': vm.method_agreement_score,
                    'data_quality_excellent': vm.data_quality_excellent,
                    'data_quality_good': vm.data_quality_good,
                    'data_quality_concerns': vm.data_quality_concerns,
                    'processing_times': {
                        'classification_time': vm.classification_time,
                        'validation_time': vm.validation_time,
                        'extraction_time': vm.extraction_time,
                        'total_time': vm.total_time
                    }
                }
            
            # Add top flood events for analysis
            all_events = []
            for severity, events in flood_events.items():
                all_events.extend(events)
            
            if all_events:
                # Sort by quality and take top 10
                top_events = sorted(all_events, 
                                    key=lambda x: x.get('quality_rank', x.get('meteorological_score', 0)), 
                                    reverse=True)[:10]
                
                validation_report['top_flood_events'] = top_events
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"✓ Validation report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            return ""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            **self.processing_stats,
            'processing_mode': self.config.processing_mode.value,
            'ffw_intersection_required': self.config.require_ffw_intersection,
            'validation_enabled': self.config.processing_mode in [ProcessingMode.PRODUCTION, ProcessingMode.VALIDATION],
            'exposure_enabled': self._exposure_enabled,
            'configuration': {
                'cross_method_validation': self.config.enable_cross_method_validation,
                'spatial_validation': self.config.enable_spatial_validation,
                'intensity_validation': self.config.enable_intensity_validation,
                'detailed_logging': self.config.enable_detailed_logging
            }
        }
        
        # Add population grid stats if available
        if self._exposure_enabled:
            try:
                from exposure import get_population_stats
                stats['population_grid_stats'] = get_population_stats()
            except Exception as e:
                logger.debug(f"Could not get population stats: {e}")
        
        return stats


    # Example usage and comprehensive testing
    def main():
    """Comprehensive testing of enhanced validation classifier with population grid"""
    
    # Test all processing modes
    test_configs = [
        ("DEVELOPMENT", ClassificationConfig(
            processing_mode=ProcessingMode.DEVELOPMENT,
            require_ffw_intersection=False,
            enable_detailed_logging=True
        )),
        ("PRODUCTION", ClassificationConfig(
            processing_mode=ProcessingMode.PRODUCTION,
            require_ffw_intersection=False,
            enable_detailed_logging=True,
            enable_cross_method_validation=True,
            enable_spatial_validation=True,
            enable_intensity_validation=True
        )),
        ("VALIDATION", ClassificationConfig(
            processing_mode=ProcessingMode.VALIDATION,
            require_ffw_intersection=False,
            enable_detailed_logging=True,
            enable_cross_method_validation=True,
            enable_spatial_validation=True,
            enable_intensity_validation=True,
            enable_hydrological_validation=True,
            save_validation_plots=True
        ))
    ]
    
    for mode_name, config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing {mode_name} Mode with Enhanced Validation + Population Grid")
        print(f"{'='*60}")
        
        # Create test data
        np.random.seed(42)
        
        # Use correct grid dimensions
        if EXPOSURE_AVAILABLE:
            nj, ni = GRID_SHAPE
        else:
            nj, ni = 3500, 7000
        
        print(f"Generating test data: {ni}×{nj} = {ni*nj:,} pixels")
        
        # Create realistic streamflow data
        unit_streamflow = np.full((nj, ni), -9999.0, dtype=np.float32)
        
        # Add land areas with background flow
        land_mask = np.random.random((nj, ni)) > 0.64
        unit_streamflow[land_mask] = np.random.exponential(0.05, land_mask.sum())
        
        # Add realistic flood events
        print("Adding realistic flood events...")
        for i in range(15):  # More events for validation testing
            center_row = np.random.randint(500, nj-500)
            center_col = np.random.randint(500, ni-500)
            
            # Create flood with realistic shape and intensity
            y, x = np.ogrid[:nj, :ni]
            distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
            
            # Variable flood sizes and intensities
            flood_radius = np.random.randint(8, 40)
            flood_intensity = np.random.uniform(1.0, 5.0)
            
            flood_mask = (distance < flood_radius) & land_mask
            
            # Add intensity gradient (higher in center)
            intensity_multiplier = np.maximum(0, 1 - distance / flood_radius)
            unit_streamflow[flood_mask] += flood_intensity * intensity_multiplier[flood_mask]
        
        # Create sample FFW polygons that overlap with some floods
        if GEOPANDAS_AVAILABLE:
            from shapely.geometry import box
            
            ffw_polygons = []
            # Create FFW areas that should overlap with some of our test floods
            ffw_polygons.append(box(-100.0, 35.0, -95.0, 40.0))  # Central US
            ffw_polygons.append(box(-85.0, 32.0, -80.0, 37.0))   # Southeast
            ffw_polygons.append(box(-120.0, 45.0, -115.0, 50.0)) # Northwest
            
            ffw_gdf = gpd.GeoDataFrame(
                {
                    'id': ['FFW_001', 'FFW_002', 'FFW_003'],
                    'severity': ['Severe', 'Severe', 'Moderate']
                }, 
                geometry=ffw_polygons, 
                crs='EPSG:4326'
            )
            print(f"Created {len(ffw_gdf)} test FFW polygons")
        else:
            ffw_gdf = None
            print("Warning: geopandas not available, testing without FFW")
        
        # Test classifier
        classifier = OptimizedFlashClassifier(config)
        
        print(f"\nStarting {mode_name} classification and validation...")
        start_time = time.time()
        
        result = classifier.classify(unit_streamflow, datetime.now(), ffw_polygons=ffw_gdf)
        
        total_time = time.time() - start_time
        
        if result:
            print(f"\n{mode_name} Results:")
            print(f"  Processing time: {total_time:.2f}s")
            print(f"  Critical pixels: {result.critical_count:,}")
            print(f"  High pixels: {result.high_count:,}")
            print(f"  Moderate pixels: {result.moderate_count:,}")
            print(f"  FFW verified pixels: {result.ffw_verified_pixels:,}")
            print(f"  FFW boosted pixels: {result.ffw_boosted_pixels:,}")
            
            # Show validation results if available
            if hasattr(result, 'validation_metrics') and result.validation_metrics:
                vm = result.validation_metrics
                print(f"\n  Validation Results:")
                print(f"    Connected components found: {vm.total_components_found}")
                print(f"    Max flood area: {vm.max_flood_area_km2:.1f} km²")
                print(f"    Intensity coherence: {vm.intensity_coherence_score:.3f}")
                print(f"    Method agreement: {vm.method_agreement_score:.3f}")
                print(f"    FFW coverage: {vm.ffw_coverage_percentage:.1f}%")
                print(f"    Data quality: {'EXCELLENT' if vm.data_quality_excellent else 'GOOD' if vm.data_quality_good else 'NEEDS ATTENTION'}")
                
                if vm.data_quality_concerns:
                    print(f"    Quality concerns:")
                    for concern in vm.data_quality_concerns:
                        print(f"      - {concern}")
                
                print(f"    Validation time: {vm.validation_time:.2f}s")
            
            # Test flood event extraction
            total_flood_pixels = result.critical_count + result.high_count + result.moderate_count
            if total_flood_pixels > 0:
                print(f"\n  Extracting flood events...")
                events_start = time.time()
                
                flood_events = classifier.extract_flood_events(result, unit_streamflow)
                
                events_time = time.time() - events_start
                total_events = sum(len(event_list) for event_list in flood_events.values())
                
                print(f"  Flood Events Extracted:")
                print(f"    Critical events: {len(flood_events.get('critical', []))}")
                print(f"    High events: {len(flood_events.get('high', []))}")
                print(f"    Moderate events: {len(flood_events.get('moderate', []))}")
                print(f"    Total events: {total_events}")
                print(f"    Extraction time: {events_time:.2f}s")
                
                # Show top events with enhanced data including home estimates
                all_events = []
                for severity, events in flood_events.items():
                    all_events.extend(events)
                
                if all_events:
                    # Sort by quality ranking
                    top_events = sorted(all_events, 
                                        key=lambda x: x.get('quality_rank', x.get('meteorological_score', 0)), 
                                        reverse=True)[:3]
                    
                    print(f"\n    Top 3 flood events by quality:")
                    for i, event in enumerate(top_events):
                        print(f"      {i+1}. {event['severity'].title()}: "
                                f"{event['max_streamflow']:.3f} m³/s/km² max, "
                                f"{event['pixel_count']} pixels, "
                                f"quality={event.get('quality_rank', event.get('meteorological_score', 0)):.1f}")
                        print(f"         Segment: {event['segment_id']}, "
                                f"Homes: {event['home_estimate']:,}, "
                                f"Score: {event['event_score']:.1f}")
                        
                        # Show enhanced metrics if available
                        if 'intensity_uniformity' in event:
                            print(f"         Uniformity: {event['intensity_uniformity']:.3f}, "
                                    f"Compactness: {event.get('spatial_compactness', 0):.3f}")
                
                # Save validation report if in validation mode
                if config.processing_mode == ProcessingMode.VALIDATION:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = classifier.save_validation_report(result, flood_events, timestamp)
                    if report_file:
                        print(f"    Validation report saved: {report_file}")
            
            # Performance assessment
            print(f"\n  Performance Assessment:")
            if total_time < 60:
                print(f"    Speed: EXCELLENT (under 1 minute)")
            elif total_time < 300:
                print(f"    Speed: GOOD (under 5 minutes)")
            elif total_time < 1500:
                print(f"    Speed: ACCEPTABLE (under 25 minutes)")
            else:
                print(f"    Speed: SLOW (over 25 minutes)")
            
            # Time utilization analysis
            if hasattr(result, 'validation_metrics') and result.validation_metrics:
                vm = result.validation_metrics
                classification_pct = (vm.classification_time / total_time * 100) if total_time > 0 else 0
                validation_pct = (vm.validation_time / total_time * 100) if total_time > 0 else 0
                extraction_pct = (vm.extraction_time / total_time * 100) if total_time > 0 else 0
                
                print(f"    Time breakdown:")
                print(f"      Classification: {classification_pct:.1f}%")
                print(f"      Validation: {validation_pct:.1f}%")
                print(f"      Extraction: {extraction_pct:.1f}%")
            
            # Business impact projection
            daily_events = total_events * 144  # 144 cycles per day
            print(f"    Daily projection: {daily_events:,} events")
            
            if daily_events <= 500:
                print(f"    Business impact: EXCELLENT (manageable volume)")
            elif daily_events <= 2000:
                print(f"    Business impact: GOOD (high but manageable)")
            else:
                print(f"    Business impact: NEEDS FILTERING (too high volume)")
            
            # Population grid assessment
            if classifier._exposure_enabled:
                print(f"\n  Population Grid Integration:")
                try:
                    from exposure import get_population_stats
                    pop_stats = get_population_stats()
                    print(f"    Total homes in grid: {pop_stats['total_homes']:,}")
                    print(f"    Grid coverage: {pop_stats['non_zero_pixels']:,} pixels")
                    print(f"    Memory usage: {pop_stats['memory_mb']:.1f} MB")
                    print(f"    ✅ Population grid integration working")
                except Exception as e:
                    print(f"    ❌ Population grid stats error: {e}")
            else:
                print(f"\n  Population Grid Integration: ❌ DISABLED")
        
        else:
            print(f"❌ {mode_name} classification failed")
        
        print(f"\n{mode_name} mode test completed in {total_time:.2f}s")
        
        # Test population grid performance if available
        if EXPOSURE_AVAILABLE and classifier._exposure_enabled:
            print(f"\n  Testing population grid performance...")
            try:
                from exposure import benchmark_lookups
                benchmark_results = benchmark_lookups(50000)
                
                print(f"    Lookup performance:")
                print(f"      Samples: {benchmark_results['samples']:,}")
                print(f"      Mean time: {benchmark_results['mean_time_ms']:.4f}ms")
                print(f"      Throughput: {benchmark_results['lookups_per_second']:,.0f} lookups/sec")
                print(f"      Target (<5ms): {'✅ PASS' if benchmark_results['performance_target_met'] else '❌ FAIL'}")
                
            except Exception as e:
                print(f"    ❌ Performance benchmark failed: {e}")


    if __name__ == "__main__":
    main()