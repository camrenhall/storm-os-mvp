#!/usr/bin/env python3
"""
Flood Classification Engine - MVP Production Version
Implements spec-compliant fixed thresholds (2/5/10 m³/s/km²) with exact scoring system
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

# Import population grid module with proper error handling
try:
    from .exposure import homes, initialize_population_grid, is_initialized, GRID_SHAPE
    EXPOSURE_AVAILABLE = True
except ImportError:
    try:
        # Fallback to pipeline-level import
        from exposure import homes, initialize_population_grid, is_initialized, GRID_SHAPE
        EXPOSURE_AVAILABLE = True
    except ImportError:
        EXPOSURE_AVAILABLE = False
        print("Warning: exposure module not available, home estimates disabled")

# Enhanced geopandas import with detailed debugging
GEOPANDAS_AVAILABLE = False
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from rasterio import features
    GEOPANDAS_AVAILABLE = True
    print(f"✅ GeoPandas successfully imported: {gpd.__version__}")
except ImportError as e:
    print(f"❌ GeoPandas import failed in flood_classifier: {e}")
    
    # Try individual component imports for debugging
    components = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('shapely', 'shapely.geometry'),
        ('fiona', 'fiona'),
        ('pyproj', 'pyproj'),
        ('rasterio', 'rasterio')
    ]
    
    print("Component import test:")
    for name, module in components:
        try:
            __import__(module)
            print(f"  ✅ {name}: OK")
        except ImportError as comp_e:
            print(f"  ❌ {name}: {comp_e}")
    
    print("Warning: geopandas not available, FFW integration disabled")
except Exception as e:
    print(f"❌ GeoPandas unexpected error: {e}")
    print("Warning: geopandas not available, FFW integration disabled")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    VALIDATION = "validation"


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for quality assurance"""
    total_components_found: int = 0
    components_merged: int = 0
    components_split: int = 0
    components_filtered: int = 0
    max_flood_area_km2: float = 0.0
    min_flood_area_km2: float = 0.0
    mean_flood_area_km2: float = 0.0
    suspicious_large_floods: int = 0
    suspicious_small_floods: int = 0
    max_streamflow_found: float = 0.0
    min_streamflow_found: float = 0.0
    intensity_coherence_score: float = 0.0
    gradient_anomalies: int = 0
    ffw_coverage_percentage: float = 0.0
    floods_outside_ffw: int = 0
    ffw_areas_without_floods: int = 0
    grid_vs_components_centroid_diff: List[float] = field(default_factory=list)
    method_agreement_score: float = 0.0
    data_quality_excellent: bool = False
    data_quality_good: bool = False
    data_quality_concerns: List[str] = field(default_factory=list)
    classification_time: float = 0.0
    validation_time: float = 0.0
    extraction_time: float = 0.0
    total_time: float = 0.0


@dataclass
class ClassificationConfig:
    """Configuration for flood classification"""
    # FFW handling - bonus system, not filtering
    require_ffw_intersection: bool = False
    enable_ffw_enhancement: bool = True
    ffw_quality_multiplier: float = 1.5
    
    # Processing mode
    processing_mode: ProcessingMode = ProcessingMode.PRODUCTION
    enable_detailed_logging: bool = True
    
    # Fixed thresholds (spec-compliant)
    use_fixed_thresholds: bool = True
    dead_data_threshold_percent: float = 1.0  # <1% active = dead data day
    
    # Enhanced validation settings
    enable_cross_method_validation: bool = True
    enable_spatial_validation: bool = True
    enable_intensity_validation: bool = True
    enable_hydrological_validation: bool = True
    
    # Quality control thresholds
    max_reasonable_flood_area_km2: float = 1000.0
    min_reasonable_flood_area_km2: float = 0.25
    max_reasonable_streamflow: float = 100.0
    max_intensity_gradient: float = 10.0
    
    # Spatial filtering
    min_flood_area_pixels: int = 9  # Changed from 2 to 9 (3x3 area)
    min_valid_pixels: int = 50000
    
    # Performance settings
    max_workers: int = 4
    save_validation_plots: bool = False
    validation_output_dir: str = "./validation_output"


@dataclass 
class ClassificationResult:
    """Classification results with comprehensive validation"""
    critical_mask: np.ndarray
    high_mask: np.ndarray  
    moderate_mask: np.ndarray
    valid_time: datetime
    total_pixels: int
    valid_pixels: int
    p98_value: float
    critical_threshold_value: float
    high_threshold_value: float
    moderate_threshold_value: float
    critical_count: int
    high_count: int
    moderate_count: int
    processing_mode: str
    normalization_method: str
    validation_metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    ffw_mask: Optional[np.ndarray] = None
    ffw_intersection_applied: bool = False
    ffw_boosted_pixels: int = 0
    ffw_verified_pixels: int = 0
    quality_degraded: bool = False


class FloodClassifier:
    """
    Production Flood Classification Engine
    Implements spec-compliant 2/5/10 m³/s/km² thresholds with exact scoring
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
        
        # Performance tracking
        self.processing_stats = {
            'last_classification_time': 0.0,
            'last_extraction_time': 0.0,
            'last_validation_time': 0.0,
            'last_hotspot_count': 0
        }
        
        # Score computation tracking
        self._score_warning_logged = False
    
    def _initialize_exposure_data(self):
        """Check if population exposure data is available (don't auto-initialize)"""
        if EXPOSURE_AVAILABLE:
            # Just check if it's already initialized, don't try to initialize here
            if is_initialized():
                logger.info("✅ Population exposure grid already initialized")
                self._exposure_enabled = True
            else:
                logger.info("ℹ️  Population exposure grid not yet initialized (will be done by pipeline)")
                self._exposure_enabled = False
        else:
            logger.warning("⚠️  Population exposure module not available")
            self._exposure_enabled = False
        
    def _setup_grid_coordinates(self):
        """Pre-compute coordinate transformation arrays"""
        if EXPOSURE_AVAILABLE:
            self.nj, self.ni = GRID_SHAPE
            logger.info(f"Using exposure module grid dimensions: {self.ni}×{self.nj}")
        else:
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
        
        logger.debug("Grid coordinates initialized")
        
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

    def calculate_spec_thresholds(self, unit_streamflow: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calculate spec-compliant FIXED thresholds: 2/5/10 m³/s/km²
        Uses adaptive fallback only when <1% of CONUS pixels exceed 0.1 m³/s/km² (dead data day)
        """
        # Get valid data
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        valid_data = unit_streamflow[valid_mask]
        
        if len(valid_data) < self.config.min_valid_pixels:
            logger.error("Insufficient valid pixels for threshold calculation")
            return None
        
        # Check for "dead data day" - less than 1% of pixels exceed 0.1 m³/s/km²
        active_pixels = (valid_data > 0.1).sum()
        active_percentage = (active_pixels / len(valid_data)) * 100
        
        if active_percentage < self.config.dead_data_threshold_percent:
            logger.warning(f"Dead data day detected ({active_percentage:.3f}% active pixels) - using adaptive percentile fallback")
            
            # FIXED: Higher fallback thresholds with spec minimums
            p98 = np.percentile(valid_data, 98.0)
            p95 = np.percentile(valid_data, 95.0) 
            p90 = np.percentile(valid_data, 90.0)
            
            fallback_thresholds = {
                'critical': max(p98, 10.0),  # Clamp at spec minimum
                'high': max(p95, 5.0),
                'moderate': max(p90, 2.0)
            }
            
            logger.info(f"FALLBACK Adaptive Thresholds (clamped at spec minimums):")
            logger.info(f"  Critical: ≥{fallback_thresholds['critical']:.3f} m³/s/km²")
            logger.info(f"  High: ≥{fallback_thresholds['high']:.3f} m³/s/km²")
            logger.info(f"  Moderate: ≥{fallback_thresholds['moderate']:.3f} m³/s/km²")
            
            return fallback_thresholds
        
        # SPEC-COMPLIANT FIXED THRESHOLDS
        thresholds = {
            'critical': 10.0,    # m³/s/km²
            'high': 5.0,
            'moderate': 2.0
        }
        
        logger.info(f"FIXED Flood Thresholds (Spec-Compliant):")
        logger.info(f"  Critical: ≥{thresholds['critical']:.1f} m³/s/km²")
        logger.info(f"  High: ≥{thresholds['high']:.1f} m³/s/km²")
        logger.info(f"  Moderate: ≥{thresholds['moderate']:.1f} m³/s/km²")
        logger.info(f"  Active data: {active_percentage:.3f}% of pixels")
        
        return thresholds

    def calculate_event_score(self, max_streamflow: float, in_ffw: bool, 
                            qpe_1h: float, home_count: int) -> int:
        """
        Calculate exact event score per spec:
        - streamflow_pts = min(int(flow/15*50), 50) (linear 0-50 across 0-15 m³/s/km²)
        - ffw_pts = 25 if in_ffw else 0
        - qpe_pts = 15 if qpe ≥ 50 mm else 0
        - exposure_pts = 10 if homes ≥ 500 else 0
        - TOTAL CLAMPED TO 100 MAX
        """
        # FIXED: Streamflow component - linear 0-50 across 0-15 m³/s/km²
        streamflow_pts = min(int(max_streamflow / 15.0 * 50), 50)
        
        # FFW bonus
        ffw_pts = 25 if in_ffw else 0
        
        # QPE bonus
        qpe_pts = 15 if qpe_1h >= 50.0 else 0
        
        # Exposure bonus
        exposure_pts = 10 if home_count >= 500 else 0
        
        # FIXED: Clamp total score to 100 max per spec
        total_score = min(streamflow_pts + ffw_pts + qpe_pts + exposure_pts, 100)
        
        if self.config.enable_detailed_logging:
            logger.debug(f"Event score: streamflow={streamflow_pts}, ffw={ffw_pts}, qpe={qpe_pts}, exposure={exposure_pts} = {total_score} (clamped)")
        
        return total_score

    def rasterize_ffw_polygons(self, ffw_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Rasterize FFW polygons for enhancement (not filtering)"""
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
            
            # Rasterize
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
            
            return rasterized
            
        except Exception as e:
            logger.error(f"FFW polygon rasterization failed: {e}")
            return np.zeros((self.nj, self.ni), dtype=np.uint8)

    def apply_flood_classification(self, unit_streamflow: np.ndarray, 
                                  thresholds: Dict[str, float],
                                  ffw_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """Apply flood classification with spec thresholds (no FFW filtering)"""
        
        # Create base valid data mask
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        
        # Apply flood thresholds (no FFW filtering)
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
            
            logger.info(f"FFW Enhancement Statistics:")
            logger.info(f"  Critical floods in FFW areas: {ffw_stats['enhanced_critical']:,}")
            logger.info(f"  High floods in FFW areas: {ffw_stats['enhanced_high']:,}")
            logger.info(f"  Total floods in FFW areas: {ffw_stats['total_in_ffw']:,}")
        
        # Log total detections
        total_floods = critical_mask.sum() + high_mask.sum() + moderate_mask.sum()
        total_pixels = unit_streamflow.size
        detection_rate = (total_floods / total_pixels) * 100
        
        logger.info(f"Flood Detection Results:")
        logger.info(f"  Critical floods: {critical_mask.sum():,} pixels")
        logger.info(f"  High floods: {high_mask.sum():,} pixels") 
        logger.info(f"  Moderate floods: {moderate_mask.sum():,} pixels")
        logger.info(f"  Total flood pixels: {total_floods:,}")
        logger.info(f"  Detection rate: {detection_rate:.3f}% of grid")
        
        return critical_mask, high_mask, moderate_mask, ffw_stats

    def apply_spatial_filtering(self, mask: np.ndarray) -> np.ndarray:
        """Apply spatial filtering to remove small isolated pixels"""
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
            filter_time = time.time() - start_time
            
            if self.config.enable_detailed_logging and removed_count > 0:
                logger.info(f"Spatial filtering: removed {removed_count:,} pixels in {filter_time:.3f}s")
            
            return filtered_mask
            
        except Exception as e:
            logger.warning(f"Spatial filtering failed: {e}")
            return mask

    def classify(self, unit_streamflow: np.ndarray, valid_time: datetime,
            ffw_polygons: Optional[gpd.GeoDataFrame] = None,
            qpe_1h_grid: Optional[np.ndarray] = None) -> Optional[ClassificationResult]:
        """
        Main classification method with spec-compliant thresholds
        """
        try:
            overall_start = time.time()
            
            # Input validation
            if unit_streamflow is None or unit_streamflow.size == 0:
                logger.error("Invalid input: empty or None unit_streamflow array")
                return None
            
            logger.info(f"Starting {self.config.processing_mode.value} flood classification for {valid_time}")
            logger.info(f"SPEC-COMPLIANT: Fixed 2/5/10 m³/s/km² thresholds with FFW enhancement")
            
            # Step 1: Rasterize FFW polygons (for enhancement, not filtering) - with graceful fallback
            ffw_mask = None
            if ffw_polygons is not None and len(ffw_polygons) > 0:
                try:
                    logger.info(f"Processing {len(ffw_polygons)} active Flash Flood Warnings for enhancement...")
                    ffw_mask = self.rasterize_ffw_polygons(ffw_polygons)
                except Exception as e:
                    logger.warning(f"FFW processing failed, continuing without FFW enhancement: {e}")
                    ffw_mask = np.zeros((self.nj, self.ni), dtype=np.uint8)
            else:
                logger.info("No FFW polygons provided - enhancement disabled but detection continues")
                ffw_mask = np.zeros((self.nj, self.ni), dtype=np.uint8)
            
            # Step 2: Calculate SPEC-COMPLIANT thresholds
            threshold_start = time.time()
            thresholds = self.calculate_spec_thresholds(unit_streamflow)
            if not thresholds:
                return None
            threshold_time = time.time() - threshold_start
            
            # Step 3: Apply flood classification (no FFW filtering)
            classification_start = time.time()
            critical_mask, high_mask, moderate_mask, ffw_stats = self.apply_flood_classification(
                unit_streamflow, thresholds, ffw_mask
            )
            classification_time = time.time() - classification_start
            
            # Step 4: Spatial filtering
            filtering_start = time.time()
            if self.config.processing_mode in [ProcessingMode.PRODUCTION, ProcessingMode.VALIDATION]:
                logger.info("Applying spatial filtering...")
                critical_mask = self.apply_spatial_filtering(critical_mask)
                high_mask = self.apply_spatial_filtering(high_mask) 
                moderate_mask = self.apply_spatial_filtering(moderate_mask)
            filtering_time = time.time() - filtering_start
            
            # Step 5: Calculate statistics
            total_pixels = unit_streamflow.size
            valid_mask = (unit_streamflow != -9999.0) & np.isfinite(unit_streamflow)
            valid_pixels = valid_mask.sum()
            
            critical_count = critical_mask.sum()
            high_count = high_mask.sum()
            moderate_count = moderate_mask.sum()
            
            # Determine quality status
            quality_degraded = len(unit_streamflow[valid_mask]) < self.config.min_valid_pixels
            
            # Step 6: Create result object
            result = ClassificationResult(
                critical_mask=critical_mask,
                high_mask=high_mask,
                moderate_mask=moderate_mask,
                ffw_mask=ffw_mask,
                valid_time=valid_time,
                total_pixels=int(total_pixels),
                valid_pixels=int(valid_pixels),
                p98_value=float(np.percentile(unit_streamflow[valid_mask], 98.0)) if valid_mask.any() else 0.0,  # FIXED: Actual 98th percentile
                critical_threshold_value=thresholds['critical'],
                high_threshold_value=thresholds['high'], 
                moderate_threshold_value=thresholds['moderate'],
                critical_count=int(critical_count),
                high_count=int(high_count),
                moderate_count=int(moderate_count),
                processing_mode=self.config.processing_mode.value,
                normalization_method="Fixed_Spec_Thresholds",
                ffw_intersection_applied=False,
                ffw_boosted_pixels=ffw_stats.get('total_in_ffw', 0),
                ffw_verified_pixels=0,
                quality_degraded=quality_degraded
            )
            
            # Performance tracking
            total_time = time.time() - overall_start
            self.processing_stats['last_classification_time'] = total_time
            
            # Results summary
            total_flood_pixels = critical_count + high_count + moderate_count
            logger.info(f"✅ Classification complete in {total_time:.2f}s:")
            logger.info(f"  Critical: {critical_count:,} pixels")
            logger.info(f"  High: {high_count:,} pixels")
            logger.info(f"  Moderate: {moderate_count:,} pixels")
            logger.info(f"  Total flood pixels: {total_flood_pixels:,}")
            logger.info(f"  Processing breakdown: thresholds={threshold_time:.2f}s, classification={classification_time:.2f}s, filtering={filtering_time:.2f}s")
            
            if ffw_stats.get('total_in_ffw', 0) > 0:
                logger.info(f"  FFW enhancement: {ffw_stats['total_in_ffw']:,} flood pixels in warning areas")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def extract_flood_events(self, result: ClassificationResult, 
                            unit_streamflow: np.ndarray,
                            qpe_1h_grid: Optional[np.ndarray] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract flood events with proper event scoring
        """
        logger.info(f"Extracting flood events using {self.config.processing_mode.value} method...")
        start_time = time.time()
        
        # Choose extraction method based on processing mode
        if self.config.processing_mode == ProcessingMode.DEVELOPMENT:
            extract_func = self._extract_hotspots_fast
        else:
            extract_func = self._extract_hotspots_precise
        
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
                    mask, 
                    unit_streamflow, 
                    severity, 
                    result.valid_time, 
                    result.normalization_method, 
                    result.ffw_mask,
                    qpe_1h_grid  # FIXED M1: Pass QPE grid through
                ): severity
                for severity, mask in tasks
            }
            
            # Collect results
            all_flood_events = {}
            for future in concurrent.futures.as_completed(future_to_severity):
                severity = future_to_severity[future]
                try:
                    events = future.result(timeout=600)
                    all_flood_events[severity] = events
                except Exception as e:
                    logger.error(f"Failed to extract {severity} flood events: {e}")
                    all_flood_events[severity] = []
        
        # Calculate timing and statistics
        processing_time = time.time() - start_time
        total_events = sum(len(events) for events in all_flood_events.values())
        
        self.processing_stats['last_extraction_time'] = processing_time
        self.processing_stats['last_hotspot_count'] = total_events
        
        logger.info(f"✓ Extracted {total_events} flood events in {processing_time:.2f}s")
        logger.info(f"  Critical events: {len(all_flood_events.get('critical', []))}")
        logger.info(f"  High events: {len(all_flood_events.get('high', []))}")
        logger.info(f"  Moderate events: {len(all_flood_events.get('moderate', []))}")
        
        return all_flood_events

    def _extract_hotspots_precise(self, mask: np.ndarray, unit_streamflow: np.ndarray, 
                             severity: str, valid_time: datetime, 
                             normalization_method: str, ffw_mask: Optional[np.ndarray] = None,
                             qpe_1h_grid: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Precise hotspot extraction with spec-compliant event scoring
        """
        if not mask.any():
            return []
        
        try:
            start_time = time.time()
            
            # Label connected components
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
               
                # FIXED: Intensity uniformity calculation to prevent zero quality scores
                if mean_streamflow > 0.001:
                    intensity_uniformity = max(0.0, 1.0 - (streamflow_std / mean_streamflow))
                else:
                    intensity_uniformity = 0.1  # Conservative fallback
                
                # Calculate meteorological score
                meteorological_score = max_streamflow * np.sqrt(pixel_count)
                
                # FIXED: Quality ranking calculation - guaranteed non-zero
                quality_rank = meteorological_score * max(0.1, intensity_uniformity)
                
                # Additional validation: Ensure quality_rank is never zero for valid floods
                if quality_rank <= 0.001 and max_streamflow > 0:
                    quality_rank = meteorological_score * 0.1
                    logger.debug(f"Applied fallback quality ranking for event {label}: {quality_rank:.2f}")
                
                # Get home estimate for this location
                home_estimate = 0
                if self._exposure_enabled:
                    try:
                        center_row = int(centroid_row)
                        center_col = int(centroid_col)
                        
                        # DIAGNOSTIC: Log the lookup attempt
                        lon, lat = self.grid_to_lonlat(center_row, center_col)
                        logger.info(f"🏠 Event {label}: grid({center_row},{center_col}) -> lat/lon({lat:.4f},{lon:.4f})")
                        
                        if 0 <= center_row < self.nj and 0 <= center_col < self.ni:
                            home_estimate = homes(center_row, center_col)
                            logger.info(f"🏠 Event {label}: {home_estimate} homes found")
                        else:
                            logger.warning(f"🏠 Event {label}: OUT OF BOUNDS")
                            
                    except Exception as e:
                        logger.error(f"🏠 Event {label}: lookup failed - {e}")
                        home_estimate = 0
                
                # Check FFW confirmation for scoring
                ffw_confirmed = False
                if ffw_mask is not None and ffw_mask.any():
                    centroid_row_int = int(centroid_row)
                    centroid_col_int = int(centroid_col)
                    
                    if (0 <= centroid_row_int < self.nj and 0 <= centroid_col_int < self.ni and
                        ffw_mask[centroid_row_int, centroid_col_int] == 1):
                        ffw_confirmed = True
                        
                # Get QPE value for this location if available
                qpe_1h_value = 0.0
                if qpe_1h_grid is not None:
                    try:
                        center_row_int = int(centroid_row)
                        center_col_int = int(centroid_col)
                        if (0 <= center_row_int < self.nj and 0 <= center_col_int < self.ni):
                            qpe_1h_value = float(qpe_1h_grid[center_row_int, center_col_int])
                            if qpe_1h_value < 0 or not np.isfinite(qpe_1h_value):
                                qpe_1h_value = 0.0
                    except Exception as e:
                        logger.debug(f"QPE lookup failed for event {label}: {e}")
                        qpe_1h_value = 0.0
                
                # CRITICAL: Calculate spec-compliant event_score with QPE
                event_score = self.calculate_event_score(
                    max_streamflow, 
                    ffw_confirmed,
                    qpe_1h_value,  # FIXED: Real QPE value
                    home_estimate
                )
                
                # VALIDATION: Ensure event_score > 0 for all valid events
                if event_score <= 0:
                    logger.error(f"Event {label} has zero event_score - this should not happen!")
                    # Force minimum score based on streamflow
                    event_score = max(1, int(max_streamflow))
                
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
                    
                    # REQUIRED: Spec-compliant metadata fields
                    'segment_id': f"{int(centroid_row)}_{int(centroid_col)}",
                    'first_seen': valid_time.isoformat() + "Z",
                    'ttl_minutes': 180,
                    'home_estimate': int(home_estimate),
                    'event_score': int(event_score),  # FIXED: Proper scoring
                    
                    # FIXED: Add missing QPE and FFW fields
                    'qpe_1h': float(qpe_1h_value),  # QPE data for downstream analysis
                    'ffw_confirmed': bool(ffw_confirmed),  # FFW boolean for auditability
                    
                    # Enhanced meteorological data
                    'max_streamflow': max_streamflow,
                    'mean_streamflow': mean_streamflow,
                    'min_streamflow': min_streamflow,
                    'streamflow_std': streamflow_std,
                    'intensity_range': max_streamflow - min_streamflow,
                    
                    # Fixed quality metrics
                    'meteorological_score': float(meteorological_score),
                    'intensity_uniformity': float(intensity_uniformity),
                    'quality_rank': float(quality_rank),
                    
                    # FFW enhancement data (keeping for backward compatibility)
                    'ffw_multiplier': 1.5 if ffw_confirmed else 1.0,
                    
                    # Ranking data
                    'intensity_rank': float(max_streamflow),
                    'size_rank': float(pixel_count)
                }
                
                hotspots.append(hotspot)
            
            # Sort by event_score (spec-compliant ranking)
            hotspots.sort(key=lambda x: x['event_score'], reverse=True)
            
            extraction_time = time.time() - start_time
            
            if self.config.enable_detailed_logging:
                logger.info(f"Precise extraction completed for {severity}:")
                logger.info(f"  Components processed: {num_features}")
                logger.info(f"  Valid events extracted: {len(hotspots)}")
                logger.info(f"  Processing time: {extraction_time:.3f}s")
                
                if hotspots:
                    top_event = hotspots[0]
                    logger.info(f"  Top event: {top_event['max_streamflow']:.3f} m³/s/km², "
                                f"{top_event['pixel_count']} pixels, score={top_event['event_score']}")
                    logger.info(f"  Top event home estimate: {top_event['home_estimate']:,} homes")
                    
                    # VALIDATION: Check that all events have non-zero scores
                    zero_score_events = [e for e in hotspots if e['event_score'] <= 0]
                    if zero_score_events:
                        logger.error(f"BUG ALERT: {len(zero_score_events)} events still have zero scores!")
                    else:
                        logger.info(f"  ✅ All {len(hotspots)} events have proper event_score > 0")
            
            return hotspots
            
        except Exception as e:
            logger.error(f"Precise hotspot extraction failed for {severity}: {e}")
            return []

    def _extract_hotspots_fast(self, mask: np.ndarray, unit_streamflow: np.ndarray,
                          severity: str, valid_time: datetime, 
                          normalization_method: str, ffw_mask: Optional[np.ndarray] = None,
                          qpe_1h_grid: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Fast hotspot extraction with proper event scoring"""
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
                max_streamflow = float(cluster_streamflow.max())
                
                # Convert to geographic coordinates
                lons, lats = self.grid_to_lonlat_vectorized(
                    np.array([int(centroid_row)]), 
                    np.array([int(centroid_col)])
                )
                
                # Get home estimate if available
                home_estimate = 0
                if self._exposure_enabled:
                    try:
                        center_row = int(centroid_row)
                        center_col = int(centroid_col)
                        
                        if self.config.enable_detailed_logging:
                            lons, lats = self.grid_to_lonlat_vectorized(
                                np.array([center_row]), np.array([center_col])
                            )
                            logger.info(f"🏠 Event {cluster_id}: grid({center_row},{center_col}) -> lat/lon({lats[0]:.4f},{lons[0]:.4f})")
                        
                        if 0 <= center_row < self.nj and 0 <= center_col < self.ni:
                            home_estimate = homes(center_row, center_col)
                            if self.config.enable_detailed_logging:
                                logger.info(f"🏠 Event {cluster_id}: {home_estimate} homes found")
                        else:
                            if self.config.enable_detailed_logging:
                                logger.warning(f"🏠 Event {cluster_id}: OUT OF BOUNDS")
                                
                    except Exception as e:
                        if self.config.enable_detailed_logging:
                            logger.error(f"🏠 Event {cluster_id}: lookup failed - {e}")
                        home_estimate = 0
                
                # Check FFW confirmation
                ffw_confirmed = False
                if ffw_mask is not None and ffw_mask.any():
                    center_row_int = int(centroid_row)
                    center_col_int = int(centroid_col)
                    if (0 <= center_row_int < self.nj and 0 <= center_col_int < self.ni and
                        ffw_mask[center_row_int, center_col_int] == 1):
                        ffw_confirmed = True
                
                # FIXED M1: Get real QPE value from grid
                qpe_1h_value = 0.0
                if qpe_1h_grid is not None:
                    try:
                        center_row_int = int(centroid_row)
                        center_col_int = int(centroid_col)
                        if (0 <= center_row_int < self.nj and 0 <= center_col_int < self.ni):
                            qpe_1h_value = float(qpe_1h_grid[center_row_int, center_col_int])
                            if qpe_1h_value < 0 or not np.isfinite(qpe_1h_value):
                                qpe_1h_value = 0.0
                    except Exception as e:
                        qpe_1h_value = 0.0
                
                # CRITICAL: Calculate proper event_score with real QPE
                event_score = self.calculate_event_score(
                    max_streamflow,
                    ffw_confirmed,
                    qpe_1h_value,  # FIXED M1: Real QPE value, not 0.0
                    home_estimate
                )
                
                # Ensure score > 0
                if event_score <= 0:
                    event_score = max(1, int(max_streamflow))
                
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
                    'max_streamflow': max_streamflow,
                    'mean_streamflow': float(cluster_streamflow.mean()),
                    'meteorological_score': float(max_streamflow * np.sqrt(pixel_count)),
                    'processing_mode': 'fast_development',
                    
                    # REQUIRED: Spec-compliant metadata fields
                    'segment_id': f"{int(centroid_row)}_{int(centroid_col)}",
                    'first_seen': valid_time.isoformat() + "Z",
                    'ttl_minutes': 180,
                    'home_estimate': int(home_estimate),
                    'event_score': int(event_score),  # FIXED: Proper scoring with QPE
                    
                    # FIXED: Add missing QPE and FFW fields
                    'qpe_1h': float(qpe_1h_value),  # QPE data for downstream analysis
                    'ffw_confirmed': bool(ffw_confirmed)  # FFW boolean for auditability
                }
                
                hotspots.append(hotspot)
            
            # Sort by event_score
            hotspots.sort(key=lambda x: x['event_score'], reverse=True)
            
            extraction_time = time.time() - start_time
            
            if self.config.enable_detailed_logging:
                logger.info(f"Fast extraction for {severity}: {len(hotspots)} events in {extraction_time:.3f}s")
                if hotspots and qpe_1h_grid is not None:
                    qpe_events = [e for e in hotspots if e.get('qpe_1h', 0) >= 50.0]
                    logger.info(f"  Events with QPE ≥50mm: {len(qpe_events)}")
            
            return hotspots
            
        except Exception as e:
            logger.error(f"Fast hotspot extraction failed for {severity}: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            **self.processing_stats,
            'processing_mode': self.config.processing_mode.value,
            'ffw_intersection_required': self.config.require_ffw_intersection,
            'exposure_enabled': self._exposure_enabled,
            'thresholds_mode': 'fixed_spec_compliant',
            'configuration': {
                'use_fixed_thresholds': self.config.use_fixed_thresholds,
                'dead_data_threshold_percent': self.config.dead_data_threshold_percent,
                'enable_ffw_enhancement': self.config.enable_ffw_enhancement,
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


def main():
    """Test the new spec-compliant flood classifier"""
    
    # Test configuration
    config = ClassificationConfig(
        processing_mode=ProcessingMode.PRODUCTION,
        use_fixed_thresholds=True,
        enable_detailed_logging=True,  # Reduce verbosity
        enable_ffw_enhancement=True,
        min_flood_area_pixels=9  # B7 fix: 3x3 minimum
    )
    
    classifier = FloodClassifier(config)
    
    print(f"Testing Spec-Compliant Flood Classifier")
    print(f"Fixed thresholds: 2/5/10 m³/s/km²")
    
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
    
    # TESTING QPE: Create test QPE grid with high rainfall areas
    qpe_test_grid = np.zeros((nj, ni), dtype=np.float32)
    
    # Add test flood events at spec thresholds
    print("Adding test flood events...")
    for i, (intensity, qpe_val) in enumerate([(2.5, 30.0), (6.0, 60.0), (12.0, 75.0)]):  # Test QPE scoring
        center_row = np.random.randint(500, nj-500)
        center_col = np.random.randint(500, ni-500)
        
        # Create 5x5 flood (meets min_flood_area_pixels=9)
        unit_streamflow[center_row-2:center_row+3, center_col-2:center_col+3] = intensity
        qpe_test_grid[center_row-2:center_row+3, center_col-2:center_col+3] = qpe_val
        print(f"  Event {i+1}: {intensity:.1f} m³/s/km², QPE {qpe_val}mm/h at ({center_row}, {center_col})")
    
    # Test classifier
    print(f"\nStarting classification...")
    start_time = time.time()
    
    result = classifier.classify(unit_streamflow, datetime.now(), qpe_1h_grid=qpe_test_grid)
    
    total_time = time.time() - start_time
    
    if result:
        print(f"\nResults:")
        print(f"  Processing time: {total_time:.2f}s")
        print(f"  Critical: {result.critical_count:,}, High: {result.high_count:,}, Moderate: {result.moderate_count:,}")
        
        # Test event extraction with QPE
        total_flood_pixels = result.critical_count + result.high_count + result.moderate_count
        if total_flood_pixels > 0:
            print(f"Extracting flood events with QPE scoring...")
            
            flood_events = classifier.extract_flood_events(result, unit_streamflow, qpe_test_grid)
            
            total_events = sum(len(event_list) for event_list in flood_events.values())
            print(f"Total events extracted: {total_events}")
            
            # Show top events
            all_events = []
            for severity, events in flood_events.items():
                all_events.extend(events)
            
            if all_events:
                top_events = sorted(all_events, key=lambda x: x['event_score'], reverse=True)[:3]
                
                print(f"Top 3 events by event_score:")
                for i, event in enumerate(top_events):
                    score = event['event_score']
                    streamflow = event['max_streamflow']
                    qpe = event.get('qpe_1h', 0)
                    print(f"  {i+1}. Score: {score}, Streamflow: {streamflow:.1f} m³/s/km², QPE: {qpe:.1f}mm/h")
                
                # VALIDATION: Check scoring constraints and QPE functionality
                max_score = max(e['event_score'] for e in all_events)
                min_score = min(e['event_score'] for e in all_events)
                zero_score_events = [e for e in all_events if e['event_score'] <= 0]
                over_100_events = [e for e in all_events if e['event_score'] > 100]
                qpe_bonus_events = [e for e in all_events if e.get('qpe_1h', 0) >= 50.0]
                
                print(f"\nValidation:")
                print(f"  Score range: {min_score} - {max_score}")
                print(f"  Zero scores: {len(zero_score_events)} ({'✓' if len(zero_score_events) == 0 else '✗'})")
                print(f"  Over 100 scores: {len(over_100_events)} ({'✓' if len(over_100_events) == 0 else '✗'})")
                print(f"  QPE bonus events (≥50mm): {len(qpe_bonus_events)} ({'✓' if len(qpe_bonus_events) > 0 else '✗'})")
                
                # Test QPE scoring specifically
                if qpe_bonus_events:
                    sample_qpe_event = qpe_bonus_events[0]
                    expected_qpe_bonus = 15
                    print(f"  Sample QPE event score: {sample_qpe_event['event_score']} (should include +15 for QPE)")
                
                if (len(zero_score_events) == 0 and len(over_100_events) == 0 and len(qpe_bonus_events) > 0):
                    print(f"✅ All validation checks passed including QPE scoring")
                else:
                    print(f"❌ Some validation checks failed")
            
        print(f"✅ Demo completed successfully")
        
    else:
        print(f"❌ Classification failed")


if __name__ == "__main__":
    main()