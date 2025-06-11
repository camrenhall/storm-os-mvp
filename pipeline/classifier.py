#!/usr/bin/env python3
"""
FLASH Classification Engine
Applies thresholds to UnitStreamflow data and generates severity classifications
Uses percentile-based method (temporary until ReturnPeriod available)
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """Configuration for FLASH classification"""
    
    # Feature flag for future ReturnPeriod integration
    use_return_period: bool = False
    
    # Percentile-based thresholds (temporary method)
    percentile_level: float = 98.0  # Use 98th percentile as reference
    critical_threshold: float = 0.75  # 75% of P98
    high_threshold: float = 0.50      # 50% of P98  
    moderate_threshold: float = 0.30  # 30% of P98
    
    # Minimum valid pixels for reliable percentile calculation
    min_valid_pixels: int = 50000
    
    # Spatial filtering
    min_blob_pixels: int = 4  # Minimum connected pixels for a flood blob
    
    # Quality control
    max_reasonable_streamflow: float = 100.0  # m³/s/km² sanity check


@dataclass 
class ClassificationResult:
    """Results from FLASH classification"""
    
    # Classification arrays (same shape as input grid)
    critical_mask: np.ndarray
    high_mask: np.ndarray  
    moderate_mask: np.ndarray
    
    # Metadata
    valid_time: datetime
    total_pixels: int
    valid_pixels: int
    
    # Threshold values used
    p98_value: float
    critical_threshold_value: float
    high_threshold_value: float
    moderate_threshold_value: float
    
    # Statistics
    critical_count: int
    high_count: int
    moderate_count: int
    
    # Quality flags
    normalization_method: str
    quality_degraded: bool = False


class FlashClassifier:
    """
    FLASH severity classification engine
    Implements percentile-based method as temporary solution
    """
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig()
        self.last_p98 = None  # Cache for fallback
        
    def calculate_percentile_thresholds(self, unit_streamflow: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calculate percentile-based thresholds for current conditions
        Returns threshold values or None if insufficient data
        """
        # Get valid (non-missing) data
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        valid_data = unit_streamflow[valid_mask]
        
        if len(valid_data) < self.config.min_valid_pixels:
            logger.warning(f"Insufficient valid pixels: {len(valid_data)} < {self.config.min_valid_pixels}")
            
            # Use cached P98 if available
            if self.last_p98 is not None:
                logger.info(f"Using cached P98 value: {self.last_p98:.3f}")
                p98 = self.last_p98
            else:
                logger.error("No cached P98 available, cannot classify")
                return None
        else:
            # Calculate 98th percentile
            p98 = np.percentile(valid_data, self.config.percentile_level)
            self.last_p98 = p98  # Cache for future use
        
        # Sanity check
        if p98 > self.config.max_reasonable_streamflow:
            logger.warning(f"P98 value {p98:.3f} seems unreasonably high")
        
        thresholds = {
            'p98': p98,
            'critical': p98 * self.config.critical_threshold,
            'high': p98 * self.config.high_threshold,  
            'moderate': p98 * self.config.moderate_threshold
        }
        
        logger.info(f"Calculated thresholds - P98: {p98:.3f}, "
                   f"Critical: ≥{thresholds['critical']:.3f}, "
                   f"High: ≥{thresholds['high']:.3f}, "
                   f"Moderate: ≥{thresholds['moderate']:.3f}")
        
        return thresholds
    
    def apply_thresholds(self, unit_streamflow: np.ndarray, 
                        thresholds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply threshold values to create classification masks
        Returns (critical_mask, high_mask, moderate_mask)
        """
        # Create base mask for valid data
        valid_mask = (unit_streamflow != -9999.0) & (unit_streamflow >= 0) & np.isfinite(unit_streamflow)
        
        # Apply thresholds (hierarchical)
        critical_mask = valid_mask & (unit_streamflow >= thresholds['critical'])
        high_mask = valid_mask & (unit_streamflow >= thresholds['high']) & (~critical_mask)
        moderate_mask = valid_mask & (unit_streamflow >= thresholds['moderate']) & (~critical_mask) & (~high_mask)
        
        return critical_mask, high_mask, moderate_mask
    
    def filter_blobs(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small isolated pixels, keep only connected blobs
        Uses scipy.ndimage for connected component analysis
        """
        if not mask.any():
            return mask
        
        try:
            # Label connected components
            labeled_array, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return mask
            
            # Count pixels in each blob
            component_sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
            
            # Keep only blobs with minimum size
            large_components = np.where(component_sizes >= self.config.min_blob_pixels)[0] + 1
            
            # Create filtered mask
            filtered_mask = np.isin(labeled_array, large_components)
            
            removed_count = mask.sum() - filtered_mask.sum()
            if removed_count > 0:
                logger.debug(f"Filtered out {removed_count} pixels in small blobs")
            
            return filtered_mask
            
        except Exception as e:
            logger.warning(f"Blob filtering failed: {e}")
            return mask
    
    def classify(self, unit_streamflow: np.ndarray, valid_time: datetime) -> Optional[ClassificationResult]:
        """
        Main classification method
        Returns ClassificationResult or None if classification fails
        """
        try:
            # Input validation
            if unit_streamflow is None or unit_streamflow.size == 0:
                logger.error("Invalid input: empty or None unit_streamflow array")
                return None
            
            logger.info(f"Classifying FLASH data for {valid_time}")
            
            # Calculate thresholds
            thresholds = self.calculate_percentile_thresholds(unit_streamflow)
            if not thresholds:
                return None
            
            # Apply thresholds to get raw masks
            critical_raw, high_raw, moderate_raw = self.apply_thresholds(unit_streamflow, thresholds)
            
            # Filter small blobs
            critical_mask = self.filter_blobs(critical_raw)
            high_mask = self.filter_blobs(high_raw) 
            moderate_mask = self.filter_blobs(moderate_raw)
            
            # Calculate statistics
            total_pixels = unit_streamflow.size
            valid_mask = (unit_streamflow != -9999.0) & np.isfinite(unit_streamflow)
            valid_pixels = valid_mask.sum()
            
            critical_count = critical_mask.sum()
            high_count = high_mask.sum()
            moderate_count = moderate_mask.sum()
            
            # Determine quality status
            quality_degraded = len(unit_streamflow[valid_mask]) < self.config.min_valid_pixels
            
            # Create result object
            result = ClassificationResult(
                critical_mask=critical_mask,
                high_mask=high_mask,
                moderate_mask=moderate_mask,
                valid_time=valid_time,
                total_pixels=int(total_pixels),
                valid_pixels=int(valid_pixels),
                p98_value=thresholds['p98'],
                critical_threshold_value=thresholds['critical'],
                high_threshold_value=thresholds['high'], 
                moderate_threshold_value=thresholds['moderate'],
                critical_count=int(critical_count),
                high_count=int(high_count),
                moderate_count=int(moderate_count),
                normalization_method="P98" if not self.config.use_return_period else "ReturnPeriod",
                quality_degraded=quality_degraded
            )
            
            logger.info(f"✓ Classification complete: "
                       f"Critical={critical_count}, High={high_count}, Moderate={moderate_count}")
            
            if quality_degraded:
                logger.warning("⚠️ Quality degraded due to insufficient valid pixels")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return None
    
    def get_flood_hotspots(self, result: ClassificationResult, 
                          severity_level: str = "critical") -> list[Dict[str, Any]]:
        """
        Extract flood hotspot locations from classification result
        Returns list of hotspot dictionaries with centroids and metadata
        """
        try:
            # Select appropriate mask
            if severity_level.lower() == "critical":
                mask = result.critical_mask
            elif severity_level.lower() == "high":
                mask = result.high_mask
            elif severity_level.lower() == "moderate":
                mask = result.moderate_mask
            else:
                raise ValueError(f"Invalid severity level: {severity_level}")
            
            if not mask.any():
                return []
            
            # Label connected components to find distinct hotspots
            labeled_array, num_features = ndimage.label(mask)
            
            hotspots = []
            
            for i in range(1, num_features + 1):
                # Get pixels for this hotspot
                hotspot_mask = (labeled_array == i)
                pixel_count = hotspot_mask.sum()
                
                # Calculate centroid (in grid coordinates)
                coords = np.where(hotspot_mask)
                centroid_row = coords[0].mean()
                centroid_col = coords[1].mean()
                
                hotspot = {
                    'hotspot_id': i,
                    'severity': severity_level,
                    'pixel_count': int(pixel_count),
                    'centroid_grid': (float(centroid_row), float(centroid_col)),
                    'valid_time': result.valid_time,
                    'normalization_method': result.normalization_method
                }
                
                hotspots.append(hotspot)
            
            logger.info(f"Found {len(hotspots)} {severity_level} hotspots")
            return hotspots
            
        except Exception as e:
            logger.error(f"Hotspot extraction failed: {e}")
            return []
    
    def create_combined_severity_mask(self, result: ClassificationResult) -> np.ndarray:
        """
        Create a single array with severity levels encoded as integers
        0=no flood, 1=moderate, 2=high, 3=critical
        """
        combined = np.zeros_like(result.moderate_mask, dtype=np.uint8)
        
        combined[result.moderate_mask] = 1
        combined[result.high_mask] = 2
        combined[result.critical_mask] = 3
        
        return combined


# Example usage and testing
def main():
    """Example usage of FlashClassifier"""
    
    # Create sample data (simulate FLASH grid)
    np.random.seed(42)  # Reproducible results
    nj, ni = 3500, 7000  # CONUS grid
    
    # Simulate unit streamflow data with realistic patterns
    unit_streamflow = np.full((nj, ni), -9999.0, dtype=np.float32)  # Start with missing values
    
    # Add some land areas with low background flow
    land_mask = np.random.random((nj, ni)) > 0.64  # ~36% land coverage (matches observations)
    unit_streamflow[land_mask] = np.random.exponential(0.05, land_mask.sum())  # Exponential distribution
    
    # Add some flood hotspots
    for _ in range(5):
        center_row = np.random.randint(500, nj-500)
        center_col = np.random.randint(500, ni-500)
        
        # Create a blob of elevated streamflow
        y, x = np.ogrid[:nj, :ni]
        distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
        flood_mask = (distance < 50) & land_mask
        unit_streamflow[flood_mask] += np.random.uniform(2.0, 8.0)
    
    # Run classification
    classifier = FlashClassifier()
    result = classifier.classify(unit_streamflow, datetime.utcnow())
    
    if result:
        print(f"✓ FLASH Classification Results:")
        print(f"  Valid time: {result.valid_time}")
        print(f"  Valid pixels: {result.valid_pixels:,} / {result.total_pixels:,} "
              f"({result.valid_pixels/result.total_pixels*100:.1f}%)")
        print(f"  P98 threshold: {result.p98_value:.3f} m³/s/km²")
        print(f"  Critical pixels: {result.critical_count:,}")
        print(f"  High pixels: {result.high_count:,}")
        print(f"  Moderate pixels: {result.moderate_count:,}")
        print(f"  Quality degraded: {result.quality_degraded}")
        
        # Extract hotspots
        critical_hotspots = classifier.get_flood_hotspots(result, "critical")
        high_hotspots = classifier.get_flood_hotspots(result, "high")
        
        print(f"  Critical hotspots: {len(critical_hotspots)}")
        print(f"  High hotspots: {len(high_hotspots)}")
        
        # Show sample hotspot
        if critical_hotspots:
            sample = critical_hotspots[0]
            print(f"  Sample critical hotspot: {sample['pixel_count']} pixels at grid {sample['centroid_grid']}")
    else:
        print("✗ Classification failed")


if __name__ == "__main__":
    main()