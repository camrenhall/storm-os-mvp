#!/usr/bin/env python3
"""
Grid Alignment and Spatial Processing
Handles coordinate transformations, FFW polygon rasterization, and spatial intersections
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    import rasterio
    from rasterio import features
    import pyproj
    SPATIAL_LIBS_AVAILABLE = True
except ImportError:
    raise ImportError("Spatial libraries required: pip install geopandas rasterio pyproj")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GridDefinition:
    """Definition of the MRMS FLASH 1km CONUS grid"""
    
    # Grid dimensions (matches MRMS FLASH)
    nj: int = 3500  # latitude points (rows)
    ni: int = 7000  # longitude points (columns)
    
    # Spatial extent (approximate CONUS bounds for 1km grid)
    # These should be refined based on actual MRMS grid definition
    west: float = -130.0   # Western boundary (longitude)
    east: float = -60.0    # Eastern boundary (longitude) 
    south: float = 20.0    # Southern boundary (latitude)
    north: float = 55.0    # Northern boundary (latitude)
    
    # CRS
    crs: str = "EPSG:4326"  # WGS84 Geographic
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.lon_res = (self.east - self.west) / self.ni
        self.lat_res = (self.north - self.south) / self.nj
        
        # Create coordinate arrays
        self.lons = np.linspace(self.west + self.lon_res/2, 
                               self.east - self.lon_res/2, self.ni)
        self.lats = np.linspace(self.south + self.lat_res/2, 
                               self.north - self.lat_res/2, self.nj)
        
        # Meshgrid for coordinate lookups
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)


class GridProcessor:
    """
    Handles spatial operations on the MRMS FLASH grid
    Provides coordinate transformations and polygon rasterization
    """
    
    def __init__(self, grid_def: GridDefinition = None):
        self.grid = grid_def or GridDefinition()
        
        # Cache for expensive operations
        self._affine_transform = None
        self._grid_bounds = None
        
    @property
    def affine_transform(self):
        """Get rasterio affine transform for the grid"""
        if self._affine_transform is None:
            try:
                from rasterio.transform import from_bounds
                self._affine_transform = from_bounds(
                    self.grid.west, self.grid.south, 
                    self.grid.east, self.grid.north,
                    self.grid.ni, self.grid.nj
                )
            except ImportError:
                # Fallback manual calculation
                self._affine_transform = self._calculate_manual_transform()
                
        return self._affine_transform
    
    def _calculate_manual_transform(self):
        """Manual affine transform calculation if rasterio unavailable"""
        # Simple affine transform: [pixel_width, 0, west, 0, -pixel_height, north]
        return [
            self.grid.lon_res,  # pixel width
            0.0,               # rotation
            self.grid.west,    # west coordinate
            0.0,               # rotation  
            -self.grid.lat_res, # pixel height (negative for north-up)
            self.grid.north    # north coordinate
        ]
    
    def grid_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid indices to longitude/latitude coordinates
        Returns (longitude, latitude)
        """
        if not (0 <= row < self.grid.nj and 0 <= col < self.grid.ni):
            raise ValueError(f"Grid indices out of bounds: ({row}, {col})")
        
        lon = self.grid.west + (col + 0.5) * self.grid.lon_res
        lat = self.grid.north - (row + 0.5) * self.grid.lat_res
        
        return lon, lat
    
    def lonlat_to_grid(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert longitude/latitude to grid indices
        Returns (row, col) - may be out of bounds
        """
        col = int((lon - self.grid.west) / self.grid.lon_res)
        row = int((self.grid.north - lat) / self.grid.lat_res)
        
        return row, col
    
    def create_grid_bounds_polygon(self) -> 'shapely.geometry.Polygon':
        """Create a polygon representing the grid bounds"""
        from shapely.geometry import box
        return box(self.grid.west, self.grid.south, self.grid.east, self.grid.north)
    
    def rasterize_polygons(self, polygons_gdf: gpd.GeoDataFrame, 
                          fill_value: int = 1, background: int = 0) -> np.ndarray:
        """
        Rasterize polygon GeoDataFrame to the FLASH grid
        Returns binary mask array where polygons = fill_value
        """
        try:
            if len(polygons_gdf) == 0:
                logger.info("No polygons to rasterize")
                return np.full((self.grid.nj, self.grid.ni), background, dtype=np.uint8)
            
            # Ensure polygons are in the same CRS as grid
            if polygons_gdf.crs != self.grid.crs:
                polygons_gdf = polygons_gdf.to_crs(self.grid.crs)
            
            # Create list of (geometry, value) tuples for rasterization
            geom_value_pairs = [(geom, fill_value) for geom in polygons_gdf.geometry 
                               if geom is not None and geom.is_valid]
            
            if not geom_value_pairs:
                logger.warning("No valid geometries found for rasterization")
                return np.full((self.grid.nj, self.grid.ni), background, dtype=np.uint8)
            
            # Rasterize using rasterio
            rasterized = features.rasterize(
                geom_value_pairs,
                out_shape=(self.grid.nj, self.grid.ni),
                transform=self.affine_transform,
                fill=background,
                dtype=np.uint8
            )
            
            polygon_pixel_count = (rasterized == fill_value).sum()
            logger.info(f"✓ Rasterized {len(geom_value_pairs)} polygons to {polygon_pixel_count:,} pixels")
            
            return rasterized
            
        except Exception as e:
            logger.error(f"Polygon rasterization failed: {e}")
            return np.full((self.grid.nj, self.grid.ni), background, dtype=np.uint8)
    
    def mask_grid_by_polygons(self, grid_array: np.ndarray, 
                             polygons_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Apply polygon mask to grid array (keep only pixels inside polygons)
        Returns masked version of input array
        """
        try:
            if len(polygons_gdf) == 0:
                # No polygons = no valid area
                return np.zeros_like(grid_array)
            
            # Create polygon mask
            polygon_mask = self.rasterize_polygons(polygons_gdf, fill_value=1, background=0)
            
            # Apply mask
            masked_array = np.where(polygon_mask == 1, grid_array, 0)
            
            masked_pixels = (masked_array != 0).sum()
            logger.info(f"Applied polygon mask: {masked_pixels:,} pixels retained")
            
            return masked_array
            
        except Exception as e:
            logger.error(f"Grid masking failed: {e}")
            return grid_array  # Return unmasked on error
    
    def extract_pixel_locations(self, mask: np.ndarray, 
                               max_pixels: int = 1000) -> List[Dict[str, Any]]:
        """
        Extract geographic locations of pixels from a boolean mask
        Returns list of pixel location dictionaries
        """
        try:
            pixel_coords = np.where(mask)
            pixel_count = len(pixel_coords[0])
            
            if pixel_count == 0:
                return []
            
            # Limit number of pixels if too many
            if pixel_count > max_pixels:
                logger.info(f"Limiting pixel extraction to {max_pixels} of {pixel_count} pixels")
                indices = np.random.choice(pixel_count, max_pixels, replace=False)
                rows = pixel_coords[0][indices]
                cols = pixel_coords[1][indices]
            else:
                rows = pixel_coords[0]
                cols = pixel_coords[1]
            
            # Convert to geographic coordinates
            locations = []
            for row, col in zip(rows, cols):
                lon, lat = self.grid_to_lonlat(int(row), int(col))
                
                location = {
                    'grid_row': int(row),
                    'grid_col': int(col),
                    'longitude': float(lon),
                    'latitude': float(lat),
                    'geometry': Point(lon, lat)
                }
                locations.append(location)
            
            logger.info(f"Extracted {len(locations)} pixel locations")
            return locations
            
        except Exception as e:
            logger.error(f"Pixel location extraction failed: {e}")
            return []
    
    def check_grid_alignment(self, test_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Test grid alignment by converting coordinates back and forth
        Useful for validating grid definition accuracy
        """
        results = {
            'test_points': len(test_points),
            'max_error_meters': 0.0,
            'mean_error_meters': 0.0,
            'alignment_ok': True
        }
        
        try:
            errors = []
            
            for lon, lat in test_points:
                # Convert to grid and back
                row, col = self.lonlat_to_grid(lon, lat)
                
                # Check if in bounds
                if 0 <= row < self.grid.nj and 0 <= col < self.grid.ni:
                    lon_back, lat_back = self.grid_to_lonlat(row, col)
                    
                    # Calculate error in meters (approximate)
                    lon_error_m = (lon_back - lon) * 111320 * np.cos(np.radians(lat))
                    lat_error_m = (lat_back - lat) * 110540
                    total_error_m = np.sqrt(lon_error_m**2 + lat_error_m**2)
                    
                    errors.append(total_error_m)
            
            if errors:
                results['max_error_meters'] = float(np.max(errors))
                results['mean_error_meters'] = float(np.mean(errors))
                results['alignment_ok'] = results['max_error_meters'] < 1000  # < 1km error
            
            logger.info(f"Grid alignment check: max error {results['max_error_meters']:.1f}m, "
                       f"mean error {results['mean_error_meters']:.1f}m")
            
        except Exception as e:
            logger.error(f"Grid alignment check failed: {e}")
            results['alignment_ok'] = False
        
        return results


# Example usage and testing
def main():
    """Example usage of GridProcessor"""
    
    # Create grid processor
    processor = GridProcessor()
    
    print(f"MRMS FLASH Grid Definition:")
    print(f"  Dimensions: {processor.grid.ni} × {processor.grid.nj}")
    print(f"  Bounds: {processor.grid.west}°W to {processor.grid.east}°W, "
          f"{processor.grid.south}°N to {processor.grid.north}°N")
    print(f"  Resolution: {processor.grid.lon_res:.4f}° × {processor.grid.lat_res:.4f}°")
    
    # Test coordinate conversions
    test_points = [
        (-100.0, 40.0),  # Center of CONUS
        (-74.0, 40.7),   # New York City
        (-118.2, 34.1),  # Los Angeles
        (-95.4, 29.8),   # Houston
        (-87.6, 41.9),   # Chicago
    ]
    
    print(f"\nTesting coordinate conversions:")
    for lon, lat in test_points:
        row, col = processor.lonlat_to_grid(lon, lat)
        
        if 0 <= row < processor.grid.nj and 0 <= col < processor.grid.ni:
            lon_back, lat_back = processor.grid_to_lonlat(row, col)
            print(f"  ({lon:.1f}, {lat:.1f}) → grid ({row}, {col}) → "
                  f"({lon_back:.3f}, {lat_back:.3f})")
        else:
            print(f"  ({lon:.1f}, {lat:.1f}) → OUT OF BOUNDS ({row}, {col})")
    
    # Test grid alignment
    alignment_results = processor.check_grid_alignment(test_points)
    print(f"\nGrid alignment: {'✓ OK' if alignment_results['alignment_ok'] else '✗ POOR'}")
    print(f"  Max error: {alignment_results['max_error_meters']:.1f} meters")
    
    # Test polygon rasterization with sample data
    try:
        # Create a sample polygon (rough outline of Texas)
        from shapely.geometry import Polygon
        texas_approx = Polygon([
            (-106.5, 25.8), (-93.5, 25.8), (-93.5, 36.5), 
            (-103.0, 36.5), (-106.5, 32.0), (-106.5, 25.8)
        ])
        
        sample_gdf = gpd.GeoDataFrame({'name': ['Texas']}, 
                                     geometry=[texas_approx], 
                                     crs='EPSG:4326')
        
        # Rasterize
        texas_mask = processor.rasterize_polygons(sample_gdf)
        texas_pixels = texas_mask.sum()
        
        print(f"\nSample polygon rasterization:")
        print(f"  Texas approximation: {texas_pixels:,} pixels")
        print(f"  Coverage: {texas_pixels / (processor.grid.ni * processor.grid.nj) * 100:.1f}% of grid")
        
    except Exception as e:
        print(f"\nPolygon test failed: {e}")


if __name__ == "__main__":
    main()