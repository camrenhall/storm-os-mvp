#!/usr/bin/env python3
"""
NWS Flash Flood Warning Client
Fetches active Flash Flood Warnings from NWS API and processes to GeoDataFrame
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import shape
    GEOPANDAS_AVAILABLE = True
except ImportError:
    raise ImportError("geopandas required: pip install geopandas")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NWSFlashFloodWarnings:
    """
    Fetch and process NWS Flash Flood Warnings for legal cover and confirmation
    Provides both raw API data and processed GeoDataFrame
    """
    
    API_URL = "https://api.weather.gov/alerts/active"
    USER_AGENT = "StormLeadEngine/1.0 (FLASH MVP)"
    
    def __init__(self, cache_dir: str = "./ffw_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = None
        
        # Cache settings
        self.cache_timeout_minutes = 5  # NWS suggests 60s cache, be conservative
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': self.USER_AGENT}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_active_ffw(self, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch active Flash Flood Warnings from NWS API
        Returns raw API response data
        """
        if not self.session:
            raise RuntimeError("Must use within async context manager")
        
        # Check cache first
        cache_file = self.cache_dir / "active_ffw.json"
        if use_cache and cache_file.exists():
            try:
                cache_age = datetime.utcnow() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(minutes=self.cache_timeout_minutes):
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    logger.debug(f"Using cached FFW data (age: {cache_age})")
                    return cached_data
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Fetch from API
        params = {'event': 'Flash Flood Warning'}
        
        try:
            logger.info("Fetching active Flash Flood Warnings from NWS API")
            async with self.session.get(self.API_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    if 'features' not in data:
                        logger.error("Invalid API response: missing 'features' key")
                        return None
                    
                    feature_count = len(data['features'])
                    logger.info(f"✓ Found {feature_count} active Flash Flood Warnings")
                    
                    # Cache the response
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        logger.debug(f"Cached FFW data to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Cache write failed: {e}")
                    
                    return data
                    
                elif response.status == 429:
                    logger.warning("Rate limited by NWS API (429)")
                    return None
                else:
                    logger.error(f"NWS API error: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to fetch FFW data: {e}")
            return None
    
    def process_to_geodataframe(self, api_data: Dict[str, Any]) -> Optional[gpd.GeoDataFrame]:
        """
        Convert NWS API response to GeoDataFrame for spatial operations
        Returns GeoDataFrame with cleaned geometries and metadata
        """
        try:
            features = api_data.get('features', [])
            if not features:
                logger.info("No active Flash Flood Warnings to process")
                return gpd.GeoDataFrame()  # Empty GeoDataFrame
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(features)
            
            if len(gdf) == 0:
                return gdf
            
            # Clean and standardize the data
            gdf = gdf.copy()
            
            # Ensure CRS is set (NWS uses WGS84)
            if gdf.crs is None:
                gdf.set_crs('EPSG:4326', inplace=True)
            
            # Extract key properties for easier access
            properties_df = gpd.GeoDataFrame([feat['properties'] for feat in features])
            
            # Add useful computed fields
            gdf['warning_id'] = gdf['id']
            gdf['event_type'] = properties_df.get('event', 'Flash Flood Warning')
            gdf['severity'] = properties_df.get('severity', 'Unknown')
            gdf['urgency'] = properties_df.get('urgency', 'Unknown')
            gdf['certainty'] = properties_df.get('certainty', 'Unknown')
            
            # Parse times
            try:
                gdf['effective_time'] = gpd.pd.to_datetime(properties_df.get('effective'), utc=True)
                gdf['expires_time'] = gpd.pd.to_datetime(properties_df.get('expires'), utc=True)
                gdf['sent_time'] = gpd.pd.to_datetime(properties_df.get('sent'), utc=True)
            except Exception as e:
                logger.warning(f"Time parsing failed: {e}")
            
            # Add areas (useful for prioritization)
            try:
                # Reproject to equal-area for area calculation (US Albers)
                gdf_albers = gdf.to_crs('EPSG:5070')  # CONUS Albers
                gdf['area_km2'] = gdf_albers.geometry.area / 1e6  # Convert m² to km²
            except Exception as e:
                logger.warning(f"Area calculation failed: {e}")
                gdf['area_km2'] = None
            
            # Clean geometries
            gdf['geometry'] = gdf['geometry'].buffer(0)  # Fix any topology issues
            
            logger.info(f"✓ Processed {len(gdf)} FFW polygons to GeoDataFrame")
            return gdf
            
        except Exception as e:
            logger.error(f"GeoDataFrame processing failed: {e}")
            return None
    
    async def get_active_warnings(self, as_geodataframe: bool = True) -> Optional[gpd.GeoDataFrame]:
        """
        High-level method to get active Flash Flood Warnings
        Returns GeoDataFrame by default, or raw API data if as_geodataframe=False
        """
        api_data = await self.fetch_active_ffw()
        if not api_data:
            return None
        
        if not as_geodataframe:
            return api_data
        
        return self.process_to_geodataframe(api_data)
    
    def filter_warnings_by_geometry(self, warnings_gdf: gpd.GeoDataFrame, 
                                   target_geometry) -> gpd.GeoDataFrame:
        """
        Filter warnings that intersect with target geometry
        Useful for checking if flood pixels are within warning areas
        """
        try:
            if len(warnings_gdf) == 0:
                return warnings_gdf
            
            # Ensure same CRS
            if hasattr(target_geometry, 'crs') and target_geometry.crs != warnings_gdf.crs:
                target_geometry = target_geometry.to_crs(warnings_gdf.crs)
            
            # Find intersections
            intersecting = warnings_gdf[warnings_gdf.geometry.intersects(target_geometry)]
            
            logger.debug(f"Found {len(intersecting)} warnings intersecting target geometry")
            return intersecting
            
        except Exception as e:
            logger.error(f"Geometry filtering failed: {e}")
            return gpd.GeoDataFrame()
    
    def cleanup_cache(self):
        """Remove old cached files"""
        try:
            cache_file = self.cache_dir / "active_ffw.json"
            if cache_file.exists():
                cache_age = datetime.utcnow() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age > timedelta(hours=1):  # Remove if older than 1 hour
                    cache_file.unlink()
                    logger.debug("Cleaned up old FFW cache")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")


# Example usage
async def main():
    """Example usage of NWSFlashFloodWarnings"""
    async with NWSFlashFloodWarnings() as ffw_client:
        # Fetch active warnings
        warnings_gdf = await ffw_client.get_active_warnings()
        
        if warnings_gdf is not None:
            if len(warnings_gdf) > 0:
                print(f"✓ Active Flash Flood Warnings: {len(warnings_gdf)}")
                print(f"  CRS: {warnings_gdf.crs}")
                print(f"  Columns: {list(warnings_gdf.columns)}")
                
                # Show summary stats
                if 'area_km2' in warnings_gdf.columns:
                    total_area = warnings_gdf['area_km2'].sum()
                    print(f"  Total warned area: {total_area:.0f} km²")
                
                # Show sample warning
                sample = warnings_gdf.iloc[0]
                print(f"  Sample warning: {sample.get('warning_id', 'Unknown ID')}")
                print(f"    Severity: {sample.get('severity', 'Unknown')}")
                print(f"    Expires: {sample.get('expires_time', 'Unknown')}")
            else:
                print("✓ No active Flash Flood Warnings")
        else:
            print("✗ Failed to fetch Flash Flood Warnings")


if __name__ == "__main__":
    asyncio.run(main())