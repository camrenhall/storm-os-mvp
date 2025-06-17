#!/usr/bin/env python3
"""
FLASH Data Ingest Module
Streams MRMS FLASH UnitStreamflow data from NCEP endpoints
"""

import asyncio
import aiohttp
import logging
import gzip
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from eccodes import (
        codes_grib_new_from_file, codes_get, codes_get_values, 
        codes_release, CodesInternalError
    )
    ECCODES_AVAILABLE = True
except ImportError:
    raise ImportError("eccodes required: pip install eccodes")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlashIngest:
    """
    FLASH UnitStreamflow data ingest from NCEP MRMS
    Handles download, decompression, and GRIB2 decoding
    """
    
    BASE_URL = "https://mrms.ncep.noaa.gov/2D/FLASH/CREST_MAXUNITSTREAMFLOW"
    EXPECTED_GRID_SIZE = 7000 * 3500  # CONUS 1km
    EXPECTED_PARAM = (0, 209, 12)  # parameterNumber, discipline, category
    
    def __init__(self, cache_dir: str = "./flash_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = None
        
        # Cache for rolling window (keep last 1 hour)
        self.cache_retention_hours = 1
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={'User-Agent': 'StormLeadEngine/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def generate_urls(self, lookback_minutes: int = 30) -> list[str]:
        """
        Generate FLASH URLs with proper retry cascade
        Always use .latest first, then timestamped backups
        """
        urls = []
        
        # Primary: .latest.grib2.gz (recommended by senior engineer)
        latest_url = f"{self.BASE_URL}/MRMS_FLASH_CREST_MAXUNITSTREAMFLOW.latest.grib2.gz"
        urls.append(latest_url)
        
        # Backup: Try recent timestamped files (up to 3 cycles back)
        now = datetime.now(datetime.UTC)
        for i in range(1, 4):  # 10, 20, 30 minutes ago
            test_time = now - timedelta(minutes=lookback_minutes + (i * 10))
            minute = (test_time.minute // 10) * 10
            test_time = test_time.replace(minute=minute, second=0, microsecond=0)
            
            fname = f"MRMS_FLASH_CREST_MAXUNITSTREAMFLOW_00.00_{test_time:%Y%m%d-%H%M}00.grib2.gz"
            backup_url = f"{self.BASE_URL}/{fname}"
            urls.append(backup_url)
            
        return urls
    
    async def download_file(self, url: str) -> Optional[bytes]:
        """Download and decompress GRIB2 file"""
        try:
            logger.info(f"Downloading: {url}")
            async with self.session.get(url) as response:
                if response.status == 200:
                    compressed_data = await response.read()
                    
                    # Size check (expect 4-6 MB compressed)
                    if len(compressed_data) < 1024 * 1024:  # < 1MB
                        logger.warning(f"File size {len(compressed_data):,} bytes seems small")
                        return None
                    
                    # Decompress
                    try:
                        decompressed_data = gzip.decompress(compressed_data)
                        logger.info(f"✓ Downloaded {len(compressed_data):,} bytes, "
                                  f"decompressed to {len(decompressed_data):,} bytes")
                        return decompressed_data
                    except Exception as e:
                        logger.error(f"Decompression failed: {e}")
                        return None
                        
                elif response.status == 404:
                    logger.debug(f"File not found (404): {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status}: {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return None
    
    def decode_grib2(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Decode GRIB2 data and extract UnitStreamflow
        Returns dict with grid data and metadata
        """
        try:
            # Write to temporary file for eccodes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            
            result = None
            
            with open(tmp_path, 'rb') as f:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    logger.error("No GRIB messages found")
                    return None
                
                try:
                    # Verify this is the expected parameter
                    param_num = codes_get(gid, 'parameterNumber')
                    discipline = codes_get(gid, 'discipline') 
                    category = codes_get(gid, 'parameterCategory')
                    
                    if (param_num, discipline, category) != self.EXPECTED_PARAM:
                        logger.error(f"Unexpected parameter: {param_num}, {discipline}, {category}")
                        return None
                    
                    # Verify grid dimensions
                    ni = codes_get(gid, 'Ni')  # longitude points
                    nj = codes_get(gid, 'Nj')  # latitude points
                    
                    if ni * nj != self.EXPECTED_GRID_SIZE:
                        logger.error(f"Unexpected grid size: {ni}×{nj} = {ni*nj}")
                        return None
                    
                    # Extract valid time
                    try:
                        valid_date = codes_get(gid, 'dataDate')
                        valid_time = codes_get(gid, 'dataTime')
                        valid_datetime = datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M")
                    except:
                        valid_datetime = datetime.now(datetime.UTC)
                    
                    # Get the data values
                    values = codes_get_values(gid)
                    unit_streamflow = values.reshape(nj, ni)  # Reshape to 2D grid
                    
                    # CRITICAL: Guard against blank/corrupt FLASH files
                    active_pixels = (unit_streamflow > 0.1).sum()
                    if active_pixels < 100:
                        raise ValueError(f"FLASH frame appears empty/corrupt: only {active_pixels} active pixels")

                    logger.debug(f"FLASH validation: {active_pixels:,} active pixels detected")
                    
                    # Basic statistics for validation
                    valid_mask = unit_streamflow != -9999.0
                    valid_count = valid_mask.sum()
                    
                    result = {
                        'unit_streamflow': unit_streamflow,
                        'valid_time': valid_datetime,
                        'grid_shape': (nj, ni),
                        'valid_points': int(valid_count),
                        'total_points': int(ni * nj),
                        'valid_percentage': float(valid_count / (ni * nj) * 100),
                        'parameter_info': {
                            'number': param_num,
                            'discipline': discipline,
                            'category': category
                        }
                    }
                    
                    if valid_count > 0:
                        valid_data = unit_streamflow[valid_mask]
                        result.update({
                            'data_min': float(valid_data.min()),
                            'data_max': float(valid_data.max()),
                            'data_mean': float(valid_data.mean())
                        })
                    
                    logger.info(f"✓ Decoded FLASH data: {valid_count:,} valid points "
                              f"({result['valid_percentage']:.1f}%), "
                              f"range [{result.get('data_min', 0):.3f}, {result.get('data_max', 0):.3f}]")
                    
                finally:
                    codes_release(gid)
            
            # Cleanup
            os.unlink(tmp_path)
            return result
            
        except Exception as e:
            logger.error(f"GRIB2 decode failed: {e}")
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return None
    
    async def fetch_latest(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest FLASH UnitStreamflow data
        Implements retry cascade as recommended
        """
        if not self.session:
            raise RuntimeError("Must use within async context manager")
        
        urls = self.generate_urls()
        
        for i, url in enumerate(urls):
            data = await self.download_file(url)
            if data:
                result = self.decode_grib2(data)
                if result:
                    result['source_url'] = url
                    result['retry_attempt'] = i
                    
                    # Cache the file for potential reuse
                    cache_file = self.cache_dir / f"flash_{result['valid_time']:%Y%m%d_%H%M}.npz"
                    try:
                        np.savez_compressed(
                            cache_file,
                            unit_streamflow=result['unit_streamflow'],
                            valid_time=result['valid_time'].isoformat(),
                            metadata=result
                        )
                        logger.debug(f"Cached to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Cache write failed: {e}")
                    
                    return result
        
        logger.error("All download attempts failed")
        return None
    
    def cleanup_cache(self):
        """Remove old cached files (keep last 1 hour)"""
        try:
            cutoff = datetime.now(datetime.UTC) - timedelta(hours=self.cache_retention_hours)
            
            for cache_file in self.cache_dir.glob("flash_*.npz"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = cache_file.stem.split('_', 1)[1]
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                    
                    if file_time < cutoff:
                        cache_file.unlink()
                        logger.debug(f"Cleaned up old cache file: {cache_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to clean cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")


# Example usage
async def main():
    """Example usage of FlashIngest"""
    async with FlashIngest() as ingest:
        # Fetch latest data
        flash_data = await ingest.fetch_latest()
        
        if flash_data:
            print(f"✓ FLASH data acquired:")
            print(f"  Valid time: {flash_data['valid_time']}")
            print(f"  Grid: {flash_data['grid_shape']}")
            print(f"  Valid points: {flash_data['valid_points']:,}")
            print(f"  Data range: [{flash_data.get('data_min', 0):.3f}, "
                  f"{flash_data.get('data_max', 0):.3f}]")
            print(f"  Source: {flash_data['source_url']}")
        else:
            print("✗ Failed to acquire FLASH data")


if __name__ == "__main__":
    asyncio.run(main())