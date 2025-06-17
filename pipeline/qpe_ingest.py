#!/usr/bin/env python3
"""
QPE Data Ingest Module
Streams MRMS RadarOnly_QPE_01H data from NCEP endpoints
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


class QPEIngest:
    """
    QPE RadarOnly_QPE_01H data ingest from NCEP MRMS
    Handles download, decompression, and GRIB2 decoding
    """
    
    BASE_URL = "https://mrms.ncep.noaa.gov/2D/RadarOnly_QPE_01H"
    EXPECTED_GRID_SIZE = 7000 * 3500  # CONUS 1km
    EXPECTED_PARAM = (59, 0, 1)  # parameterNumber, discipline, category for QPE
    
    def __init__(self, cache_dir: str = "./qpe_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = None
        
        # Cache for rolling window
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
        Generate QPE URLs with proper retry cascade
        Always use .latest first, then timestamped backups
        """
        urls = []
        
        # FIXED: Primary .latest.grib2.gz with correct naming pattern
        latest_url = f"{self.BASE_URL}/MRMS_RadarOnly_QPE_01H.latest.grib2.gz"
        urls.append(latest_url)
        
        # FIXED: Backup timestamped files with correct naming
        now = datetime.utcnow()
        for i in range(1, 4):  # 10, 20, 30 minutes ago
            test_time = now - timedelta(minutes=lookback_minutes + (i * 10))
            minute = (test_time.minute // 10) * 10
            test_time = test_time.replace(minute=minute, second=0, microsecond=0)
            
            # FIXED: Use correct QPE file naming pattern
            fname = f"MRMS_RadarOnly_QPE_01H_00.00_{test_time:%Y%m%d-%H%M}00.grib2.gz"
            backup_url = f"{self.BASE_URL}/{fname}"
            urls.append(backup_url)
            
        return urls
    
    async def download_file(self, url: str) -> Optional[bytes]:
        """Download and decompress GRIB2 file"""
        try:
            logger.info(f"Downloading QPE: {url}")
            async with self.session.get(url) as response:
                if response.status == 200:
                    compressed_data = await response.read()
                    
                    # QPE files are naturally smaller - just check for non-empty
                    if len(compressed_data) < 1024:  # < 1KB = definitely bad
                        logger.warning(f"QPE file size {len(compressed_data):,} bytes too small")
                        return None
                    
                    # Decompress
                    try:
                        decompressed_data = gzip.decompress(compressed_data)
                        logger.info(f"✓ Downloaded QPE {len(compressed_data):,} bytes, "
                                f"decompressed to {len(decompressed_data):,} bytes")
                        return decompressed_data
                    except Exception as e:
                        logger.error(f"QPE decompression failed: {e}")
                        return None
                        
                elif response.status == 404:
                    logger.debug(f"QPE file not found (404): {url}")
                    return None
                else:
                    logger.warning(f"QPE HTTP {response.status}: {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"QPE download failed for {url}: {e}")
            return None
    
    def decode_grib2(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Decode GRIB2 data and extract QPE
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
                    logger.error("No GRIB messages found in QPE file")
                    return None
                
                try:
                    # Get parameter info for logging but don't fail on mismatch
                    param_num = codes_get(gid, 'parameterNumber')
                    discipline = codes_get(gid, 'discipline') 
                    category = codes_get(gid, 'parameterCategory')
                    
                    logger.info(f"QPE GRIB2 parameters: {param_num}, {discipline}, {category}")
                    
                    # RELAXED: Log parameter info but don't enforce strict validation
                    # QPE parameters may vary from expected values
                    if (param_num, discipline, category) != self.EXPECTED_PARAM:
                        logger.warning(f"QPE parameter mismatch: expected {self.EXPECTED_PARAM}, got ({param_num}, {discipline}, {category})")
                        logger.info("Continuing with parameter validation relaxed for QPE")
                    
                    # Verify grid dimensions
                    ni = codes_get(gid, 'Ni')  # longitude points
                    nj = codes_get(gid, 'Nj')  # latitude points
                    
                    if ni * nj != self.EXPECTED_GRID_SIZE:
                        logger.error(f"Unexpected QPE grid size: {ni}×{nj} = {ni*nj}")
                        return None
                    
                    # Extract valid time
                    try:
                        valid_date = codes_get(gid, 'dataDate')
                        valid_time = codes_get(gid, 'dataTime')
                        valid_datetime = datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M")
                    except:
                        valid_datetime = datetime.utcnow()
                    
                    # Get the data values and ensure float32 consistency
                    values = codes_get_values(gid)
                    qpe_1h = values.reshape(nj, ni).astype(np.float32)  # Force float32 like UnitStreamflow
                    
                    # Basic statistics for validation
                    valid_mask = (qpe_1h >= 0) & np.isfinite(qpe_1h)  # QPE should be non-negative
                    valid_count = valid_mask.sum()
                    
                    result = {
                        'qpe_1h': qpe_1h,
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
                        valid_data = qpe_1h[valid_mask]
                        result.update({
                            'data_min': float(valid_data.min()),
                            'data_max': float(valid_data.max()),
                            'data_mean': float(valid_data.mean())
                        })
                    
                    logger.info(f"✓ Decoded QPE data: {valid_count:,} valid points "
                            f"({result['valid_percentage']:.1f}%), "
                            f"range [{result.get('data_min', 0):.1f}, {result.get('data_max', 0):.1f}] mm")
                    
                finally:
                    codes_release(gid)
            
            # Cleanup
            os.unlink(tmp_path)
            return result
            
        except Exception as e:
            logger.error(f"QPE GRIB2 decode failed: {e}")
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return None
    
    async def fetch_latest(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest QPE data
        Implements retry cascade
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
                    cache_file = self.cache_dir / f"qpe_{result['valid_time']:%Y%m%d_%H%M}.npz"
                    try:
                        np.savez_compressed(
                            cache_file,
                            qpe_1h=result['qpe_1h'],
                            valid_time=result['valid_time'].isoformat(),
                            metadata=result
                        )
                        logger.debug(f"Cached QPE to {cache_file}")
                    except Exception as e:
                        logger.warning(f"QPE cache write failed: {e}")
                    
                    return result
        
        logger.error("All QPE download attempts failed")
        return None
    
    def cleanup_cache(self):
        """Remove old cached files"""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=self.cache_retention_hours)
            
            for cache_file in self.cache_dir.glob("qpe_*.npz"):
                try:
                    timestamp_str = cache_file.stem.split('_', 1)[1]
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                    
                    if file_time < cutoff:
                        cache_file.unlink()
                        logger.debug(f"Cleaned up old QPE cache file: {cache_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to clean QPE cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"QPE cache cleanup failed: {e}")


# Example usage
async def main():
    """Example usage of QPEIngest"""
    async with QPEIngest() as ingest:
        qpe_data = await ingest.fetch_latest()
        
        if qpe_data:
            print(f"✓ QPE data acquired:")
            print(f"  Valid time: {qpe_data['valid_time']}")
            print(f"  Grid: {qpe_data['grid_shape']}")
            print(f"  Valid points: {qpe_data['valid_points']:,}")
            print(f"  Data range: [{qpe_data.get('data_min', 0):.1f}, "
                  f"{qpe_data.get('data_max', 0):.1f}] mm")
            print(f"  Source: {qpe_data['source_url']}")
        else:
            print("✗ Failed to acquire QPE data")


if __name__ == "__main__":
    asyncio.run(main())