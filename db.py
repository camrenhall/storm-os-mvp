#!/usr/bin/env python3
"""
Database connection and operations for Flood-Lead Intelligence MVP
Handles asyncpg pool management and bulk operations to Neon
"""

import os
import asyncpg
import asyncio
import logging
from typing import List, Tuple, Any, Optional
from datetime import datetime
from dateutil.parser import isoparse

logger = logging.getLogger(__name__)

# Database connection pool - module level singleton
_db_pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    """Get or create database connection pool"""
    global _db_pool
    
    if _db_pool is None:
        env = os.getenv('ENV', 'dev')
        dsn_var = 'DATABASE_URL_PROD' if env == 'prod' else 'DATABASE_URL_DEV'
        dsn = os.environ[dsn_var]
        
        logger.info(f"Creating database pool for {env} environment")
        
        # Fixed: Remove ssl='require' to avoid deprecation warning (rely on DSN)
        _db_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=4,  # Render cron opens, writes, exits → small pool
            timeout=15,
            statement_cache_size=100,
            command_timeout=30,  # Neon closes idle <5 min; stay quick
        )
        
        logger.info("✓ Database pool created successfully")
    
    return _db_pool

async def close_pool():
    """Close database connection pool"""
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
        logger.info("Database pool closed")

# Column definitions for flood_pixels_raw table
FLOOD_PIXELS_COLUMNS = (
    'segment_id', 'score', 'homes', 'qpe_1h', 'ffw',
    'lon', 'lat', 'first_seen', 'geom'
)

def to_row(event_dict: dict) -> Tuple[Any, ...]:
    """
    Convert flood event dict to database row tuple
    Maps from FloodClassifier output to flood_pixels_raw schema
    """
    # Fixed: Safer segment_id encoding to avoid collisions
    segment_id_str = event_dict['segment_id']
    if '_' in segment_id_str:
        row, col = map(int, segment_id_str.split('_'))
        segment_id = row * 10000 + col  # Documented encoding: row*10000 + col
    else:
        segment_id = int(segment_id_str)
    
    # Fixed: Use dateutil for more lenient timestamp parsing
    first_seen_str = event_dict['first_seen']
    first_seen = isoparse(first_seen_str)
    
    # Fixed: Include WKT geometry string for text-mode COPY
    lon = event_dict['longitude']
    lat = event_dict['latitude']
    geom_wkt = f'SRID=4326;POINT({lon} {lat})'
    
    return (
        segment_id,                           # bigint
        event_dict['event_score'],           # smallint (score)
        event_dict['home_estimate'],         # integer (homes)
        event_dict['qpe_1h'],               # numeric
        event_dict['ffw_confirmed'],        # boolean
        lon,                                # numeric (lon)
        lat,                                # numeric (lat)
        first_seen,                         # timestamptz
        geom_wkt                            # geometry as WKT string
    )

async def dump_to_db(event_rows: List[dict], retry_count: int = 0) -> bool:
    """
    Bulk insert flood events to database using copy_records_to_table
    Returns True on success, False on failure after retries
    """
    if not event_rows:
        logger.info("No flood events to persist")
        return True
    
    try:
        pool = await get_pool()
        
        # Fixed: Use copy_records_to_table with proper binary format
        rows = [to_row(event_dict) for event_dict in event_rows]
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.copy_records_to_table(
                    'flood_pixels_raw', 
                    records=rows, 
                    columns=FLOOD_PIXELS_COLUMNS
                )
        
        logger.info(f"✓ {len(event_rows)} flood pixels persisted to Neon")
        return True
        
    except Exception as e:
        logger.error(f"Database write failed (attempt {retry_count + 1}): {e}")
        
        if retry_count == 0:
            # First failure - retry once with backoff
            logger.info("Retrying database write in 30 seconds...")
            await asyncio.sleep(30)
            return await dump_to_db(event_rows, retry_count + 1)
        else:
            # Second failure - log error and raise for cron monitoring
            logger.error("Database write failed after retry - raising exception for monitoring")
            raise Exception(f"Database write failed after 2 attempts: {e}")

async def test_connection() -> bool:
    """Test database connectivity"""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            logger.info("✓ Database connection test successful")
            return result == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False