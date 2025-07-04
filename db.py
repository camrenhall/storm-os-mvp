#!/usr/bin/env python3
"""
Database connection and operations for Flood-Lead Intelligence MVP
Handles asyncpg pool management and bulk operations to Neon
FIXED: Schema alignment with actual database structure and syntax fixes
"""

import os
import asyncpg
import asyncio
import logging
import uuid
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

# FIXED: Column definitions matching actual database schema
# Schema: segment_id, score, homes, qpe_1h, ffw, geom, first_seen
# Note: 'id' is auto-generated bigserial, 'first_seen' has DEFAULT now()
FLOOD_PIXELS_COLUMNS = (
    'segment_id', 'score', 'homes', 'qpe_1h', 'ffw', 'geom', 'first_seen'
)

def to_row(event_dict: dict) -> Tuple[Any, ...]:
    """
    Convert flood event dict to database row tuple
    FIXED: Maps to actual database schema (no separate lon/lat columns)
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
    run_ts     = isoparse(event_dict["run_timestamp"])
    run_id     = uuid.UUID(event_dict["run_id"])  # safe if already UUID
    
    
    # FIXED: Create geometry WKT string directly (no separate lon/lat)
    lon = event_dict['longitude']
    lat = event_dict['latitude']
    geom_wkt = f'SRID=4326;POINT({lon} {lat})'
    
    return (
        segment_id,                           # bigint
        event_dict['event_score'],           # smallint (score)
        event_dict['home_estimate'],         # integer (homes)
        event_dict['qpe_1h'],               # numeric
        event_dict['ffw_confirmed'],        # boolean
        geom_wkt,                           # geometry as WKT string
        first_seen,                        # timestamptz
        run_ts,                            # NEW
        run_id                             # NEW (uuid)
        
    )

async def dump_to_db(event_rows: List[dict], retry_count: int = 0) -> bool:
    """
    Bulk insert flood events to database using executemany
    FIXED: Uses executemany for guaranteed PostGIS geometry compatibility
    Returns True on success, False on failure after retries
    """
    if not event_rows:
        logger.info("No flood events to persist")
        return True
    
    try:
        pool = await get_pool()
        
        # Convert event dictionaries to database rows
        rows = [to_row(event_dict) for event_dict in event_rows]
        
        # FIXED: Simple static SQL string to avoid f-string syntax issues
        insert_sql = """
            INSERT INTO flood_pixels_raw
                        (segment_id, score, homes, qpe_1h, ffw, geom,
                        first_seen, run_timestamp, run_id)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        """
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # FIXED: Use executemany for guaranteed compatibility with PostGIS geometry
                # This approach works reliably with geometry WKT strings
                await conn.executemany(insert_sql, rows)
        
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