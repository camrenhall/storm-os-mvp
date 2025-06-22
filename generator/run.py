#!/usr/bin/env python3
"""
Generator Container - Main Entry Point
30-minute orchestrated pipeline: MRMS data → flood classification → database
FIXED: Proper async context manager usage for data ingestion clients
"""

import asyncio
import asyncpg
import logging
import time
import os
import sys
from typing import Dict, Any, List, Optional

# Fixed: Correct import paths for your directory structure
from pipeline.flash_ingest import FlashIngest
from pipeline.qpe_ingest import QPEIngest
from pipeline.ffw_client import NWSFlashFloodWarnings
from pipeline.flood_classifier import FloodClassifier, ClassificationConfig, ProcessingMode
from pipeline.exposure import initialize_population_grid, is_initialized

# Import database module (top-level)
from db import dump_to_db, test_connection, close_pool

# Uniqueness Tracing for DB

from uuid import uuid4
from datetime import datetime, timezone

PIPELINE_RUN_ID  = uuid4()
PIPELINE_RUN_TS  = datetime.now(tz=timezone.utc)

# Setup logging
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )

logger = logging.getLogger(__name__)

class GeneratorPipeline:
    """
    Main Generator Pipeline Orchestrator
    Coordinates all data sources and processing steps
    """
    
    def __init__(self):
        self.cache_dir = os.getenv('CACHE_DIR', '/tmp/mrms_cache')
        self.noaa_base = os.getenv('NOAA_MRMS_BASE', 'https://mrms.ncep.noaa.gov')
        
        # Fixed: Conservative logging configuration for production
        self.classifier_config = ClassificationConfig(
            processing_mode=ProcessingMode.PRODUCTION,
            use_fixed_thresholds=True,
            enable_detailed_logging=True,
            enable_ffw_enhancement=True,
            min_flood_area_pixels=9  # 3x3 minimum per spec
        )
        self.classifier = FloodClassifier(self.classifier_config)
        
        # Fixed: Initialize fresh stats each time
        self.reset_stats()
    
    def reset_stats(self):
        """Reset performance tracking stats"""
        self.stats = {
            'pipeline_start_time': None,
            'data_ingestion_time': 0.0,
            'classification_time': 0.0,
            'extraction_time': 0.0,
            'database_time': 0.0,
            'total_time': 0.0
        }
    
    async def initialize_dependencies(self):
        """Initialize population grid and test database connectivity"""
        logger.info("Initializing Generator dependencies...")
        
        # Initialize population exposure grid with correct path
        if not is_initialized():
            try:
                logger.info("Initializing population exposure grid...")
                initialize_population_grid(
                    parquet_path="pipeline/pixel_exposure_conus.parquet",
                    allow_regional_data=True
                )
                logger.info("✓ Population exposure grid initialized")
            except Exception as e:
                logger.error(f"Population grid initialization failed: {e}")
                # Continue without exposure data
        
        # Test database connectivity
        db_ok = await test_connection()
        if not db_ok:
            raise Exception("Database connectivity test failed")
        
        logger.info("✓ All dependencies initialized successfully")
    
    async def ingest_meteorological_data(self) -> Dict[str, Any]:
        """
        Parallel ingestion of FLASH, QPE, and FFW data
        FIXED: Proper async context manager usage
        Returns dict with all meteorological inputs
        """
        logger.info("Starting parallel meteorological data ingestion...")
        ingestion_start = time.time()
        
        # FIXED: Use proper async context managers for all clients
        async def fetch_flash():
            try:
                async with FlashIngest(cache_dir=f"{self.cache_dir}/flash") as flash_client:
                    return await flash_client.fetch_latest()
            except Exception as e:
                logger.error(f"FLASH fetch failed: {e}")
                return e
        
        async def fetch_qpe():
            try:
                async with QPEIngest(cache_dir=f"{self.cache_dir}/qpe") as qpe_client:
                    return await qpe_client.fetch_latest()
            except Exception as e:
                logger.warning(f"QPE fetch failed: {e}")
                return e
        
        async def fetch_ffw():
            try:
                async with NWSFlashFloodWarnings(cache_dir=f"{self.cache_dir}/ffw") as ffw_client:
                    return await ffw_client.get_active_warnings()
            except Exception as e:
                logger.warning(f"FFW fetch failed: {e}")
                return e
        
        # Execute all downloads in parallel
        try:
            flash_result, qpe_result, ffw_result = await asyncio.gather(
                fetch_flash(),
                fetch_qpe(), 
                fetch_ffw(),
                return_exceptions=True
            )
            
            # Handle individual failures gracefully
            meteorological_data = {}
            
            # Process FLASH result (critical - must succeed)
            if isinstance(flash_result, Exception):
                logger.error(f"FLASH data ingestion failed: {flash_result}")
                raise Exception("FLASH data required for pipeline")
            elif flash_result is None:
                logger.error("FLASH data ingestion returned None")
                raise Exception("FLASH data required for pipeline")
            else:
                meteorological_data['flash'] = flash_result
                logger.info(f"✓ FLASH data acquired: {flash_result['valid_time']}")
            
            # Process QPE result (optional - can continue without)
            if isinstance(qpe_result, Exception):
                logger.warning(f"QPE data ingestion failed: {qpe_result}")
                meteorological_data['qpe'] = None
            elif qpe_result is None:
                logger.warning("QPE data not available")
                meteorological_data['qpe'] = None
            else:
                meteorological_data['qpe'] = qpe_result
                logger.info(f"✓ QPE data acquired: {qpe_result['valid_time']}")
            
            # Process FFW result (optional - can continue without)
            if isinstance(ffw_result, Exception):
                logger.warning(f"FFW data ingestion failed: {ffw_result}")
                meteorological_data['ffw'] = None
            elif ffw_result is None or len(ffw_result) == 0:
                logger.info("No active Flash Flood Warnings")
                meteorological_data['ffw'] = None
            else:
                meteorological_data['ffw'] = ffw_result
                logger.info(f"✓ FFW data acquired: {len(ffw_result)} active warnings")
            
            self.stats['data_ingestion_time'] = time.time() - ingestion_start
            return meteorological_data
            
        except Exception as e:
            logger.error(f"Meteorological data ingestion failed: {e}")
            raise
    
    async def process_flood_classification(self, meteorological_data: Dict[str, Any]) -> List[dict]:
        """
        Run flood classification and event extraction
        Returns list of flood event dictionaries
        """
        logger.info("Starting flood classification and event extraction...")
        classification_start = time.time()
        
        try:
            # Extract data components
            flash_data = meteorological_data['flash']
            qpe_data = meteorological_data.get('qpe')
            ffw_data = meteorological_data.get('ffw')
            
            unit_streamflow = flash_data['unit_streamflow']
            valid_time = flash_data['valid_time']
            
            # Optional data grids
            qpe_grid = qpe_data['qpe_1h'] if qpe_data else None
            ffw_polygons = ffw_data if ffw_data is not None else None
            
            # Run classification
            logger.info(f"Classifying floods for {valid_time} using spec-compliant 2/5/10 m³/s/km² thresholds")
            
            classification_result = self.classifier.classify(
                unit_streamflow=unit_streamflow,
                valid_time=valid_time,
                ffw_polygons=ffw_polygons,
                qpe_1h_grid=qpe_grid
            )
            
            if not classification_result:
                logger.error("Flood classification failed")
                return []
            
            self.stats['classification_time'] = time.time() - classification_start
            
            # Extract flood events
            extraction_start = time.time()
            
            flood_events = self.classifier.extract_flood_events(
                classification_result,
                unit_streamflow,
                qpe_grid
            )
            
            # Flatten all events into single list
            all_events = []
            for severity, events in flood_events.items():
                all_events.extend(events)
            
            self.stats['extraction_time'] = time.time() - extraction_start
            
            logger.info(f"✓ Classification complete: {len(all_events)} flood events extracted")
            logger.info(f"  Critical: {len(flood_events.get('critical', []))}")
            logger.info(f"  High: {len(flood_events.get('high', []))}")
            logger.info(f"  Moderate: {len(flood_events.get('moderate', []))}")
            
            return all_events
            
        except Exception as e:
            logger.error(f"Flood classification failed: {e}")
            raise
    
    async def persist_flood_events(self, flood_events: List[dict]) -> bool:
        """
        Persist flood events to database
        Returns success status
        """
        if not flood_events:
            logger.info("No flood events to persist")
            return True
        
        # attach provenance columns
        for ev in flood_events:
            ev["run_timestamp"] = self.run_ts.isoformat()
            ev["run_id"]       = str(self.run_id)
        
        logger.info(f"Persisting {len(flood_events)} flood events to database...")
        database_start = time.time()
        
        try:
            success = await dump_to_db(flood_events)
            self.stats['database_time'] = time.time() - database_start
            
            if success:
                logger.info(f"✓ {len(flood_events)} flood events successfully persisted")
            else:
                logger.error("Database persistence failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Database persistence failed: {e}")
            raise
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete 30-minute pipeline cycle
        Returns performance statistics
        """
        # Fixed: Reset stats at start of each run
        self.reset_stats()
        
        self._lock_id = 987654            # arbitrary but unique
        conn = await asyncpg.connect(dsn=os.getenv("DATABASE_URL_DEV"))
        if not await conn.fetchval("SELECT pg_try_advisory_lock($1)", self._lock_id):
            logger.warning("Another Generator is already running – aborting this cycle")
            await conn.close()
            return {"success": False, "reason": "lock_contended"}
        self._advisory_conn = conn        # hold until finally: to release
        
        logger.info("=" * 60)
        logger.info("STARTING FLOOD-LEAD INTELLIGENCE GENERATOR PIPELINE")
        logger.info("=" * 60)
        
        self.stats['pipeline_start_time'] = time.time()
        
        try:
            # Step 1: Initialize dependencies
            await self.initialize_dependencies()
            
            # Step 2: Ingest meteorological data (FIXED: proper async context managers)
            meteorological_data = await self.ingest_meteorological_data()
            
            # Step 3: Process flood classification
            flood_events = await self.process_flood_classification(meteorological_data)
            
            # Step 4: Persist to database
            success = await self.persist_flood_events(flood_events)
            
            # Calculate final statistics
            self.stats['total_time'] = time.time() - self.stats['pipeline_start_time']
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Performance Summary:")
            logger.info(f"  Data ingestion: {self.stats['data_ingestion_time']:.2f}s")
            logger.info(f"  Classification: {self.stats['classification_time']:.2f}s")
            logger.info(f"  Event extraction: {self.stats['extraction_time']:.2f}s")
            logger.info(f"  Database persistence: {self.stats['database_time']:.2f}s")
            logger.info(f"  Total pipeline time: {self.stats['total_time']:.2f}s")
            logger.info(f"  Events processed: {len(flood_events)}")
            logger.info("=" * 60)
            
            return {
                'success': success,
                'events_processed': len(flood_events),
                'performance_stats': self.stats
            }
            
        except Exception as e:
            self.stats['total_time'] = time.time() - self.stats['pipeline_start_time']
            logger.error("=" * 60)
            logger.error("PIPELINE FAILED")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            logger.error(f"Total runtime before failure: {self.stats['total_time']:.2f}s")
            logger.error("=" * 60)
            raise
        
        finally:
            if hasattr(self, "_advisory_conn"):
            await self._advisory_conn.execute("SELECT pg_advisory_unlock($1)", self._lock_id)
            await self._advisory_conn.close()
            # Cleanup database connections
            await close_pool()

async def main():
    """Main entry point for Generator container"""
    setup_logging()
    
    logger.info(f"Generator starting - ENV={os.getenv('ENV', 'dev')}")
    
    try:
        pipeline = GeneratorPipeline()
        result = await pipeline.run_complete_pipeline()
        
        if result['success']:
            logger.info(f"Generator cycle completed successfully")
            sys.exit(0)
        else:
            logger.error(f"Generator cycle failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Generator failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())