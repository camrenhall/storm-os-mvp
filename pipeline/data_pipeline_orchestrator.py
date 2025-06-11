#!/usr/bin/env python3
"""
FLASH Data Pipeline Orchestrator
Coordinates data ingestion, classification, and spatial processing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import traceback

# Import our modules (adjust imports based on your project structure)
from flash_ingest import FlashIngest
from ffw_client import NWSFlashFloodWarnings
from classifier import FlashClassifier, ClassificationConfig, ClassificationResult
from grid_align import GridProcessor, GridDefinition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline"""
    
    # Processing intervals
    update_interval_minutes: int = 10  # How often to run the pipeline
    max_retry_attempts: int = 3
    
    # Quality thresholds
    min_valid_pixels_required: int = 50000
    max_age_minutes: int = 30  # Maximum age of FLASH data to accept
    
    # Output settings
    output_dir: str = "./pipeline_output"
    save_intermediate_results: bool = True
    
    # Classification settings
    classification_config: ClassificationConfig = None
    
    # Daily SMS limits (placeholder for future integration)
    daily_sms_limit: int = 200
    
    def __post_init__(self):
        if self.classification_config is None:
            self.classification_config = ClassificationConfig()


@dataclass
class PipelineResult:
    """Results from a complete pipeline run"""
    
    # Timing
    pipeline_start_time: datetime
    pipeline_end_time: datetime
    processing_duration_seconds: float
    
    # Data status
    flash_data_acquired: bool
    flash_valid_time: Optional[datetime] = None
    flash_data_age_minutes: Optional[float] = None
    
    ffw_data_acquired: bool
    active_ffw_count: int = 0
    
    # Classification results
    classification_successful: bool
    classification_result: Optional[ClassificationResult] = None
    
    # Hotspots detected
    critical_hotspots: List[Dict[str, Any]] = None
    high_hotspots: List[Dict[str, Any]] = None
    moderate_hotspots: List[Dict[str, Any]] = None
    
    # Spatial processing
    ffw_confirmed_critical_count: int = 0
    ffw_confirmed_high_count: int = 0
    
    # Quality flags
    quality_degraded: bool = False
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.critical_hotspots is None:
            self.critical_hotspots = []
        if self.high_hotspots is None:
            self.high_hotspots = []
        if self.moderate_hotspots is None:
            self.moderate_hotspots = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class FlashDataPipeline:
    """
    Main data pipeline orchestrator
    Coordinates all data processing components
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.flash_ingest = None
        self.ffw_client = None
        self.classifier = FlashClassifier(self.config.classification_config)
        self.grid_processor = GridProcessor()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.last_successful_run = None
        self.consecutive_failures = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.flash_ingest = FlashIngest()
        self.ffw_client = NWSFlashFloodWarnings()
        
        await self.flash_ingest.__aenter__()
        await self.ffw_client.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.flash_ingest:
            await self.flash_ingest.__aexit__(exc_type, exc_val, exc_tb)
        if self.ffw_client:
            await self.ffw_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def acquire_flash_data(self) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """
        Acquire FLASH UnitStreamflow data
        Returns (success, data_dict, errors)
        """
        errors = []
        
        try:
            logger.info("Acquiring FLASH data...")
            flash_data = await self.flash_ingest.fetch_latest()
            
            if not flash_data:
                errors.append("Failed to fetch FLASH data from all sources")
                return False, None, errors
            
            # Check data age
            data_age = datetime.utcnow() - flash_data['valid_time']
            age_minutes = data_age.total_seconds() / 60
            
            if age_minutes > self.config.max_age_minutes:
                errors.append(f"FLASH data too old: {age_minutes:.1f} minutes")
                return False, None, errors
            
            # Check data quality
            if flash_data['valid_points'] < self.config.min_valid_pixels_required:
                errors.append(f"Insufficient valid pixels: {flash_data['valid_points']}")
                return False, None, errors
            
            logger.info(f"✓ FLASH data acquired: {flash_data['valid_points']:,} valid pixels, "
                       f"age {age_minutes:.1f} minutes")
            
            return True, flash_data, errors
            
        except Exception as e:
            error_msg = f"FLASH data acquisition failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return False, None, errors
    
    async def acquire_ffw_data(self) -> Tuple[bool, Optional[Any], List[str]]:
        """
        Acquire NWS Flash Flood Warning data
        Returns (success, geodataframe, errors)
        """
        errors = []
        
        try:
            logger.info("Acquiring FFW data...")
            warnings_gdf = await self.ffw_client.get_active_warnings()
            
            if warnings_gdf is None:
                errors.append("Failed to fetch FFW data from NWS API")
                return False, None, errors
            
            warning_count = len(warnings_gdf)
            logger.info(f"✓ FFW data acquired: {warning_count} active warnings")
            
            return True, warnings_gdf, errors
            
        except Exception as e:
            error_msg = f"FFW data acquisition failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return False, None, errors
    
    def classify_flood_severity(self, flash_data: Dict[str, Any]) -> Tuple[bool, Optional[ClassificationResult], List[str]]:
        """
        Apply flood severity classification to FLASH data
        Returns (success, classification_result, errors)
        """
        errors = []
        
        try:
            logger.info("Classifying flood severity...")
            
            unit_streamflow = flash_data['unit_streamflow']
            valid_time = flash_data['valid_time']
            
            result = self.classifier.classify(unit_streamflow, valid_time)
            
            if not result:
                errors.append("Classification failed to produce results")
                return False, None, errors
            
            logger.info(f"✓ Classification complete: "
                       f"Critical={result.critical_count}, "
                       f"High={result.high_count}, "
                       f"Moderate={result.moderate_count}")
            
            return True, result, errors
            
        except Exception as e:
            error_msg = f"Classification failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return False, None, errors
    
    def extract_hotspots(self, classification_result: ClassificationResult) -> Tuple[List, List, List]:
        """
        Extract hotspots for each severity level
        Returns (critical_hotspots, high_hotspots, moderate_hotspots)
        """
        try:
            critical_hotspots = self.classifier.get_flood_hotspots(classification_result, "critical")
            high_hotspots = self.classifier.get_flood_hotspots(classification_result, "high") 
            moderate_hotspots = self.classifier.get_flood_hotspots(classification_result, "moderate")
            
            # Add geographic coordinates to hotspots
            for hotspots_list in [critical_hotspots, high_hotspots, moderate_hotspots]:
                for hotspot in hotspots_list:
                    row, col = hotspot['centroid_grid']
                    lon, lat = self.grid_processor.grid_to_lonlat(int(row), int(col))
                    hotspot['longitude'] = lon
                    hotspot['latitude'] = lat
            
            total_hotspots = len(critical_hotspots) + len(high_hotspots) + len(moderate_hotspots)
            logger.info(f"✓ Extracted {total_hotspots} total hotspots")
            
            return critical_hotspots, high_hotspots, moderate_hotspots
            
        except Exception as e:
            logger.error(f"Hotspot extraction failed: {e}")
            return [], [], []
    
    def apply_ffw_confirmation(self, hotspots: List[Dict], warnings_gdf) -> int:
        """
        Check which hotspots are confirmed by active FFW polygons
        Returns count of confirmed hotspots
        """
        if len(warnings_gdf) == 0 or not hotspots:
            return 0
        
        try:
            confirmed_count = 0
            
            for hotspot in hotspots:
                # Create point geometry for hotspot
                from shapely.geometry import Point
                hotspot_point = Point(hotspot['longitude'], hotspot['latitude'])
                
                # Check if point intersects any warning polygon
                intersects = warnings_gdf.geometry.intersects(hotspot_point).any()
                
                if intersects:
                    hotspot['ffw_confirmed'] = True
                    confirmed_count += 1
                else:
                    hotspot['ffw_confirmed'] = False
            
            logger.info(f"✓ FFW confirmation: {confirmed_count}/{len(hotspots)} hotspots confirmed")
            return confirmed_count
            
        except Exception as e:
            logger.error(f"FFW confirmation failed: {e}")
            return 0
    
    def save_results(self, result: PipelineResult):
        """Save pipeline results to JSON file"""
        try:
            timestamp = result.pipeline_start_time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"pipeline_result_{timestamp}.json"
            
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            
            # Handle datetime serialization
            for key, value in result_dict.items():
                if isinstance(value, datetime):
                    result_dict[key] = value.isoformat()
            
            # Handle nested datetime objects
            if result_dict['classification_result']:
                if 'valid_time' in result_dict['classification_result']:
                    result_dict['classification_result']['valid_time'] = \
                        result_dict['classification_result']['valid_time'].isoformat()
            
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"✓ Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def run_pipeline(self) -> PipelineResult:
        """
        Execute complete data pipeline
        Returns PipelineResult with all processing outcomes
        """
        start_time = datetime.utcnow()
        logger.info(f"=== Starting FLASH Data Pipeline at {start_time} ===")
        
        # Initialize result object
        result = PipelineResult(
            pipeline_start_time=start_time,
            pipeline_end_time=start_time,  # Will update at end
            processing_duration_seconds=0.0,
            flash_data_acquired=False,
            ffw_data_acquired=False,
            classification_successful=False
        )
        
        try:
            # Step 1: Acquire FLASH data
            flash_success, flash_data, flash_errors = await self.acquire_flash_data()
            result.flash_data_acquired = flash_success
            result.errors.extend(flash_errors)
            
            if flash_success:
                result.flash_valid_time = flash_data['valid_time']
                result.flash_data_age_minutes = (start_time - flash_data['valid_time']).total_seconds() / 60
            
            # Step 2: Acquire FFW data (parallel to classification)
            ffw_success, warnings_gdf, ffw_errors = await self.acquire_ffw_data()
            result.ffw_data_acquired = ffw_success
            result.active_ffw_count = len(warnings_gdf) if ffw_success else 0
            result.errors.extend(ffw_errors)
            
            # Step 3: Classification (only if FLASH data available)
            if flash_success:
                class_success, classification_result, class_errors = self.classify_flood_severity(flash_data)
                result.classification_successful = class_success
                result.classification_result = classification_result
                result.errors.extend(class_errors)
                
                if class_success:
                    result.quality_degraded = classification_result.quality_degraded
                    
                    # Step 4: Extract hotspots
                    critical_hs, high_hs, moderate_hs = self.extract_hotspots(classification_result)
                    result.critical_hotspots = critical_hs
                    result.high_hotspots = high_hs
                    result.moderate_hotspots = moderate_hs
                    
                    # Step 5: FFW confirmation (only if both data sources available)
                    if ffw_success:
                        result.ffw_confirmed_critical_count = self.apply_ffw_confirmation(critical_hs, warnings_gdf)
                        result.ffw_confirmed_high_count = self.apply_ffw_confirmation(high_hs, warnings_gdf)
            
            # Update pipeline status
            if result.flash_data_acquired and result.classification_successful:
                self.last_successful_run = start_time
                self.consecutive_failures = 0
                logger.info("✓ Pipeline completed successfully")
            else:
                self.consecutive_failures += 1
                logger.warning(f"Pipeline completed with issues (failure #{self.consecutive_failures})")
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            self.consecutive_failures += 1
        
        # Finalize result
        end_time = datetime.utcnow()
        result.pipeline_end_time = end_time
        result.processing_duration_seconds = (end_time - start_time).total_seconds()
        
        # Save results if configured
        if self.config.save_intermediate_results:
            self.save_results(result)
        
        logger.info(f"=== Pipeline completed in {result.processing_duration_seconds:.1f}s ===")
        return result
    
    async def run_continuous(self, max_iterations: int = None):
        """
        Run pipeline continuously at configured intervals
        """
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                try:
                    # Run pipeline
                    result = await self.run_pipeline()
                    
                    # Log summary
                    logger.info(f"Iteration {iteration + 1}: "
                               f"Critical={len(result.critical_hotspots)}, "
                               f"High={len(result.high_hotspots)}, "
                               f"Errors={len(result.errors)}")
                    
                    iteration += 1
                    
                    # Wait for next iteration
                    if max_iterations is None or iteration < max_iterations:
                        sleep_seconds = self.config.update_interval_minutes * 60
                        logger.info(f"Sleeping for {sleep_seconds}s until next pipeline run...")
                        await asyncio.sleep(sleep_seconds)
                        
                except KeyboardInterrupt:
                    logger.info("Pipeline interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Pipeline iteration failed: {e}")
                    # Still sleep before retrying
                    await asyncio.sleep(60)  # 1 minute on error
                    
        except Exception as e:
            logger.error(f"Continuous pipeline failed: {e}")


# Example usage
async def main():
    """Example usage of FlashDataPipeline"""
    
    # Configure pipeline
    config = PipelineConfig(
        update_interval_minutes=10,
        save_intermediate_results=True,
        output_dir="./test_pipeline_output"
    )
    
    # Run pipeline
    async with FlashDataPipeline(config) as pipeline:
        # Single run
        result = await pipeline.run_pipeline()
        
        print(f"\n=== Pipeline Results ===")
        print(f"Processing time: {result.processing_duration_seconds:.1f}s")
        print(f"FLASH data: {'✓' if result.flash_data_acquired else '✗'}")
        print(f"FFW data: {'✓' if result.ffw_data_acquired else '✗'}")
        print(f"Classification: {'✓' if result.classification_successful else '✗'}")
        print(f"Critical hotspots: {len(result.critical_hotspots)}")
        print(f"High hotspots: {len(result.high_hotspots)}")
        print(f"FFW confirmed critical: {result.ffw_confirmed_critical_count}")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())