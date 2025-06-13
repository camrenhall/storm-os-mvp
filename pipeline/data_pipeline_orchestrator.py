#!/usr/bin/env python3
"""
FLASH Data Pipeline Orchestrator - UPDATED VERSION
Properly integrates FFW data with classifier for complete flood detection pipeline
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
from optimized_classifier import OptimizedFlashClassifier, ClassificationConfig, ClassificationResult
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
    
    # UPDATED: More aggressive classification settings for production
    classification_config: ClassificationConfig = None
    
    # Daily SMS limits (placeholder for future integration)
    daily_sms_limit: int = 200
    
    def __post_init__(self):
        if self.classification_config is None:
            # Production-ready settings with adaptive thresholds
            self.classification_config = ClassificationConfig(
                max_critical_hotspots_per_cycle=20,    # ~480/day max
                max_high_hotspots_per_cycle=100,       # Buffer for active periods
                max_total_hotspots_per_cycle=150,      # Hard cap
                enable_fast_hotspot_extraction=True,   # Speed optimization
                max_workers=4                          # Parallel processing
            )


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
    
    # UPDATED: FFW-specific metrics
    ffw_boosted_pixels: int = 0
    threshold_escalation_applied: bool = False
    original_vs_final_critical_threshold: Tuple[float, float] = (0.0, 0.0)
    
    # Performance metrics
    classification_time_seconds: float = 0.0
    hotspot_extraction_time_seconds: float = 0.0
    
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
    Main data pipeline orchestrator - UPDATED VERSION
    Properly integrates FFW data with classification engine
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.flash_ingest = None
        self.ffw_client = None
        self.classifier = OptimizedFlashClassifier(self.config.classification_config)
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
    
    def classify_flood_severity(self, flash_data: Dict[str, Any], 
                               ffw_gdf: Optional[Any] = None) -> Tuple[bool, Optional[ClassificationResult], List[str]]:
        """
        Apply flood severity classification to FLASH data WITH FFW integration
        Returns (success, classification_result, errors)
        """
        errors = []
        
        try:
            logger.info("Classifying flood severity with FFW integration...")
            
            unit_streamflow = flash_data['unit_streamflow']
            valid_time = flash_data['valid_time']
            
            # UPDATED: Pass FFW polygons to classifier
            start_time = datetime.now()
            result = self.classifier.classify(
                unit_streamflow, 
                valid_time,
                ffw_polygons=ffw_gdf  # KEY INTEGRATION POINT
            )
            classification_time = (datetime.now() - start_time).total_seconds()
            
            if not result:
                errors.append("Classification failed to produce results")
                return False, None, errors
            
            # Enhanced logging with FFW and adaptive threshold info
            logger.info(f"✓ Classification complete: "
                       f"Critical={result.critical_count}, "
                       f"High={result.high_count}, "
                       f"Moderate={result.moderate_count}")
            
            if result.ffw_boosted_pixels > 0:
                logger.info(f"  FFW boost: {result.ffw_boosted_pixels:,} pixels elevated to CRITICAL")
            
            if result.threshold_escalation_applied:
                logger.info(f"  Adaptive thresholds: Critical raised from "
                          f"{result.original_critical_threshold:.3f} to {result.critical_threshold_value:.3f}")
            
            logger.info(f"  Classification time: {classification_time:.2f}s")
            
            return True, result, errors
            
        except Exception as e:
            error_msg = f"Classification failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return False, None, errors
    
    def extract_hotspots(self, classification_result: ClassificationResult) -> Tuple[List, List, List, float]:
        """
        Extract hotspots for each severity level - UPDATED with timing
        Returns (critical_hotspots, high_hotspots, moderate_hotspots, extraction_time)
        """
        try:
            start_time = datetime.now()
            
            # Use optimized extraction method
            all_hotspots_dict = self.classifier.get_flood_hotspots_optimized(classification_result)
            
            critical_hotspots = all_hotspots_dict.get('critical', [])
            high_hotspots = all_hotspots_dict.get('high', [])
            moderate_hotspots = all_hotspots_dict.get('moderate', [])
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            total_hotspots = len(critical_hotspots) + len(high_hotspots) + len(moderate_hotspots)
            logger.info(f"✓ Extracted {total_hotspots} hotspots in {extraction_time:.2f}s")
            
            # Performance check
            if extraction_time > 60:
                logger.warning(f"⚠️  Hotspot extraction took {extraction_time:.1f}s - performance issue!")
            
            return critical_hotspots, high_hotspots, moderate_hotspots, extraction_time
            
        except Exception as e:
            logger.error(f"Hotspot extraction failed: {e}")
            return [], [], [], 0.0
    
    def validate_results(self, result: PipelineResult) -> List[str]:
        """
        Validate pipeline results and generate warnings
        Returns list of validation warnings
        """
        warnings = []
        
        # Check hotspot counts against targets
        total_hotspots = len(result.critical_hotspots) + len(result.high_hotspots)
        target_max = self.config.classification_config.max_total_hotspots_per_cycle
        
        if total_hotspots > target_max:
            warnings.append(f"Hotspot count ({total_hotspots}) exceeds target ({target_max})")
        
        # Check processing time
        if result.processing_duration_seconds > 300:  # 5 minutes
            warnings.append(f"Pipeline took {result.processing_duration_seconds:.1f}s - too slow for 10min cycle")
        
        # Check for excessive FFW boost
        if result.ffw_boosted_pixels > 10000:
            warnings.append(f"Excessive FFW boost: {result.ffw_boosted_pixels:,} pixels")
        
        # Check data freshness
        if result.flash_data_age_minutes and result.flash_data_age_minutes > 20:
            warnings.append(f"FLASH data is {result.flash_data_age_minutes:.1f} minutes old")
        
        return warnings
    
    def save_results(self, result: PipelineResult):
        """Save pipeline results to JSON file with enhanced metadata"""
        try:
            timestamp = result.pipeline_start_time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"pipeline_result_{timestamp}.json"
            
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            
            # Handle datetime serialization
            for key, value in result_dict.items():
                if isinstance(value, datetime):
                    result_dict[key] = value.isoformat()
            
            # Handle nested datetime objects in classification result
            if result_dict['classification_result']:
                class_result = result_dict['classification_result']
                if 'valid_time' in class_result:
                    class_result['valid_time'] = class_result['valid_time'].isoformat()
            
            # Add pipeline performance summary
            result_dict['performance_summary'] = {
                'total_time_seconds': result.processing_duration_seconds,
                'classification_time_seconds': result.classification_time_seconds,
                'hotspot_extraction_time_seconds': result.hotspot_extraction_time_seconds,
                'hotspots_per_second': len(result.critical_hotspots + result.high_hotspots) / max(result.hotspot_extraction_time_seconds, 0.1),
                'within_10min_cycle': result.processing_duration_seconds < 600
            }
            
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"✓ Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def run_pipeline(self) -> PipelineResult:
        """
        Execute complete data pipeline with FFW integration
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
            
            # Step 3: Classification WITH FFW INTEGRATION
            if flash_success:
                class_start = datetime.now()
                class_success, classification_result, class_errors = self.classify_flood_severity(
                    flash_data, 
                    warnings_gdf if ffw_success else None  # Pass FFW data
                )
                result.classification_time_seconds = (datetime.now() - class_start).total_seconds()
                
                result.classification_successful = class_success
                result.classification_result = classification_result
                result.errors.extend(class_errors)
                
                if class_success:
                    result.quality_degraded = classification_result.quality_degraded
                    result.ffw_boosted_pixels = classification_result.ffw_boosted_pixels
                    result.threshold_escalation_applied = classification_result.threshold_escalation_applied
                    result.original_vs_final_critical_threshold = (
                        classification_result.original_critical_threshold,
                        classification_result.critical_threshold_value
                    )
                    
                    # Step 4: Extract hotspots
                    critical_hs, high_hs, moderate_hs, extraction_time = self.extract_hotspots(classification_result)
                    result.critical_hotspots = critical_hs
                    result.high_hotspots = high_hs
                    result.moderate_hotspots = moderate_hs
                    result.hotspot_extraction_time_seconds = extraction_time
            
            # Step 5: Validate results
            validation_warnings = self.validate_results(result)
            result.warnings.extend(validation_warnings)
            
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
        
        # Performance summary
        logger.info(f"=== Pipeline completed in {result.processing_duration_seconds:.1f}s ===")
        logger.info(f"  Classification: {result.classification_time_seconds:.1f}s")
        logger.info(f"  Hotspot extraction: {result.hotspot_extraction_time_seconds:.1f}s")
        logger.info(f"  Total hotspots: {len(result.critical_hotspots + result.high_hotspots + result.moderate_hotspots)}")
        
        if result.warnings:
            logger.warning(f"  Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                logger.warning(f"    - {warning}")
        
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
                               f"Time={result.processing_duration_seconds:.1f}s, "
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
    """Example usage of updated FlashDataPipeline with FFW integration"""
    
    # Configure pipeline with production-ready settings
    config = PipelineConfig(
        update_interval_minutes=10,
        save_intermediate_results=True,
        output_dir="./production_pipeline_output",
        classification_config=ClassificationConfig(
            max_critical_hotspots_per_cycle=25,    # Conservative limit
            max_high_hotspots_per_cycle=100,       
            max_total_hotspots_per_cycle=150,      
            enable_fast_hotspot_extraction=True,   # Speed optimization ON
            max_workers=4
        )
    )
    
    # Run pipeline
    async with FlashDataPipeline(config) as pipeline:
        # Single run
        result = await pipeline.run_pipeline()
        
        print(f"\n=== Enhanced Pipeline Results ===")
        print(f"Processing time: {result.processing_duration_seconds:.1f}s")
        print(f"  Classification: {result.classification_time_seconds:.1f}s")
        print(f"  Hotspot extraction: {result.hotspot_extraction_time_seconds:.1f}s")
        print(f"FLASH data: {'✓' if result.flash_data_acquired else '✗'}")
        print(f"FFW data: {'✓' if result.ffw_data_acquired else '✗'} ({result.active_ffw_count} warnings)")
        print(f"Classification: {'✓' if result.classification_successful else '✗'}")
        
        print(f"Hotspots detected:")
        print(f"  Critical: {len(result.critical_hotspots)}")
        print(f"  High: {len(result.high_hotspots)}")
        print(f"  Moderate: {len(result.moderate_hotspots)}")
        
        if result.ffw_boosted_pixels > 0:
            print(f"FFW boost: {result.ffw_boosted_pixels:,} pixels elevated to CRITICAL")
        
        if result.threshold_escalation_applied:
            orig, final = result.original_vs_final_critical_threshold
            print(f"Adaptive thresholds: Critical raised from {orig:.3f} to {final:.3f}")
        
        # Performance assessment
        if result.processing_duration_seconds < 60:
            print("✅ PERFORMANCE: Excellent (under 1 minute)")
        elif result.processing_duration_seconds < 300:
            print("⚠️  PERFORMANCE: Acceptable (under 5 minutes)")
        else:
            print("❌ PERFORMANCE: Poor (over 5 minutes)")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print(f"Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  - {warning}")


if __name__ == "__main__":
    asyncio.run(main())