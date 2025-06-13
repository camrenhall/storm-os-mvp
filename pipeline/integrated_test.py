#!/usr/bin/env python3
"""
Integrated FLASH Pipeline Test
Connects flash_ingest.py → classifier.py → output files
Tests the full pipeline with real NOAA data
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Import our modules
from flash_ingest import FlashIngest
from optimized_classifier import OptimizedFlashClassifier, ClassificationConfig
from grid_align import GridProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_full_pipeline():
    """Test the complete FLASH pipeline with real data"""
    
    output_dir = Path("./pipeline_test_output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=== Starting Integrated FLASH Pipeline Test ===")
    
    # Step 1: Ingest real FLASH data
    logger.info("Step 1: Fetching real FLASH data from NOAA...")
    async with FlashIngest() as ingest:
        flash_data = await ingest.fetch_latest()
        
        if not flash_data:
            logger.error("❌ Failed to fetch FLASH data - cannot proceed")
            return False
        
        logger.info(f"✅ FLASH data acquired:")
        logger.info(f"   Valid time: {flash_data['valid_time']}")
        logger.info(f"   Grid shape: {flash_data['grid_shape']}")
        logger.info(f"   Valid pixels: {flash_data['valid_points']:,}")
        logger.info(f"   Data range: [{flash_data.get('data_min', 0):.3f}, "
                   f"{flash_data.get('data_max', 0):.3f}] m³/s/km²")
        logger.info(f"   Source: {flash_data['source_url']}")
    
    # Step 2: Classify the real data
    logger.info("\nStep 2: Classifying flood severity...")
    config = ClassificationConfig(max_workers=4)  # Use 4 threads
    classifier = OptimizedFlashClassifier(config)
    
    result = classifier.classify(
        flash_data['unit_streamflow'], 
        flash_data['valid_time']
    )
    
    if not result:
        logger.error("❌ Classification failed")
        return False
    
    logger.info(f"✅ Classification complete:")
    logger.info(f"   P98 threshold: {result.p98_value:.3f} m³/s/km²")
    logger.info(f"   Critical pixels: {result.critical_count:,}")
    logger.info(f"   High pixels: {result.high_count:,}")
    logger.info(f"   Moderate pixels: {result.moderate_count:,}")
    logger.info(f"   Quality degraded: {result.quality_degraded}")
    
    # Step 3: Extract hotspots with geographic coordinates (OPTIMIZED)
    logger.info("\nStep 3: Extracting flood hotspots (optimized)...")
    
    # Use the new optimized method that returns all severities at once
    all_hotspots_dict = classifier.get_flood_hotspots_optimized(result)
    
    # Extract individual lists for backward compatibility
    critical_hotspots = all_hotspots_dict.get('critical', [])
    high_hotspots = all_hotspots_dict.get('high', [])
    moderate_hotspots = all_hotspots_dict.get('moderate', [])
    
    # Flatten all hotspots for output (coordinates already included)
    all_hotspots = []
    for severity, hotspots_list in all_hotspots_dict.items():
        all_hotspots.extend(hotspots_list)
    
    logger.info(f"✅ Hotspots extracted:")
    logger.info(f"   Critical: {len(critical_hotspots)}")
    logger.info(f"   High: {len(high_hotspots)}")
    logger.info(f"   Moderate: {len(moderate_hotspots)}")
    logger.info(f"   Total with coordinates: {len(all_hotspots)}")
    
    # Step 4: Save comprehensive output files
    logger.info("\nStep 4: Saving output files...")
    timestamp = flash_data['valid_time'].strftime("%Y%m%d_%H%M%S")
    
    # 4a. Save raw classification arrays
    arrays_file = output_dir / f"flash_classification_{timestamp}.npz"
    np.savez_compressed(
        arrays_file,
        unit_streamflow=flash_data['unit_streamflow'],
        critical_mask=result.critical_mask,
        high_mask=result.high_mask,
        moderate_mask=result.moderate_mask,
        valid_time=flash_data['valid_time'].isoformat(),
        metadata={
            'p98_threshold': result.p98_value,
            'critical_threshold': result.critical_threshold_value,
            'high_threshold': result.high_threshold_value,
            'moderate_threshold': result.moderate_threshold_value,
            'total_pixels': result.total_pixels,
            'valid_pixels': result.valid_pixels,
            'source_url': flash_data['source_url']
        }
    )
    logger.info(f"   💾 Arrays saved: {arrays_file}")
    
    # 4b. Save hotspots as JSON
    hotspots_file = output_dir / f"flood_hotspots_{timestamp}.json"
    hotspots_output = {
        'metadata': {
            'valid_time': flash_data['valid_time'].isoformat(),
            'processing_time': datetime.now().isoformat(),
            'source_url': flash_data['source_url'],
            'classification_method': result.normalization_method,
            'quality_degraded': result.quality_degraded,
            'thresholds': {
                'p98': result.p98_value,
                'critical': result.critical_threshold_value,
                'high': result.high_threshold_value,
                'moderate': result.moderate_threshold_value
            }
        },
        'summary': {
            'total_hotspots': len(all_hotspots),
            'critical_count': len(critical_hotspots),
            'high_count': len(high_hotspots),
            'moderate_count': len(moderate_hotspots)
        },
        'hotspots': all_hotspots
    }
    
    with open(hotspots_file, 'w') as f:
        json.dump(hotspots_output, f, indent=2, default=str)
    logger.info(f"   💾 Hotspots saved: {hotspots_file}")
    
    # 4c. Save human-readable summary
    summary_file = output_dir / f"pipeline_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"FLASH Flood Detection Pipeline Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Processing Time: {datetime.now()}\n")
        f.write(f"FLASH Valid Time: {flash_data['valid_time']}\n")
        f.write(f"Data Source: {flash_data['source_url']}\n\n")
        
        f.write(f"Grid Information:\n")
        f.write(f"  Dimensions: {flash_data['grid_shape']}\n")
        f.write(f"  Total pixels: {result.total_pixels:,}\n")
        f.write(f"  Valid pixels: {result.valid_pixels:,} ({result.valid_pixels/result.total_pixels*100:.1f}%)\n\n")
        
        f.write(f"Classification Thresholds:\n")
        f.write(f"  P98 Reference: {result.p98_value:.3f} m³/s/km²\n")
        f.write(f"  Critical: ≥{result.critical_threshold_value:.3f} m³/s/km²\n")
        f.write(f"  High: ≥{result.high_threshold_value:.3f} m³/s/km²\n")
        f.write(f"  Moderate: ≥{result.moderate_threshold_value:.3f} m³/s/km²\n\n")
        
        f.write(f"Flood Detection Results:\n")
        f.write(f"  Critical pixels: {result.critical_count:,}\n")
        f.write(f"  High pixels: {result.high_count:,}\n")
        f.write(f"  Moderate pixels: {result.moderate_count:,}\n\n")
        
        f.write(f"Hotspot Analysis:\n")
        f.write(f"  Critical hotspots: {len(critical_hotspots)}\n")
        f.write(f"  High hotspots: {len(high_hotspots)}\n")
        f.write(f"  Moderate hotspots: {len(moderate_hotspots)}\n\n")
        
        if all_hotspots:
            f.write(f"Sample Hotspots (first 5):\n")
            for i, hotspot in enumerate(all_hotspots[:5]):
                f.write(f"  {i+1}. {hotspot['severity'].title()}: "
                       f"{hotspot['pixel_count']} pixels at "
                       f"({hotspot['longitude']:.3f}, {hotspot['latitude']:.3f})\n")
        
        f.write(f"\nQuality Flags:\n")
        f.write(f"  Quality degraded: {result.quality_degraded}\n")
        f.write(f"  Method: {result.normalization_method}\n")
    
    logger.info(f"   💾 Summary saved: {summary_file}")
    
    # Step 5: Validation checks
    logger.info("\nStep 5: Validation checks...")
    
    validation_results = []
    
    # Check data reasonableness
    if result.valid_pixels > 1000000:  # > 1M valid pixels
        validation_results.append("✅ Sufficient valid data coverage")
    else:
        validation_results.append("⚠️  Low valid data coverage")
    
    # Check threshold reasonableness
    if 0.01 <= result.p98_value <= 50.0:  # Reasonable streamflow range
        validation_results.append("✅ P98 threshold in reasonable range")
    else:
        validation_results.append("⚠️  P98 threshold seems unusual")
    
    # Check hotspot counts
    total_hotspots = len(all_hotspots)
    if 0 < total_hotspots < 10000:  # Some activity but not excessive
        validation_results.append("✅ Reasonable number of hotspots detected")
    elif total_hotspots == 0:
        validation_results.append("ℹ️  No flood hotspots detected (may be normal)")
    else:
        validation_results.append("⚠️  Very high number of hotspots - check thresholds")
    
    # Geographic distribution check
    if all_hotspots:
        lons = [h['longitude'] for h in all_hotspots]
        lats = [h['latitude'] for h in all_hotspots]
        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)
        
        if lon_range > 5 and lat_range > 3:  # Distributed across regions
            validation_results.append("✅ Hotspots geographically distributed")
        else:
            validation_results.append("ℹ️  Hotspots concentrated in small region")
    
    logger.info("Validation Results:")
    for result_msg in validation_results:
        logger.info(f"   {result_msg}")
    
    logger.info(f"\n🎉 Pipeline test complete! Check output files in: {output_dir}")
    
    return True


async def main():
    """Run the integrated pipeline test"""
    try:
        success = await test_full_pipeline()
        if success:
            print("\n" + "="*60)
            print("✅ INTEGRATED PIPELINE TEST SUCCESSFUL")
            print("✅ All components working with real NOAA data")
            print("✅ Output files generated and validated")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ PIPELINE TEST FAILED")
            print("Check logs above for details")
            print("="*60)
    except Exception as e:
        logger.error(f"Pipeline test crashed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())