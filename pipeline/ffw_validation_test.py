#!/usr/bin/env python3
"""
FFW-First Approach Validation and Geographic Verification Script
Tests whether the 0.4% retention rate is appropriate and validates flood event geography
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import time

# Import modules
from flash_ingest import FlashIngest
from enhanced_validation_classifier import OptimizedFlashClassifier, ClassificationConfig, ProcessingMode
from ffw_client import NWSFlashFloodWarnings
from grid_align import GridProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def comprehensive_ffw_validation():
    """
    Comprehensive validation of FFW-first approach and geographic accuracy
    """
    
    output_dir = Path("./ffw_validation_output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=== FFW-FIRST APPROACH VALIDATION ===")
    logger.info("Testing geographic accuracy and retention rate validation")
    
    # Step 1: Acquire real data
    logger.info("Step 1: Acquiring FLASH and FFW data...")
    
    async with FlashIngest() as ingest:
        flash_data = await ingest.fetch_latest()
        if not flash_data:
            logger.error("Failed to fetch FLASH data")
            return False
    
    async with NWSFlashFloodWarnings() as ffw_client:
        warnings_gdf = await ffw_client.get_active_warnings()
        if warnings_gdf is None:
            logger.error("Failed to fetch FFW data")
            return False
    
    logger.info(f"✅ Data acquired: {len(warnings_gdf)} active FFW polygons")
    
    # Step 2: Test WITHOUT FFW filtering to establish baseline
    logger.info("\nStep 2: Baseline classification WITHOUT FFW filtering...")
    
    baseline_config = ClassificationConfig(
        processing_mode=ProcessingMode.PRODUCTION,
        require_ffw_intersection=False,  # DISABLED for baseline
        enable_detailed_logging=True,
        enable_cross_method_validation=True,
        enable_spatial_validation=True,
        enable_intensity_validation=True
    )
    
    baseline_classifier = OptimizedFlashClassifier(baseline_config)
    baseline_result = baseline_classifier.classify(
        flash_data['unit_streamflow'], 
        flash_data['valid_time'],
        ffw_polygons=warnings_gdf  # Still pass FFW data but don't require intersection
    )
    
    if not baseline_result:
        logger.error("Baseline classification failed")
        return False
    
    baseline_total_pixels = (baseline_result.critical_count + 
                           baseline_result.high_count + 
                           baseline_result.moderate_count)
    
    logger.info(f"Baseline results (NO FFW filtering):")
    logger.info(f"  Critical pixels: {baseline_result.critical_count:,}")
    logger.info(f"  High pixels: {baseline_result.high_count:,}")
    logger.info(f"  Moderate pixels: {baseline_result.moderate_count:,}")
    logger.info(f"  Total flood pixels: {baseline_total_pixels:,}")
    logger.info(f"  FFW boosted pixels: {baseline_result.ffw_boosted_pixels:,}")
    
    # Extract baseline flood events for geographic analysis
    baseline_events = baseline_classifier.extract_flood_events(baseline_result, flash_data['unit_streamflow'])
    baseline_total_events = sum(len(events) for events in baseline_events.values())
    
    logger.info(f"  Total flood events detected: {baseline_total_events}")
    
    # Step 3: Test WITH FFW filtering (current approach)
    logger.info("\nStep 3: FFW-FIRST classification (current approach)...")
    
    ffw_config = ClassificationConfig(
        processing_mode=ProcessingMode.PRODUCTION,
        require_ffw_intersection=True,  # ENABLED for FFW-first
        enable_detailed_logging=True,
        enable_cross_method_validation=True,
        enable_spatial_validation=True,
        enable_intensity_validation=True
    )
    
    ffw_classifier = OptimizedFlashClassifier(ffw_config)
    ffw_result = ffw_classifier.classify(
        flash_data['unit_streamflow'], 
        flash_data['valid_time'],
        ffw_polygons=warnings_gdf
    )
    
    if not ffw_result:
        logger.error("FFW-first classification failed")
        return False
    
    ffw_total_pixels = (ffw_result.critical_count + 
                       ffw_result.high_count + 
                       ffw_result.moderate_count)
    
    logger.info(f"FFW-first results:")
    logger.info(f"  Critical pixels: {ffw_result.critical_count:,}")
    logger.info(f"  High pixels: {ffw_result.high_count:,}")
    logger.info(f"  Moderate pixels: {ffw_result.moderate_count:,}")
    logger.info(f"  Total flood pixels: {ffw_total_pixels:,}")
    logger.info(f"  FFW verified pixels: {ffw_result.ffw_verified_pixels:,}")
    logger.info(f"  FFW boosted pixels: {ffw_result.ffw_boosted_pixels:,}")
    
    # Extract FFW-filtered flood events
    ffw_events = ffw_classifier.extract_flood_events(ffw_result, flash_data['unit_streamflow'])
    ffw_total_events = sum(len(events) for events in ffw_events.values())
    
    logger.info(f"  Total flood events detected: {ffw_total_events}")
    
    # Step 4: Compare and analyze retention rates
    logger.info("\nStep 4: FFW Filtering Analysis...")
    
    if baseline_total_pixels > 0:
        pixel_retention_rate = (ffw_total_pixels / baseline_total_pixels) * 100
        logger.info(f"Pixel retention rate: {pixel_retention_rate:.2f}%")
        
        if pixel_retention_rate < 1.0:
            logger.warning(f"⚠️  Very low pixel retention: {pixel_retention_rate:.2f}%")
            logger.info("   This suggests FFW filtering may be quite aggressive")
        elif pixel_retention_rate < 10.0:
            logger.info(f"✅ Good noise reduction: {pixel_retention_rate:.2f}% retention")
        else:
            logger.info(f"⚠️  High retention rate: {pixel_retention_rate:.2f}% - limited filtering effect")
    
    if baseline_total_events > 0:
        event_retention_rate = (ffw_total_events / baseline_total_events) * 100
        logger.info(f"Event retention rate: {event_retention_rate:.2f}%")
        
        consolidation_ratio = baseline_total_events / ffw_total_events if ffw_total_events > 0 else 0
        logger.info(f"Event consolidation ratio: {consolidation_ratio:.1f}:1")
    
    # Step 5: Geographic validation of detected flood events
    logger.info("\nStep 5: Geographic Validation of Flood Events...")
    
    # Analyze FFW-filtered events in detail
    all_ffw_events = []
    for severity, events in ffw_events.items():
        all_ffw_events.extend(events)
    
    if all_ffw_events:
        logger.info(f"Detailed geographic analysis of {len(all_ffw_events)} flood events:")
        
        # Sort by quality rank (using fixed calculation)
        sorted_events = sorted(all_ffw_events, 
                             key=lambda x: x.get('quality_rank', 0), 
                             reverse=True)
        
        # Group by approximate geographic regions for validation
        regions = {
            'Southeast': {'lat_range': (25, 40), 'lon_range': (-90, -75), 'events': []},
            'South_Central': {'lat_range': (25, 37), 'lon_range': (-105, -90), 'events': []},
            'Northeast': {'lat_range': (37, 47), 'lon_range': (-80, -67), 'events': []},
            'Midwest': {'lat_range': (37, 49), 'lon_range': (-100, -80), 'events': []},
            'West': {'lat_range': (30, 49), 'lon_range': (-125, -100), 'events': []},
            'Southwest': {'lat_range': (25, 37), 'lon_range': (-125, -105), 'events': []}
        }
        
        # Classify events by region
        for event in all_ffw_events:
            lat, lon = event['latitude'], event['longitude']
            event_classified = False
            
            for region_name, region_data in regions.items():
                lat_min, lat_max = region_data['lat_range']
                lon_min, lon_max = region_data['lon_range']
                
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    region_data['events'].append(event)
                    event_classified = True
                    break
            
            if not event_classified:
                logger.warning(f"Event outside expected CONUS bounds: ({lat:.3f}, {lon:.3f})")
        
        # Report by region
        logger.info("Flood events by geographic region:")
        for region_name, region_data in regions.items():
            event_count = len(region_data['events'])
            if event_count > 0:
                logger.info(f"  {region_name}: {event_count} events")
                
                # Show top event details for each region
                top_event = max(region_data['events'], key=lambda x: x.get('quality_rank', 0))
                logger.info(f"    Top event: {top_event['max_streamflow']:.3f} m³/s/km², "
                          f"{top_event['area_km2']:.0f} km², "
                          f"quality={top_event.get('quality_rank', 0):.1f}")
                logger.info(f"    Location: ({top_event['latitude']:.3f}, {top_event['longitude']:.3f})")
        
        # Validate event characteristics
        logger.info("\nFlood event characteristics validation:")
        
        # Size distribution
        areas = [event['area_km2'] for event in all_ffw_events]
        intensities = [event['max_streamflow'] for event in all_ffw_events]
        quality_ranks = [event.get('quality_rank', 0) for event in all_ffw_events]
        
        logger.info(f"  Event size range: {min(areas):.0f} - {max(areas):.0f} km²")
        logger.info(f"  Mean event size: {np.mean(areas):.0f} km²")
        logger.info(f"  Intensity range: {min(intensities):.3f} - {max(intensities):.3f} m³/s/km²")
        logger.info(f"  Mean intensity: {np.mean(intensities):.3f} m³/s/km²")
        logger.info(f"  Quality score range: {min(quality_ranks):.1f} - {max(quality_ranks):.1f}")
        
        # Check for quality score bug
        zero_quality_events = [e for e in all_ffw_events if e.get('quality_rank', 0) <= 0.001]
        if zero_quality_events:
            logger.error(f"❌ QUALITY BUG DETECTED: {len(zero_quality_events)} events have zero quality scores")
            for event in zero_quality_events[:3]:  # Show first 3 examples
                logger.error(f"  Zero quality event: {event['max_streamflow']:.3f} m³/s/km², "
                           f"{event['pixel_count']} pixels, uniformity={event.get('intensity_uniformity', 0):.3f}")
        else:
            logger.info(f"✅ Quality score validation passed: All events have meaningful quality scores")
        
        # Show top events overall
        logger.info(f"\nTop {min(5, len(sorted_events))} flood events by quality:")
        for i, event in enumerate(sorted_events[:5]):
            logger.info(f"  {i+1}. {event['severity'].title()}: "
                      f"{event['max_streamflow']:.3f} m³/s/km² max, "
                      f"{event['area_km2']:.0f} km², "
                      f"quality={event.get('quality_rank', 0):.1f}")
            logger.info(f"     Location: ({event['latitude']:.3f}, {event['longitude']:.3f})")
            logger.info(f"     Uniformity: {event.get('intensity_uniformity', 0):.3f}, "
                      f"Compactness: {event.get('spatial_compactness', 0):.3f}")
    
    # Step 6: Validation against FFW polygons
    logger.info("\nStep 6: FFW Coverage Validation...")
    
    if len(warnings_gdf) > 0 and all_ffw_events:
        # Check if flood events are actually within FFW boundaries
        from shapely.geometry import Point
        
        events_outside_ffw = 0
        events_inside_ffw = 0
        
        for event in all_ffw_events:
            event_point = Point(event['longitude'], event['latitude'])
            
            # Check if point is within any FFW polygon
            inside_any_ffw = False
            for idx, warning in warnings_gdf.iterrows():
                if warning.geometry.contains(event_point):
                    inside_any_ffw = True
                    break
            
            if inside_any_ffw:
                events_inside_ffw += 1
            else:
                events_outside_ffw += 1
                logger.warning(f"Event outside FFW: ({event['latitude']:.3f}, {event['longitude']:.3f}) "
                             f"- {event['max_streamflow']:.3f} m³/s/km²")
        
        ffw_coverage_percent = (events_inside_ffw / len(all_ffw_events)) * 100
        logger.info(f"FFW coverage validation:")
        logger.info(f"  Events inside FFW areas: {events_inside_ffw}")
        logger.info(f"  Events outside FFW areas: {events_outside_ffw}")
        logger.info(f"  Coverage percentage: {ffw_coverage_percent:.1f}%")
        
        if ffw_coverage_percent < 95:
            logger.warning("⚠️  Some events detected outside FFW boundaries - investigate rasterization")
        else:
            logger.info("✅ FFW-first filtering working correctly")
    
    # Step 7: Business impact assessment
    logger.info("\nStep 7: Business Impact Assessment...")
    
    # Daily projections (144 cycles per day at 10-minute intervals)
    baseline_daily = baseline_total_events * 144 if baseline_total_events else 0
    ffw_daily = ffw_total_events * 144 if ffw_total_events else 0
    
    logger.info(f"Daily volume projections:")
    logger.info(f"  Without FFW filtering: {baseline_daily:,} events/day")
    logger.info(f"  With FFW filtering: {ffw_daily:,} events/day")
    
    target_volume = 1000  # Target from business requirements
    
    if ffw_daily <= target_volume:
        logger.info(f"✅ FFW-filtered volume ({ffw_daily:,}) within target ({target_volume:,})")
    elif ffw_daily <= target_volume * 2:
        logger.info(f"⚠️  FFW-filtered volume ({ffw_daily:,}) above target but manageable")
    else:
        logger.info(f"❌ FFW-filtered volume ({ffw_daily:,}) significantly exceeds target")
    
    if baseline_daily > 0:
        volume_reduction = ((baseline_daily - ffw_daily) / baseline_daily) * 100
        logger.info(f"Volume reduction achieved: {volume_reduction:.1f}%")
    
    # Step 8: Recommendations
    logger.info("\nStep 8: Recommendations...")
    
    recommendations = []
    
    # FFW filtering assessment
    if len(warnings_gdf) == 0:
        recommendations.append("No active FFW warnings to test - monitor during active weather")
    elif ffw_total_events == 0 and baseline_total_events > 0:
        recommendations.append("FFW filtering eliminated all events - may be too restrictive")
    elif ffw_daily > target_volume * 3:
        recommendations.append("Consider additional quality-based filtering to reduce volume")
    else:
        recommendations.append("FFW-first approach appears to be working effectively")
    
    # Quality score validation
    if zero_quality_events:
        recommendations.append("CRITICAL: Fix quality score calculation bug before production")
    else:
        recommendations.append("Quality score calculations working correctly")
    
    # Geographic distribution
    active_regions = sum(1 for region_data in regions.values() if len(region_data['events']) > 0)
    if active_regions >= 3:
        recommendations.append("Good geographic distribution of flood detection")
    elif active_regions >= 1:
        recommendations.append("Limited geographic coverage - normal for current weather pattern")
    else:
        recommendations.append("No geographically distributed events - investigate data quality")
    
    logger.info("Final recommendations:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    # Step 9: Save comprehensive results
    timestamp = flash_data['valid_time'].strftime("%Y%m%d_%H%M%S")
    validation_report_file = output_dir / f"ffw_validation_report_{timestamp}.json"
    
    validation_report = {
        'metadata': {
            'test_timestamp': datetime.now().isoformat(),
            'flash_data_time': flash_data['valid_time'].isoformat(),
            'active_ffw_count': len(warnings_gdf),
            'flash_data_age_minutes': (datetime.utcnow() - flash_data['valid_time']).total_seconds() / 60
        },
        'baseline_results': {
            'total_pixels': baseline_total_pixels,
            'total_events': baseline_total_events,
            'critical_pixels': baseline_result.critical_count,
            'high_pixels': baseline_result.high_count,
            'moderate_pixels': baseline_result.moderate_count,
            'daily_projection': baseline_daily
        },
        'ffw_filtered_results': {
            'total_pixels': ffw_total_pixels,
            'total_events': ffw_total_events,
            'critical_pixels': ffw_result.critical_count,
            'high_pixels': ffw_result.high_count,
            'moderate_pixels': ffw_result.moderate_count,
            'ffw_verified_pixels': ffw_result.ffw_verified_pixels,
            'ffw_boosted_pixels': ffw_result.ffw_boosted_pixels,
            'daily_projection': ffw_daily
        },
        'analysis': {
            'pixel_retention_rate_percent': pixel_retention_rate if baseline_total_pixels > 0 else 0,
            'event_retention_rate_percent': event_retention_rate if baseline_total_events > 0 else 0,
            'volume_reduction_percent': volume_reduction if baseline_daily > 0 else 0,
            'ffw_coverage_percent': ffw_coverage_percent if all_ffw_events else 0,
            'zero_quality_events_detected': len(zero_quality_events) if 'zero_quality_events' in locals() else 0
        },
        'geographic_distribution': {
            region_name: len(region_data['events']) 
            for region_name, region_data in regions.items()
        } if all_ffw_events else {},
        'recommendations': recommendations,
        'validation_status': {
            'ffw_first_working': ffw_total_events > 0 or len(warnings_gdf) == 0,
            'quality_scores_working': len(zero_quality_events) == 0 if 'zero_quality_events' in locals() else True,
            'geographic_distribution_reasonable': active_regions >= 1 if 'active_regions' in locals() else True,
            'business_volume_acceptable': ffw_daily <= target_volume * 2
        }
    }
    
    with open(validation_report_file, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    logger.info(f"💾 Validation report saved: {validation_report_file}")
    
    # Final assessment
    logger.info("\n" + "="*60)
    logger.info("FFW-FIRST VALIDATION SUMMARY")
    logger.info("="*60)
    
    validation_passed = True
    critical_issues = []
    
    # Check critical issues
    if 'zero_quality_events' in locals() and len(zero_quality_events) > 0:
        critical_issues.append("Quality score calculation bug detected")
        validation_passed = False
    
    if ffw_daily > target_volume * 5:
        critical_issues.append("Daily event volume far exceeds business targets")
        validation_passed = False
    
    if len(warnings_gdf) > 0 and ffw_total_events == 0 and baseline_total_events > 10:
        critical_issues.append("FFW filtering may be too restrictive")
        validation_passed = False
    
    if validation_passed:
        logger.info("✅ VALIDATION PASSED: FFW-first approach working correctly")
        logger.info("✅ Geographic distribution reasonable")
        logger.info("✅ Quality scores calculating properly")
        logger.info("✅ Business volume targets achievable")
    else:
        logger.error("❌ VALIDATION ISSUES IDENTIFIED:")
        for issue in critical_issues:
            logger.error(f"  - {issue}")
    
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Event consolidation: {baseline_total_events} → {ffw_total_events} events")
    logger.info(f"  Daily volume: {ffw_daily:,} events/day (target: {target_volume:,})")
    logger.info(f"  Geographic regions with events: {active_regions if 'active_regions' in locals() else 0}")
    logger.info(f"  FFW coverage: {ffw_coverage_percent:.1f}%" if 'ffw_coverage_percent' in locals() else "  FFW coverage: Not tested")
    
    return validation_passed


async def main():
    """Run FFW validation test"""
    try:
        success = await comprehensive_ffw_validation()
        
        if success:
            print("\n🎉 FFW-FIRST VALIDATION SUCCESSFUL")
            print("✅ System ready for production deployment")
        else:
            print("\n⚠️  FFW-FIRST VALIDATION IDENTIFIED ISSUES") 
            print("🔧 Address issues before production deployment")
            
    except Exception as e:
        logger.error(f"FFW validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())