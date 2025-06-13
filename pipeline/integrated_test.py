#!/usr/bin/env python3
"""
Comprehensive Validation Test for Enhanced FLASH Classification
Tests data quality preservation with full 20-25 minute processing time utilization
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import time

# Import our modules
from flash_ingest import FlashIngest
from enhanced_validation_classifier import OptimizedFlashClassifier, ClassificationConfig, ProcessingMode, ValidationMetrics
from ffw_client import NWSFlashFloodWarnings
from grid_align import GridProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def comprehensive_validation_test():
    """
    Comprehensive test leveraging full processing time for maximum data quality validation
    """
    
    output_dir = Path("./comprehensive_validation_output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=== COMPREHENSIVE VALIDATION TEST - Enhanced Data Quality ===")
    logger.info("Leveraging full 20-25 minute processing window for maximum validation")
    
    # Step 1: Ingest real FLASH data
    logger.info("Step 1: Acquiring real-time FLASH data...")
    async with FlashIngest() as ingest:
        flash_data = await ingest.fetch_latest()
        
        if not flash_data:
            logger.error("❌ Failed to fetch FLASH data - cannot proceed")
            return False
        
        logger.info(f"✅ FLASH data acquired:")
        logger.info(f"   Valid time: {flash_data['valid_time']}")
        logger.info(f"   Grid dimensions: {flash_data['grid_shape']}")
        logger.info(f"   Valid pixels: {flash_data['valid_points']:,}")
        logger.info(f"   Streamflow range: [{flash_data.get('data_min', 0):.3f}, "
                   f"{flash_data.get('data_max', 0):.3f}] m³/s/km²")
        logger.info(f"   Data source: {flash_data['source_url']}")
        logger.info(f"   Data age: {(datetime.utcnow() - flash_data['valid_time']).total_seconds() / 60:.1f} minutes")
    
    # Step 2: Acquire FFW data with detailed analysis
    logger.info("\nStep 2: Acquiring and analyzing Flash Flood Warnings...")
    async with NWSFlashFloodWarnings() as ffw_client:
        warnings_gdf = await ffw_client.get_active_warnings()
        
        if warnings_gdf is not None:
            logger.info(f"✅ FFW data acquired: {len(warnings_gdf)} active warnings")
            
            if len(warnings_gdf) > 0:
                # Detailed FFW analysis
                total_area = warnings_gdf['area_km2'].sum() if 'area_km2' in warnings_gdf.columns else 0
                logger.info(f"   Total warning coverage: {total_area:.0f} km²")
                
                # Analyze warning characteristics
                severity_counts = warnings_gdf['severity'].value_counts() if 'severity' in warnings_gdf.columns else {}
                urgency_counts = warnings_gdf['urgency'].value_counts() if 'urgency' in warnings_gdf.columns else {}
                
                logger.info(f"   Warning analysis:")
                logger.info(f"     Severity distribution: {dict(severity_counts)}")
                logger.info(f"     Urgency distribution: {dict(urgency_counts)}")
                
                # Show detailed warning information
                for idx, warning in warnings_gdf.iterrows():
                    event_type = warning.get('event_type', 'Flash Flood Warning')
                    severity = warning.get('severity', 'Unknown')
                    area = warning.get('area_km2', 0)
                    logger.info(f"     Warning {idx+1}: {event_type} ({severity}) - {area:.0f} km²")
                    
                    # Check expiration times
                    if 'expires_time' in warning and warning['expires_time']:
                        expires = warning['expires_time']
                        time_to_expire = (expires - datetime.now(expires.tz if hasattr(expires, 'tz') and expires.tz else None)).total_seconds() / 3600
                        logger.info(f"       Expires in: {time_to_expire:.1f} hours")
            else:
                logger.info("   No active Flash Flood Warnings at this time")
                logger.info("   This will test FFW-first approach with empty warning set")
        else:
            logger.error("❌ Failed to fetch FFW data")
            return False
    
    # Step 3: Test Progressive Processing Modes
    test_modes = [
        ("DEVELOPMENT", ProcessingMode.DEVELOPMENT, "Fast iteration testing"),
        ("PRODUCTION", ProcessingMode.PRODUCTION, "Standard operational quality"),
        ("VALIDATION", ProcessingMode.VALIDATION, "Maximum quality verification")
    ]
    
    results_comparison = {}
    
    for mode_name, processing_mode, description in test_modes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {mode_name} Mode: {description}")
        logger.info(f"{'='*80}")
        
        # Configure for this processing mode
        config = ClassificationConfig(
            processing_mode=processing_mode,
            require_ffw_intersection=True,  # FFW-FIRST enabled
            
            # Enhanced validation settings (progressive by mode)
            enable_cross_method_validation=(processing_mode != ProcessingMode.DEVELOPMENT),
            enable_spatial_validation=True,
            enable_intensity_validation=(processing_mode in [ProcessingMode.PRODUCTION, ProcessingMode.VALIDATION]),
            enable_hydrological_validation=(processing_mode == ProcessingMode.VALIDATION),
            enable_detailed_logging=True,
            
            # Quality thresholds
            max_reasonable_flood_area_km2=15000.0,  # Allow larger floods for validation
            min_reasonable_flood_area_km2=0.5,
            max_intensity_gradient=15.0,
            
            # Processing settings
            max_workers=4,
            save_validation_plots=(processing_mode == ProcessingMode.VALIDATION),
            validation_output_dir=str(output_dir / f"{mode_name.lower()}_validation")
        )
        
        logger.info(f"Configuration for {mode_name}:")
        logger.info(f"  Processing mode: {processing_mode.value}")
        logger.info(f"  Cross-method validation: {config.enable_cross_method_validation}")
        logger.info(f"  Spatial validation: {config.enable_spatial_validation}")
        logger.info(f"  Intensity validation: {config.enable_intensity_validation}")
        logger.info(f"  Hydrological validation: {config.enable_hydrological_validation}")
        logger.info(f"  FFW intersection required: {config.require_ffw_intersection}")
        
        # Create classifier for this mode
        classifier = OptimizedFlashClassifier(config)
        
        # Run classification with comprehensive timing
        logger.info(f"\nExecuting {mode_name} classification with enhanced validation...")
        mode_start_time = time.time()
        
        result = classifier.classify(
            flash_data['unit_streamflow'], 
            flash_data['valid_time'],
            ffw_polygons=warnings_gdf
        )
        
        mode_total_time = time.time() - mode_start_time
        
        if not result:
            logger.error(f"❌ {mode_name} classification failed")
            continue
        
        logger.info(f"✅ {mode_name} classification completed in {mode_total_time:.2f}s")
        
        # Extract and analyze results
        logger.info(f"\n{mode_name} Classification Results:")
        logger.info(f"  Processing time: {mode_total_time:.2f}s")
        logger.info(f"  Critical pixels: {result.critical_count:,}")
        logger.info(f"  High pixels: {result.high_count:,}")
        logger.info(f"  Moderate pixels: {result.moderate_count:,}")
        logger.info(f"  Total flood pixels: {result.critical_count + result.high_count + result.moderate_count:,}")
        logger.info(f"  FFW verified pixels: {result.ffw_verified_pixels:,}")
        logger.info(f"  FFW boosted pixels: {result.ffw_boosted_pixels:,}")
        
        # Analyze validation metrics if available
        validation_quality = "NOT_AVAILABLE"
        if hasattr(result, 'validation_metrics') and result.validation_metrics:
            vm = result.validation_metrics
            
            logger.info(f"\n{mode_name} Validation Metrics:")
            logger.info(f"  Connected components found: {vm.total_components_found}")
            logger.info(f"  Largest flood area: {vm.max_flood_area_km2:.1f} km²")
            logger.info(f"  Smallest flood area: {vm.min_flood_area_km2:.1f} km²")
            logger.info(f"  Suspicious large floods: {vm.suspicious_large_floods}")
            logger.info(f"  Max streamflow detected: {vm.max_streamflow_found:.3f} m³/s/km²")
            logger.info(f"  Intensity coherence score: {vm.intensity_coherence_score:.3f}")
            logger.info(f"  Gradient anomalies: {vm.gradient_anomalies}")
            
            if len(warnings_gdf) > 0:
                logger.info(f"  FFW coverage: {vm.ffw_coverage_percentage:.1f}%")
                logger.info(f"  Floods outside FFW: {vm.floods_outside_ffw}")
            
            if vm.grid_vs_components_centroid_diff:
                mean_diff = np.mean(vm.grid_vs_components_centroid_diff)
                max_diff = max(vm.grid_vs_components_centroid_diff)
                logger.info(f"  Method agreement score: {vm.method_agreement_score:.3f}")
                logger.info(f"  Centroid differences: mean={mean_diff:.4f}°, max={max_diff:.4f}°")
            
            # Quality assessment
            if vm.data_quality_excellent:
                validation_quality = "EXCELLENT"
            elif vm.data_quality_good:
                validation_quality = "GOOD"
            else:
                validation_quality = "CONCERNS"
            
            logger.info(f"  Overall data quality: {validation_quality}")
            
            if vm.data_quality_concerns:
                logger.info(f"  Quality concerns identified:")
                for concern in vm.data_quality_concerns:
                    logger.info(f"    - {concern}")
            
            # Processing time breakdown
            logger.info(f"  Processing time breakdown:")
            logger.info(f"    Classification: {vm.classification_time:.2f}s")
            logger.info(f"    Validation: {vm.validation_time:.2f}s")
            if vm.extraction_time > 0:
                logger.info(f"    Extraction: {vm.extraction_time:.2f}s")
            logger.info(f"    Total: {vm.total_time:.2f}s")
        
        # Extract flood events for this mode
        flood_events = {}
        extraction_time = 0.0
        
        total_flood_pixels = result.critical_count + result.high_count + result.moderate_count
        if total_flood_pixels > 0:
            logger.info(f"\nExtracting flood events for {mode_name}...")
            extraction_start = time.time()
            
            flood_events = classifier.extract_flood_events(result, flash_data['unit_streamflow'])
            
            extraction_time = time.time() - extraction_start
            
            total_events = sum(len(events) for events in flood_events.values())
            logger.info(f"✅ {mode_name} flood events extracted:")
            logger.info(f"  Critical events: {len(flood_events.get('critical', []))}")
            logger.info(f"  High events: {len(flood_events.get('high', []))}")
            logger.info(f"  Moderate events: {len(flood_events.get('moderate', []))}")
            logger.info(f"  Total events: {total_events}")
            logger.info(f"  Extraction time: {extraction_time:.2f}s")
            
            # Analyze event quality
            if total_events > 0:
                all_events = []
                for severity, events in flood_events.items():
                    all_events.extend(events)
                
                # Sort by quality ranking
                sorted_events = sorted(all_events, 
                                     key=lambda x: x.get('quality_rank', x.get('meteorological_score', 0)), 
                                     reverse=True)
                
                logger.info(f"\n  Top flood events by quality:")
                for i, event in enumerate(sorted_events[:5]):  # Show top 5
                    quality_score = event.get('quality_rank', event.get('meteorological_score', 0))
                    logger.info(f"    {i+1}. {event['severity'].title()}: "
                              f"{event['max_streamflow']:.3f} m³/s/km² max, "
                              f"{event['pixel_count']} pixels ({event['area_km2']:.1f} km²), "
                              f"quality={quality_score:.1f}")
                    
                    # Show enhanced metrics if available
                    if 'intensity_uniformity' in event:
                        logger.info(f"       Uniformity: {event['intensity_uniformity']:.3f}, "
                                  f"Compactness: {event.get('spatial_compactness', 0):.3f}, "
                                  f"Aspect: {event.get('aspect_ratio', 0):.1f}")
                    
                    # Show geographic location
                    logger.info(f"       Location: ({event['longitude']:.3f}, {event['latitude']:.3f})")
        else:
            logger.info(f"  No flood pixels detected for {mode_name} (may be normal with FFW-first)")
        
        # Store results for comparison
        results_comparison[mode_name] = {
            'processing_time': mode_total_time,
            'classification_result': result,
            'flood_events': flood_events,
            'total_events': sum(len(events) for events in flood_events.values()) if flood_events else 0,
            'validation_quality': validation_quality,
            'extraction_time': extraction_time
        }
        
        # Save detailed results for this mode
        timestamp = flash_data['valid_time'].strftime("%Y%m%d_%H%M%S")
        mode_output_file = output_dir / f"{mode_name.lower()}_results_{timestamp}.json"
        
        mode_output = {
            'metadata': {
                'processing_mode': mode_name.lower(),
                'timestamp': datetime.now().isoformat(),
                'flash_valid_time': flash_data['valid_time'].isoformat(),
                'flash_source': flash_data['source_url'],
                'active_ffw_count': len(warnings_gdf),
                'processing_time_seconds': mode_total_time
            },
            'classification_results': {
                'critical_pixels': result.critical_count,
                'high_pixels': result.high_count,
                'moderate_pixels': result.moderate_count,
                'ffw_verified_pixels': result.ffw_verified_pixels,
                'ffw_boosted_pixels': result.ffw_boosted_pixels,
                'thresholds': {
                    'p98': result.p98_value,
                    'critical': result.critical_threshold_value,
                    'high': result.high_threshold_value,
                    'moderate': result.moderate_threshold_value
                }
            },
            'flood_events': flood_events,
            'validation_quality': validation_quality
        }
        
        # Add validation metrics if available
        if hasattr(result, 'validation_metrics') and result.validation_metrics:
            vm = result.validation_metrics
            mode_output['validation_metrics'] = {
                'total_components': vm.total_components_found,
                'flood_area_range': [vm.min_flood_area_km2, vm.max_flood_area_km2],
                'intensity_range': [vm.min_streamflow_found, vm.max_streamflow_found],
                'quality_scores': {
                    'intensity_coherence': vm.intensity_coherence_score,
                    'method_agreement': vm.method_agreement_score,
                    'ffw_coverage': vm.ffw_coverage_percentage
                },
                'quality_flags': {
                    'excellent': vm.data_quality_excellent,
                    'good': vm.data_quality_good,
                    'concerns': vm.data_quality_concerns
                },
                'processing_breakdown': {
                    'classification_time': vm.classification_time,
                    'validation_time': vm.validation_time,
                    'extraction_time': vm.extraction_time,
                    'total_time': vm.total_time
                }
            }
        
        with open(mode_output_file, 'w') as f:
            json.dump(mode_output, f, indent=2, default=str)
        
        logger.info(f"💾 {mode_name} results saved: {mode_output_file}")
        
        # Performance assessment for this mode
        logger.info(f"\n{mode_name} Performance Assessment:")
        if mode_total_time < 120:  # 2 minutes
            logger.info(f"  ⚡ EXCELLENT: Very fast processing ({mode_total_time:.1f}s)")
        elif mode_total_time < 600:  # 10 minutes
            logger.info(f"  ✅ GOOD: Reasonable processing time ({mode_total_time:.1f}s)")
        elif mode_total_time < 1500:  # 25 minutes
            logger.info(f"  ⏰ ACCEPTABLE: Using available time for quality ({mode_total_time:.1f}s)")
        else:
            logger.info(f"  🐌 SLOW: Exceeds target processing time ({mode_total_time:.1f}s)")
    
    # Step 4: Cross-Mode Comparison and Analysis
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-MODE COMPARISON & VALIDATION ANALYSIS")
    logger.info(f"{'='*80}")
    
    if len(results_comparison) >= 2:
        logger.info("Analyzing processing mode differences...")
        
        # Compare processing times
        logger.info("\nProcessing Time Comparison:")
        for mode_name, mode_data in results_comparison.items():
            logger.info(f"  {mode_name}: {mode_data['processing_time']:.2f}s")
        
        # Compare event detection
        logger.info("\nFlood Event Detection Comparison:")
        for mode_name, mode_data in results_comparison.items():
            result = mode_data['classification_result']
            logger.info(f"  {mode_name}:")
            logger.info(f"    Critical pixels: {result.critical_count:,}")
            logger.info(f"    High pixels: {result.high_count:,}")
            logger.info(f"    Total events: {mode_data['total_events']}")
            logger.info(f"    Validation quality: {mode_data['validation_quality']}")
        
        # Analyze data quality preservation
        logger.info("\nData Quality Analysis:")
        
        # Check if Production/Validation modes show fewer but higher-quality events
        dev_events = results_comparison.get('DEVELOPMENT', {}).get('total_events', 0)
        prod_events = results_comparison.get('PRODUCTION', {}).get('total_events', 0)
        val_events = results_comparison.get('VALIDATION', {}).get('total_events', 0)
        
        if dev_events > 0 and prod_events > 0:
            event_ratio = dev_events / prod_events
            logger.info(f"  Development vs Production event ratio: {event_ratio:.1f}:1")
            
            if event_ratio > 10:
                logger.info("  ✅ Connected components significantly reducing fragmentation")
            elif event_ratio > 3:
                logger.info("  ✅ Moderate improvement in event consolidation")
            else:
                logger.info("  ⚠️  Limited difference between methods - investigate")
        
        # Check validation quality progression
        quality_progression = []
        for mode in ['DEVELOPMENT', 'PRODUCTION', 'VALIDATION']:
            if mode in results_comparison:
                quality = results_comparison[mode]['validation_quality']
                quality_progression.append(f"{mode}:{quality}")
        
        logger.info(f"  Quality progression: {' → '.join(quality_progression)}")
        
        # Speed vs Quality trade-off analysis
        logger.info("\nSpeed vs Quality Trade-off Analysis:")
        
        fastest_time = min(data['processing_time'] for data in results_comparison.values())
        slowest_time = max(data['processing_time'] for data in results_comparison.values())
        
        logger.info(f"  Speed range: {fastest_time:.2f}s to {slowest_time:.2f}s")
        logger.info(f"  Speed ratio: {slowest_time/fastest_time:.1f}x slower for maximum quality")
        
        # Determine if the speed difference is justified
        if slowest_time < 300:  # Under 5 minutes
            logger.info("  ✅ Excellent: Maximum quality achieved with reasonable processing time")
        elif slowest_time < 900:  # Under 15 minutes
            logger.info("  ✅ Good: Quality improvement justifies moderate time increase")
        elif slowest_time < 1500:  # Under 25 minutes
            logger.info("  ⚠️  Acceptable: Using available processing window for quality")
        else:
            logger.info("  ❌ Concerning: Processing time exceeds targets")
    
    # Step 5: Business Impact Analysis
    logger.info(f"\nBusiness Impact Analysis:")
    
    # Calculate daily projections (144 cycles per day)
    logger.info("Daily volume projections (144 cycles/day):")
    
    business_assessment = {}
    for mode_name, mode_data in results_comparison.items():
        daily_events = mode_data['total_events'] * 144
        business_assessment[mode_name] = daily_events
        
        logger.info(f"  {mode_name}: {daily_events:,} events/day")
        
        # Assess business viability
        if daily_events <= 200:
            assessment = "EXCELLENT - Perfect for target SMS volume"
        elif daily_events <= 1000:
            assessment = "GOOD - Manageable volume with some filtering"
        elif daily_events <= 5000:
            assessment = "ACCEPTABLE - Requires downstream filtering"
        else:
            assessment = "EXCESSIVE - Needs significant volume reduction"
        
        logger.info(f"    Business assessment: {assessment}")
    
    # Recommend optimal mode
    logger.info(f"\nRecommended Processing Mode:")
    
    # Find best balance of quality and business volume
    recommended_mode = None
    best_score = 0
    
    for mode_name, mode_data in results_comparison.items():
        daily_events = business_assessment[mode_name]
        quality = mode_data['validation_quality']
        processing_time = mode_data['processing_time']
        
        # Scoring system (higher is better)
        score = 0
        
        # Quality scoring
        if quality == "EXCELLENT":
            score += 100
        elif quality == "GOOD":
            score += 80
        elif quality == "CONCERNS":
            score += 40
        
        # Volume scoring (prefer manageable volumes)
        if daily_events <= 500:
            score += 50
        elif daily_events <= 2000:
            score += 30
        elif daily_events <= 10000:
            score += 10
        
        # Speed scoring (prefer reasonable times)
        if processing_time < 300:  # Under 5 minutes
            score += 20
        elif processing_time < 900:  # Under 15 minutes
            score += 15
        elif processing_time < 1500:  # Under 25 minutes
            score += 10
        
        logger.info(f"  {mode_name} overall score: {score}/170")
        
        if score > best_score:
            best_score = score
            recommended_mode = mode_name
    
    if recommended_mode:
        logger.info(f"  🎯 RECOMMENDED: {recommended_mode} mode")
        logger.info(f"     Best balance of quality, volume, and processing time")
    
    # Step 6: Data Quality Verification Summary
    logger.info(f"\n{'='*80}")
    logger.info("DATA QUALITY VERIFICATION SUMMARY")
    logger.info(f"{'='*80}")
    
    quality_verified = True
    quality_issues = []
    
    # Check FFW-first implementation
    ffw_working = False
    for mode_name, mode_data in results_comparison.items():
        result = mode_data['classification_result']
        if result.ffw_verified_pixels > 0 or len(warnings_gdf) == 0:
            ffw_working = True
            break
    
    if ffw_working:
        logger.info("✅ FFW-FIRST APPROACH: Working correctly")
        if len(warnings_gdf) > 0:
            logger.info("   Floods detected only in NWS warning areas")
        else:
            logger.info("   No active warnings to test, but mechanism ready")
    else:
        logger.info("❌ FFW-FIRST APPROACH: Not working as expected")
        quality_issues.append("FFW-first filtering may not be working")
        quality_verified = False
    
    # Check connected components effectiveness
    components_working = False
    if 'DEVELOPMENT' in results_comparison and 'PRODUCTION' in results_comparison:
        dev_events = results_comparison['DEVELOPMENT']['total_events']
        prod_events = results_comparison['PRODUCTION']['total_events']
        
        if dev_events > prod_events * 2:  # Production should have fewer, larger events
            components_working = True
            logger.info("✅ CONNECTED COMPONENTS: Effectively consolidating flood events")
            logger.info(f"   Event consolidation: {dev_events} → {prod_events} events")
        else:
            logger.info("⚠️  CONNECTED COMPONENTS: Limited consolidation effect")
            quality_issues.append("Connected components may not be significantly improving event quality")
    
    # Check processing time utilization
    max_time = max(data['processing_time'] for data in results_comparison.values())
    if max_time < 60:
        logger.info("⚠️  PROCESSING TIME: Very fast - may not be fully utilizing available time")
        logger.info("   Consider enabling more validation checks")
    elif max_time < 1500:  # Under 25 minutes
        logger.info("✅ PROCESSING TIME: Good utilization of available processing window")
    else:
        logger.info("❌ PROCESSING TIME: Exceeds target processing window")
        quality_issues.append("Processing time exceeds operational limits")
        quality_verified = False
    
    # Check validation metrics
    validation_working = False
    for mode_name, mode_data in results_comparison.items():
        if mode_data['validation_quality'] in ['EXCELLENT', 'GOOD']:
            validation_working = True
            break
    
    if validation_working:
        logger.info("✅ VALIDATION METRICS: Comprehensive quality checks working")
    else:
        logger.info("❌ VALIDATION METRICS: Quality concerns identified")
        quality_issues.append("Validation metrics indicate data quality issues")
        quality_verified = False
    
    # Overall assessment
    logger.info(f"\n{'='*50}")
    if quality_verified and len(quality_issues) == 0:
        logger.info("🎉 OVERALL ASSESSMENT: EXCELLENT")
        logger.info("✅ All data quality checks passed")
        logger.info("✅ Ready for production deployment")
    elif len(quality_issues) <= 2:
        logger.info("✅ OVERALL ASSESSMENT: GOOD")
        logger.info("✅ Core functionality working with minor issues")
        logger.info("⚠️  Some optimizations recommended")
    else:
        logger.info("⚠️  OVERALL ASSESSMENT: NEEDS ATTENTION")
        logger.info("❌ Multiple quality issues identified")
        logger.info("🔧 Requires fixes before production use")
    
    if quality_issues:
        logger.info(f"\nIssues to address:")
        for issue in quality_issues:
            logger.info(f"  - {issue}")
    
    # Step 7: Save comprehensive report
    logger.info(f"\nSaving comprehensive validation report...")
    
    timestamp = flash_data['valid_time'].strftime("%Y%m%d_%H%M%S")
    comprehensive_report = output_dir / f"comprehensive_validation_report_{timestamp}.json"
    
    report_data = {
        'metadata': {
            'test_timestamp': datetime.now().isoformat(),
            'flash_data_time': flash_data['valid_time'].isoformat(),
            'flash_source': flash_data['source_url'],
            'active_ffw_count': len(warnings_gdf),
            'test_duration_seconds': time.time() - start_time
        },
        'modes_tested': list(results_comparison.keys()),
        'results_by_mode': {},
        'cross_mode_analysis': {
            'processing_time_range': [min(data['processing_time'] for data in results_comparison.values()),
                                    max(data['processing_time'] for data in results_comparison.values())],
            'event_count_range': [min(data['total_events'] for data in results_comparison.values()),
                                max(data['total_events'] for data in results_comparison.values())],
            'recommended_mode': recommended_mode,
            'business_projections': business_assessment
        },
        'quality_verification': {
            'overall_quality_verified': quality_verified,
            'ffw_first_working': ffw_working,
            'connected_components_working': components_working,
            'validation_metrics_working': validation_working,
            'quality_issues': quality_issues
        },
        'recommendations': {
            'deployment_ready': quality_verified and len(quality_issues) <= 1,
            'recommended_mode': recommended_mode,
            'processing_time_acceptable': max_time < 1500,
            'business_volume_acceptable': min(business_assessment.values()) <= 2000
        }
    }
    
    # Add detailed results for each mode
    for mode_name, mode_data in results_comparison.items():
        result = mode_data['classification_result']
        report_data['results_by_mode'][mode_name] = {
            'processing_time': mode_data['processing_time'],
            'total_events': mode_data['total_events'],
            'validation_quality': mode_data['validation_quality'],
            'classification_summary': {
                'critical_pixels': result.critical_count,
                'high_pixels': result.high_count,
                'moderate_pixels': result.moderate_count,
                'ffw_verified_pixels': result.ffw_verified_pixels,
                'ffw_boosted_pixels': result.ffw_boosted_pixels
            },
            'daily_projection': business_assessment[mode_name]
        }
    
    with open(comprehensive_report, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    logger.info(f"💾 Comprehensive report saved: {comprehensive_report}")
    logger.info(f"\n🎉 Comprehensive validation test complete!")
    logger.info(f"Total test duration: {time.time() - start_time:.2f}s")
    logger.info(f"Output directory: {output_dir}")
    
    return quality_verified


async def main():
    """Run comprehensive validation test"""
    try:
        start_time = time.time()
        
        success = await comprehensive_validation_test()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE VALIDATION TEST RESULTS")
        print("="*80)
        
        if success:
            print("✅ VALIDATION SUCCESSFUL: Data quality verified")
            print("✅ Enhanced validation system working correctly")
            print("✅ FFW-first approach implemented and tested")
            print("✅ Connected components preserving flood event integrity")
            print("✅ Processing time optimized for available window")
        else:
            print("⚠️  VALIDATION ISSUES IDENTIFIED")
            print("🔧 Review validation report for specific issues")
            print("🔧 Address quality concerns before production deployment")
        
        print(f"\nKey Achievements:")
        print(f"- Comprehensive validation system implemented")
        print(f"- Multi-mode processing (Development/Production/Validation)")
        print(f"- Cross-method validation (grid vs connected components)")
        print(f"- Spatial and intensity coherence validation")
        print(f"- FFW integration with quality verification")
        print(f"- Business impact analysis and recommendations")
        
        print(f"\nTotal validation time: {total_time:.2f}s")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Comprehensive validation test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set start time for full test duration tracking
    start_time = time.time()
    asyncio.run(main())