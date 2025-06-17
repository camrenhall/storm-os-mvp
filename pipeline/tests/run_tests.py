#!/usr/bin/env python3
"""
Test runner for flood classification production readiness
"""

import pytest
import sys
from pathlib import Path

def main():
    """Run the full test suite"""
    
    # Add parent directory to path for imports (pipeline modules)
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    # Also add the pipeline subdirectory specifically
    pipeline_dir = parent_dir / "pipeline"
    if pipeline_dir.exists():
        sys.path.insert(0, str(pipeline_dir))
    
    # Test arguments
    args = [
        "--verbose",
        "--durations=10",
        "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        args.extend(["--cov=../pipeline", "--cov-report=term-missing"])
    except ImportError:
        print("pytest-cov not available, skipping coverage")
    
    # Run fast tests first
    print("=" * 60)
    print("PHASE 1: Running fast unit tests")
    print("=" * 60)
    
    fast_result = pytest.main(args + [
        "-m", "not slow",
        "test_g2_threshold_scoring.py",
        "test_g3_metadata_integrity.py", 
        "test_g4_fallback_adaptive.py",
        "test_g5_spatial_filtering.py",
        "test_g7_edge_cases.py"
    ])
    
    if fast_result != 0:
        print("❌ Fast tests failed, stopping")
        return fast_result
    
    print("\n" + "=" * 60)
    print("PHASE 2: Running slow integration tests")
    print("=" * 60)
    
    slow_result = pytest.main(args + [
        "-m", "slow",
        "test_g1_fixtures_decoding.py",
        "test_g6_performance.py"
    ])
    
    if slow_result != 0:
        print("❌ Slow tests failed")
        return slow_result
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - PRODUCTION READY")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())