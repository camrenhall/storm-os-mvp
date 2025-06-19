#!/usr/bin/env python3
"""
Flood-Lead Intelligence Generator Package
Orchestrates 30-minute data ingestion and flood detection pipeline
"""

__version__ = "1.0.0"
__author__ = "Flood-Lead Intelligence Team"

from .run import main as run_generator

__all__ = ['run_generator']