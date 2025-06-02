"""MIMIC-IV Analysis Agent Module.

A comprehensive toolkit for analyzing MIMIC-IV clinical database.
This module provides:
- Data loading and preprocessing utilities
- Core analytical functions for predictive modeling
- Patient trajectory analysis
- Order pattern detection
- Clinical interpretation tools
- Exploratory data analysis capabilities
- Visualization components
"""

from .configurations.settings import setup_logging, logger

# Set up default logging when the package is imported
setup_logging()


__all__ = ["logger"]

__author__ = "Artin Majdi"
