"""MIMIC-IV Data Loading Module for Order Pattern Analysis.

This module provides specialized data loading and preprocessing
utilities for order pattern analysis in MIMIC-IV clinical data.

Features:
- Efficient data loading for MIMIC-IV tables
- Pre-processing specific to provider order patterns
- Data transformations for clustering and pattern analysis
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data_loader import DataLoader

__all__ = ['DataLoader']
