"""MIMIC-IV Data Loading and Preprocessing Module.

This module provides utilities for loading and preprocessing MIMIC-IV clinical data.
It includes:
- MIMICDataLoader class for efficient data loading
- Convenience functions for common data operations
- Preprocessing utilities for clinical data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data_loader import MIMICDataLoader


__all__ = [
    'MIMICDataLoader',
]
