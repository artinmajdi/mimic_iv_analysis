"""MIMIC-IV Model Module.

This module provides machine learning models for analyzing MIMIC-IV data.
It includes:
- Prediction models for clinical outcomes
- Feature importance analysis
- Model evaluation utilities
- Model serialization and loading functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import from parent module's model_module.py if available
try:
    from .. import model_module
except ImportError:
    pass

__all__ = []
