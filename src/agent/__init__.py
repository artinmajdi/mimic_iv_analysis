"""MIMIC-IV Analysis Package.

A comprehensive toolkit for analyzing MIMIC-IV clinical database.
"""

# Import submodules to make them available when importing the package
from .data import data_loader
from .core import (
    predictive_modeling_module,
    patient_trajectory_module,
    order_pattern_module,
    clinical_interpretation_module,
    eda_module
)
from .configurations import load_config
from .visualization import app
from .utils import *
from .app import StreamlitApp

__all__ = [
    # Data modules
    'data_loader',

    # Configuration
    'load_config',

    # Core analytical modules
    'predictive_modeling_module',
    'patient_trajectory_module',
    'order_pattern_module',
    'clinical_interpretation_module',
    'eda_module',

    # Visualization
    'StreamlitApp'
]
