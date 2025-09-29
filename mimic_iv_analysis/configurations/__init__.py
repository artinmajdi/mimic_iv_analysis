"""MIMIC-IV Analysis Configuration Module.

This module provides configuration management utilities for the MIMIC-IV
analysis package, enabling consistent settings across different components.

Features:
- Configuration loading from YAML files
- Default configuration management
- Configuration validation
"""

import os
import yaml

from .settings import logger
from .params import (   TableNames,
                        ColumnNames,
                        pyarrow_dtypes_map,
                        DEFAULT_MIMIC_PATH,
                        DEFAULT_NUM_SUBJECTS,
                        RANDOM_STATE,
                        SUBJECT_ID_COL,
                        DEFAULT_STUDY_TABLES_LIST,
                        DataFrameType
                        )

def load_config(config_path=None):
    """Load configuration from a YAML file.

    Args:
        config_path (str, optional): Path to the config file. If None, uses default location.

    Returns:
        dict: Configuration settings
    """
    if config_path is None:
        # Default to looking for config in the utils directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.yaml')

    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

__all__ = [
    'load_config',
    'logger',

    # Params
    'TableNames',
    'ColumnNames',
    'DEFAULT_MIMIC_PATH',
    'DEFAULT_NUM_SUBJECTS',
    'RANDOM_STATE',
    'SUBJECT_ID_COL',
    'DEFAULT_STUDY_TABLES_LIST',
    'DataFrameType',
    'pyarrow_dtypes_map',
]
