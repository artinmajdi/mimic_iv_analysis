"""MIMIC-IV Analysis Package.

A comprehensive toolkit for analyzing MIMIC-IV clinical database,
providing modules for data preprocessing, visualization, and analysis.

This package contains the following modules:
- pipeline: Data preprocessing and pipeline components
- agent: Main analysis agent with predictive modeling capabilities
- agent2: Alternative analysis agent with specialized visualizations
- claude: Claude AI integration for advanced analysis

For usage examples, see the documentation or refer to the example scripts.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import claude, agent, agent2, pipeline


__version__ = "0.1.0"
__author__ = "Artin Majdi"
