"""
Filtering module for MIMIC-IV data.

This module provides functionality for filtering MIMIC-IV data based on
inclusion and exclusion criteria from the MIMIC-IV dataset tables.
"""

# Standard library imports
import logging
from typing import Dict, List, Any, Optional, Tuple, Set

# Data processing imports
import pandas as pd
import dask.dataframe as dd
import numpy as np

from mimic_iv_analysis.io.data_loader import TableNamesHOSP, TableNamesICU

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Filtering:
	"""
	Class for applying inclusion and exclusion filters to MIMIC-IV data.

	This class provides methods to filter pandas DataFrames containing MIMIC-IV data
	based on various inclusion and exclusion criteria from the MIMIC-IV dataset tables.
	It handles the relationships between different tables and applies filters efficiently.
	"""

	def __init__(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNamesHOSP | TableNamesICU):
		"""Initialize the Filtering class."""

		self.df = df
		self.table_name = table_name


	def render(self) -> pd.DataFrame | dd.DataFrame:

		if self.table_name == TableNamesHOSP.PATIENTS:
			return self.df[(self.df.anchor_age >= 18.0) & (self.df.anchor_age <= 75.0)]

		if self.table_name in [TableNamesHOSP.DIAGNOSES_ICD, TableNamesHOSP.D_ICD_DIAGNOSES]:
			return self.df[self.df.icd_version == 10]

		if self.table_name == TableNamesHOSP.POE:
			return self.df.drop(columns=['discontinue_of_poe_id', 'discontinued_by_poe_id'])

		if self.table_name == TableNamesHOSP.ADMISSIONS:

			# Get admission IDs where patient is alive
			self.df = self.df[self.df.deathtime.isnull()]

			# Get admission IDs with valid admission and discharge times
			self.df = self.df.dropna(subset=['admittime', 'dischtime'])

			# Additional validation: dischtime should be after admittime
			self.df = self.df[self.df['dischtime'] > self.df['admittime']]

			return self.df


		return self.df
