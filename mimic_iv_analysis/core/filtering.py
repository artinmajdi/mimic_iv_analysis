"""
Filtering module for MIMIC-IV data.

This module provides functionality for filtering MIMIC-IV data based on
inclusion and exclusion criteria from the MIMIC-IV dataset tables.
"""

# Standard library imports
from typing import Dict, List, Any, Optional, Tuple, Set

# Data processing imports
import pandas as pd
import dask.dataframe as dd
import numpy as np

from mimic_iv_analysis import logger
from mimic_iv_analysis.configurations.params import TableNamesHOSP, TableNamesICU


class Filtering:
	"""
	Class for applying inclusion and exclusion filters to MIMIC-IV data.

	This class provides methods to filter pandas DataFrames containing MIMIC-IV data
	based on various inclusion and exclusion criteria from the MIMIC-IV dataset tables.
	It handles the relationships between different tables and applies filters efficiently.
	"""

	def __init__(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNamesHOSP):
		"""Initialize the Filtering class."""

		self.df = df
		self.table_name = table_name


	def render(self) -> pd.DataFrame | dd.DataFrame:

		# TODO: apply this filter too
		if self.table_name == TableNamesHOSP.PATIENTS:
			self.df = self.df[(self.df.anchor_age >= 18.0) & (self.df.anchor_age <= 75.0)]
			self.df = self.df[self.df.anchor_year_group == '2017 - 2019']


		elif self.table_name == TableNamesHOSP.DIAGNOSES_ICD:
			# Filter for rows where icd_version is 10
			# TODO: why when i add this, I get errors.
			self.df = self.df[self.df.icd_version == 10]

			# Filter for rows where seq_num is 1, 2, or 3
			# self.df = self.df[self.df.seq_num.astype(int).isin([1, 2, 3])]

			# Filter for rows where the value in the column icd_code starts with "E11"
			self.df = self.df[self.df.icd_code.str.startswith('E11')]


		elif self.table_name == TableNamesHOSP.D_ICD_DIAGNOSES:
			self.df = self.df[self.df.icd_version == 10]


		elif self.table_name == TableNamesHOSP.POE:
			self.df = self.df.drop(columns=['discontinue_of_poe_id', 'discontinued_by_poe_id'])


		elif self.table_name == TableNamesHOSP.ADMISSIONS:

			# Get admission IDs where patient is alive
			self.df = self.df[(self.df.deathtime.isnull()) | (self.df.hospital_expire_flag == 0)]

			# Get admission IDs with valid admission and discharge times
			self.df = self.df.dropna(subset=['admittime', 'dischtime'])

			# Additional validation: dischtime should be after admittime
			self.df = self.df[self.df['dischtime'] > self.df['admittime']]

			# # Exclude admission types like "EW EMER.", "URGENT", or "ELECTIVE"
			# self.df = self.df[~self.df.admission_type.isin(['EW EMER.', 'URGENT', 'ELECTIVE'])]


		self.df = self.df.reset_index(drop=True)
		return self.df
