"""
Filtering module for MIMIC-IV data.

This module provides functionality for filtering MIMIC-IV data based on
inclusion and exclusion criteria from the MIMIC-IV dataset tables.
"""

import pandas as pd
import dask.dataframe as dd

from mimic_iv_analysis import logger
from mimic_iv_analysis.configurations.params import TableNames, TableNames


class Filtering:
	"""
	Class for applying inclusion and exclusion filters to MIMIC-IV data.

	This class provides methods to filter pandas DataFrames containing MIMIC-IV data
	based on various inclusion and exclusion criteria from the MIMIC-IV dataset tables.
	It handles the relationships between different tables and applies filters efficiently.
	"""

	def __init__(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNames):
		"""Initialize the Filtering class."""

		self.df = df
		self.table_name = table_name


	def render(self) -> pd.DataFrame | dd.DataFrame:

		if self.table_name == TableNames.PATIENTS:
			self.df = self.df[(self.df.anchor_age >= 18.0) & (self.df.anchor_age <= 75.0)]
			self.df = self.df[self.df.anchor_year_group == '2017 - 2019']


		elif self.table_name == TableNames.DIAGNOSES_ICD:
			# Filter for rows where icd_version is 10
			self.df = self.df[self.df.icd_version.isin([10,'10'])]

			# Filter for rows where seq_num is 1, 2, or 3
			# self.df = self.df[self.df.seq_num.astype(int).isin([1, 2, 3])]

			# Filter for rows where the value in the column icd_code starts with "E11"
			self.df = self.df[self.df.icd_code.str.startswith('E11')]


		elif self.table_name == TableNames.D_ICD_DIAGNOSES:
			self.df = self.df[self.df.icd_version.isin([10,'10'])]


		elif self.table_name == TableNames.POE:
			self.df = self.df.drop(columns=['discontinue_of_poe_id', 'discontinued_by_poe_id'])


		elif self.table_name == TableNames.ADMISSIONS:

			# Get admission IDs where patient is alive
			self.df = self.df[(self.df.deathtime.isnull()) | (self.df.hospital_expire_flag == 0)]

			# Get admission IDs with valid admission and discharge times
			self.df = self.df.dropna(subset=['admittime', 'dischtime'])

			# Additional validation: dischtime should be after admittime
			self.df = self.df[self.df['dischtime'] > self.df['admittime']]

			# Exclude admission types like "EW EMER.", "URGENT", or "ELECTIVE"
			# self.df = self.df[~self.df.admission_type.isin(['EW EMER.', 'URGENT', 'ELECTIVE'])]

		elif self.table_name == TableNames.TRANSFERS:
			# self.df = self.df.dropna(subset=['hadm_id'])
			self.df = self.df[self.df.hadm_id != '']
			# if 'hadm_id' in self.df.columns:
			# 	self.df['hadm_id'] = self.df['hadm_id'].astype('int64')


		self.df = self.df.reset_index(drop=True)
		return self.df
