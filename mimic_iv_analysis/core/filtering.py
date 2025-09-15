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
	Class for applying inclusion and exclusion filter_params to MIMIC-IV data.

	This class provides methods to filter pandas DataFrames containing MIMIC-IV data
	based on various inclusion and exclusion criteria from the MIMIC-IV dataset tables.
	It handles the relationships between different tables and applies filter_params efficiently.
	"""

	def __init__(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNames, filter_params: dict = {}):
		"""Initialize the Filtering class."""

		self.df = df
		self.table_name = table_name
		self.filter_params = filter_params


	def render(self) -> pd.DataFrame | dd.DataFrame:

		if self.table_name == TableNames.PATIENTS:

			anchor_age        = (self.df.anchor_age >= 18.0) & (self.df.anchor_age <= 75.0)
			anchor_year_group = self.df.anchor_year_group.isin(['2017 - 2019'])
			dod               = self.df.dod.isnull()
			self.df           = self.df[anchor_age & anchor_year_group & dod]


			# Exclude admission types like “Emergency”, “Urgent”, or “Elective”
			# ADMISSION_TYPES     = ['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION']
			# self.df = self.df[ ~self.df.admission_type.isin(['EMERGENCY', 'URGENT', 'ELECTIVE']) ]

		elif self.table_name == TableNames.DIAGNOSES_ICD:

			icd_version = self.df.icd_version.isin(['10'])
			seq_num     = self.df.seq_num.isin(['1', '2', '3'])
			icd_code    = self.df.icd_code.str.startswith('E11')
			self.df     = self.df[icd_version & seq_num & icd_code]

		elif self.table_name == TableNames.D_ICD_DIAGNOSES:
			self.df = self.df[ self.df.icd_version.isin([10,'10']) ]


		elif self.table_name == TableNames.POE:

			if self.table_name.value in self.filter_params:

				poe_filters_params = self.filter_params[self.table_name.value]

				# Filter columns
				self.df = self.df[ poe_filters_params['selected_columns'] ]

				if poe_filters_params['apply_order_type']:
					self.df = self.df[ self.df.order_type.isin(poe_filters_params['order_type']) ]

				if poe_filters_params['apply_transaction_type']:
					self.df = self.df[ self.df.transaction_type.isin(poe_filters_params['transaction_type']) ]


		elif self.table_name == TableNames.ADMISSIONS:

			if self.table_name.value in self.filter_params:

				admissions_filters = self.filter_params[self.table_name.value]

				# Filter columns
				self.df = self.df[ admissions_filters['selected_columns'] ]


				# Valid admission and discharge times
				if admissions_filters['valid_admission_discharge']:
					self.df = self.df.dropna(subset=['admittime', 'dischtime'])


				# Patient is alive
				if admissions_filters['exclude_in_hospital_death']:
					self.df = self.df[ (self.df.deathtime.isnull()) | (self.df.hospital_expire_flag == 0) ]


				# Discharge time is after admission time
				if admissions_filters['discharge_after_admission']:
					self.df = self.df[ self.df['dischtime'] > self.df['admittime'] ]


				# Apply admission types
				if admissions_filters['apply_admission_type']:
					self.df = self.df[ self.df.admission_type.isin(admissions_filters['admission_type']) ]

				# Apply admission location
				if admissions_filters['apply_admission_location']:
					self.df = self.df[ self.df.admission_location.isin(admissions_filters['admission_location']) ]

			# Exclude admission types like "EW EMER.", "URGENT", or "ELECTIVE"
			# self.df = self.df[~self.df.admission_type.isin(['EW EMER.', 'URGENT', 'ELECTIVE'])]


		elif self.table_name == TableNames.TRANSFERS:
			# self.df = self.df.dropna(subset=['hadm_id'])
			self.df = self.df[self.df.hadm_id != '']
			# if 'hadm_id' in self.df.columns:
			# 	self.df['hadm_id'] = self.df['hadm_id'].astype('int64')


			# Identify inpatient locations using transfers table and careunit column (to find “Med”, “Surg”, “Ortho”, “Card”, “GU”, “MICU”, “SICU”, “CSRU”


		self.df = self.df.reset_index(drop=True)
		return self.df
