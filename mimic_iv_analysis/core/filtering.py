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
			df_pre = self.df.compute()
   
			self.df = self.df[(self.df.anchor_age >= 18.0) & (self.df.anchor_age <= 75.0)]
			self.df = self.df[self.df.anchor_year_group == '2017 - 2019']
			self.df = self.df[ self.df.dod.isnull() ]
   
			df_post = self.df.compute()

			logger.info(f"Filtered {df_pre.shape[0]} rows to {df_post.shape[0]} rows")

		elif self.table_name == TableNames.DIAGNOSES_ICD:
			# Filter for rows where icd_version is 10
			self.df = self.df[self.df.icd_version.isin([10,'10'])]

			# TODO: add this filter.
			# Filter for rows where seq_num is 1, 2, or 3
			# self.df = self.df[self.df.seq_num.astype(int).isin([1, 2, 3])]

			# Filter for rows where the value in the column icd_code starts with "E11"
			self.df = self.df[self.df.icd_code.str.startswith('E11')]


		elif self.table_name == TableNames.D_ICD_DIAGNOSES:
			self.df = self.df[self.df.icd_version.isin([10,'10'])]


		elif self.table_name == TableNames.POE:
			
			if self.table_name.value in self.filter_params:

				poe_filters = self.filter_params[self.table_name.value]

				# Filter columns
				self.df = self.df[ poe_filters['selected_columns'] ]

				# Filter order types
				if poe_filters['apply_order_type']:
					self.df = self.df[ self.df.order_type.isin(poe_filters['order_type']) ]

				# Filter transaction types
				if poe_filters['apply_transaction_type']:
					self.df = self.df[ self.df.transaction_type.isin(poe_filters['transaction_type']) ]


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


		self.df = self.df.reset_index(drop=True)
		return self.df
