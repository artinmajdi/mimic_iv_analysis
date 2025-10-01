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

		elif self.table_name == TableNames.DIAGNOSES_ICD:

			icd_version = self.df.icd_version.isin([10])
			seq_num     = self.df.seq_num.isin([1,2,3])
			icd_code    = self.df.icd_code.str.startswith('E11')
			self.df     = self.df[icd_version & seq_num & icd_code]

		elif self.table_name == TableNames.D_ICD_DIAGNOSES:
			self.df = self.df[ self.df.icd_version.isin([10]) ]

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

			# Filter columns
			# self.df = self.df[ admissions_filters['selected_columns'] ]

			self.df = self.df.dropna(subset=['admittime', 'dischtime'])

			# Patient is alive
			exclude_in_hospital_death = (self.df.deathtime.isnull()) | (self.df.hospital_expire_flag == 0)

			# Discharge time is after admission time
			discharge_after_admission = self.df['dischtime'] > self.df['admittime']

			# Exclude admission types like “Emergency”, “Urgent”, or “Elective”
			admission_type    = ~self.df.admission_type.isin(['EW EMER.', 'DIRECT EMER.', 'URGENT', 'ELECTIVE'])

			self.df = self.df[ exclude_in_hospital_death & discharge_after_admission & admission_type]


		elif self.table_name == TableNames.TRANSFERS:

			# df2 = self.df.compute()

			# empty_cells = self.df.hadm_id != ''
			self.df = self.df[ ~self.df.hadm_id.isnull()]

			careunit = self.df.careunit.isin(['Medicine'])
			self.df = self.df[careunit]

		elif self.table_name == TableNames.MICROBIOLOGYEVENTS:
			self.df = self.df.drop(columns=['comments'])

		elif self.table_name == TableNames.LABEVENTS:
			self.df = self.df.drop(columns=['hadm_id', 'value', 'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments'])

		# Reset index
		self.df = self.df.reset_index(drop=True)
		return self.df
