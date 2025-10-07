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
		"""
		Apply filtering criteria to MIMIC-IV data based on table-specific inclusion and exclusion rules.
		
		Returns:
			pd.DataFrame | dd.DataFrame: Filtered dataframe with applied criteria
		
		Table-specific filtering criteria:
		
		PATIENTS:
			Inclusions:
				- anchor_age between 18 and 75 years
				- anchor_year_group: '2017 - 2019'
				- dod (date of death) is null (alive patients)
		
		DIAGNOSES_ICD:
			Inclusions:
				- icd_version: 10
				- seq_num: 1, 2, or 3
				- icd_code: starts with 'E11' (diabetes-related codes)
		
		D_ICD_DIAGNOSES:
			Inclusions:
				- icd_version: 10
		
		POE (Provider Order Entry):
			Column selection (when table in filter_params):
				- Keeps only: poe_id, poe_seq, subject_id, hadm_id, ordertime, order_type, order_subtype, order_provider_id
		
		ADMISSIONS:
			Column exclusions:
				- Removes: admit_provider_id, insurance, language, marital_status, 
				  race, edregtime, edouttime
			Row exclusions:
				- Null values in admittime or dischtime
				- In-hospital deaths (deathtime not null AND hospital_expire_flag = 1)
				- Discharge time before admission time
				- Admission types: 'EW EMER.', 'DIRECT EMER.', 'URGENT', 'ELECTIVE'
		
		TRANSFERS:
			Row exclusions:
				- Null values in hadm_id
			Note: Careunit filtering to 'Medicine' is commented out
		
		MICROBIOLOGYEVENTS:
			Column exclusions:
				- Removes: comments
		
		LABEVENTS:
			Column selection:
				- Keeps only: labevent_id, subject_id, hadm_id, itemid, order_provider_id, charttime
			Note: Row filtering for null hadm_id and order_provider_id is commented out
		
		PRESCRIPTIONS:
			Column exclusions:
				- Removes: pharmacy_id, starttime, stoptime, drug_type, formulary_drug_cd
		
		OMR (Outpatient Medication Reconciliation):
			Inclusions:
				- seq_num: 1, 2, or 3
		"""

		# ============================================================================
		# 							patients
		# ============================================================================
		if self.table_name == TableNames.PATIENTS:

			# Only keep rows with anchor_age between 18 and 75, anchor_year_group 2017 - 2019, and dod is null
			anchor_age        = (self.df.anchor_age >= 18.0) & (self.df.anchor_age <= 75.0)
			anchor_year_group = self.df.anchor_year_group.isin(['2017 - 2019'])
			dod               = self.df.dod.isnull()

			self.df           = self.df[anchor_age & anchor_year_group & dod]

		# ============================================================================
		# 							diagnoses_icd
		# ============================================================================
		elif self.table_name == TableNames.DIAGNOSES_ICD:

			# Only keep rows with icd_version 10, seq_num 1, 2, 3, and icd_code starting with E11
			icd_version = self.df.icd_version.isin([10])
			seq_num     = self.df.seq_num.isin([1,2,3])
			icd_code    = self.df.icd_code.str.startswith('E11')
			self.df     = self.df[icd_version & seq_num & icd_code]

		# ============================================================================
		# 							d_icd_diagnoses
		# ============================================================================
		elif self.table_name == TableNames.D_ICD_DIAGNOSES:
			# Only keep rows with icd_version 10
			self.df = self.df[ self.df.icd_version.isin([10]) ]

		# ============================================================================
		# 							poe
		# ============================================================================
		elif self.table_name == TableNames.POE:

			if self.table_name.value in self.filter_params:
				# Only include these columns
				self.df = self.df[ ["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime", "order_type", "order_subtype", "order_provider_id"] ]

		# ============================================================================
		# 							admissions
		# ============================================================================
		elif self.table_name == TableNames.ADMISSIONS:

			# Drop columns
			self.df = self.df.drop(columns=['admit_provider_id', 'insurance','language', 'marital_status', 'race', 'edregtime', 'edouttime'])

			# Drop rows with null values in admittime and dischtime
			self.df = self.df.dropna(subset=['admittime', 'dischtime'])

			# Patient is alive: Drop rows with null values in deathtime or hospital_expire_flag is 0
			exclude_in_hospital_death = (self.df.deathtime.isnull()) | (self.df.hospital_expire_flag == 0)

			# Discharge time is after admission time: Drop rows with dischtime before admittime
			discharge_after_admission = self.df['dischtime'] > self.df['admittime']

			# Exclude admission types like “Emergency”, “Urgent”, or “Elective”
			admission_type = ~self.df.admission_type.isin(['EW EMER.', 'DIRECT EMER.', 'URGENT', 'ELECTIVE'])

			self.df = self.df[ exclude_in_hospital_death & discharge_after_admission & admission_type]

		# ============================================================================
		# 							transfers
		# ============================================================================
		elif self.table_name == TableNames.TRANSFERS:

			# Drop rows with null values in hadm_id
			# empty_cells = self.df.hadm_id != ''
			self.df = self.df[ ~self.df.hadm_id.isnull()]

			# Only keep the rows with careunit in ['Medicine']
			# careunit = self.df.careunit.isin(['Medicine'])
			# self.df = self.df[careunit]

		# ============================================================================
		# 							microbiologyevents
		# ============================================================================
		elif self.table_name == TableNames.MICROBIOLOGYEVENTS:
			# Drop comments column
			self.df = self.df.drop(columns=['comments'])

		# ============================================================================
		# 							labevents
		# ============================================================================
		elif self.table_name == TableNames.LABEVENTS:
			# Only keep these columns
			self.df = self.df[['labevent_id', 'subject_id', 'hadm_id', 'itemid', 'order_provider_id', 'charttime']]

			# Drop rows with null values in hadm_id
			# hadm_id                = ~self.df.hadm_id.isnull()
			# order_provider_id_null = ~self.df.order_provider_id.isnull()

			# Drop rows with null values in hadm_id and order_provider_id
			# self.df = self.df[hadm_id & order_provider_id_null]
			# self.df = self.df[hadm_id]

		# ============================================================================
		# 							prescriptions
		# ============================================================================
		elif self.table_name == TableNames.PRESCRIPTIONS:
			# Drop columns
			self.df = self.df.drop(columns=['pharmacy_id', 'starttime', 'stoptime', 'drug_type', 'formulary_drug_cd'])

		# ============================================================================
		# 							omr
		# ============================================================================
		elif self.table_name == TableNames.OMR:
			# Only keep rows with seq_num 1, 2, 3
			seq_num     = self.df.seq_num.isin([1,2,3])
			self.df     = self.df[seq_num]


		# Reset index
		self.df = self.df.reset_index(drop=True)
		return self.df
