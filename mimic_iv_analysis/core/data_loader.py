# Standard library imports
import os
import glob
import logging
from typing import Dict, Optional, Tuple, List, Any

# Data processing imports
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd

# Streamlit import
import streamlit as st

# Import filtering functionality
from mimic_iv_analysis.core.filtering import Filtering

# Utility functions
def convert_string_dtypes(df):
    """Convert pandas StringDtype to object type to avoid Arrow conversion issues in Streamlit.

    Args:
        df: Input DataFrame (pandas or Dask)

    Returns:
        DataFrame with StringDtype columns converted to object type
    """
    if hasattr(df, 'compute'):
        # For Dask DataFrame, we need to compute it first
        return df
    else:
        # For pandas DataFrame
        for col in df.columns:
            if hasattr(df[col], 'dtype') and str(df[col].dtype) == 'string':
                df[col] = df[col].astype('object')
        return df
# Constants
DEFAULT_MIMIC_PATH    = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

class MIMICDataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""

	def __init__(self):
		"""Initialize the MIMICDataLoader class."""
		self.filtering = Filtering()


	def _format_size(self, size_mb: float) -> str:
		"""Format file size into KB, MB, or GB string."""
		if size_mb < 0.001:
			return "(< 1 KB)"
		elif size_mb < 1:
			return f"({size_mb * 1024:.1f} KB)"
		elif size_mb < 1000:
			return f"({size_mb:.1f} MB)"
		else:
			return f"({size_mb / 1000:.1f} GB)"


	def scan_mimic_directory(self, mimic_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
		"""Scans the MIMIC-IV directory structure and returns table info."""
		available_tables    = {}
		file_paths          = {}
		file_sizes          = {}
		table_display_names = {}

		if not os.path.exists(mimic_path):
			return available_tables, file_paths, file_sizes, table_display_names

		modules = ['hosp', 'icu']
		for module in modules:
			module_path = os.path.join(mimic_path, module)
			if os.path.exists(module_path):
				available_tables[module] = []
				current_module_tables = {}

				csv_files = glob.glob(os.path.join(module_path, '*.csv'))
				csv_gz_files = glob.glob(os.path.join(module_path, '*.csv.gz'))

				all_files = csv_files + csv_gz_files

				for file_path in all_files:
					is_gz = file_path.endswith('.csv.gz')
					table_name = os.path.basename(file_path).replace('.csv.gz' if is_gz else '.csv', '')

					if table_name in current_module_tables and not is_gz:
						current_module_tables[table_name] = file_path

					elif table_name not in current_module_tables:
						current_module_tables[table_name] = file_path

				for table_name, file_path in current_module_tables.items():
					available_tables[module].append(table_name)
					file_paths[(module, table_name)] = file_path
					try:
						file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

					except OSError:
						file_size_mb = 0

					file_sizes[(module, table_name)] = file_size_mb
					size_str = self._format_size(file_size_mb)
					table_display_names[(module, table_name)] = f"{table_name} {size_str}"

				available_tables[module].sort()

		return available_tables, file_paths, file_sizes, table_display_names


	def load_mimic_table(self, file_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1', use_dask: bool = False, filter_params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[pd.DataFrame], int]:
		"""Loads a specific MIMIC-IV table, handling large files and sampling.

		Args:
			file_path: Path to the CSV/CSV.GZ file
			sample_size: Number of rows to sample for large files
			encoding: File encoding
			use_dask: Whether to use Dask for loading (better for very large files)

		Returns:
			Tuple containing:
				- DataFrame with the loaded data (may be sampled)
				- Total row count in the original file
		"""
		try:
			file_size_mb  = os.path.getsize(file_path) / (1024 * 1024)
			is_compressed = file_path.endswith('.gz')
			compression   = 'gzip' if is_compressed else None
			read_params   = { 'encoding': encoding, 'compression': compression, 'low_memory': False }

			if use_dask:
				# Using Dask for large file processing
				try:
					# First attempt to detect datetime columns by reading a small sample with pandas
					sample_df = pd.read_csv(file_path, nrows=5, **read_params)

					# Identify potential datetime columns
					datetime_cols = []
					for col in sample_df.columns:
						if 'time' in col.lower() or 'date' in col.lower():
							datetime_cols.append(col)

					# Create a dtype dictionary for problematic columns
					dtype_dict = {col: 'object' for col in datetime_cols}

					# Update read_params with the dtype dictionary
					dask_read_params = read_params.copy()
					dask_read_params['dtype'] = dtype_dict

					# Load the data with Dask using the updated parameters
					dask_df = dd.read_csv(file_path, **dask_read_params)

					# Get total row count
					total_rows = int(dask_df.shape[0].compute())

				except Exception as e:
					logging.warning(f"Specific column error with Dask: {str(e)}")
					logging.info("Falling back to generic object dtypes for all columns")

					# Fallback: use object dtype for all columns
					dask_read_params = read_params.copy()
					dask_read_params['dtype'] = 'object'

					# Try again with all columns as objects
					dask_df = dd.read_csv(file_path, **dask_read_params)
					total_rows = int(dask_df.shape[0].compute())

				# For large files or when sampling is needed
				if total_rows > sample_size:
					# Convert to pandas with sampling
					if file_size_mb > LARGE_FILE_THRESHOLD_MB:
						# For very large files, use head() which is more efficient
						df = dask_df.head(sample_size, compute=True)
					else:
						# For smaller files, we can compute the whole thing and then sample
						df = dask_df.compute()
						df = df.sample(sample_size, random_state=RANDOM_STATE)
				else:
					# If file is smaller than sample size, load it all
					df = dask_df.compute()
			else:
				# Traditional pandas approach
				# Get total row count without loading entire file
				total_rows = 0
				with pd.read_csv(file_path, chunksize=10000, **read_params) as reader:
					for chunk in reader:
						total_rows += len(chunk)

				# Load actual data (with sampling if needed)
				if file_size_mb > LARGE_FILE_THRESHOLD_MB:
					df = pd.read_csv(file_path, nrows=sample_size, **read_params)
				else:
					df = pd.read_csv(file_path, **read_params)
					if len(df) > sample_size:
						df = df.sample(sample_size, random_state=RANDOM_STATE)

			# Apply filtering if filter_params are provided
			if filter_params is not None and df is not None:
				df = self.apply_filters(df, filter_params)
				# Update total_rows to reflect filtered count
				total_rows = len(df) if not use_dask else int(df.shape[0].compute() if hasattr(df, 'compute') else df.shape[0])

			# Convert StringDtype columns to avoid PyArrow conversion issues in Streamlit
			if df is not None:
				df = convert_string_dtypes(df)

			return df, total_rows
		except Exception as e:
			logging.error(f"Error loading data from {os.path.basename(file_path)}: {str(e)}")
			return None, 0


	def apply_filters(self, df: pd.DataFrame, filter_params: Dict[str, Any]) -> pd.DataFrame:
		"""
		Apply filters to the DataFrame based on the provided filter parameters.

		This method loads necessary related tables (patients, admissions, diagnoses_icd, transfers)
		and applies the specified filters using the Filtering class.

		Args:
			df: DataFrame to filter
			filter_params: Dictionary containing filter parameters

		Returns:
			Filtered DataFrame
		"""
		logging.info("Applying filters to DataFrame...")

		# Check if filtering is enabled
		if not any(filter_params.get(f, False) for f in [
			'apply_encounter_timeframe', 'apply_age_range', 'apply_t2dm_diagnosis',
			'apply_valid_admission_discharge', 'apply_inpatient_stay', 'exclude_in_hospital_death'
		]):
			logging.info("No filters enabled. Returning original DataFrame.")
			return df

		# Load necessary related tables if they're not already in the DataFrame
		patients_df = None
		admissions_df = None
		diagnoses_df = None
		transfers_df = None

		# Check if we need to load the patients table
		if filter_params.get('apply_encounter_timeframe', False) or filter_params.get('apply_age_range', False):
			if 'anchor_year_group' not in df.columns or 'anchor_age' not in df.columns:
				patients_path = os.path.join(st.session_state.mimic_path, 'hosp', 'patients.csv.gz')
				if os.path.exists(patients_path):
					patients_df, _ = self.load_mimic_table(patients_path, filter_params=None)
				else:
					logging.warning("Patients table not found. Some filters may not be applied.")

		# Check if we need to load the admissions table
		if filter_params.get('apply_valid_admission_discharge', False) or \
		   filter_params.get('apply_inpatient_stay', False) or \
		   filter_params.get('exclude_in_hospital_death', False):
			if 'admittime' not in df.columns or 'dischtime' not in df.columns or \
			   'admission_type' not in df.columns or 'deathtime' not in df.columns:
				admissions_path = os.path.join(st.session_state.mimic_path, 'hosp', 'admissions.csv.gz')
				if os.path.exists(admissions_path):
					admissions_df, _ = self.load_mimic_table(admissions_path, filter_params=None)
				else:
					logging.warning("Admissions table not found. Some filters may not be applied.")

		# Check if we need to load the diagnoses_icd table
		if filter_params.get('apply_t2dm_diagnosis', False):
			if 'icd_code' not in df.columns or 'seq_num' not in df.columns:
				diagnoses_path = os.path.join(st.session_state.mimic_path, 'hosp', 'diagnoses_icd.csv.gz')
				if os.path.exists(diagnoses_path):
					diagnoses_df, _ = self.load_mimic_table(diagnoses_path, filter_params=None)
				else:
					logging.warning("Diagnoses_icd table not found. T2DM diagnosis filter may not be applied.")

		# Check if we need to load the transfers table
		if filter_params.get('apply_inpatient_stay', False) and filter_params.get('require_inpatient_transfer', False):
			if 'careunit' not in df.columns:
				transfers_path = os.path.join(st.session_state.mimic_path, 'hosp', 'transfers.csv.gz')
				if os.path.exists(transfers_path):
					transfers_df, _ = self.load_mimic_table(transfers_path, filter_params=None)
				else:
					logging.warning("Transfers table not found. Inpatient transfer filter may not be applied.")

		# Apply filters using the Filtering class
		filtered_df = self.filtering.apply_filters(
			df=df,
			filter_params=filter_params,
			patients_df=patients_df,
			admissions_df=admissions_df,
			diagnoses_df=diagnoses_df,
			transfers_df=transfers_df
		)

		return filtered_df

	def get_table_info(self, module: str, table_name: str) -> str:
		"""Get descriptive information about a specific MIMIC-IV table."""
		table_info = {
			('hosp', 'admissions'): "Patient hospital admissions information",
			('hosp', 'patients'): "Patient demographic data",
			('hosp', 'labevents'): "Laboratory measurements (large file)",
			('hosp', 'microbiologyevents'): "Microbiology test results",
			('hosp', 'pharmacy'): "Pharmacy orders",
			('hosp', 'prescriptions'): "Medication prescriptions",
			('hosp', 'procedures_icd'): "Patient procedures",
			('hosp', 'diagnoses_icd'): "Patient diagnoses",
			('hosp', 'emar'): "Electronic medication administration records",
			('hosp', 'emar_detail'): "Detailed medication administration data",
			('hosp', 'poe'): "Provider order entries",
			('hosp', 'poe_detail'): "Detailed order information",
			('hosp', 'd_hcpcs'): "HCPCS code definitions",
			('hosp', 'd_icd_diagnoses'): "ICD diagnosis code definitions",
			('hosp', 'd_icd_procedures'): "ICD procedure code definitions",
			('hosp', 'd_labitems'): "Laboratory test definitions",
			('hosp', 'hcpcsevents'): "HCPCS events",
			('hosp', 'drgcodes'): "Diagnosis-related group codes",
			('hosp', 'services'): "Hospital services",
			('hosp', 'transfers'): "Patient transfers",
			('hosp', 'provider'): "Provider information",
			('hosp', 'omr'): "Order monitoring results",
			('icu', 'chartevents'): "Patient charting data (vital signs, etc.)",
			('icu', 'datetimeevents'): "Date/time-based events",
			('icu', 'inputevents'): "Patient intake data",
			('icu', 'outputevents'): "Patient output data",
			('icu', 'procedureevents'): "ICU procedures",
			('icu', 'ingredientevents'): "Detailed medication ingredients",
			('icu', 'd_items'): "Dictionary of ICU items",
			('icu', 'icustays'): "ICU stay information",
			('icu', 'caregiver'): "Caregiver information"
		}
		return table_info.get((module, table_name), "No description available")


	def convert_to_parquet(self, df: pd.DataFrame, table_name: str, current_file_path: str) -> Optional[str]:
		"""Converts the loaded DataFrame to Parquet format. Returns path or None."""
		# TODO: learng about the difference between pyarrow and dask. then if needed, include teh pyarrow logic to the code
		try:
			parquet_dir = os.path.join(os.path.dirname(current_file_path), 'parquet_files')
			os.makedirs(parquet_dir, exist_ok=True)
			parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")
			table = pa.Table.from_pandas(df)
			pq.write_table(table, parquet_file)
			return parquet_file
		except Exception as e:
			logging.error(f"Error converting {table_name} to Parquet: {str(e)}")
			return None

