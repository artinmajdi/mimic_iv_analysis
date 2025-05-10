# Standard library imports
import os
import glob
import logging
from typing import Dict, Optional, Tuple, List, Any, Union
import traceback

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

class DataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""

	def __init__(self):
		"""Initialize the MIMICDataLoader class."""
		self.filtering = Filtering()
		self.mimic_path = DEFAULT_MIMIC_PATH

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


	def load_mimic_table(self, file_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1', use_dask: bool = False, max_chunks: Optional[int] = None) -> Tuple[Optional[Union[pd.DataFrame, dd.DataFrame]], int]:
		"""Loads a specific MIMIC-IV table, handling large files and sampling.

		Args:
			file_path: Path to the CSV/CSV.GZ file
			sample_size: Number of rows to sample for large files
			encoding: File encoding
			use_dask: Whether to use Dask for loading (better for very large files)
			max_chunks: Maximum number of chunks to process (for debugging or limiting memory usage)

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
				chunks = []
				chunk_count = 0
				total_rows_processed = 0
				with pd.read_csv(file_path, chunksize=100000, **read_params) as reader: # chunksize can be tuned
					for chunk in reader:
						chunks.append(chunk)
						chunk_count += 1
						total_rows_processed += len(chunk)
						if max_chunks is not None and max_chunks != -1 and chunk_count >= max_chunks:
							logging.info(f"Reached max_chunks ({max_chunks}) for {os.path.basename(file_path)}. Processed {chunk_count} chunks.")
							break
				total_rows = total_rows_processed

				# Load actual data (with sampling if needed)
				if sample_size is not None and file_size_mb > LARGE_FILE_THRESHOLD_MB:
					df = pd.read_csv(file_path, nrows=sample_size, **read_params)
				else:
					df = pd.concat(chunks, ignore_index=True)
					# Apply sampling only if sample_size is specified and df is larger
					if sample_size is not None and len(df) > sample_size:
						df = df.sample(sample_size, random_state=RANDOM_STATE)

			# Convert StringDtype columns to avoid PyArrow conversion issues in Streamlit
			if df is not None:
				df = convert_string_dtypes(df)

			return df, total_rows
		except Exception as e:
			logging.error(f"Error loading data from {os.path.basename(file_path)}: {str(e)}")
			return None, 0


	def apply_filters(self, df: pd.DataFrame, filter_params: Dict[str, Any], mimic_path: str=None) -> pd.DataFrame:
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

		if mimic_path is not None:
			self.mimic_path = mimic_path

		logging.info("Applying filters to DataFrame...")

		# Check if filtering is enabled
		if not any(filter_params.get(f, False) for f in [
			'apply_encounter_timeframe', 'apply_age_range', 'apply_t2dm_diagnosis',
			'apply_valid_admission_discharge', 'apply_inpatient_stay', 'exclude_in_hospital_death'
		]):
			logging.info("No filters enabled. Returning original DataFrame.")
			return df

		# Load necessary related tables if they're not already in the DataFrame
		patients_df   = None
		admissions_df = None
		diagnoses_df  = None
		transfers_df  = None

		# Check if we need to load the patients table
		if filter_params.get('apply_encounter_timeframe', False) or filter_params.get('apply_age_range', False):

			if ('anchor_year_group' not in df.columns) or ('anchor_age' not in df.columns):
				# Determine the path for the patients table
				patients_path_gz = os.path.join(mimic_path, 'hosp', 'patients.csv.gz')
				patients_path_csv = os.path.join(mimic_path, 'hosp', 'patients.csv')

				if os.path.exists(patients_path_gz):
					patients_file_to_load = patients_path_gz
				elif os.path.exists(patients_path_csv):
					patients_file_to_load = patients_path_csv
				else:
					patients_file_to_load = None
					logging.warning("Patients file (.csv.gz or .csv) not found at expected paths. Some filters may not be applied.")

				if patients_file_to_load:
					patients_df, _ = self.load_mimic_table(patients_file_to_load, sample_size=None, max_chunks=-1)
					if patients_df is not None:
						if 'age' not in patients_df.columns and 'anchor_age' in patients_df.columns:
							patients_df['age'] = patients_df['anchor_age']
							logging.info("Created 'age' column from 'anchor_age' in patients_df for filtering.")
						elif 'age' not in patients_df.columns:
							logging.warning("'anchor_age' column not found in loaded patients_df. Cannot create 'age' column.")
					# If patients_df is None after load_mimic_table, it means loading failed.
					elif patients_df is None:
						logging.warning(f"Failed to load patients table from {patients_file_to_load}. Age filter may not be applied.")

		# Check if we need to load the admissions table
		if 	filter_params.get('apply_valid_admission_discharge', False) or \
			filter_params.get('apply_inpatient_stay', False) or \
			filter_params.get('exclude_in_hospital_death', False):

			if ('admittime' not in df.columns) or ('dischtime' not in df.columns) or ('admission_type' not in df.columns) or ('deathtime' not in df.columns):
				admissions_path = os.path.join(mimic_path, 'hosp', 'admissions.csv.gz')

				if os.path.exists(admissions_path):
					# Load full table, filtering happens later
					admissions_df, _ = self.load_mimic_table(admissions_path)
				else:
					logging.warning("Admissions table not found. Some filters may not be applied.")

		# Check if we need to load the diagnoses_icd table
		if filter_params.get('apply_t2dm_diagnosis', False):
			if ('icd_code' not in df.columns) or ('seq_num' not in df.columns):
				diagnoses_path = os.path.join(mimic_path, 'hosp', 'diagnoses_icd.csv.gz')

				if os.path.exists(diagnoses_path):
					# Load full table, filtering happens later
					diagnoses_df, _ = self.load_mimic_table(diagnoses_path)
				else:
					logging.warning("Diagnoses_icd table not found. T2DM diagnosis filter may not be applied.")

		# Check if we need to load the transfers table
		if filter_params.get('apply_inpatient_stay', False) and filter_params.get('require_inpatient_transfer', False):
			if 'careunit' not in df.columns:
				transfers_path = os.path.join(mimic_path, 'hosp', 'transfers.csv.gz')
				if os.path.exists(transfers_path):
					# Load full table, filtering happens later
					transfers_df, _ = self.load_mimic_table(transfers_path)
				else:
					logging.warning("Transfers table not found. Inpatient transfer filter may not be applied.")

		# Apply filters using the Filtering class
		filtered_df = self.filtering.apply_filters(
			df            = df,
			filter_params = filter_params,
			patients_df   = patients_df,
			admissions_df = admissions_df,
			diagnoses_df  = diagnoses_df,
			transfers_df  = transfers_df
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


	def convert_to_parquet(self, df: Union[pd.DataFrame, dd.DataFrame], table_name: str, current_file_path: str) -> Optional[str]:
		"""Converts the loaded DataFrame to Parquet format. Returns path or None."""
		try:
			# Use current_file_path directly as the base directory for 'parquet_files'
			parquet_dir = os.path.join(current_file_path, 'parquet_files')
			os.makedirs(parquet_dir, exist_ok=True)
			parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")

			if isinstance(df, pd.DataFrame):
				table = pa.Table.from_pandas(df)
				pq.write_table(table, parquet_file)
				logging.info(f"Pandas DataFrame {table_name} converted to Parquet: {parquet_file}")
			elif isinstance(df, dd.DataFrame):
				# For Dask, to_parquet can write a directory or a single file.
				# To ensure a single file is written (consistent with pandas behavior here),
				# we can compute and then write, or use single_file=True if available/desired.
				# For simplicity, let's compute and then use pandas-like writing if it's small enough
				# or use Dask's direct to_parquet for larger Dask dataframes.
				# A more robust Dask approach is df.to_parquet(parquet_file, write_index=False, engine='pyarrow')
				# which will create a directory if df has multiple partitions.
				# For now, assuming we want a single file output for this utility:
				df.to_parquet(parquet_file, write_index=False, engine='pyarrow') # Removed single_file=True
				logging.info(f"Dask DataFrame {table_name} converted to Parquet: {parquet_file}")
			else:
				logging.error(f"Unsupported DataFrame type for Parquet conversion: {type(df)}")
				return None
			return parquet_file
		except Exception as e:
			logging.error(f"Error converting {table_name} to Parquet: {str(e)}")
			return None


	def load_reference_table(self, mimic_path: str, module: str, table_name: str,
					encoding: str = 'latin-1') -> pd.DataFrame:
		"""Loads a reference/dictionary table in full.

		Args:
			mimic_path: Path to the MIMIC-IV dataset
			module: Module name ('hosp' or 'icu')
			table_name: Name of the table to load
			encoding: File encoding

		Returns:
			DataFrame containing the full reference table
		"""
		# Check for uncompressed CSV first (prioritize over compressed)
		table_path_csv = os.path.join(mimic_path, module, f"{table_name}.csv")
		table_path_gz  = os.path.join(mimic_path, module, f"{table_name}.csv.gz")

		logging.info(f"Looking for reference table at: {table_path_csv} or {table_path_gz}")

		# Prioritize uncompressed CSV over gzipped format
		if os.path.exists(table_path_csv):
			table_path = table_path_csv
			logging.info(f"Found uncompressed CSV file: {table_path}")

		elif os.path.exists(table_path_gz):
			table_path = table_path_gz
			logging.info(f"Found gzipped file: {table_path}")

		else:
			logging.error(f"Reference table {table_name} not found at {table_path_csv} or {table_path_gz}!")
			return pd.DataFrame()

		try:

			logging.info(f"Loading reference table {table_name} from {table_path}...")
			compression = 'gzip' if table_path.endswith('.gz') else None

			# For reference tables, we load the complete file (no sampling)
			df = pd.read_csv(table_path, encoding=encoding, compression=compression)

			if df is not None:
				df = convert_string_dtypes(df)
				logging.info(f"Successfully loaded {table_name} with {len(df)} rows and {len(df.columns)} columns")
				return df
			else:
				logging.error(f"Failed to load reference table {table_name} - empty DataFrame returned")
				return pd.DataFrame()

		except Exception as e:
			logging.error(f"Error loading reference table {table_name}: {str(e)}")
			logging.error(traceback.format_exc())
			return pd.DataFrame()


	def load_patient_cohort(self, mimic_path: str, sample_size: int, encoding: str = 'latin-1',
						use_dask: bool = False, max_chunks: Optional[int] = None) -> Tuple[pd.DataFrame, set]:
		"""Loads and samples the patients table to create a base cohort.

		Args:
			mimic_path: Path to the MIMIC-IV dataset
			sample_size: Number of patients to sample
			encoding: File encoding
			use_dask: Whether to use Dask for loading
			max_chunks: Maximum number of chunks to process (for debugging or limiting memory usage)

		Returns:
			Tuple containing:
				- DataFrame with the sampled patients
				- Set of sampled subject_ids for filtering other tables
		"""

		# Prioritize uncompressed CSV over gzipped format
		patients_path_csv = os.path.join(mimic_path, 'hosp', 'patients.csv')
		patients_path_gz = os.path.join(mimic_path, 'hosp', 'patients.csv.gz')

		if os.path.exists(patients_path_csv):
			patients_path = patients_path_csv
			logging.info(f"Using uncompressed patients file: {patients_path}")

		elif os.path.exists(patients_path_gz):
			patients_path = patients_path_gz
			logging.info(f"Using compressed patients file: {patients_path}")

		else:
			logging.error("Patients table not found!")
			return pd.DataFrame(), set()

		logging.info(f"Loading patients table with sample_size={sample_size}...")

		try:

			# Load with pandas
			args = { 'encoding': encoding, 'compression': 'gzip' if patients_path.endswith('.gz') else None }

			if use_dask:
				# Load with Dask, avoiding the blocksize warning
				dask_read_params = args.copy()
				dask_read_params['low_memory'] = False

				try:
					# Attempt 1: Infer datetime columns and set to object
					sample_df = pd.read_csv(patients_path, nrows=5, **dask_read_params)
					datetime_cols = [col for col in sample_df.columns if 'time' in col.lower() or 'date' in col.lower()]
					dtype_dict = {col: 'object' for col in datetime_cols}

					current_dask_read_params = dask_read_params.copy()
					current_dask_read_params['dtype'] = dtype_dict

					logging.info(f"Loading patients with Dask, attempting dtypes: {dtype_dict} based on column names.")
					df = dd.read_csv(patients_path, blocksize=None, **current_dask_read_params)

				except Exception as e:
					logging.warning(f"Dask dtype specification based on column names failed for patients: {str(e)}. Falling back to all object dtypes for Dask reading.")
					# Attempt 2: Fallback to all object dtypes
					fallback_dask_read_params = dask_read_params.copy()
					fallback_dask_read_params['dtype'] = 'object'
					df = dd.read_csv(patients_path, blocksize=None, **fallback_dask_read_params)

				# Compute the full DataFrame if sample size is larger than total
				df_patients = df.compute()

			else: # Not using Dask, use pandas
				# Load in chunks, respecting max_chunks
				chunks = []
				chunk_count = 0
				total_rows_processed = 0
				with pd.read_csv(patients_path, chunksize=100000, **args) as reader:
					for chunk in reader:
						chunks.append(chunk)
						chunk_count += 1
						total_rows_processed += len(chunk)
						if max_chunks is not None and max_chunks != -1 and chunk_count >= max_chunks:
							logging.info(f"Reached max_chunks ({max_chunks}) for patients.csv. Processed {chunk_count} chunks.")
							break
				
				if not chunks:
					logging.error("No data loaded from patients.csv with pandas chunking.")
					return pd.DataFrame(), set()
				
				df_loaded_patients = pd.concat(chunks, ignore_index=True)
				total_patients_processed = len(df_loaded_patients)

				# Apply sampling (using head as per original logic) if needed
				if sample_size and sample_size < total_patients_processed:
					df_patients = df_loaded_patients.head(sample_size)
					logging.info(f"Took first {len(df_patients)} patients from {total_patients_processed} processed rows.")
				else:
					df_patients = df_loaded_patients
					logging.info(f"Loaded all {total_patients_processed} processed patient rows (or sample_size was not applicable).")

			df_patients = convert_string_dtypes(df_patients)

			sampled_subject_ids = set(df_patients['subject_id'])
			logging.info(f"Loaded patients: {len(df_patients)} (processed {total_patients_processed} before sampling)")
			return df_patients, sampled_subject_ids

		except Exception as e:
			logging.error(f"Error loading patients table: {str(e)}")
			logging.error(traceback.format_exc())
			return pd.DataFrame(), set()


	def load_filtered_table(self, mimic_path: str, module: str, table_name: str, filter_column: str, filter_values: set, encoding: str = 'latin-1', use_dask: bool = False, max_chunks: Optional[int] = None) -> pd.DataFrame:
		"""
			Loads a table and filters it based on specified column values.

			Args:
				mimic_path   : Path to the MIMIC-IV dataset
				module       : Module name ('hosp' or 'icu')
				table_name   : Name of the table to load
				filter_column: Column name to filter on
				filter_values: Set of values to include
				encoding     : File encoding
				use_dask     : Whether to use Dask for loading
				max_chunks   : Maximum number of chunks to process (for debugging or limiting memory usage)

			Returns:
				DataFrame containing the filtered table
			"""

		def _get_table_path() -> Optional[str]:

			# Prioritize uncompressed CSV over gzipped format
			table_path_csv = os.path.join(mimic_path, module, f"{table_name}.csv")
			table_path_gz  = os.path.join(mimic_path, module, f"{table_name}.csv.gz")

			logging.info(f"Looking for table {table_name} at: {table_path_csv} or {table_path_gz}")

			if os.path.exists(table_path_csv):
				table_path = table_path_csv
				logging.info(f"Found uncompressed CSV file: {table_path}")

			elif os.path.exists(table_path_gz):
				table_path = table_path_gz
				logging.info(f"Found gzipped file: {table_path}")

			else:
				logging.error(f"Table {table_name} not found!")
				return None

			logging.info(f"Loading and filtering {table_name} table...")
			logging.info(f"Filtering on column {filter_column} with {len(filter_values)} unique values")

			return table_path

		# Get the table path
		table_path = _get_table_path()

		if table_path is None:
			return pd.DataFrame()

		args = { 'encoding': encoding, 'compression': 'gzip' if table_path.endswith('.gz') else None }

		if use_dask:
			# More robust Dask dtype handling, similar to load_mimic_table
			dask_read_params = args.copy()
			dask_read_params['low_memory'] = False

			try:
				# Attempt 1: Infer datetime columns and set to object
				sample_df = pd.read_csv(table_path, nrows=5, **dask_read_params)
				datetime_cols = [col for col in sample_df.columns if 'time' in col.lower() or 'date' in col.lower()]
				dtype_dict = {col: 'object' for col in datetime_cols}

				current_dask_read_params = dask_read_params.copy()
				current_dask_read_params['dtype'] = dtype_dict

				logging.info(f"Loading {table_name} with Dask, attempting dtypes: {dtype_dict} based on column names.")
				df = dd.read_csv(table_path, blocksize=None, **current_dask_read_params)

			except Exception as e:
				logging.warning(f"Dask dtype specification based on column names failed for {table_name}: {str(e)}. Falling back to all object dtypes for Dask reading.")
				# Attempt 2: Fallback to all object dtypes
				fallback_dask_read_params = dask_read_params.copy()
				fallback_dask_read_params['dtype'] = 'object'
				df = dd.read_csv(table_path, blocksize=None, **fallback_dask_read_params)

			# Check if filter column exists
			if filter_column not in df.columns:
				logging.warning(f"Filter column {filter_column} not found in {table_name}")
				return pd.DataFrame()

			# Filter the DataFrame (this is efficient in Dask as it's a lazy operation)
			filtered_df = df[df[filter_column].isin(list(filter_values))]

			# Compute the filtered result
			result_df = filtered_df.compute()
			logging.info(f"Filtered {table_name} to {len(result_df)} rows using Dask")

			return convert_string_dtypes(result_df)


		# For pandas: Use chunking to filter while loading to minimize memory usage
		filtered_chunks = []

		logging.info(f"Loading {table_name} with pandas chunk processing...")

		with pd.read_csv(table_path, chunksize=100000, **args) as reader:

			# Process file in chunks to minimize memory usage
			chunk_count = 0
			for chunk in reader:
				chunk_count += 1

				# Check filter column on first chunk
				if chunk_count == 1 and filter_column not in chunk.columns:
					logging.warning(f"Filter column {filter_column} not found in {table_name}")
					return pd.DataFrame()

				# Filter each chunk
				filtered_chunk = chunk[chunk[filter_column].isin(filter_values)]

				# Only keep chunks with matching rows
				if len(filtered_chunk) > 0:
					filtered_chunks.append(filtered_chunk)

				if max_chunks is not None and max_chunks != -1 and chunk_count >= max_chunks:
					logging.info(f"Reached max_chunks ({max_chunks}) for {table_name} during filtered load. Processed {chunk_count} chunks.")
					break

				# Log progress periodically
				if chunk_count % 10 == 0:
					logging.info(f"Processed {chunk_count} chunks of {table_name}...")

		# Combine all filtered chunks
		if filtered_chunks:
			result_df = pd.concat(filtered_chunks, ignore_index=True)
			logging.info(f"Loaded {len(result_df)} rows from {table_name} after filtering")

			return convert_string_dtypes(result_df)


		logging.warning(f"No rows found in {table_name} after filtering")
		return pd.DataFrame()


	def merge_tables(self,
						left_df : Union[pd.DataFrame, dd.DataFrame],
						right_df: Union[pd.DataFrame, dd.DataFrame],
						on      : List[str],
						how     : str = 'left') -> Union[pd.DataFrame, dd.DataFrame]:
		"""Merges two DataFrames on specified columns.

		Args:
			left_df : Left DataFrame (pandas or Dask)
			right_df: Right DataFrame (pandas or Dask)
			on      : Column(s) to join on
			how     : Type of merge to perform

		Returns:
			Merged DataFrame (pandas or Dask, matches input type if consistent)
		"""
		is_dask_left = isinstance(left_df, dd.DataFrame)
		is_dask_right = isinstance(right_df, dd.DataFrame)

		if is_dask_left != is_dask_right:
			logging.error("Cannot merge a Dask DataFrame with a pandas DataFrame. Both must be of the same type. Returning left DataFrame.")
			return left_df

		# Handle empty pandas DataFrames explicitly
		if not is_dask_left and (left_df.empty or right_df.empty):
			logging.warning("One or both pandas DataFrames are empty. Returning left DataFrame.")
			return left_df

		# Check if join columns exist in both tables
		missing_cols = []
		for col in on:

			if col not in left_df.columns:
				missing_cols.append(f"{col} missing in left table")

			if col not in right_df.columns:
				missing_cols.append(f"{col} missing in right table")

		if missing_cols:
			logging.warning(f"Cannot merge tables: {', '.join(missing_cols)}. Returning left DataFrame.")
			return left_df

		# Perform the merge
		if is_dask_left:  # Both are Dask DataFrames
			result = dd.merge(left_df, right_df, on=on, how=how)
			# Logging row counts for Dask DataFrames requires computation.
			# Log partition info instead or a general Dask merge message.
			logging.info(f"Dask merge performed on columns: {on}. Left npartitions: {left_df.npartitions}, Right npartitions: {right_df.npartitions}, Result npartitions: {result.npartitions}")
		else:  # Both are pandas DataFrames
			result = left_df.merge(right_df, on=on, how=how)
			logging.info(f"Pandas merged_table: {len(left_df)} rows + {len(right_df)} rows → {len(result)} rows on columns: {on}")

		return result


	def load_connected_tables(self, mimic_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1', use_dask: bool = False, merged_view: bool = False, max_chunks: Optional[int] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
		"""
			Loads and connects the six main MIMIC-IV tables: patients, admissions, diagnoses_icd, d_icd_diagnoses, poe, and poe_detail.

			This method coordinates the loading of all tables in the proper order to maintain
			data integrity throughout the connected tables.

			Args:
				mimic_path: Path to the MIMIC-IV dataset
				sample_size: Number of rows to sample for patients table
				encoding: File encoding
				use_dask: Whether to use Dask for loading larger tables
				merged_view: If True, merges tables into a single comprehensive DataFrame
				max_chunks: Maximum number of chunks to process for underlying table loads

			Returns:
				If merged_view=False: Dictionary containing the loaded tables
				If merged_view=True: Tuple containing (Dictionary of tables, Merged DataFrame)
		"""

		logging.info("Loading connected MIMIC-IV tables...")
		tables = self._get_individual_tables(mimic_path=mimic_path, sample_size=sample_size, encoding=encoding, use_dask=use_dask, max_chunks=max_chunks)

		# If merged view is requested, create a comprehensive merged dataframe
		merged_df = pd.DataFrame()
		if merged_view:
			logging.info("Merged view requested by caller, creating comprehensive merged dataframe...")
			merged_df = self.create_merged_view(tables)
		else:
			logging.info("Merged view not requested by caller.")

		return tables, merged_df



	def _get_individual_tables(self, mimic_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1', use_dask: bool = False, max_chunks: Optional[int] = None) -> Dict[str, pd.DataFrame]:

		tables = {}

		# Prepare args for table loading functions, excluding max_chunks for load_reference_table
		cohort_args = { 'mimic_path': mimic_path, 'encoding': encoding, 'use_dask': use_dask, 'max_chunks': max_chunks }
		filter_args = { 'mimic_path': mimic_path, 'module':'hosp', 'encoding': encoding, 'use_dask': use_dask, 'max_chunks': max_chunks } # module will be overridden per call if needed for icu

		# Step 1: Load reference table d_icd_diagnoses (always load in full, no max_chunks)
		tables['d_icd_diagnoses'] = self.load_reference_table(
			mimic_path=mimic_path,
			module='hosp',
			table_name='d_icd_diagnoses',
			encoding=encoding
		)

		if tables['d_icd_diagnoses'].empty:
			return {}  # Cannot proceed without reference table

		# Step 2: Load and sample patients table
		tables['patients'], sampled_subject_ids = self.load_patient_cohort(sample_size=sample_size, **cohort_args)

		if tables['patients'].empty:
			return {}  # Cannot proceed without patients

		# Step 3: Load admissions for sampled patients
		tables['admissions'] = self.load_filtered_table(table_name='admissions', filter_column='subject_id', filter_values=sampled_subject_ids, **filter_args )

		if tables['admissions'].empty:
			logging.warning("No admissions found for sampled patients")



		# Get hadm_ids for filtering subsequent tables
		sampled_hadm_ids = set(tables['admissions']['hadm_id']) if 'hadm_id' in tables['admissions'].columns else set()


		# Step 4: Load diagnoses for sampled admissions
		if sampled_hadm_ids:
			tables['diagnoses_icd'] = self.load_filtered_table(table_name='diagnoses_icd', filter_column='hadm_id', filter_values=sampled_hadm_ids, **filter_args )

			# Step 5: Link diagnoses with their descriptions
			if not tables['diagnoses_icd'].empty:
				tables['diagnoses_icd'] = self.merge_tables( left_df=tables['diagnoses_icd'], right_df=tables['d_icd_diagnoses'], on=['icd_code', 'icd_version'], how='left' )

		else:
			tables['diagnoses_icd'] = pd.DataFrame()
			logging.warning("No diagnoses loaded: missing admission IDs")


		# Step 6: Load POE (Provider Order Entry) for sampled admissions
		if sampled_hadm_ids:
			tables['poe'] = self.load_filtered_table(table_name='poe', filter_column='hadm_id', filter_values=sampled_hadm_ids, **filter_args )

			# Get poe_ids for filtering poe_detail
			sampled_poe_ids = set(tables['poe']['poe_id']) if 'poe_id' in tables['poe'].columns else set()

			# Step 7: Load POE_detail for orders in our cohort
			if sampled_poe_ids:
				tables['poe_detail'] = self.load_filtered_table(table_name='poe_detail', filter_column='poe_id', filter_values=sampled_poe_ids, **filter_args )
			else:
				tables['poe_detail'] = pd.DataFrame()
				logging.warning("No POE details loaded: missing POE IDs")

		else:
			tables['poe'] = pd.DataFrame()
			tables['poe_detail'] = pd.DataFrame()
			logging.warning("No POE data loaded: missing admission IDs")

		logging.info("Completed loading connected MIMIC-IV tables")

		return tables


	def create_merged_view(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
		"""
		Creates a comprehensive merged view of connected MIMIC-IV tables.

		Args:
			tables: Dictionary containing loaded MIMIC-IV tables

		Returns:
			Merged DataFrame
		"""

		logging.info("Creating merged view of connected tables...")

		# Extract the tables for readability
		patients_df        = tables.get('patients', pd.DataFrame())
		admissions_df      = tables.get('admissions', pd.DataFrame())
		diagnoses_df       = tables.get('diagnoses_icd', pd.DataFrame())
		d_icd_diagnoses_df = tables.get('d_icd_diagnoses', pd.DataFrame())
		poe_df             = tables.get('poe', pd.DataFrame())
		poe_detail_df      = tables.get('poe_detail', pd.DataFrame())

		# Check if we have our core tables
		if not patients_df.empty and not admissions_df.empty:

			# 1. Start with patients + admissions
			logging.info("Merging patients and admissions tables...")

			merged_df = self.merge_tables( left_df=patients_df, right_df=admissions_df, on=['subject_id'], how='inner' )

			# 2. Add diagnoses if available
			if not diagnoses_df.empty:
				logging.info("Adding diagnoses information...")

				# Prepare diagnoses with descriptions
				if not d_icd_diagnoses_df.empty and 'icd_code' in diagnoses_df.columns:
					diagnoses_with_desc = self.merge_tables( left_df=diagnoses_df, right_df=d_icd_diagnoses_df, on=['icd_code', 'icd_version'], how='left' )

				else:
					diagnoses_with_desc = diagnoses_df

				# Take the first diagnosis for each admission to avoid duplicating rows
				logging.info("Selecting primary diagnoses for each admission...")

				if 'seq_num' in diagnoses_with_desc.columns:

					# Get primary diagnoses
					first_diag = diagnoses_with_desc.sort_values('seq_num').groupby(['subject_id', 'hadm_id']).first().reset_index()

					# Merge primary diagnoses with the main dataset
					merged_df = self.merge_tables( left_df=merged_df, right_df=first_diag, on=['subject_id', 'hadm_id'], how='left' )


			# 3. Add first POE record for each admission if available
			if not poe_df.empty:
				logging.info("Adding provider order information...")

				if 'ordertime' in poe_df.columns:
					# Take the first order for each admission
					first_order = poe_df.sort_values('ordertime').groupby(['subject_id', 'hadm_id']).first().reset_index()

					# Merge first order with the main dataset
					merged_df = self.merge_tables( left_df=merged_df, right_df=first_order, on=['subject_id', 'hadm_id'], how='left' )

					# 4. Add POE details if available
					if not poe_detail_df.empty:
						logging.info("Adding provider order details...")

						# Reshape order details from tall to wide format for key fields
						if 'field_name' in poe_detail_df.columns and 'field_value' in poe_detail_df.columns:

							# Get most important fields
							important_fields = poe_detail_df.field_name.value_counts().head(10).index.tolist()

							# Filter to just the important fields
							poe_detail_filtered = poe_detail_df[poe_detail_df.field_name.isin(important_fields)]

							# Pivot to get one row per order with columns for each field
							try:

								poe_detail_wide = poe_detail_filtered.pivot_table(
									index   = ['poe_id'],
									columns = 'field_name',
									values  = 'field_value',
									aggfunc = 'first'
								).reset_index()

								# Merge into main dataframe if pivot was successful
								if not poe_detail_wide.empty:
									merged_df = self.merge_tables( left_df=merged_df, right_df=poe_detail_wide, on=['poe_id'], how='left' )

							except Exception as e:

								logging.warning(f"Could not pivot POE details: {e}")

								# Just merge in the first detail record for each POE
								first_detail = poe_detail_df.groupby('poe_id').first().reset_index()

								merged_df = self.merge_tables( left_df=merged_df, right_df=first_detail, on=['poe_id'], how='left' )

			# Finalize merged dataframe
			logging.info(f"Created merged view with {len(merged_df)} rows and {len(merged_df.columns)} columns")
			return merged_df


		logging.warning("Cannot create merged view: missing core tables")
		return pd.DataFrame()
