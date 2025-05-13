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
from tqdm import tqdm
from pathlib import Path
from functools import cached_property
import enum

# TODO: I should create a jupyter notebook, and load the tables there, and do the merging manually, make sure it wokrs. then update this code accordingly (update it manually)

# Import filtering functionality
from mimic_iv_analysis.core.filtering import Filtering

# Utility functions
# def convert_string_dtypes(df):
# 	"""Convert pandas StringDtype to object type to avoid Arrow conversion issues in Streamlit.

# 	Args:
# 		df: Input DataFrame (pandas or Dask)

# 	Returns:
# 		DataFrame with StringDtype columns converted to object type
# 	"""
# 	if df is None:
# 		return df

# 	if hasattr(df, 'compute'):
# 		# For Dask DataFrame, apply the conversion without computing
# 		string_cols = [col for col in df.columns if str(df[col].dtype) == 'string']
# 		if string_cols:
# 			return df.map_partitions(lambda partition:
# 				partition.assign(**{col: partition[col].astype('object') for col in string_cols})
# 			)
# 		return df

# 	# For pandas DataFrame
# 	for col in df.columns:
# 		if hasattr(df[col], 'dtype') and str(df[col].dtype) == 'string':
# 			df[col] = df[col].astype('object')
# 	return df

# Constants
DEFAULT_MIMIC_PATH    = Path("/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1")
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

SUBJECT_ID_COL        = 'subject_id'



class TableNamesHOSP(enum.Enum):
	ADMISSIONS         = 'admissions'
	D_HCPCS            = 'd_hcpcs'
	D_ICD_DIAGNOSES    = 'd_icd_diagnoses'
	D_ICD_PROCEDURES   = 'd_icd_procedures'
	D_LABITEMS         = 'd_labitems'
	DIAGNOSES_ICD      = 'diagnoses_icd'
	DRGCODES           = 'drgcodes'
	EMAR               = 'emar'
	EMAR_DETAIL        = 'emar_detail'
	HCPCSEVENTS        = 'hcpcsevents'
	LABEVENTS          = 'labevents'
	MICROBIOLOGYEVENTS = 'microbiologyevents'
	OMR                = 'omr'
	PATIENTS           = 'patients'
	PHARMACY           = 'pharmacy'
	POE                = 'poe'
	POE_DETAIL         = 'poe_detail'
	PRESCRIPTIONS      = 'prescriptions'
	PROCEDURES_ICD     = 'procedures_icd'
	PROVIDER           = 'provider'
	SERVICES           = 'services'
	TRANSFERS          = 'transfers'

	@classmethod
	def values(cls):
		return [member.value for member in cls]

	@property
	def description(self):

		tables_descriptions = {
			('hosp', 'admissions')        : "Patient hospital admissions information",
			('hosp', 'patients')          : "Patient demographic data",
			('hosp', 'labevents')         : "Laboratory measurements (large file)",
			('hosp', 'microbiologyevents'): "Microbiology test results",
			('hosp', 'pharmacy')          : "Pharmacy orders",
			('hosp', 'prescriptions')     : "Medication prescriptions",
			('hosp', 'procedures_icd')    : "Patient procedures",
			('hosp', 'diagnoses_icd')     : "Patient diagnoses",
			('hosp', 'emar')              : "Electronic medication administration records",
			('hosp', 'emar_detail')       : "Detailed medication administration data",
			('hosp', 'poe')               : "Provider order entries",
			('hosp', 'poe_detail')        : "Detailed order information",
			('hosp', 'd_hcpcs')           : "HCPCS code definitions",
			('hosp', 'd_icd_diagnoses')   : "ICD diagnosis code definitions",
			('hosp', 'd_icd_procedures')  : "ICD procedure code definitions",
			('hosp', 'd_labitems')        : "Laboratory test definitions",
			('hosp', 'hcpcsevents')       : "HCPCS events",
			('hosp', 'drgcodes')          : "Diagnosis-related group codes",
			('hosp', 'services')          : "Hospital services",
			('hosp', 'transfers')         : "Patient transfers",
			('hosp', 'provider')          : "Provider information",
			('hosp', 'omr')               : "Order monitoring results"
		}

		return tables_descriptions.get(('hosp', self.value))

class TableNamesICU(enum.Enum):
	CAREGIVER          = 'caregiver'
	CHARTEVENTS        = 'chartevents'
	DATETIMEEVENTS     = 'datetimeevents'
	D_ITEMS            = 'd_items'
	ICUSTAYS           = 'icustays'
	INGREDIENTEVENTS   = 'ingredientevents'
	INPUTEVENTS        = 'inputevents'
	OUTPUTEVENTS       = 'outputevents'
	PROCEDUREEVENTS    = 'procedureevents'

	@classmethod
	def values(cls):
		return [member.value for member in cls]

	@property
	def description(self):

		tables_descriptions = {
			('icu', 'chartevents')        : "Patient charting data (vital signs, etc.)",
			('icu', 'datetimeevents')     : "Date/time-based events",
			('icu', 'inputevents')        : "Patient intake data",
			('icu', 'outputevents')       : "Patient output data",
			('icu', 'procedureevents')    : "ICU procedures",
			('icu', 'ingredientevents')   : "Detailed medication ingredients",
			('icu', 'd_items')            : "Dictionary of ICU items",
			('icu', 'icustays')           : "ICU stay information",
			('icu', 'caregiver')          : "Caregiver information"
		}

		return tables_descriptions.get(('icu', self.value))


dtypes_all = {
	'discontinued_by_poe_id': 'object',
	'long_description'      : 'string',
	'icd_code'              : 'string',
	'drg_type'              : 'category',
	'enter_provider_id'     : 'string',
	'hadm_id'               : 'int',
	'icustay_id'            : 'int',
	'leave_provider_id'     : 'string',
	'poe_id'                : 'string',
	'emar_id'               : 'string',
	'subject_id'            : 'int64',
	'pharmacy_id'           : 'string',
	'interpretation'        : 'object',
	'org_name'              : 'object',
	'quantity'              : 'object',
	'infusion_type'         : 'object',
	'sliding_scale'         : 'object',
	'fill_quantity'         : 'object',
	'expiration_unit'       : 'category',
	'duration_interval'     : 'category',
	'dispensation'          : 'category',
	'expirationdate'        : 'object',
	'one_hr_max'            : 'object',
	'infusion_type'         : 'object',
	'sliding_scale'         : 'object',
	'lockout_interval'      : 'object',
	'basal_rate'            : 'object',
	'form_unit_disp'        : 'category',
	'route'                 : 'category',
	'dose_unit_rx'          : 'category',
	'drug_type'             : 'category',
	'form_rx'               : 'object',
	'form_val_disp'         : 'object',
	'gsn'                   : 'object',
	'dose_val_rx'           : 'object',
	'prev_service'          : 'object',
	'curr_service'          : 'category'}

parse_dates_all = [
			'admittime',
			'dischtime',
			'deathtime',
			'edregtime',
			'edouttime',
			'charttime',
			'scheduletime',
			'storetime',
			'storedate']



class DataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""

	DEFAULT_STUDY_TABLES_LIST = [	TableNamesHOSP.PATIENTS.value,
									TableNamesICU.ADMISSIONS.value,
									TableNamesHOSP.DIAGNOSES_ICD.value,
									TableNamesICU.D_ICD_DIAGNOSES.value,
									TableNamesICU.POE.value,
									TableNamesICU.POE_DETAIL.value]


	def __init__(self, 	mimic_path        : Path = DEFAULT_MIMIC_PATH,
						study_tables_list : Optional[List[TableNamesHOSP | TableNamesICU]] = None):

		# MIMIC_IV v3.1 path
		self.mimic_path = mimic_path

		# Tables to load. Use list provided by user or default list
		self.study_table_list = self.DEFAULT_STUDY_TABLES_LIST if study_tables_list is None else [table.value for table in study_tables_list]

		# Class variables
		self.filtering           : Filtering              = Filtering()
		self._subject_ids_list   : List[str]              = []
		self._patients_file_path : Optional[str]          = None
		self.dataset_info_df     : Optional[pd.DataFrame] = None


	def scan_mimic_directory(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
		"""Scans the MIMIC-IV directory structure and returns a DataFrame with table info.

		Returns:
			pd.DataFrame: DataFrame containing columns:
				- module      : The module name (hosp/icu)
				- table_name  : Name of the table
				- file_path   : Full path to the file
				- file_size_mb: Size of file in MB
				- display_name: Formatted display name with size
		"""

		def _get_list_of_available_tables(module_path: Path) -> Dict[str, Path]:
			"""Lists unique table files from a module path."""

			POSSIBLE_FILE_TYPES = ['.parquet', '.csv', '.csv.gz']

			def _get_all_files() -> List[str]:

				filenames = []
				for suffix in POSSIBLE_FILE_TYPES:

					tables_path_list = glob.glob(os.path.join(module_path, f'*{suffix}'))
					if not tables_path_list:
						continue

					filenames.extend([os.path.basename(table_path).replace(suffix, '') for table_path in tables_path_list])

				return list(set(filenames))

			def _get_priority_file(table_name: str) -> Optional[Path]:

				# First priority is parquet
				if (module_path / f'{table_name}.parquet').exists():
					return module_path / f'{table_name}.parquet'

				# Second priority is csv
				if (module_path / f'{table_name}.csv').exists():
					return module_path / f'{table_name}.csv'

				# Third priority is csv.gz
				if (module_path / f'{table_name}.csv.gz').exists():
					return module_path / f'{table_name}.csv.gz'

				# If none exist, return None
				return None

			filenames = _get_all_files()

			return {table_name: _get_priority_file(table_name) for table_name in filenames}


		def _get_available_tables_info(available_tables_dict: Dict[str, Path], module: str):
			"""Extracts table information from a dictionary of table files."""

			def _format_size(size_mb: float) -> str:
				if size_mb < 1e-3:
					return "(< 1 KB)"
				if size_mb < 1:
					return f"({size_mb * 1024:.1f} KB)"
				if size_mb < 1000:
					return f"({size_mb:.1f} MB)"
				return f"({size_mb / 1000:.1f} GB)"

			nonlocal dataset_info

			dataset_info['available_tables'][module] = []

			# Iterate through all tables in the module
			for table_name, file_path in available_tables_dict.items():

				if file_path is None or not file_path.exists():
					continue

				# Add to available tables
				dataset_info['available_tables'][module].append(table_name)

				# Store file path
				dataset_info['file_paths'][(module, table_name)] = file_path

				try:
					file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

				except OSError as e:
					logging.warning(f"Error getting file size for {file_path}: {e}")
					file_size_mb = 0

				# Store file size
				dataset_info['file_sizes'][(module, table_name)] = file_size_mb

				# Format size string
				size_str = _format_size(file_size_mb)

				# Store display name
				dataset_info['table_display_names'][(module, table_name)] = f"{table_name} {size_str}"

				# Store file suffix
				suffix = file_path.suffix
				dataset_info['file_suffix'][(module, table_name)] = 'csv.gz' if suffix == '.gz' else suffix

		def _get_info_as_dataframe() -> pd.DataFrame:

			table_info = []
			for module in dataset_info['available_tables']:
				for table_name in dataset_info['available_tables'][module]:

					file_path = dataset_info['file_paths'][(module, table_name)]

					table_info.append({
						'module'      : module,
						'table_name'  : table_name,
						'file_path'   : file_path,
						'file_size_mb': dataset_info['file_sizes'][(module, table_name)],
						'display_name': dataset_info['table_display_names'][(module, table_name)],
						'suffix'      : dataset_info['file_suffix'][(module, table_name)]
					})

			# Convert to DataFrame
			dataset_info_df = pd.DataFrame(table_info)

			# Add mimic path as an attribute
			dataset_info_df.attrs['mimic_path'] = self.mimic_path

			return dataset_info_df

		# Initialize dataset info
		dataset_info = {
			'available_tables'   : {},
			'file_paths'         : {},
			'file_sizes'         : {},
			'table_display_names': {},
			'file_suffix'        : {},
		}

		# If the mimic path does not exist, return an empty DataFrame
		if not self.mimic_path.exists():
			return pd.DataFrame(columns=['module', 'table_name', 'file_path', 'file_size_mb', 'display_name'])

		# Iterate through modules
		modules = ['hosp', 'icu']
		for module in modules:

			# Get module path
			module_path: Path = self.mimic_path / module

			# if the module does not exist, skip it
			if not module_path.exists():
				continue

			# Get available tables:
			available_tables_dict = _get_list_of_available_tables(module_path)
			# df = pd.DataFrame(available_tables_dict, index=['path']).T.reset_index().rename(columns={'index':'name'})

			# If no tables found, skip this module
			if not available_tables_dict:
				continue

			# Get available tables info
			_get_available_tables_info(available_tables_dict, module)

		# Convert to DataFrame
		self.dataset_info_df = _get_info_as_dataframe()

		return self.dataset_info_df, dataset_info




	def _load_csv_table_with_correct_column_datatypes(self, file_path: Path):


		# Check if file exists
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"CSV file not found: {file_path}")

		# First read a small sample to get column names without type conversion
		sample_df = pd.read_csv(file_path, nrows=5)
		columns   = sample_df.columns.tolist()

		# Filter dtypes and parse_dates to only include existing columns
		dtypes      = {col: dtype for col, dtype in dtypes_all.items() if col in columns}
		parse_dates = [col for col in parse_dates_all if col in columns]

		# Now read with dask using correct types
		df = dd.read_csv(
			urlpath        = file_path,
			dtype          = dtypes,
			parse_dates    = parse_dates if parse_dates else None,
			assume_missing = True,
			blocksize      = None if file_path.suffix == '.gz' else '200MB'
		)

		return df


	def save_all_tables_as_parquet(self):
		""" Checks if parquet versions exist and creates them if needed. """

		if self.dataset_info_df is None:
			self.dataset_info_df , _ = self.scan_mimic_directory()


		dataset_info_df = self.dataset_info_df[self.dataset_info_df.table_name.isin(self.study_tables_list)]


		for i in tqdm(range(len(dataset_info_df)), desc="Saving tables as parquet"):

			row        = dataset_info_df.iloc[i]
			table_name = row.table_name
			file_path  = row.file_path

			if (tables_list is not None) and (table_name not in tables_list):
				continue

			tqdm.write(f"Processing {table_name}")
			self.save_as_parquet(source_csv_path=file_path)


	def save_as_parquet(self, source_csv_path: Path, df: Optional[pd.DataFrame]=None, target_parquet_path: Optional[Path]=None):
		""" Saves a DataFrame as a Parquet file in the parquet_files directory. """

		def _fix_source_csv_path() -> Path:
			""" Fixes the source csv path if it is a parquet file. """

			if source_csv_path.name.endswith('.parquet'):

				csv_dir = source_csv_path.parent / source_csv_path.name.replace('.parquet', '.csv')
				gz_dir  = source_csv_path.parent / source_csv_path.name.replace('.parquet', '.csv.gz')

				if csv_dir.exists():
					return csv_dir

				if gz_dir.exists():
					return gz_dir

				else:
					raise ValueError(f"Cannot find csv or csv.gz file for {source_csv_path}")

			return source_csv_path


		try:

			if df is None:

				source_csv_path = _fix_source_csv_path()

				# Load table
				df = self._load_csv_table_with_correct_column_datatypes(file_path=source_csv_path)

			# Get parquet directory
			if target_parquet_path is None:
				suffix = '.csv.gz' if source_csv_path.name.endswith('.gz') else '.csv'
				target_parquet_path = source_csv_path.parent / source_csv_path.name.replace(suffix, '.parquet')

			# Save to parquet
			df.to_parquet(target_parquet_path, engine='pyarrow')

		except Exception as e:
			logging.error(f"Failed to save {source_csv_path} to parquet: {str(e)}")


	@cached_property
	def subject_ids_list(self) -> List[Any]:
		""" Returns a list of unique subject_ids found in patients.csv. """

		# Load subject IDs if not already loaded or if the list is empty
		if not self._subject_ids_list:
			self._update_subject_ids_list()

		return self._subject_ids_list


	def _update_subject_ids_list(self):
		logging.info("Loading subject IDs from patients table (this will be cached)")

		# Scan directory if not already done
		if self.dataset_info_df is None:
			self.dataset_info_df , _ = self.scan_mimic_directory()

		# Get patients.csv file path
		md = self.dataset_info_df.module     == 'hosp'
		tn = self.dataset_info_df.table_name == TableNamesHOSP.PATIENTS.value

		patients_df_file_path = Path( self.dataset_info_df[ md & tn ].file_path.values[0] )


		# Check if patients.csv exists
		if not patients_df_file_path.exists():
			logging.warning(f"patients.csv not found at {patients_df_file_path}. Cannot load subject IDs.")
			self._subject_ids_list = []
			return

		# Load patients table
		patients_df = self.load_table_full(file_path=patients_df_file_path)

		# Get subject IDs
		self._subject_ids_list = patients_df[SUBJECT_ID_COL].unique().compute().tolist()


	def sampled_subject_ids_list(self, num_subjects: int = 100, random_selection: bool = False) -> List[Any]:
		"""Returns a list of subject_ids for sampling. """
		if not self.subject_ids_list or num_subjects <= 0:
			return []

		if random_selection:
			return random.sample(self.subject_ids_list, num_subjects)

		return self.subject_ids_list[:num_subjects]


	def load_table_full(self, file_path: Path) -> pd.DataFrame:

		# The parquet files are already saved with the correct datatypes
		if file_path.suffix == '.parquet':
			return dd.read_parquet(file_path)

		return self._load_csv_table_with_correct_column_datatypes(file_path)


	def load_table_for_sampled_subject_ids(self, file_path: Path, subject_ids: List[Any]) -> pd.DataFrame:
		"""Loads a table for a list of sampled subject_ids."""
		table_df = self.load_table_full(file_path)
		return table_df[table_df['subject_id'].isin(subject_ids)]


	def load_mimic_table(self,
							file_path         : Path,
							sample_size       : int = DEFAULT_SAMPLE_SIZE,
							encoding          : str = 'latin-1',
							use_dask          : bool = False,
							max_chunks        : Optional[int] = None,
							target_subject_ids: Optional[List[Any]] = 'subject_id'
						) -> Tuple[Optional[Union[pd.DataFrame, dd.DataFrame]], int]:
		"""Loads a specific MIMIC-IV table, handling large files and sampling.

		Args:
			file_path: Path to the CSV/CSV.GZ file
			sample_size: Number of rows to sample for large files (used if target_subject_ids is not effective)
			encoding: File encoding
			use_dask: Whether to use Dask for loading
			max_chunks: Maximum number of chunks to process (for pandas chunking or Dask debugging)
			target_subject_ids: Optional list of subject_ids to filter the table by.

		Returns:
			Tuple containing:
				- DataFrame with the loaded data (may be sampled or filtered by subject_id)
				- Total row count (can be estimate or count after filtering, context dependent)
		"""

		def _has_subject_id_col():
			"""Attempts to check if the file has the SUBJECT_ID_COL without loading the entire file."""
			try:
				header_df = pd.read_csv(file_path, nrows=0, **read_params)
				return SUBJECT_ID_COL in header_df.columns

			except Exception as e:
				logging.warning(f"Could not read header for {os.path.basename(file_path)}: {e}. Assuming no '{SUBJECT_ID_COL}'.")
				return False


		def _load_large_table():

			def load_with_dask():
				"""Loads a large file by filtering by subject_ids using Dask."""

				logging.info(f"Using Dask to filter by subject_id for {os.path.basename(file_path)}.")
				try:
					# Define dtypes for known columns to help Dask if possible
					# This part might need to be more dynamic or passed in if we want to optimize for all tables
					dtypes_for_dask = {SUBJECT_ID_COL: 'Int64', 'expirationdate': 'datetime64[ns]'} # Example, might need more

					# For very large files, only load subject_id and other essential columns if specified elsewhere
					# For now, loads all columns then filters.
					if file_path.suffix == '.parquet':
						ddf = dd.read_parquet(file_path)
					else:
						ddf = dd.read_csv(file_path, dtype=dtypes_for_dask, **read_params)

					# Filter by target_subject_ids
					filtered_ddf = ddf[ddf[SUBJECT_ID_COL].isin(target_subject_ids)]

					# Convert Dask DataFrame to pandas DataFrame
					df_result = filtered_ddf.compute()

					# Calculate total rows loaded
					total_rows_loaded = len(df_result)

					# Log the result
					logging.info(f"Loaded {total_rows_loaded} rows for {len(target_subject_ids)} subjects from {os.path.basename(file_path)} using Dask.")
					return df_result , total_rows_loaded # convert_string_dtypes(df_result)

				except Exception as e:
					logging.error(f"Dask filtering by subject_id failed for {os.path.basename(file_path)}: {e}. Falling back.")
					traceback.print_exc()
					# Fallback if Dask direct filter fails; will go to standard Dask load or pandas load below

			def _load_with_pandas_chunking():
				"""Loads a large file by filtering by subject_ids using pandas chunking."""

				logging.info(f"Using Pandas chunking to filter by subject_id for {os.path.basename(file_path)}.")

				chunks_for_target_ids = []
				processed_chunks      = 0

				# Read the file in chunks
				with pd.read_csv(file_path, chunksize=100000, **read_params) as reader:

					# Iterate through chunks
					for chunk in reader:

						# Increment processed chunks counter
						processed_chunks += 1

						# Filter chunk by target_subject_ids
						filtered_chunk = chunk[chunk[SUBJECT_ID_COL].isin(target_subject_ids)]

						# Add to chunks if not empty
						if not filtered_chunk.empty:
							chunks_for_target_ids.append(filtered_chunk)

						# Check if we've reached max chunks
						if max_chunks is not None and max_chunks != -1 and processed_chunks >= max_chunks:
							logging.info(f"Reached max_chunks ({max_chunks}) during subject_id filtering for {os.path.basename(file_path)}.")
							break

					# Combine filtered chunks into final DataFrame
					if chunks_for_target_ids:
						df_result = pd.concat(chunks_for_target_ids, ignore_index=True)
					else:
						df_result = pd.DataFrame(columns=header_df.columns) # Empty df with correct columns

					# Calculate total rows loaded
					total_rows_loaded = len(df_result)

					# Log the result
					logging.info(f"Loaded {total_rows_loaded} rows for {len(target_subject_ids)} subjects from {os.path.basename(file_path)} using Pandas chunking.")

				return df_result, total_rows_loaded # convert_string_dtypes(df_result)


			logging.info(f"Attempting to load {len(target_subject_ids)} subject_ids for large file: {os.path.basename(file_path)}")

			return load_with_dask() if use_dask else _load_with_pandas_chunking()


		def _load_table_normal_logic():

			def _load_with_dask():
				# Using Dask for large file processing (or if explicitly requested)
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
						if target_subject_ids and has_subject_id_col and SUBJECT_ID_COL in df.columns: # Apply subject filter if small file
							df = df[df[SUBJECT_ID_COL].isin(target_subject_ids)]
						df = df.sample(sample_size, random_state=RANDOM_STATE)
				else:
					# If file is smaller than sample size, load it all
					df = dask_df.compute()
					if target_subject_ids and has_subject_id_col and SUBJECT_ID_COL in df.columns: # Apply subject filter if small file
						df = df[df[SUBJECT_ID_COL].isin(target_subject_ids)]

					return df, total_rows # convert_string_dtypes(df)

			def _load_with_pandas():
				# Traditional pandas approach
				# Get total row count without loading entire file
				total_rows           = 0
				chunks               = []
				chunk_count          = 0
				total_rows_processed = 0

				# Load chunks
				with pd.read_csv(file_path, chunksize=100000, **read_params) as reader: # chunksize can be tuned

					# Iterate through chunks
					for chunk in reader:
						chunks.append(chunk)
						chunk_count += 1
						total_rows_processed += len(chunk)

						# Log progress
						logging.info(f"Processed chunk {chunk_count} for {os.path.basename(file_path)}. Total rows processed: {total_rows_processed}")

						# Check if we've reached max chunks
						if max_chunks is not None and max_chunks != -1 and chunk_count >= max_chunks:
							logging.info(f"Reached max_chunks ({max_chunks}) for {os.path.basename(file_path)}. Processed {chunk_count} chunks.")
							break

				# Calculate total rows
				total_rows = total_rows_processed

				# Load actual data (with sampling if needed)
				if sample_size is not None and file_size_mb > LARGE_FILE_THRESHOLD_MB:
					df = pd.read_csv(file_path, nrows=sample_size, **read_params)
				else:
					df = pd.concat(chunks, ignore_index=True)

				# Apply subject_id filtering if it's a small file or fallback
				if target_subject_ids and has_subject_id_col and SUBJECT_ID_COL in df.columns:
					df = df[df[SUBJECT_ID_COL].isin(target_subject_ids)]

				# Apply sampling only if sample_size is specified and df is larger
				if sample_size is not None and len(df) > sample_size:
					df = df.sample(sample_size, random_state=RANDOM_STATE)

				return df, total_rows # convert_string_dtypes(df)

			logging.info(f"Proceeding with standard load for {os.path.basename(file_path)} (use_dask={use_dask}, sample_size={sample_size}).")

			df, total_rows = _load_with_dask() if use_dask else _load_with_pandas()

			return df, total_rows


		try:
			logging.info(f"Loading data from {os.path.basename(file_path)}...")

			file_size_mb  = os.path.getsize(file_path) / (1024 * 1024)
			is_compressed = file_path.endswith('.gz')
			compression   = 'gzip' if is_compressed else None
			read_params   = { 'encoding': encoding, 'compression': compression, 'low_memory': False }

			# Check if the file has the SUBJECT_ID_COL
			has_subject_id_col = _has_subject_id_col()


			# Primary filtering logic: If target_subject_ids are provided and the file is large and has subject_id
			if target_subject_ids and has_subject_id_col and file_size_mb > LARGE_FILE_THRESHOLD_MB:
				return _load_large_table()

			# Standard loading logic (if not filtered by subject_ids for large files or if it's a small file)
			return _load_table_normal_logic()


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
		is_dask_left  = isinstance(left_df, dd.DataFrame)
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






if __name__ == '__main__':

	MIMIC_DATA_PATH = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"

	data_loader = DataLoader()

	# Scan the directory
	dataset_info_df, _ = data_loader.scan_mimic_directory()

	subject_ids_list = data_loader.sampled_subject_ids_list(num_samples=10, random_selection=True)
	print(subject_ids_list)

	# TODO (next step):
	# 	1. Filter the large tables with the subject_ids_list.
	# 	2. Load multiple tables and merge them into one.
	# 	3. Save the merged table as a parquet file.
