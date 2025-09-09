# Standard library imports
import os
import glob
from pathlib import Path
from functools import lru_cache, cached_property
from typing import Dict, Optional, Tuple, List, Any, Literal
import warnings

# Data processing imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import humanize
from tqdm import tqdm

from mimic_iv_analysis import logger
from mimic_iv_analysis.core.filtering import Filtering
from mimic_iv_analysis.configurations import (  TableNames,
												pyarrow_dtypes_map,
												COLUMN_TYPES,
												DATETIME_COLUMNS,
												DEFAULT_MIMIC_PATH,
												DEFAULT_NUM_SUBJECTS,
												SUBJECT_ID_COL,
												DEFAULT_STUDY_TABLES_LIST,
												DataFrameType)


class DataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""

	def __init__(self, mimic_path: Path = DEFAULT_MIMIC_PATH, study_tables_list: Optional[List[TableNames]] = None, apply_filtering: bool = True, include_transfers: bool = False):
		# Initialize persisted resources tracking
		self._persisted_resources = {}

		# MIMIC_IV v3.1 path
		self.mimic_path      = Path(mimic_path)
		self.apply_filtering = apply_filtering
		self.include_transfers = include_transfers

		# Tables to load. Use list provided by user or default list
		self.study_table_list = set(study_tables_list or DEFAULT_STUDY_TABLES_LIST)

		if not include_transfers:
			self.study_table_list -= {TableNames.TRANSFERS}

		# Class variables
		self._all_subject_ids       : List[int]                = []
		self.tables_info_df         : Optional[pd.DataFrame]  = None
		self.tables_info_dict       : Optional[Dict[str, Any]] = None
		self.partial_subject_id_list: Optional[List[int]]      = None

	@lru_cache(maxsize=None)
	def scan_mimic_directory(self):
		"""Scans the MIMIC-IV directory structure and updates the tables_info_df and tables_info_dict attributes.

			tables_info_df is a DataFrame containing info:
				pd.DataFrame: DataFrame containing columns:
					- module      : The module name (hosp/icu)
					- table_name  : Name of the table
					- file_path   : Full path to the file
					- file_size   : Size of file in MB
					- display_name: Formatted display name with size
					- suffix      : File suffix (csv, csv.gz, parquet)
					- columns_list: List of columns in the table

			tables_info_dict is a dictionary containing info:
				Dict[str, Any]: Dictionary containing keys:
					- available_tables   : Dictionary of available tables
					- file_paths         : Dictionary of file paths
					- file_sizes         : Dictionary of file sizes
					- table_display_names: Dictionary of table display names
					- suffix             : Dictionary of file suffixes
					- columns_list       : Dictionary of column lists
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

		def _get_available_tables_info(available_tables_dict: Dict[str, Path], module: Literal['hosp', 'icu']):
			"""Extracts table information from a dictionary of table files."""

			def _get_file_size_in_bytes(file_path: Path) -> int:
				if file_path.suffix == '.parquet':
					return sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
				return file_path.stat().st_size

			tables_info_dict['available_tables'][module] = []

			# Iterate through all tables in the module
			for table_name, file_path in available_tables_dict.items():

				if file_path is None or not file_path.exists():
					continue

				# Add to available tables
				tables_info_dict['available_tables'][module].append(table_name)

				# Store file path
				tables_info_dict['file_paths'][(module, table_name)] = file_path

				# Store file size
				tables_info_dict['file_sizes'][(module, table_name)] = _get_file_size_in_bytes(file_path)

				# Store display name
				tables_info_dict['table_display_names'][(module, table_name)] = (
					f"{table_name} {humanize.naturalsize(_get_file_size_in_bytes(file_path))}"
				)

				# Store file suffix
				suffix = file_path.suffix
				tables_info_dict['suffix'][(module, table_name)] = 'csv.gz' if suffix == '.gz' else suffix

				# Store columns
				if suffix == '.parquet':
					df = dd.read_parquet(file_path, split_row_groups=True)
				else:
					df = pd.read_csv(file_path, nrows=1)
				tables_info_dict['columns_list'][(module, table_name)] = set(df.columns.tolist())

		def _get_info_as_dataframe() -> pd.DataFrame:
			table_info = []
			for module in tables_info_dict['available_tables']:
				for table_name in tables_info_dict['available_tables'][module]:

					file_path = tables_info_dict['file_paths'][(module, table_name)]

					table_info.append({
						'module'      : module,
						'table_name'  : table_name,
						'file_path'   : file_path,
						'file_size'   : tables_info_dict['file_sizes'][(module, table_name)],
						'display_name': tables_info_dict['table_display_names'][(module, table_name)],
						'suffix'      : tables_info_dict['suffix'][(module, table_name)],
						'columns_list': tables_info_dict['columns_list'][(module, table_name)]
					})

			# Convert to DataFrame
			dataset_info_df = pd.DataFrame(table_info)

			# Add mimic path as an attribute
			dataset_info_df.attrs['mimic_path'] = self.mimic_path

			return dataset_info_df

		def _iterate_through_modules():
			modules = ['hosp', 'icu']
			for module in modules:

				# Get module path
				module_path: Path = self.mimic_path / module

				# if the module does not exist, skip it
				if not module_path.exists():
					continue

				# Get available tables:
				available_tables_dict = _get_list_of_available_tables(module_path)

				# If no tables found, skip this module
				if not available_tables_dict:
					continue

				# Get available tables info
				_get_available_tables_info(available_tables_dict, module)
    
		if self.mimic_path is None or not self.mimic_path.exists():
			self.tables_info_dict = None
			self.tables_info_df = None
			return

		# Initialize dataset info
		tables_info_dict = {
			'available_tables'   : {},
			'file_paths'         : {},
			'file_sizes'         : {},
			'table_display_names': {},
			'suffix'             : {},
			'columns_list'       : {},
		}

		_iterate_through_modules()

		# Convert to DataFrame
		self.tables_info_df = _get_info_as_dataframe()
		self.tables_info_dict = tables_info_dict

	@property
	def study_tables_info(self) -> pd.DataFrame:
		"""Returns a DataFrame containing info for tables in the study."""

		if self.tables_info_df is None:
			self.scan_mimic_directory()

		# Get tables in the study
		study_tables = [table.value for table in self.study_table_list]

		return self.tables_info_df[self.tables_info_df.table_name.isin(study_tables)]

	@property
	def _list_of_tables_w_subject_id_column(self) -> List[TableNames]:
		"""Returns a list of tables that have subject_id column."""
		tables_list = self.study_tables_info[
			self.study_tables_info.columns_list.apply(lambda x: 'subject_id' in x)
		].table_name.tolist()

		return [TableNames(table_name) for table_name in tables_list]

	@staticmethod
	def _get_column_dtype(file_path: Optional[Path] = None, columns_list: Optional[List[str]] = None) -> Tuple[Dict[str, str], List[str]]:
		"""Determine the best dtype for a column based on its name and table."""

		if file_path is None and columns_list is None:
			raise ValueError("Either file_path or columns_list must be provided.")


		if file_path is not None:
			columns_list = pd.read_csv(file_path, nrows=1).columns.tolist()

		dtypes      = {col: dtype for col, dtype in COLUMN_TYPES.items() if col in columns_list}
		parse_dates = [col for col in DATETIME_COLUMNS if col in columns_list]

		# Check if the file being loaded is the transfers table
		# if file_path is not None and TableNames.TRANSFERS.value in file_path.name:
		# 	# If so, remove hadm_id from the dtypes to avoid type error on load
		# 	if 'hadm_id' in dtypes:
		# 		del dtypes['hadm_id']

		return dtypes, parse_dates

	def _load_unfiltered_csv_table(self, file_path: Path, use_dask: bool = True) -> pd.DataFrame | dd.DataFrame:

		# Check if file exists
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"CSV file not found: {file_path}")

		if file_path.suffix not in ['.csv', '.gz', '.csv.gz']:
			logger.warning(f"File {file_path} is not a CSV file. Skipping.")
			return pd.DataFrame()

		# First read a small sample to get column names without type conversion
		dtypes, parse_dates = self._get_column_dtype(file_path=file_path)

		# Read with either dask or pandas based on user choice
		if use_dask:
			df = dd.read_csv(
				urlpath        = file_path,
				dtype          = dtypes,
				parse_dates    = parse_dates if parse_dates else None,
				assume_missing = True,
				blocksize      = None if file_path.suffix == '.gz' else '200MB'
			)
		else:
			df = pd.read_csv(
				filepath_or_buffer = file_path,
				dtype       = dtypes,
				parse_dates = parse_dates if parse_dates else None
			)

		return df

	def _get_file_path(self, table_name: TableNames) -> Path:
		"""Get the file path for a table."""

		if self.tables_info_df is None:
			self.scan_mimic_directory()

		return Path(self.tables_info_df[
				(self.tables_info_df.table_name == table_name.value) &
				(self.tables_info_df.module == table_name.module)
			]['file_path'].iloc[0])

	def all_subject_ids(self, table_name: TableNames, df: Optional[pd.DataFrame | dd.DataFrame] = None) -> List[int]:
		"""Returns a list of unique subject_ids found in the admission table."""

		def _load_unique_subject_ids_for_table(df: pd.DataFrame | dd.DataFrame):

			csv_tag = table_name.value
			if table_name == TableNames.MERGED and self.include_transfers:
				csv_tag += '_w_transfers'

			subject_ids_path = self.mimic_path / 'subject_ids' / f'{csv_tag}_subject_ids.csv'
			subject_ids_path.parent.mkdir(parents=True, exist_ok=True)

			if self.tables_info_df is None:
				self.scan_mimic_directory()

			if subject_ids_path.exists():
				subject_ids = pd.read_csv(subject_ids_path)
				self._all_subject_ids = subject_ids['subject_id'].values.tolist()

			else:
				if df is None:
					df = self._load_full_table_single(table_name=table_name, use_dask=True)

				subject_ids = df['subject_id'].unique().compute()
				subject_ids.to_csv(subject_ids_path, index=False)
				self._all_subject_ids = subject_ids.values.tolist()

		if not self._all_subject_ids:
			_load_unique_subject_ids_for_table(df=df)

		return self._all_subject_ids

	def get_merged_table_subject_id_intersection(self, num_subjects: int = DEFAULT_NUM_SUBJECTS) -> List[int]:
		"""Find the intersection of subject_ids across all merged table components and return a subset."""
		
		# Get all tables that have subject_id column and are part of merged table
		tables_with_subject_id = [table for table in self.merged_table_components 
									if table in self.tables_w_subject_id_column]
		
		if not tables_with_subject_id:
			logger.warning("No tables with subject_id found in merged table components")
			return []
		
		logger.info(f"Finding subject_id intersection across {len(tables_with_subject_id)} tables")
		
		# Start with subject_ids from the first table
		intersection_set = None
		
		for table_name in tables_with_subject_id:
			# Load only subject_id column to minimize memory usage
			file_path = self._get_file_path(table_name)
			
			if file_path.suffix == '.parquet':
				# For parquet, we can select specific columns efficiently
				df_subject_ids = dd.read_parquet(file_path, columns=['subject_id'], split_row_groups=True)
			else:
				# For CSV, read only the subject_id column
				if file_path.suffix in ['.csv', '.gz', '.csv.gz']:
					df_subject_ids = dd.read_csv(
						urlpath        = file_path,
						usecols        = ['subject_id'],
						dtype          = {'subject_id': 'int64'},
						assume_missing = True,
						blocksize      = None if str(file_path).endswith('.gz') else '200MB'
					)
				else:
					# Fallback: load via helper then select column
					df             = self._load_unfiltered_csv_table(file_path, use_dask=True)
					df_subject_ids = df[['subject_id']]
			
			# Get unique subject_ids for this table
			unique_subject_ids = set(df_subject_ids['subject_id'].unique().compute())
			
			if intersection_set is None:
				intersection_set = unique_subject_ids
			else:
				intersection_set = intersection_set.intersection(unique_subject_ids)
			
			logger.info(f"After {table_name.value}: {len(intersection_set)} subject_ids in intersection")
		
		if not intersection_set:
			logger.warning("No common subject_ids found across all tables")
			return []
		
		# Convert to sorted list and take the requested number
		intersection_list = sorted(list(intersection_set))
		
		if num_subjects <= 0:
			return []
		elif num_subjects >= len(intersection_list):
			return intersection_list
		else:
			return intersection_list[:num_subjects]


	def load_all_study_tables_full(self, use_dask: bool = True) -> Dict[str, pd.DataFrame | dd.DataFrame]:

		tables_dict = {}
		persisted_tables = {}  # Track persisted DataFrames for cleanup
		
		try:
			for _, row in self.study_tables_info.iterrows():
				table_name = TableNames(row.table_name)

				if table_name is TableNames.MERGED:
					raise ValueError("merged table can not be part of the merged table")

				df = self._load_full_table_single(table_name=table_name, use_dask=use_dask)
				
				# Persist Dask DataFrames for efficient reuse
				if use_dask and isinstance(df, dd.DataFrame):
					df_persisted = df.persist()
					tables_dict[table_name.value] = df_persisted
					persisted_tables[table_name.value] = df_persisted
					logger.info(f"Persisted table: {table_name.value}")
				else:
					tables_dict[table_name.value] = df

			# Store persisted tables for potential cleanup
			self._persisted_resources.update(persisted_tables)
			
		except Exception as e:
			logger.error(f"Error in load_all_study_tables_full: {str(e)}")
			# Cleanup on error
			self._cleanup_persisted_resources(persisted_tables)
			raise

		return tables_dict

	def _cleanup_persisted_resources(self, resources_dict: Dict[str, dd.DataFrame] = None):
		"""Clean up persisted Dask DataFrames to free memory."""
		try:
			if resources_dict is None:
				resources_dict = self._persisted_resources
			
			for name, df in resources_dict.items():
				if isinstance(df, dd.DataFrame):
					try:
						# Clear the persisted data from memory
						df.clear_divisions()
						logger.info(f"Cleaned up persisted table: {name}")
					except Exception as cleanup_error:
						logger.warning(f"Error cleaning up {name}: {str(cleanup_error)}")
			
			# Clear the tracking dictionary
			if resources_dict is self._persisted_resources:
				self._persisted_resources.clear()
				
		except Exception as e:
			logger.error(f"Error in cleanup_persisted_resources: {str(e)}")

	def _load_full_table_single(self, table_name: TableNames, use_dask: bool = True) -> pd.DataFrame | dd.DataFrame:

		if self.tables_info_df is None:
			self.scan_mimic_directory()

		file_path = self._get_file_path(table_name=table_name)

		# For parquet files, respect the use_dask flag
		if file_path.suffix == '.parquet':
			if use_dask:
				return dd.read_parquet(file_path, split_row_groups=True)
			else:
				return pd.read_parquet(file_path)

		df = self._load_unfiltered_csv_table(file_path, use_dask=use_dask)

		if self.apply_filtering:
			df = Filtering(df=df, table_name=table_name).render()

		return df

	def partial_loading(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNames, num_subjects: int = DEFAULT_NUM_SUBJECTS) -> pd.DataFrame | dd.DataFrame:

		def get_partial_subject_id_list(random_selection: bool = False ) -> List[int]:

			subject_ids = self.all_subject_ids(table_name=table_name, df=df)

			# If no subject IDs or num_subjects is non-positive, return an empty list
			if not subject_ids or num_subjects <= 0:
				return []

			# If num_subjects is greater than the number of available subject IDs, return all subject IDs
			if num_subjects > len(subject_ids):
				return subject_ids

			if random_selection:
				self.partial_subject_id_list = np.random.choice(a=subject_ids, size=num_subjects, replace=False).tolist()
			else:
				self.partial_subject_id_list = subject_ids[:num_subjects]

			return self.partial_subject_id_list

		if 'subject_id' not in df.columns:
			logger.info(f"Table {table_name.value} does not have a subject_id column. "
						f"Partial loading is not possible. Skipping partial loading.")
			return df

		subject_ids_set = set( get_partial_subject_id_list(random_selection=False) )

		logger.info(f"Filtering {table_name.value} by subject_id for {num_subjects} subjects.")

		# Use map_partitions for Dask DataFrame or direct isin for pandas
		if isinstance(df, dd.DataFrame):
			return df.map_partitions(lambda part: part[part['subject_id'].isin(subject_ids_set)])

		return df[df['subject_id'].isin(subject_ids_set)]

	def load(self, table_name: TableNames, partial_loading: bool = False, num_subjects: int = DEFAULT_NUM_SUBJECTS, use_dask:bool = True, tables_dict:Optional[Dict[str, pd.DataFrame | dd.DataFrame]] = None) -> pd.DataFrame | dd.DataFrame:

		if table_name is TableNames.MERGED:
			# Use optimized path when partial loading is requested (select subject_ids first, then load only needed rows)
			if partial_loading and tables_dict is None:
				return self._load_full_tables_merged_optimized(num_subjects=num_subjects, use_dask=use_dask)
				
			# Fall back to regular merge (uses provided tables_dict when available)
			df = self._load_full_tables_merged(tables_dict=tables_dict, use_dask=use_dask)
		else:
			df = self._load_full_table_single(table_name=table_name, use_dask=use_dask)

		# Apply legacy row-level filtering only for non-merged tables
		if partial_loading and table_name is not TableNames.MERGED:
			df = self.partial_loading(df=df, table_name=table_name, num_subjects=num_subjects)

		return df

	def _load_full_tables_merged(self, tables_dict: Optional[Dict[str, pd.DataFrame | dd.DataFrame]] = None, use_dask: bool = True) -> pd.DataFrame | dd.DataFrame:
		""" Load and merge tables. """

		if tables_dict is None:
			tables_dict = self.load_all_study_tables_full(use_dask=use_dask)

		# Get tables
		patients_df        = tables_dict[TableNames.PATIENTS.value] # 2.8MB
		admissions_df      = tables_dict[TableNames.ADMISSIONS.value] # 19.9MB
		diagnoses_icd_df   = tables_dict[TableNames.DIAGNOSES_ICD.value] # 33.6MB
		d_icd_diagnoses_df = tables_dict[TableNames.D_ICD_DIAGNOSES.value] # 876KB
		poe_df             = tables_dict[TableNames.POE.value] # 606MB
		poe_detail_df      = tables_dict[TableNames.POE_DETAIL.value] # 55MB

		persisted_intermediates = {}  # Track intermediate results for cleanup
		
		try:
			# Merge tables with persist() for intermediate results
			df12 = patients_df.merge(admissions_df, on='subject_id', how='inner')
			
			# Persist intermediate result if using Dask
			if use_dask and isinstance(df12, dd.DataFrame):
				df12 = df12.persist()
				persisted_intermediates['patients_admissions'] = df12

			if TableNames.TRANSFERS.value in tables_dict:
				transfers_df = tables_dict[TableNames.TRANSFERS.value]
				df123 = df12.merge(transfers_df, on=['subject_id', 'hadm_id'], how='inner')
				
				# Persist intermediate result
				if use_dask and isinstance(df123, dd.DataFrame):
					df123 = df123.persist()
					persisted_intermediates['with_transfers'] = df123
			else:
				df123 = df12

			diagnoses_merged = diagnoses_icd_df.merge(d_icd_diagnoses_df, on=['icd_code', 'icd_version'], how='inner')
			
			# Persist diagnoses merge result
			if use_dask and isinstance(diagnoses_merged, dd.DataFrame):
				diagnoses_merged = diagnoses_merged.persist()
				persisted_intermediates['diagnoses_merged'] = diagnoses_merged
			
			merged_wo_poe = df123.merge(diagnoses_merged, on=['subject_id', 'hadm_id'], how='inner')
			
			# Persist before final merge
			if use_dask and isinstance(merged_wo_poe, dd.DataFrame):
				merged_wo_poe = merged_wo_poe.persist()
				persisted_intermediates['merged_wo_poe'] = merged_wo_poe

			# The reason for 'left' is that we want to keep all the rows from poe table.
			# The poe_detail table for unknown reasons, has fewer rows than poe table.
			poe_and_details = poe_df.merge(poe_detail_df, on=['poe_id', 'poe_seq', 'subject_id'], how='left')
			
			# Persist POE merge result
			if use_dask and isinstance(poe_and_details, dd.DataFrame):
				poe_and_details = poe_and_details.persist()
				persisted_intermediates['poe_and_details'] = poe_and_details
			
			merged_full_study = merged_wo_poe.merge(poe_and_details, on=['subject_id', 'hadm_id'], how='inner')

			# Store intermediate persisted results for potential cleanup
			self._persisted_resources.update(persisted_intermediates)
			
		except Exception as e:
			logger.error(f"Error in _load_full_tables_merged: {str(e)}")
			# Cleanup intermediate results on error
			self._cleanup_persisted_resources(persisted_intermediates)
			raise

		return merged_full_study

	def load_filtered_study_tables_by_subjects(self, subject_ids: List[int], use_dask: bool = True) -> Dict[str, pd.DataFrame | dd.DataFrame]:
		"""Load only rows for the given subject_ids for each study table, keeping descriptor tables unfiltered."""

		if self.tables_info_df is None:
			self.scan_mimic_directory()

		subject_ids_set = set(subject_ids)
		tables_dict: Dict[str, pd.DataFrame | dd.DataFrame] = {}

		for _, row in self.study_tables_info.iterrows():
			table_name = TableNames(row.table_name)

			if table_name is TableNames.MERGED:
				raise ValueError("merged table can not be part of the merged table")

			# Load table
			df = self._load_full_table_single(table_name=table_name, use_dask=use_dask)

			# Apply subject_id filtering when available
			if 'subject_id' in df.columns:
				if isinstance(df, dd.DataFrame):
					df = df.map_partitions(lambda part: part[part['subject_id'].isin(subject_ids_set)])
				else:
					df = df[df['subject_id'].isin(subject_ids_set)]

			tables_dict[table_name.value] = df

		return tables_dict

	def _load_full_tables_merged_optimized(self, num_subjects: int = DEFAULT_NUM_SUBJECTS, use_dask: bool = True) -> pd.DataFrame | dd.DataFrame:
		"""Optimized merged loading: select subject_ids first, load filtered tables, then merge."""

		# 1) Compute intersection and select N subject_ids
		selected_subject_ids = self.get_merged_table_subject_id_intersection(num_subjects=num_subjects)
		if not selected_subject_ids:
			logger.warning("No subject_ids selected for optimized merged loading; falling back to full merged load")
			return self._load_full_tables_merged(use_dask=use_dask)

		# 2) Load only rows for selected subject_ids across component tables
		filtered_tables = self.load_filtered_study_tables_by_subjects(subject_ids=selected_subject_ids, use_dask=use_dask)

		# 3) Merge filtered tables using the same logic as the regular merger
		merged = self._load_full_tables_merged(tables_dict=filtered_tables, use_dask=use_dask)

		# Persist selected ids for potential reuse by other methods
		self.partial_subject_id_list = selected_subject_ids

		return merged


	@property
	def tables_w_subject_id_column(self) -> List[TableNames]:
		"""Tables that have a subject_id column."""
		return  [	TableNames.PATIENTS,
					TableNames.ADMISSIONS,
					TableNames.TRANSFERS,
					TableNames.DIAGNOSES_ICD,
					TableNames.POE,
					TableNames.POE_DETAIL]

	@property
	def merged_table_components(self) -> List[TableNames]:
		"""Tables that are components of the merged table."""
		return [
			TableNames.PATIENTS,
			TableNames.ADMISSIONS,
			TableNames.TRANSFERS,
			TableNames.DIAGNOSES_ICD,
			TableNames.D_ICD_DIAGNOSES,
			TableNames.POE,
			TableNames.POE_DETAIL
		]


class ExampleDataLoader(DataLoader):
	"""ExampleDataLoader class for loading example data."""

	def __init__(self, partial_loading: bool = False, num_subjects: int = 100, random_selection: bool = False, use_dask: bool = True, apply_filtering: bool = True):

		super().__init__(apply_filtering=apply_filtering)

		self.partial_loading = partial_loading
		self.num_subjects    = num_subjects
		self.random_selection = random_selection
		self.use_dask        = use_dask

		self.scan_mimic_directory()
		self.tables_dict = self.load_all_study_tables_full(use_dask=use_dask)

		# with warnings.catch_warnings():
		# 	warnings.simplefilter("ignore")

	def counter(self):
		"""Print row and subject ID counts for each table."""

		def get_nrows(table_name):
			df = self.tables_dict[table_name.value]
			return humanize.intcomma(df.shape[0].compute() if isinstance(df, dd.DataFrame) else df.shape[0])

		def get_nsubject_ids(table_name):
			df = self.tables_dict[table_name.value]
			if 'subject_id' not in df.columns:
				return "N/A"
			# INFO: if returns errors, use df.subject_id.unique().shape[0].compute() instead
			return humanize.intcomma(
				df.subject_id.nunique().compute() if isinstance(df, dd.DataFrame)
				else df.subject_id.nunique()
			)

		# Format the output in a tabular format
		print(f"{'Table':<15} | {'Rows':<10} | {'Subject IDs':<10}")
		print(f"{'-'*15} | {'-'*10} | {'-'*10}")
		print(f"{'patients':<15} | {get_nrows(TableNames.PATIENTS):<10} | {get_nsubject_ids(TableNames.PATIENTS):<10}")
		print(f"{'admissions':<15} | {get_nrows(TableNames.ADMISSIONS):<10} | {get_nsubject_ids(TableNames.ADMISSIONS):<10}")
		print(f"{'diagnoses_icd':<15} | {get_nrows(TableNames.DIAGNOSES_ICD):<10} | {get_nsubject_ids(TableNames.DIAGNOSES_ICD):<10}")
		print(f"{'poe':<15} | {get_nrows(TableNames.POE):<10} | {get_nsubject_ids(TableNames.POE):<10}")
		print(f"{'poe_detail':<15} | {get_nrows(TableNames.POE_DETAIL):<10} | {get_nsubject_ids(TableNames.POE_DETAIL):<10}")

	def study_table_info(self):
		"""Get info about study tables."""
		return self.study_tables_info

	def merge_two_tables(self, table1: TableNames, table2: TableNames, on: Tuple[str], how: Literal['inner', 'left', 'right', 'outer'] = 'inner'):
		"""Merge two tables."""
		df1 = self.tables_dict[table1.value]
		df2 = self.tables_dict[table2.value]

		# Ensure compatible types for merge columns
		for col in on:
			if col in df1.columns and col in df2.columns:

				# Convert to same type in both dataframes
				if col.endswith('_id') and col not in ['poe_id', 'emar_id', 'pharmacy_id']:
					df1[col] = df1[col].astype('int64')
					df2[col] = df2[col].astype('int64')

				elif col in ['icd_code', 'icd_version']:
					df1[col] = df1[col].astype('string')
					df2[col] = df2[col].astype('string')

				elif col in ['poe_id', 'emar_id', 'pharmacy_id'] or col.endswith('provider_id'):
					df1[col] = df1[col].astype('string')
					df2[col] = df2[col].astype('string')

		return df1.merge(df2, on=on, how=how)

	def save_as_parquet(self, table_name: TableNames):
		"""Save a table as Parquet."""
		ParquetConverter(data_loader=self).save_as_parquet(table_name=table_name)

	def n_rows_after_merge(self):
		"""Print row counts after merges."""
		patients_df        = self.tables_dict[TableNames.PATIENTS.value]
		admissions_df      = self.tables_dict[TableNames.ADMISSIONS.value]
		diagnoses_icd_df   = self.tables_dict[TableNames.DIAGNOSES_ICD.value]
		d_icd_diagnoses_df = self.tables_dict[TableNames.D_ICD_DIAGNOSES.value]
		poe_detail_df      = self.tables_dict[TableNames.POE_DETAIL.value]

		# Ensure compatible types
		patients_df        = self.ensure_compatible_types(patients_df, ['subject_id'])
		admissions_df      = self.ensure_compatible_types(admissions_df, ['subject_id', 'hadm_id'])
		diagnoses_icd_df   = self.ensure_compatible_types(diagnoses_icd_df, ['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
		d_icd_diagnoses_df = self.ensure_compatible_types(d_icd_diagnoses_df, ['icd_code', 'icd_version'])
		poe_df             = self.ensure_compatible_types(poe_df, ['subject_id', 'hadm_id', 'poe_id', 'poe_seq'])
		poe_detail_df      = self.ensure_compatible_types(poe_detail_df, ['subject_id', 'poe_id', 'poe_seq'])

		df12              = patients_df.merge(admissions_df, on='subject_id', how='inner')
		df34              = diagnoses_icd_df.merge(d_icd_diagnoses_df, on=('icd_code', 'icd_version'), how='inner')
		poe_and_details   = poe_df.merge(poe_detail_df, on=('poe_id', 'poe_seq', 'subject_id'), how='left')
		merged_wo_poe     = df12.merge(df34, on=('subject_id', 'hadm_id'), how='inner')
		merged_full_study = merged_wo_poe.merge(poe_and_details, on=('subject_id', 'hadm_id'), how='inner')

		def get_count(df):
			return df.shape[0].compute() if isinstance(df, dd.DataFrame) else df.shape[0]

		print(f"{'DataFrame':<15} {'Count':<10} {'DataFrame':<15} {'Count':<10} {'DataFrame':<15} {'Count':<10}")
		print("-" * 70)
		print(f"{'df12':<15} {get_count(df12):<10} {'patients':<15} {get_count(patients_df):<10} {'admissions':<15} {get_count(admissions_df):<10}")
		print(f"{'df34':<15} {get_count(df34):<10} {'diagnoses_icd':<15} {get_count(diagnoses_icd_df):<10} {'d_icd_diagnoses':<15} {get_count(d_icd_diagnoses_df):<10}")
		print(f"{'poe_and_details':<15} {get_count(poe_and_details):<10} {'poe':<15} {get_count(poe_df):<10} {'poe_detail':<15} {get_count(poe_detail_df):<10}")
		print(f"{'merged_wo_poe':<15} {get_count(merged_wo_poe):<10} {'df34':<15} {get_count(df34):<10} {'df12':<15} {get_count(df12):<10}")
		print(f"{'merged_full_study':<15} {get_count(merged_full_study):<10} {'poe_and_details':<15} {get_count(poe_and_details):<10} {'merged_wo_poe':<15} {get_count(merged_wo_poe):<10}")

	def load_table(self, table_name: TableNames):
		"""Load a single table."""
		return self.tables_dict[table_name.value]

	def load_all_study_tables(self):
		"""Load all study tables."""
		return self.tables_dict

	def load_merged_tables(self):
		"""Load merged tables."""
		return self.data_loader.load_merged_tables(
			tables_dict    = self.tables_dict,
			partial_loading = self.partial_loading,
			num_subjects    = self.num_subjects,
			use_dask        = self.use_dask)


class ParquetConverter:
	"""Handles conversion of CSV/CSV.GZ files to Parquet format with appropriate schemas."""

	def __init__(self, data_loader: DataLoader):
		self.data_loader = data_loader

	def _get_csv_file_path(self, table_name: TableNames) -> Tuple[Path, str]:
		"""
		Gets the CSV file path for a table.

		Args:
			table_name: The table to get the file path for

		Returns:
			Tuple of (file path, suffix)
		"""
		def _fix_source_csv_path(source_path: Path) -> Tuple[Path, str]:
			"""Fixes the source csv path if it is a parquet file."""

			if source_path.name.endswith('.parquet'):

				csv_path = source_path.parent / source_path.name.replace('.parquet', '.csv')
				gz_path = source_path.parent / source_path.name.replace('.parquet', '.csv.gz')

				if csv_path.exists():
					return csv_path, '.csv'

				if gz_path.exists():
					return gz_path, '.csv.gz'

				raise ValueError(f"Cannot find csv or csv.gz file for {source_path}")

			suffix = '.csv.gz' if source_path.name.endswith('.gz') else '.csv'

			return source_path, suffix

		if self.data_loader.tables_info_df is None:
			self.data_loader.scan_mimic_directory()


		source_path = Path(self.data_loader.tables_info_df[(self.data_loader.tables_info_df.table_name == table_name.value)]['file_path'].values[0])

		return _fix_source_csv_path(source_path)

	def _create_table_schema(self, df: pd.DataFrame | dd.DataFrame) -> pa.Schema:
		"""
		Create a PyArrow schema for a table, inferring types for unspecified columns.
		It prioritizes manually defined types from COLUMN_TYPES and DATETIME_COLUMNS.
		"""

		# For Dask, use the metadata for schema inference; for pandas, a small sample is enough
		meta_df = df._meta if isinstance(df, dd.DataFrame) else df.head(1)

		# Infer a base schema from the DataFrame's structure to include all columns
		try:
			base_schema = pa.Schema.from_pandas(meta_df, preserve_index=False)
		except Exception:
			# Fallback for complex types that might cause issues with from_pandas
			base_schema = pa.Table.from_pandas(meta_df, preserve_index=False).schema

		# Get custom types from configurations
		custom_dtypes, parse_dates = DataLoader._get_column_dtype(columns_list=df.columns.tolist())

		# Create a dictionary for quick lookup of custom pyarrow types
		custom_pyarrow_types = {col: pyarrow_dtypes_map[dtype] for col, dtype in custom_dtypes.items()}
		custom_pyarrow_types.update({col: pa.timestamp('ns') for col in parse_dates})

		# Rebuild the schema, replacing inferred types with our custom ones where specified
		fields = []
		for field in base_schema:
			if field.name in custom_pyarrow_types:
				# Use the custom type if available
				fields.append(pa.field(field.name, custom_pyarrow_types[field.name]))
			else:
				# Otherwise, use the automatically inferred type
				fields.append(field)

		# # Get all columns from the DataFrame
		# all_columns = df.columns.tolist()

		# # Get custom types from configurations
		# dtypes, parse_dates = DataLoader._get_column_dtype(columns_list=all_columns)

		# # Create a dictionary for quick lookup of custom pyarrow types
		# custom_pyarrow_types = {col: pyarrow_dtypes_map[dtype] for col, dtype in dtypes.items()}
		# custom_pyarrow_types.update({col: pa.timestamp('ns') for col in parse_dates})

		# # Create fields for all columns
		# fields = []
		# for col in all_columns:
		# 	if col in custom_pyarrow_types:
		# 		# Use the custom type if available
		# 		fields.append(pa.field(col, custom_pyarrow_types[col]))
		# 	else:
		# 		# Default to string type for columns not explicitly defined
		# 		fields.append(pa.field(col, pa.string()))

		return pa.schema(fields)

	def save_as_parquet(self, table_name: TableNames, df: Optional[pd.DataFrame | dd.DataFrame] = None, target_parquet_path: Optional[Path] = None, use_dask: bool = True) -> None:
		"""
		Saves a DataFrame as a Parquet file.

		Args:
			table_name: Table name to save as parquet
			df: Optional DataFrame to save (if None, loads from source_path)
			target_parquet_path: Optional target path for the parquet file
			use_dask: Whether to use Dask for loading
		"""
		if df is None or target_parquet_path is None:

			# Get csv file path
			csv_file_path, suffix = self._get_csv_file_path(table_name)

			# Load the CSV file
			if df is None:
				df = self.data_loader._load_unfiltered_csv_table(file_path=csv_file_path, use_dask=use_dask)

			# Get parquet directory
			if target_parquet_path is None:
				target_parquet_path = csv_file_path.parent / csv_file_path.name.replace(suffix, '.parquet')

		# Create schema
		schema = self._create_table_schema(df)

		# if table_name == TableNames.TRANSFERS:
		# 	df = df.dropna(subset=['hadm_id'])
		# 	if 'hadm_id' in df.columns:
		# 		df['hadm_id'] = df['hadm_id'].astype('int64')

		# Save to parquet
		if isinstance(df, dd.DataFrame):
			df.to_parquet(target_parquet_path, schema=schema, engine='pyarrow', compression='snappy')
		else:
			table = pa.Table.from_pandas(df, schema=schema)
			pq.write_table(table, target_parquet_path, compression='snappy')

	def save_all_tables_as_parquet(self, tables_list: Optional[List[TableNames]] = None) -> None:
		"""
		Save all tables as Parquet files.

		Args:
			tables_list: List of table names to convert
		"""
		# If no tables list is provided, use the study table list
		if tables_list is None:
			tables_list = self.data_loader.study_table_list

		# Save tables as parquet
		for table_name in tqdm(tables_list, desc="Saving tables as parquet"):
			self.save_as_parquet(table_name=table_name)



if __name__ == '__main__':

	loader = DataLoader(mimic_path=DEFAULT_MIMIC_PATH, apply_filtering=True, include_transfers=True)
	df_merged = loader.load(table_name=TableNames.MERGED, partial_loading=False)
	subject_ids = loader.all_subject_ids(df=df_merged, table_name=TableNames.MERGED)

	# Convert admissions table to Parquet
	# converter = ParquetConverter(data_loader=loader)
	# converter.save_as_parquet(table_name=TableNames.ADMISSIONS, use_dask=True)
	# converter.save_all_tables_as_parquet()

	# 1. Load a table fully
	# logger.info("Loading 'patients' table fully...")
	# patients_df = loader.load_one_table(TableNames.PATIENTS, partial_loading=False, use_dask=True)
	# merged_tables = data_loader.load_merged_tables(partial_loading=True, num_subjects=10)
	# example = ExampleDataLoader(partial_loading=True, num_subjects=10, apply_filtering=True)
	# example.load_merged_tables()

	print('done')
