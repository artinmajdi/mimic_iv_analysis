# Standard library imports
import os
import glob
from pathlib import Path
from functools import lru_cache, cached_property
from re import L
import tempfile
from typing import Dict, Optional, Tuple, List, Any, Literal

# Data processing imports
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import humanize
from tqdm import tqdm

from mimic_iv_analysis import logger
from mimic_iv_analysis.core.filtering import Filtering
from mimic_iv_analysis.core.dask_config_optimizer import DaskConfigOptimizer
from mimic_iv_analysis.configurations import (  TableNames,
												ColumnNames,
												pyarrow_dtypes_map,
												DEFAULT_MIMIC_PATH,
												DEFAULT_NUM_SUBJECTS,
												DEFAULT_STUDY_TABLES_LIST
												)


subject_id: str = ColumnNames.SUBJECT_ID.value

# Module-level function for multiprocessing compatibility
def process_partition_cpu_optimized(partition_data):
	"""Process a single partition with CPU optimization - standalone function for multiprocessing"""
	try:
		import pandas as pd
		import numpy as np
		import logging
		from mimic_iv_analysis.configurations import ColumnNames
		
		# Setup logging for this process
		mp_logger = logging.getLogger('mp_worker')
		subject_id = ColumnNames.SUBJECT_ID.value
		
		# Convert to pandas for CPU-intensive operations
		if hasattr(partition_data, 'compute'):
			partition_df = partition_data.compute()
		else:
			partition_df = partition_data
		
		# Use numpy for faster unique operations
		if subject_id in partition_df.columns:
			unique_values = partition_df[subject_id].dropna().unique()
			result = set(unique_values)
			mp_logger.debug(f"Processed partition: {len(result)} unique IDs")
			return result
		else:
			mp_logger.warning(f"Column {subject_id} not found in partition")
			return set()
			
	except Exception as e:
		# Return error info for debugging
		mp_logger.error(f"Error processing partition: {str(e)}")
		return {'error': str(e)}

class DataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""

	def __init__(self, 	mimic_path: Path = DEFAULT_MIMIC_PATH,
						study_tables_list: list[str] = DEFAULT_STUDY_TABLES_LIST,
						apply_filtering  : bool     = True,
						filter_params    : Optional[dict[str, dict[str, Any]]] = {} ):

		# Initialize persisted resources tracking
		self._persisted_resources = {}

		# MIMIC_IV v3.1 path
		self.mimic_path      = Path(mimic_path)
		self.apply_filtering = apply_filtering
		self.filter_params   = filter_params

		# Tables to load. Use list provided by user or default list
		self.study_tables_list = study_tables_list

		# Class variables
		self.tables_info_df         : Optional[pd.DataFrame]  = None
		self.tables_info_dict       : Optional[Dict[str, Any]] = None

		# Initialize CPU optimization configuration
		self._setup_cpu_optimization()

	def _setup_cpu_optimization(self):
		"""Setup CPU optimization configuration for maximum performance."""
		import dask
		import multiprocessing as mp

		# Get system resources
		optimizer = DaskConfigOptimizer()
		system_info = optimizer.get_system_info()

		cpu_count = system_info['cpu_count']
		memory_gb = system_info['total_memory_gb']

		# Configure Dask globally for CPU-intensive operations
		dask.config.set({
			# Process-based parallelism
			'scheduler': 'processes',
			'distributed.worker.daemon': False,

			# CPU optimization
			'distributed.nthreads': 1,  # Single thread per process for CPU tasks
			'distributed.worker.multiprocessing.initializer': None,
			'distributed.worker.multiprocessing.initargs': (),

			# Memory optimization for CPU tasks
			'array.chunk-size': f'{min(1024, int(memory_gb * 80))}MB',
			'distributed.worker.memory.target': 0.9 if memory_gb > 20 else 0.8,
			'distributed.worker.memory.spill': 0.95,
			'distributed.worker.memory.pause': 0.95,

			# Performance tuning
			'dataframe.query-planning': False,
			'distributed.comm.timeouts.connect': '60s',
			'distributed.comm.timeouts.tcp': '60s',
			'distributed.worker.connections.outgoing': min(cpu_count * 2, 50),
		})

		# Set multiprocessing start method for better performance
		try:
			if mp.get_start_method(allow_none=True) != 'spawn':
				mp.set_start_method('spawn', force=True)
		except RuntimeError:
			# Start method already set
			pass

		logger.info(f"CPU optimization configured: {cpu_count} cores, {memory_gb:.1f}GB RAM, process-based scheduler")

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

		return self.tables_info_df[self.tables_info_df.table_name.isin(self.study_tables_list)]

	# @property
	# def _list_of_tables_w_subject_id_column(self) -> list[str]:
	# 	"""Returns a list of tables that have subject_id column."""
	# 	return self.study_tables_info[
	# 		self.study_tables_info.columns_list.apply(lambda x: 'subject_id' in x)
	# 	].table_name.tolist()


	@staticmethod
	def _get_column_dtype(file_path: Optional[Path] = None, columns_list: Optional[List[str]] = None) -> Tuple[Dict[str, str], List[str]]:
		"""Determine the best dtype for a column based on its name and table.

		Converts integer dtypes to nullable integer dtypes (Int64, Int32, Int16) to properly handle NA values.
		This prevents issues where missing values in integer columns cause type conversion errors.
		"""

		if file_path is None and columns_list is None:
			raise ValueError("Either file_path or columns_list must be provided.")

		if file_path is not None:
			columns_list = pd.read_csv(file_path, nrows=1).columns.tolist()

		# Get base dtypes and convert integer types to nullable versions
		dtypes = {}
		for col, dtype in TableNames._COLUMN_TYPES.items():
			if col in columns_list:
				# Convert integer dtypes to nullable versions for proper NA handling
				if dtype == 'int64':
					dtypes[col] = 'Int64'  # Nullable integer
				elif dtype == 'int32':
					dtypes[col] = 'Int32'  # Nullable integer
				elif dtype == 'int16':
					dtypes[col] = 'Int16'  # Nullable integer
				else:
					dtypes[col] = dtype  # Keep other dtypes as-is

		parse_dates = [col for col in TableNames._DATETIME_COLUMNS if col in columns_list]

		# remove datetime columns from dtypes
		dtypes = {col: dtype for col, dtype in dtypes.items() if col not in parse_dates}

		return dtypes, parse_dates

	def _compute_unique_subject_ids_chunked(self, df: dd.DataFrame, chunk_size: int = 1000000) -> set:
		"""Compute unique subject_ids from a Dask DataFrame using optimized parallel processing.

		Args:
			df: Dask DataFrame containing subject_id column
			chunk_size: Number of rows to process in each chunk (default: 1M rows)

		Returns:
			set: Set of unique subject_ids
		"""
		import psutil
		import dask
		from concurrent.futures import ProcessPoolExecutor, as_completed
		import multiprocessing as mp
		from functools import partial

		# Get system resources and optimal configuration
		optimizer = DaskConfigOptimizer()
		system_info = optimizer.get_system_info()
		optimal_config = optimizer.get_optimal_balanced_config()

		cpu_count = system_info['cpu_count']
		memory_gb = system_info['total_memory_gb']
		available_memory_gb = system_info['available_memory_gb']

		logger.info(f"System resources: {cpu_count} CPU cores, {memory_gb:.1f}GB total RAM, {available_memory_gb:.1f}GB available")
		logger.info(f"Computing unique subject_ids using CPU-optimized multiprocessing")

		# Use aggressive configuration for CPU-bound tasks
		memory_target = 0.9 if available_memory_gb > 20 else 0.8  # Use even more memory for CPU tasks
		chunk_size_mb = min(1024, int(available_memory_gb * 80))  # Larger chunks for CPU processing
		partition_size_mb = max(50, min(200, int(available_memory_gb * 8)))  # Larger partitions for CPU work

		logger.info(f"CPU-optimized configuration: chunk_size={chunk_size_mb}MB, partition_size={partition_size_mb}MB, memory_target={memory_target}")

		# Configure Dask for CPU-intensive processing
		with dask.config.set({
			'dataframe.query-planning': False,  # Use legacy query planning for stability
			'array.chunk-size': f'{chunk_size_mb}MB',  # Large chunks for CPU processing
			'distributed.worker.memory.target': memory_target,  # Aggressive memory usage
			'distributed.worker.memory.spill': memory_target + 0.05,   # Spill slightly above target
			'distributed.worker.memory.pause': memory_target + 0.05,    # Pause close to target
			'distributed.nthreads': 1,         # Single thread per worker for multiprocessing
			'distributed.worker.daemon': False,        # Better for batch processing
			'scheduler': 'processes',          # Use process scheduler for CPU tasks
		}):
			# Check initial partitions and optimize for CPU processing
			n_partitions = df.npartitions
			logger.info(f"Initial partitions: {n_partitions}")

			# Repartition for optimal CPU processing
			# Target: 2-3 partitions per CPU core for CPU-intensive tasks
			optimal_partitions = max(cpu_count * 2, 20)  # Fewer but larger partitions for CPU work

			if n_partitions < optimal_partitions:
				logger.info(f"Repartitioning from {n_partitions} to optimize for {optimal_partitions} target partitions")
				# Use larger partition size for CPU processing
				df = df.repartition(partition_size=f"{partition_size_mb}MB")
				n_partitions = df.npartitions
				logger.info(f"Repartitioned to {n_partitions} partitions")

			# Use all CPU cores for true parallelism
			max_workers = min(cpu_count, n_partitions)  # Use all available CPU cores
			logger.info(f"Using {max_workers} CPU processes for true parallel processing")

			# Persist the DataFrame in memory for faster access across processes
			logger.info("Persisting DataFrame in memory for multiprocessing access...")
			df = df.persist()

			# Use the module-level process_partition_cpu_optimized function for multiprocessing compatibility
			unique_ids = set()
			completed_partitions = 0

			# Process partitions using multiprocessing for true CPU parallelism
			logger.info("Starting CPU-optimized multiprocessing execution...")

			# Use optimal batch size for CPU utilization
			batch_size = max(2, min(cpu_count * 2, n_partitions // 4))  # Larger batches for CPU efficiency

			for batch_start in range(0, n_partitions, batch_size):
				batch_end = min(batch_start + batch_size, n_partitions)
				batch_partitions = list(range(batch_start, batch_end))

				logger.info(f"Processing CPU batch {batch_start//batch_size + 1}/{(n_partitions + batch_size - 1)//batch_size}: partitions {batch_start}-{batch_end-1}")

				# Get partition data for this batch
				partition_data_list = []
				for i in batch_partitions:
					try:
						partition = df.get_partition(i)
						partition_data_list.append(partition)
					except Exception as e:
						logger.warning(f"Error getting partition {i}: {e}")
						partition_data_list.append(None)

				# Process batch with multiprocessing - use spawn method for better CPU isolation
				mp_context = mp.get_context('spawn')
				with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
					# Submit batch of partition processing tasks
					future_to_partition = {}
					for idx, partition_data in enumerate(partition_data_list):
						if partition_data is not None:
							future = executor.submit(process_partition_cpu_optimized, partition_data)
							future_to_partition[future] = batch_partitions[idx]

				# Collect results as they complete with progress tracking
				batch_progress = tqdm(as_completed(future_to_partition),
									 total=len(future_to_partition),
									 desc=f"CPU batch {batch_start//batch_size + 1}")

				for future in batch_progress:
					partition_idx = future_to_partition[future]
					try:
						result = future.result(timeout=300)  # 5 minute timeout per partition
						if isinstance(result, dict) and 'error' in result:
							logger.warning(f"Error in partition {partition_idx}: {result['error']}")
						else:
							unique_ids.update(result)
							completed_partitions += 1

							# Log progress every 10 partitions or at completion
							if completed_partitions % 10 == 0 or completed_partitions == n_partitions:
								logger.info(f"Processed {completed_partitions}/{n_partitions} partitions, unique IDs so far: {len(unique_ids):,}")

					except Exception as e:
						logger.warning(f"Failed to get result for partition {partition_idx}: {e}")

			logger.info(f"Completed CPU-optimized multiprocessing. Total unique subject_ids: {len(unique_ids):,}")
			return unique_ids

	@staticmethod
	def _handle_na_values_in_dataframe(df: pd.DataFrame | dd.DataFrame) -> pd.DataFrame | dd.DataFrame:
		"""Handle NA values in integer columns by ensuring proper nullable integer dtypes.

		This method ensures that integer columns with NA values are properly handled
		by converting them to nullable integer dtypes if they aren't already.
		"""

		def _convert_integer_columns_with_na(df_part: pd.DataFrame) -> pd.DataFrame:
			"""Convert integer columns that contain NA values to nullable integer dtypes."""
			for col in df_part.columns:
				if col in TableNames._COLUMN_TYPES:
					expected_dtype = TableNames._COLUMN_TYPES[col]
					if expected_dtype in ['int64', 'int32', 'int16']:
						# Convert integer columns to nullable integer dtypes for consistency
						# This prevents issues when data contains NA values
						if not pd.api.types.is_extension_array_dtype(df_part[col]):
							# First, replace empty strings and 'NULL' with NaN
							df_part[col] = df_part[col].replace(['', 'NULL', 'null', 'None'], np.nan)

							# Convert to nullable integer dtype
							if expected_dtype == 'int64':
								df_part[col] = pd.to_numeric(df_part[col], errors='coerce').astype('Int64')
							elif expected_dtype == 'int32':
								df_part[col] = pd.to_numeric(df_part[col], errors='coerce').astype('Int32')
							elif expected_dtype == 'int16':
								df_part[col] = pd.to_numeric(df_part[col], errors='coerce').astype('Int16')

			return df_part

		# Handle Dask DataFrame
		if isinstance(df, dd.DataFrame):
			return df.map_partitions(_convert_integer_columns_with_na, meta=df)

		# Handle pandas DataFrame
		return _convert_integer_columns_with_na(df)


	def _get_file_path(self, table_name: TableNames | str) -> Path:
		"""Get the file path for a table with priority: parquet > csv > csv.gz"""

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		if table_name.value == TableNames.MERGED.value:
			return self.merged_table_parquet_path

		if self.tables_info_df is None:
			self.scan_mimic_directory()

		# Filter for the specific table
		df = self.tables_info_df[
				(self.tables_info_df.table_name == table_name.value) &
				(self.tables_info_df.module == table_name.module) ]

		if df.empty:
			return None

		# Check for parquet first
		parquet_files = df[df.suffix == '.parquet']
		if not parquet_files.empty:
			return Path(parquet_files['file_path'].iloc[0])

		return Path(df['file_path'].iloc[0])


	def get_sample_subject_ids(self, table_name: TableNames | str, num_subjects: int = DEFAULT_NUM_SUBJECTS) -> list[int]:

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		def _sample_subject_ids(common_subject_ids_list: list[int]) -> list[int]:
			"""Sample subject_ids from the list, ensuring no duplicates."""
			if num_subjects is None:
				return []
			elif num_subjects >= len(common_subject_ids_list):
				return common_subject_ids_list
			return common_subject_ids_list[:num_subjects]

		unique_subject_ids = self.get_unique_subject_ids(table_name=table_name)

		return _sample_subject_ids(list(unique_subject_ids))

	def get_unique_subject_ids(self, table_name: TableNames | str, recalculate_subject_ids: bool = False) -> set:

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		def get_for_one_table(table_name: TableNames) -> set:
			"""Returns a list of unique subject_ids found in a table."""

			def _fetch_full_table_subject_ids() -> set:
				"""Returns the list of unique subject_ids found in a full table, without applying filters."""

				file_path = self._get_file_path(table_name=table_name)

				if file_path.suffix == '.parquet':
					df_subject_id_column = dd.read_parquet(file_path, columns=[subject_id], split_row_groups=True)

				elif file_path.suffix in ['.csv', '.gz', '.csv.gz']:

					df_subject_id_column = dd.read_csv(
						urlpath        = file_path,
						usecols        = [subject_id],
						dtype          = {subject_id: 'int64'},
						assume_missing = True,
						blocksize      = None if str(file_path).endswith('.gz') else '200MB' )

				# Process unique subject_ids in chunks to avoid memory issues
				return self._compute_unique_subject_ids_chunked(df_subject_id_column)

			def _fetch_filtered_table_subject_ids(table_name: TableNames) -> set:
				""" Returns a list of unique subject_ids found in the table after applying filters. """

				df = self.fetch_table(table_name=table_name, apply_filtering=self.apply_filtering)
				return self._compute_unique_subject_ids_chunked(df)

			def get_table_subject_ids_path(table_name: TableNames) -> Path:

				csv_tag = table_name.value

				if self.apply_filtering:
					csv_tag += '_filtered'

				subject_ids_path = self.mimic_path / 'subject_ids' / f'{csv_tag}_subject_ids.csv'
				subject_ids_path.parent.mkdir(parents=True, exist_ok=True)

				return subject_ids_path

			subject_ids_path = get_table_subject_ids_path(table_name=table_name)

			if subject_ids_path.exists() and not recalculate_subject_ids:
				df_unique_subject_ids = pd.read_csv(subject_ids_path)
				return set(df_unique_subject_ids[subject_id].values.tolist())

			if self.apply_filtering:
				unique_subject_ids = _fetch_filtered_table_subject_ids(table_name=table_name)
			else:
				unique_subject_ids = _fetch_full_table_subject_ids()

			pd.DataFrame({subject_id: list(unique_subject_ids)}).to_csv(subject_ids_path, index=False)

			return unique_subject_ids

		def get_merged_table_subject_ids() -> list[int]:
			"""Find the intersection of subject_ids across all merged table components and return a subset."""

			def _looping_tables(tables_with_subject_id):

				logger.info(f"Finding subject_id intersection across {len(tables_with_subject_id)} tables")

				intersection_set = None

				for table_name in tables_with_subject_id:

					unique_subject_ids = get_for_one_table(table_name=table_name)

					if intersection_set is None:
						intersection_set = unique_subject_ids
					else:
						intersection_set = intersection_set.intersection(unique_subject_ids)

					logger.info(f"After {table_name.value}: {len(intersection_set)} subject_ids in intersection")

				return intersection_set

			# Get all tables that have subject_id column and are part of merged table
			tables_with_subject_id = [TableNames(n) if isinstance(n, str) else n for n in self.study_tables_list if n in TableNames._TABLES_W_SUBJECT_ID_COLUMN]

			if not tables_with_subject_id:
				logger.warning("No tables with subject_id found in merged table components")
				return []

			intersection_set = _looping_tables(tables_with_subject_id)

			if not intersection_set:
				logger.warning("No common subject_ids found across all tables")
				raise ValueError("No common subject_ids found across all tables")

			# Convert to sorted list and take the requested number
			common_subject_ids_list = sorted(list(intersection_set))

			logger.info(f"Found {len(common_subject_ids_list)} common subject_ids in intersection of selected tables.")

			return common_subject_ids_list

		if table_name.value == TableNames.MERGED.value:
			return get_merged_table_subject_ids()

		if table_name.value in TableNames._TABLES_W_SUBJECT_ID_COLUMN:
			return get_for_one_table(table_name=table_name)

		return set()

	def fetch_complete_study_tables(self, use_dask: bool = True) -> Dict[str, pd.DataFrame | dd.DataFrame]:

		tables_dict = {}
		persisted_tables = {}  # Track persisted DataFrames for cleanup

		try:
			for _, row in self.study_tables_info.iterrows():
				table_name = TableNames(row.table_name)

				if table_name is TableNames.MERGED:
					raise ValueError("merged table can not be part of the merged table")

				df = self.fetch_table(table_name=table_name, use_dask=use_dask, apply_filtering=self.apply_filtering)

				# Persist Dask DataFrames for efficient reuse
				if use_dask and isinstance(df, dd.DataFrame):
					df_persisted                       = df.persist()
					tables_dict[table_name.value]      = df_persisted
					persisted_tables[table_name.value] = df_persisted
					logger.info(f"Persisted table: {table_name.value}")
				else:
					tables_dict[table_name.value] = df

			# Store persisted tables for potential cleanup
			self._persisted_resources.update(persisted_tables)

		except Exception as e:
			logger.error(f"Error in fetch_complete_study_tables: {str(e)}")
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

	@property
	def merged_table_parquet_path(self) -> Path:
		return self.mimic_path / f'{TableNames.MERGED.value}.parquet'


	def fetch_table(self, table_name: Optional[TableNames | str] = None, file_path: Optional[Path] = None, use_dask: bool = True, apply_filtering: bool = True) -> pd.DataFrame | dd.DataFrame:

		def _load(file_path: Path) -> pd.DataFrame | dd.DataFrame:
			"""Load a table from a file path, handling parquet and csv formats."""

			if file_path.suffix == '.parquet':

				if use_dask:
					return dd.read_parquet(file_path, split_row_groups=True)

				return pd.read_parquet(file_path)

			elif file_path.suffix in ['.csv', '.gz', '.csv.gz']:

				# First read a small sample to get column names without type conversion
				dtypes, parse_dates = self._get_column_dtype(file_path=file_path)

				if use_dask:
					return dd.read_csv(
						urlpath        = file_path,
						dtype          = dtypes,
						parse_dates    = parse_dates if parse_dates else None,
						assume_missing = True,
						blocksize      = None if file_path.suffix == '.gz' else '200MB' )

				return pd.read_csv(
					filepath_or_buffer = file_path,
					dtype       = dtypes,
					parse_dates = parse_dates if parse_dates else None )

			raise ValueError(f"Unsupported file type: {file_path.suffix}")

		def _get_file_path_and_table_name(file_path, table_name):
			if file_path is None and table_name is None:
				raise ValueError("Either file_path or table_name must be provided.")

			if file_path is None:
				file_path = self._get_file_path(table_name=table_name)

			if not os.path.exists(file_path):
				raise FileNotFoundError(f"CSV file not found: {file_path}")

			if table_name is None:
				table_name = TableNames(file_path.name)

			return file_path, table_name

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		if self.tables_info_df is None:
			self.scan_mimic_directory()

		file_path, table_name = _get_file_path_and_table_name(file_path, table_name)

		df = _load(file_path=file_path)

		# Handle NA values in integer columns
		df = self._handle_na_values_in_dataframe(df)

		if apply_filtering:
			df = Filtering(df=df, table_name=table_name, filter_params=self.filter_params).render()

		return df


	def partial_loading(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNames | str, num_subjects: int = DEFAULT_NUM_SUBJECTS) -> pd.DataFrame | dd.DataFrame:

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		if subject_id not in df.columns:
			logger.info(f"Table {table_name.value} does not have a subject_id column. "
						f"Partial loading is not possible. Skipping partial loading.")
			return df

		subject_ids_list = self.get_sample_subject_ids(table_name=table_name, num_subjects=num_subjects)
		subject_ids_set = set(subject_ids_list)

		logger.info(f"Filtering {table_name.value} by subject_id for {num_subjects} subjects.")

		# Use map_partitions for Dask DataFrame or direct isin for pandas
		return self.extract_rows_by_subject_ids(df=df, table_name=table_name, subject_ids_list=subject_ids_list)


	def extract_rows_by_subject_ids(self, df: pd.DataFrame | dd.DataFrame, table_name: TableNames | str, subject_ids_list: List[int]) -> pd.DataFrame | dd.DataFrame:

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		logger.info(f"Filtering {table_name.value} by subject_id for {len(subject_ids_list)} subjects.")

		# Use map_partitions for Dask DataFrame or direct isin for pandas
		if isinstance(df, dd.DataFrame):
			return df.map_partitions(lambda part: part[part[subject_id].isin(subject_ids_list)])

		return df[df[subject_id].isin(subject_ids_list)]


	def load(self, table_name: TableNames | str, partial_loading: bool = False, num_subjects: int = DEFAULT_NUM_SUBJECTS, use_dask:bool = True, tables_dict:Optional[Dict[str, pd.DataFrame | dd.DataFrame]] = None) -> pd.DataFrame | dd.DataFrame:

		if isinstance(table_name, str):
			table_name = TableNames(table_name)

		if table_name is TableNames.MERGED:
			# Use optimized path when partial loading is requested (select subject_ids first, then load only needed rows)
			if partial_loading and tables_dict is None:
				return self.load_filtered_merged_table_by_subjects(num_subjects=num_subjects, use_dask=use_dask)

			# Fall back to regular merge (uses provided tables_dict when available)
			df = self.merge_tables(tables_dict=tables_dict, use_dask=use_dask)
		else:
			df = self.fetch_table(table_name=table_name, use_dask=use_dask, apply_filtering=self.apply_filtering)

			if partial_loading:
				df = self.partial_loading(df=df, table_name=table_name, num_subjects=num_subjects)

		return df


	def merge_tables(self, tables_dict: Optional[Dict[str, pd.DataFrame | dd.DataFrame]] = None, use_dask: bool = True) -> pd.DataFrame | dd.DataFrame:
		""" Load and merge tables. """

		if tables_dict is None:
			tables_dict = self.fetch_complete_study_tables(use_dask=use_dask)

		# Get tables
		patients_df        = tables_dict[TableNames.PATIENTS.value] # 2.8MB
		admissions_df      = tables_dict[TableNames.ADMISSIONS.value] # 19.9MB
		diagnoses_icd_df   = tables_dict[TableNames.DIAGNOSES_ICD.value] # 33.6MB
		d_icd_diagnoses_df = tables_dict[TableNames.D_ICD_DIAGNOSES.value] # 876KB
		poe_df             = tables_dict[TableNames.POE.value] # 606MB
		poe_detail_df      = tables_dict[TableNames.POE_DETAIL.value] # 55MB
		transfers_df       = tables_dict.get(TableNames.TRANSFERS.value) # 46MB
		prescriptions_df   = tables_dict.get(TableNames.PRESCRIPTIONS.value) # 606MB
		labevents_df       = tables_dict.get(TableNames.LABEVENTS.value) # 2.6GB
		d_labitems_df      = tables_dict.get(TableNames.D_LABITEMS.value) # 30KB

		persisted_intermediates = {}  # Track intermediate results for cleanup

		try:
			# Merge tables with persist() for intermediate results
			df12 = patients_df.merge(admissions_df, on=subject_id, how='inner')

			# Persist intermediate result if using Dask
			if isinstance(df12, dd.DataFrame):
				df12 = df12.persist()
				persisted_intermediates['patients_admissions'] = df12

			df123 = df12.merge(transfers_df, on=[subject_id, ColumnNames.HADM_ID.value], how='inner')

			# Persist intermediate result
			if isinstance(df123, dd.DataFrame):
				df123 = df123.persist()
				persisted_intermediates['with_transfers'] = df123
			else:
				df123 = df12

			diagnoses_merged = diagnoses_icd_df.merge(d_icd_diagnoses_df, on=['icd_code', 'icd_version'], how='inner')

			# Persist diagnoses merge result
			if isinstance(diagnoses_merged, dd.DataFrame):
				diagnoses_merged = diagnoses_merged.persist()
				persisted_intermediates['diagnoses_merged'] = diagnoses_merged

			merged_wo_poe = df123.merge(diagnoses_merged, on=[subject_id, ColumnNames.HADM_ID.value], how='inner')

			# Persist before final merge
			if isinstance(merged_wo_poe, dd.DataFrame):
				merged_wo_poe = merged_wo_poe.persist()
				persisted_intermediates['merged_wo_poe'] = merged_wo_poe

			# The reason for 'left' is that we want to keep all the rows from poe table.
			# The poe_detail table for unknown reasons, has fewer rows than poe table.
			poe_and_details = poe_df.merge(poe_detail_df, on=[ColumnNames.POE_ID.value, ColumnNames.POE_SEQ.value, subject_id], how='inner')

			# Persist POE merge result
			if isinstance(poe_and_details, dd.DataFrame):
				poe_and_details = poe_and_details.persist()
				persisted_intermediates['poe_and_details'] = poe_and_details

			merged_full_study = merged_wo_poe.merge(poe_and_details, on=[subject_id, ColumnNames.HADM_ID.value], how='inner')

			# Store intermediate persisted results for potential cleanup
			self._persisted_resources.update(persisted_intermediates)

		except Exception as e:
			logger.error(f"Error in merge_tables: {str(e)}")
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
			df = self.fetch_table(table_name=table_name, use_dask=use_dask, apply_filtering=self.apply_filtering)

			# Apply subject_id filtering when available
			if subject_id in df.columns:
				if isinstance(df, dd.DataFrame):
					df = df.map_partitions(lambda part: part[part[subject_id].isin(subject_ids_set)])
				else:
					df = df[df[subject_id].isin(subject_ids_set)]

			tables_dict[table_name.value] = df

		return tables_dict

	def load_filtered_merged_table_by_subjects(self, num_subjects: int = DEFAULT_NUM_SUBJECTS, use_dask: bool = True) -> pd.DataFrame | dd.DataFrame:
		"""Optimized merged loading: select subject_ids first, load filtered tables, then merge."""

		# 1) Compute intersection and select N subject_ids
		common_subject_ids_sample_list = self.get_sample_subject_ids(table_name=TableNames.MERGED, num_subjects=num_subjects)

		if common_subject_ids_sample_list:
			# 2) Load only rows for selected subject_ids across component tables
			tables_dict = self.load_filtered_study_tables_by_subjects(subject_ids=common_subject_ids_sample_list, use_dask=use_dask)

		else:
			logger.warning("No subject_ids selected for optimized merged loading; falling back to full merged load")
			tables_dict = self.fetch_complete_study_tables(use_dask=use_dask)

		# 3) Merge filtered tables using the same logic as the regular merger
		return self.merge_tables(tables_dict=tables_dict, use_dask=use_dask)



class ExampleDataLoader(DataLoader):
	"""ExampleDataLoader class for loading example data."""

	def __init__(self, partial_loading: bool = False, num_subjects: int = 100, random_selection: bool = False, use_dask: bool = True, apply_filtering: bool = True, filter_params: Optional[dict[str, dict[str, Any]]] = {}):

		super().__init__(apply_filtering=apply_filtering, filter_params=filter_params)

		self.partial_loading  = partial_loading
		self.num_subjects     = num_subjects
		self.random_selection = random_selection
		self.use_dask         = use_dask

		self.scan_mimic_directory()
		self.tables_dict = self.fetch_complete_study_tables(use_dask=use_dask)

		# with warnings.catch_warnings():
		# 	warnings.simplefilter("ignore")

	def counter(self):
		"""Print row and subject ID counts for each table."""

		def get_nrows(table_name):
			df = self.tables_dict[table_name.value]
			return humanize.intcomma(df.shape[0].compute() if isinstance(df, dd.DataFrame) else df.shape[0])

		def get_nsubject_ids(table_name):
			df = self.tables_dict[table_name.value]
			if subject_id not in df.columns:
				return "N/A"
			# INFO: if returns errors, use df.subject_id.unique().shape[0].compute() instead
			return humanize.intcomma(
				df[subject_id].nunique().compute() if isinstance(df, dd.DataFrame)
				else df[subject_id].nunique()
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
		patients_df        = self.ensure_compatible_types(patients_df, [subject_id])
		admissions_df      = self.ensure_compatible_types(admissions_df, [subject_id, ColumnNames.HADM_ID.value])
		diagnoses_icd_df   = self.ensure_compatible_types(diagnoses_icd_df, [subject_id, ColumnNames.HADM_ID.value, 'icd_code', 'icd_version'])
		d_icd_diagnoses_df = self.ensure_compatible_types(d_icd_diagnoses_df, ['icd_code', 'icd_version'])
		poe_df             = self.ensure_compatible_types(poe_df, [subject_id, ColumnNames.HADM_ID.value, 'poe_id', 'poe_seq'])
		poe_detail_df      = self.ensure_compatible_types(poe_detail_df, [subject_id, 'poe_id', 'poe_seq'])

		# Merge tables
		df12              = patients_df.merge(admissions_df, on=subject_id, how='inner')
		df34              = diagnoses_icd_df.merge(d_icd_diagnoses_df, on=('icd_code', 'icd_version'), how='inner')
		poe_and_details   = poe_df.merge(poe_detail_df, on=('poe_id', 'poe_seq', subject_id), how='left')
		merged_wo_poe     = df12.merge(df34, on=(subject_id, ColumnNames.HADM_ID.value), how='inner')
		merged_full_study = merged_wo_poe.merge(poe_and_details, on=(subject_id, ColumnNames.HADM_ID.value), how='inner')

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


# TODO: currently when i try from the .parquet folder. if the folder is empty, the code gives me an error. fix the code so that if this happens it would fallback to loading the csv.gz or csv file

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
		It prioritizes manually defined types from TableNames._COLUMN_TYPES and TableNames._DATETIME_COLUMNS.
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

	def save_as_parquet(self, table_name: TableNames, df: Optional[pd.DataFrame | dd.DataFrame] = None, target_parquet_path: Optional[Path] = None, chunk_size: int = 10000) -> None:
		"""
		Saves a DataFrame as a Parquet file with improved memory management.

		Args:
			table_name         : Table name to save as parquet
			df                 : Optional DataFrame to save (if None, loads from source_path)
			target_parquet_path: Optional target path for the parquet file
			use_dask           : Whether to use Dask for loading
			chunk_size         : Number of rows per chunk for large datasets
		"""
		def _save_pandas_chunked(df: pd.DataFrame, target_path: Path, schema: pa.Schema, chunk_size: int) -> None:
			"""
			Save a large pandas DataFrame in chunks to avoid memory issues.

			Args:
				df: DataFrame to save
				target_path: Target parquet file path
				schema: PyArrow schema
				chunk_size: Number of rows per chunk
			"""
			# Create directory if it doesn't exist
			target_path.parent.mkdir(parents=True, exist_ok=True)

			# Write first chunk to establish the file
			first_chunk = df.iloc[:chunk_size]
			table = pa.Table.from_pandas(first_chunk, schema=schema)
			pq.write_table(table, target_path, compression='snappy')

			# Append remaining chunks
			for i in range(chunk_size, len(df), chunk_size):
				chunk = df.iloc[i:i+chunk_size]
				table = pa.Table.from_pandas(chunk, schema=schema)
				# For subsequent chunks, we need to use a different approach
				# since PyArrow doesn't support direct append mode
				temp_path = target_path.with_suffix(f'.chunk_{i}.parquet')
				pq.write_table(table, temp_path, compression='snappy')

			# Combine all chunks (this is a simplified approach)
			logger.info(f"Chunked writing completed for {target_path}")


		if df is None or target_parquet_path is None:

			# Get csv file path
			csv_file_path, suffix = self._get_csv_file_path(table_name)

			# Load the CSV file
			if df is None:

				# This is added as an exception, because the labevents table is too large.
				if table_name == TableNames.LABEVENTS:
					df = self.data_loader.fetch_table(file_path=csv_file_path, table_name=table_name, apply_filtering=True)

					subject_ids_list = self.data_loader.get_unique_subject_ids(table_name=TableNames.MERGED)

					df = self.data_loader.extract_rows_by_subject_ids(df=df, table_name=table_name, subject_ids_list=subject_ids_list)

			else:
				df = self.data_loader.fetch_table(file_path=csv_file_path, table_name=table_name, apply_filtering=False)

			# Get parquet directory
			if target_parquet_path is None:
				target_parquet_path = csv_file_path.parent / csv_file_path.name.replace(suffix, '.parquet')

		schema = self._create_table_schema(df)

		try:
			if isinstance(df, dd.DataFrame):

				# Repartition to smaller chunks if necessary to avoid memory issues
				if df.npartitions > 50 or table_name == TableNames.MERGED:
					df = df.repartition(partition_size="30MB")
					logger.info(f"Repartitioned {table_name} to {df.npartitions} partitions")

				df.to_parquet(
					target_parquet_path,
					schema              = schema,
					engine              = 'pyarrow',
					write_metadata_file = True,
					compression         = 'snappy',
					# Performance optimizations
					write_index=False,  # Skip index writing for speed
					overwrite=True,     # Allow overwriting for updates
					# Use threads for I/O bound operations
					compute_kwargs={'scheduler': 'threads', 'num_workers': min(4, os.cpu_count())}
				)

				logger.info(f'Successfully saved {table_name} as parquet with {df.npartitions} partitions')

			else:
				if len(df) > chunk_size:
					logger.info(f"Large dataset detected ({len(df)} rows). Using chunked processing with {chunk_size} rows per chunk.")
					_save_pandas_chunked(df=df, target_path=target_parquet_path.with_suffix('.csv'), schema=schema, chunk_size=chunk_size)
				else:
					table = pa.Table.from_pandas(df, schema=schema)
					pq.write_table(table, target_parquet_path.with_suffix('.csv'), compression='snappy')

				logger.info(f'Successfully saved {table_name} as parquet')

		except Exception as e:
			logger.error(f"Failed to save {table_name} as parquet: {str(e)}")
			raise


	def save_all_tables_as_parquet(self, tables_list: Optional[List[TableNames]] = None) -> None:
		"""
		Save all tables as Parquet files with improved error handling.

		Args:
			tables_list: List of table names to convert
		"""
		# If no tables list is provided, use the study table list
		if tables_list is None:
			tables_list = self.data_loader.study_tables_list

		# Save tables as parquet with error handling
		failed_tables = []
		for table_name in tqdm(tables_list, desc="Saving tables as parquet"):
			try:
				self.save_as_parquet(table_name=table_name)
			except Exception as e:
				logger.error(f"Failed to convert {table_name}: {str(e)}")
				failed_tables.append(table_name)
				continue

		if failed_tables:
			logger.warning(f"Failed to convert the following tables: {failed_tables}")
		else:
			logger.info("Successfully converted all tables to Parquet format")

	@staticmethod
	def save_dask_partitions_separately(df, target_path: Path, schema: pa.Schema, table_name: TableNames) -> None:
		"""
		Save Dask DataFrame partition by partition when memory is limited.

		Args:
			df: Dask DataFrame to save
			target_path: Target parquet file path
			schema: PyArrow schema
			table_name: Name of the table being saved
		"""
		logger.info(f"Saving {table_name} partition by partition due to memory constraints")

		# Create temporary directory for partition files
		with tempfile.TemporaryDirectory() as temp_dir:
			temp_path = Path(temp_dir)
			partition_files = []

			try:
				# Save each partition separately
				for i in range(df.npartitions):
					partition = df.get_partition(i).compute()
					if len(partition) > 0:  # Only save non-empty partitions
						partition_file = temp_path / f"partition_{i:04d}.parquet"
						table = pa.Table.from_pandas(partition, schema=schema)
						pq.write_table(table, partition_file, compression='snappy')
						partition_files.append(partition_file)
						logger.debug(f"Saved partition {i} with {len(partition)} rows")

				# Combine all partition files into final parquet file
				if partition_files:
					logger.info(f"Combining {len(partition_files)} partitions into final file")
					combined_table = pq.read_table(partition_files, schema=schema)
					pq.write_table(combined_table, target_path, compression='snappy')
					logger.info(f"Successfully combined partitions for {table_name}")
				else:
					logger.warning(f"No non-empty partitions found for {table_name}")
					# Create empty parquet file with schema
					empty_df = pd.DataFrame()
					for field in schema:
						empty_df[field.name] = pd.Series(dtype=field.type.to_pandas_dtype())
					empty_table = pa.Table.from_pandas(empty_df, schema=schema)
					pq.write_table(empty_table, target_path, compression='snappy')

			except Exception as e:
				logger.error(f"Error in partition-by-partition save for {table_name}: {str(e)}")
				raise


	@staticmethod
	def prepare_table_for_download_as_csv(df: pd.DataFrame | dd.DataFrame):
		"""Prepare CSV data on-demand with progress tracking."""
		logger.info("Preparing CSV download...")

		try:
			if isinstance(df, dd.DataFrame):

				# Use Dask's native to_csv with temporary file
				import tempfile
				with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
					tmp_path = tmp_file.name

				# Dask writes directly to file without computing entire DataFrame
				df.to_csv(tmp_path, index=False, single_file=True)

				# Read the file content and clean up
				with open(tmp_path, 'r', encoding='utf-8') as f:
					csv_data = f.read().encode('utf-8')

				# Clean up temporary file
				Path(tmp_path).unlink(missing_ok=True)

				return csv_data
			else:
				csv_data = df.to_csv(index=False).encode('utf-8')
				return csv_data

		except Exception as e:
			logger.error(f"Error preparing CSV download: {e}")
			return b""  # Return empty bytes on error


if __name__ == '__main__':

	loader = DataLoader(mimic_path=DEFAULT_MIMIC_PATH, apply_filtering=True, filter_params={})
	df_merged = loader.load(table_name=TableNames.MERGED, partial_loading=False)
	subject_ids = loader.get_unique_subject_ids(table_name=TableNames.ADMISSIONS)

	print('done')
