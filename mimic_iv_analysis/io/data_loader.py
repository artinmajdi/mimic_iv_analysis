# Standard library imports
import os
import glob
import logging
import traceback
from pathlib import Path
from functools import lru_cache, cached_property
from typing import Dict, Optional, Tuple, List, Any, Union, Literal
import warnings
import enum

# Data processing imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import humanize
from tqdm import tqdm

# Our modules
from mimic_iv_analysis.core.params import ( TableNamesHOSP,
                                            TableNamesICU,
                                            dtypes_all,
                                            parse_dates_all,
                                            convert_table_names_to_enum_class,
                                            DEFAULT_MIMIC_PATH,
                                            DEFAULT_NUM_SUBJECTS,
                                            SUBJECT_ID_COL,
                                            DEFAULT_STUDY_TABLES_LIST)

from mimic_iv_analysis.core.filtering import Filtering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class DataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""




	def __init__(self,mimic_path: Path = DEFAULT_MIMIC_PATH,
					study_tables_list : Optional[List[TableNamesHOSP | TableNamesICU]] = None):

		# MIMIC_IV v3.1 path
		self.mimic_path = mimic_path

		# Tables to load. Use list provided by user or default list
		self.study_table_list = study_tables_list or DEFAULT_STUDY_TABLES_LIST

		# Class variables
		self._all_subject_ids : List[dtypes_all['subject_id']] = []
		self.tables_info_df   : Optional[pd.DataFrame]         = None # will be updated inside the scan_mimic_directory method()
		self.tables_info_dict : Optional[Dict[str, Any]]       = None # will be updated inside the scan_mimic_directory method()

		self.partial_subject_id_list: Optional[List[dtypes_all['subject_id']]] = None


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
			""" Lists unique table files from a module path. """

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
				tables_info_dict['table_display_names'][(module, table_name)] = f"{table_name} {humanize.naturalsize(_get_file_size_in_bytes(file_path))}"

				# Store file suffix
				suffix = file_path.suffix
				tables_info_dict['suffix'][(module, table_name)] = 'csv.gz' if suffix == '.gz' else suffix

				# Store columns
				if suffix == '.parquet':
					df = dd.read_parquet(file_path)
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

		# Initialize dataset info
		tables_info_dict = {
			'available_tables'   : {},
			'file_paths'         : {},
			'file_sizes'         : {},
			'table_display_names': {},
			'suffix'             : {},
			'columns_list'       : {},
		}

		# If the mimic path does not exist, return an empty DataFrame
		if not self.mimic_path.exists():
			self.tables_info_df = pd.DataFrame(columns=tables_info_dict.keys())
			return None, None

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

			# If no tables found, skip this module
			if not available_tables_dict:
				continue

			# Get available tables info
			_get_available_tables_info(available_tables_dict, module)

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
	def _list_of_tables_w_subject_id_column(self) -> List[str]:
		"""Returns a list of tables that have subject_id column."""
		tables_list = self.study_tables_info[self.study_tables_info.columns_list.apply(lambda x: 'subject_id' in x)].table_name.tolist()

		return [convert_table_names_to_enum_class(name=table_name, module='hosp') for table_name in tables_list]


	def load_csv_table_with_correct_column_datatypes(self, file_path: Path, use_dask: bool = True):

		# Check if file exists
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"CSV file not found: {file_path}")

		if file_path.suffix not in ['.csv', '.gz', '.csv.gz']:
			logging.warning(f"File {file_path} is not a CSV file. Skipping.")
			return pd.DataFrame()

		# First read a small sample to get column names without type conversion
		sample_df = pd.read_csv(file_path, nrows=5)
		columns   = sample_df.columns.tolist()

		# Filter dtypes and parse_dates to only include existing columns
		dtypes      = {col: dtype for col, dtype in dtypes_all.items() if col in columns}
		parse_dates = [col for col in parse_dates_all if col in columns]

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
				dtype          = dtypes,
				parse_dates    = parse_dates if parse_dates else None
			)

		return df


	def _get_file_path(self, table_name: TableNamesHOSP | TableNamesICU) -> Path:

		return Path(self.tables_info_df[
											(self.tables_info_df.table_name == table_name.value) &
											(self.tables_info_df.module == table_name.module)
										]['file_path'].iloc[0])


	@lru_cache(maxsize=None)
	def _load_unique_subject_ids_for_table(self, table_name: TableNamesHOSP | TableNamesICU = TableNamesHOSP.ADMISSIONS) -> List[dtypes_all['subject_id']]:
		"""Load unique subject IDs from the table_name.

		Retrieves all unique subject IDs from the table_name and stores them
		in the _subject_ids_list attribute. If the table_name cannot be found,
		an empty list will be stored instead.

		This method is called by the all_subject_ids property when needed.
		"""

		logging.info(f"Loading subject IDs from {table_name.value} table (this will be cached)")

		# Scan directory if not already done
		if self.tables_info_df is None:
			self.scan_mimic_directory()

		self._all_subject_ids = self.load_table(table_name=table_name, partial_loading=False)['subject_id'].unique().compute().tolist()

		return self._all_subject_ids


	@cached_property
	def all_subject_ids(self) -> List[dtypes_all['subject_id']]:
		""" Returns a list of unique subject_ids found in the admission table. """

		# TODO: I changed this to get the intersection of all subject IDs found in the tables that have subject_id column. but now, it returns empty. why? (could i do the filtering after I merged all the tables?)
		def _load_unique_subject_id_common_between_all_tables():
			subject_ids = []
			for table_name in self._list_of_tables_w_subject_id_column:
				subject_ids.append(set(self.load_table(table_name=table_name, partial_loading=False)['subject_id'].unique().compute().tolist()))

			# get the intersection of all subject IDs
			self._all_subject_ids = list(set.intersection(*subject_ids))


		# Load subject IDs if not already loaded or if the list is empty
		if not self._all_subject_ids:

			_ = self._load_unique_subject_ids_for_table()

			# self._load_unique_subject_id_common_between_all_tables()


		return self._all_subject_ids


	def get_partial_subject_id_list_for_partial_loading(self, num_subjects: int = DEFAULT_NUM_SUBJECTS, random_selection: bool = False) -> List[dtypes_all['subject_id']]:
		"""
			Returns a subset of subject IDs for sampling.

			Args:
				num_subjects    : Number of subject IDs to return. If 0 or negative, returns None.
				random_selection: If True, randomly selects the subject IDs. Otherwise, takes the first N.

			Returns:
				List of subject IDs for sampling, or None if the subject ID list is empty or num_subjects <= 0.
		"""

		# If no subject IDs or num_subjects is non-positive, return an empty list
		if not self.all_subject_ids or num_subjects <= 0:
			return []

		# If num_subjects is greater than the number of available subject IDs, return all subject IDs
		if num_subjects > len(self.all_subject_ids):
			return self.all_subject_ids

		# Randomly select subject IDs if random_selection is True
		if random_selection:
			self.partial_subject_id_list = np.random.choice(a=self.all_subject_ids, size=num_subjects).tolist()
		else:
			# Otherwise, return the first N subject IDs
			self.partial_subject_id_list = self.all_subject_ids[:num_subjects]

		return self.partial_subject_id_list


	def load_all_study_tables(self,
								partial_loading  : bool                                     = False,
								num_subjects     : int                                      = DEFAULT_NUM_SUBJECTS,
								random_selection : bool                                     = False,
								subject_ids      : Optional[List[dtypes_all['subject_id']]] = None,
								use_dask         : bool                                     = True
								) -> Dict[str, pd.DataFrame | dd.DataFrame]:

		# Get subject IDs for partial loading
		if partial_loading and (subject_ids is None):
			subject_ids = self.get_partial_subject_id_list_for_partial_loading(num_subjects=num_subjects, random_selection=random_selection)

		tables_dict = {}
		for _, row in self.study_tables_info.iterrows():

			table_name = convert_table_names_to_enum_class(name=row.table_name, module=row.module)

			tables_dict[table_name.value] = self.load_table(table_name=table_name, partial_loading=partial_loading, subject_ids=subject_ids, use_dask=use_dask)

		return tables_dict


	def load_table(self,
						table_name     : TableNamesHOSP | TableNamesICU,
						partial_loading: bool = False,
						subject_ids    : Optional[List[dtypes_all['subject_id']]] = None,
						use_dask       : bool = True
						) -> pd.DataFrame | dd.DataFrame:

		def _load_table_full() -> pd.DataFrame | dd.DataFrame:

			file_path = self._get_file_path(table_name=table_name)

			# For parquet files, respect the use_dask flag
			if file_path.suffix == '.parquet':
				if use_dask:
					return dd.read_parquet(file_path)
				else:
					return pd.read_parquet(file_path)

			return self.load_csv_table_with_correct_column_datatypes(file_path, use_dask=use_dask)

		def _partial_loading(df):

			if subject_ids is None:
				raise ValueError("partial_loading is True but both subject_ids is None")


			if 'subject_id' not in df.columns:
				logging.info(f"Table {table_name.value} does not have a subject_id column. Partial loading is not possible. Skipping partial loading.")
				return df

			logging.info(f"Filtering {table_name.value} by subject_id for {len(subject_ids)} subjects.")

			# Convert subject_ids to a set for faster lookups
			subject_ids_set = set(subject_ids)

			# Use map_partitions for Dask DataFrame or direct isin for pandas
			if isinstance(df, dd.DataFrame):
				return df.map_partitions(lambda part: part[part['subject_id'].isin(subject_ids_set)])

			return df[df['subject_id'].isin(subject_ids_set)]


		def _get_n_rows(df):
			return df.size.compute() / len(df.columns) if isinstance(df, dd.DataFrame) else df.shape[0]

		logging.info(f"Loading ----- {table_name.value} ----- table.")

		# Load table
		df = _load_table_full()
		logging.info(f"Loading full table: {humanize.intcomma(_get_n_rows(df))} rows.")

		df = Filtering(df=df, table_name=table_name).render()


		logging.info(f"Applied filters: {humanize.intcomma(_get_n_rows(df))} rows.")

		if partial_loading:
			df = _partial_loading(df)
			logging.info(f"Applied partial loading: {_get_n_rows(df)} rows.")

		return df


	def load_with_pandas_chunking(self,
										file_path         : Path,
										target_subject_ids: List[dtypes_all['subject_id']],
										max_chunks        : Optional[int] = None,
										read_params       : Optional[Dict[str, Any]] = None
										) -> Tuple[pd.DataFrame, int]:
		"""Loads a large file by filtering by subject_ids using pandas chunking."""

		logging.info(f"Using Pandas chunking to filter by subject_id for {os.path.basename(file_path)}.")

		# Initialize variables
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


	def load_merged_tables(self,
							partial_loading  : bool = False,
							num_subjects     : int  = DEFAULT_NUM_SUBJECTS,
							random_selection : bool = False,
							use_dask         : bool = True,
							tables_dict      : Optional[Dict[str, pd.DataFrame | dd.DataFrame]] = None
							) -> pd.DataFrame:

		if tables_dict is None:
			tables_dict = self.load_all_study_tables(	partial_loading  = partial_loading,
														num_subjects     = num_subjects,
														random_selection = random_selection,
														use_dask         = use_dask)

		# Get tables
		patients_df        = tables_dict[TableNamesHOSP.PATIENTS.value]
		admissions_df      = tables_dict[TableNamesHOSP.ADMISSIONS.value]
		diagnoses_icd_df   = tables_dict[TableNamesHOSP.DIAGNOSES_ICD.value]
		poe_df             = tables_dict[TableNamesHOSP.POE.value]
		d_icd_diagnoses_df = tables_dict[TableNamesHOSP.D_ICD_DIAGNOSES.value]
		poe_detail_df      = tables_dict[TableNamesHOSP.POE_DETAIL.value]

		if TableNamesHOSP.TRANSFERS.value in tables_dict:
			transfers_df       = tables_dict[TableNamesHOSP.TRANSFERS.value]


		# Merge tables
		df12 = patients_df.merge(admissions_df, on='subject_id', how='inner')

		if TableNamesHOSP.TRANSFERS.value in tables_dict:
			df123 = df12.merge(transfers_df, on=('subject_id', 'hadm_id'), how='inner')
		else:
			df123 = df12

		diagnoses_merged = diagnoses_icd_df.merge(d_icd_diagnoses_df, on=('icd_code', 'icd_version'), how='inner')
		merged_wo_poe    = df123.merge(diagnoses_merged, on=('subject_id', 'hadm_id'), how='inner')

		# The reason for 'left' is that we want to keep all the rows from poe table. The poe_detail table for unknown reasons, has fewer rows than poe table.
		poe_and_details   = poe_df.merge(poe_detail_df, on=('poe_id', 'poe_seq', 'subject_id'), how='left')

		merged_full_study = merged_wo_poe.merge(poe_and_details, on=('subject_id', 'hadm_id'), how='inner')

		return {'merged_wo_poe': merged_wo_poe, 'merged_full_study': merged_full_study, 'poe_and_details': poe_and_details}


class ExampleDataLoader:
	"""ExampleDataLoader class for loading example data."""

	def __init__(self, partial_loading: bool = False, num_subjects: int = 100, random_selection: bool = False, use_dask: bool = True):
		self.data_loader = DataLoader()
		self.data_loader.scan_mimic_directory()

		if partial_loading:
			self.tables_dict = self.data_loader.load_all_study_tables(partial_loading=True, num_subjects=num_subjects, random_selection=random_selection, use_dask=use_dask)
		else:
			self.tables_dict = self.data_loader.load_all_study_tables(partial_loading=False, use_dask=use_dask)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")


	def counter(self):

		get_nrows        = lambda table_name: humanize.intcomma(self.tables_dict[table_name.value].shape[0].compute())
		get_nsubject_ids = lambda table_name: humanize.intcomma(self.tables_dict[table_name.value].subject_id.unique().shape[0].compute())

		# Format the output in a tabular format
		print(f"{'Table':<15} | {'Rows':<10} | {'Subject IDs':<10}")
		print(f"{'-'*15} | {'-'*10} | {'-'*10}")
		print(f"{'patients':<15} | {get_nrows(TableNamesHOSP.PATIENTS):<10} | {get_nsubject_ids(TableNamesHOSP.PATIENTS):<10}")
		print(f"{'admissions':<15} | {get_nrows(TableNamesHOSP.ADMISSIONS):<10} | {get_nsubject_ids(TableNamesHOSP.ADMISSIONS):<10}")
		print(f"{'diagnoses_icd':<15} | {get_nrows(TableNamesHOSP.DIAGNOSES_ICD):<10} | {get_nsubject_ids(TableNamesHOSP.DIAGNOSES_ICD):<10}")
		print(f"{'poe':<15} | {get_nrows(TableNamesHOSP.POE):<10} | {get_nsubject_ids(TableNamesHOSP.POE):<10}")
		print(f"{'poe_detail':<15} | {get_nrows(TableNamesHOSP.POE_DETAIL):<10} | {get_nsubject_ids(TableNamesHOSP.POE_DETAIL):<10}")


	def study_table_info(self):
		return self.data_loader.study_tables_info


	def merge_two_tables(self, table1: TableNamesHOSP | TableNamesICU, table2: TableNamesHOSP | TableNamesICU, on: Tuple[str], how: Literal['inner', 'left', 'right', 'outer'] = 'inner'):
		return self.tables_dict[table1.value].merge(self.tables_dict[table2.value], on=on, how=how)


	def save_as_parquet(self, table_name: TableNamesHOSP | TableNamesICU):
		ParquetConverter(data_loader=self.data_loader).save_as_parquet(table_name=table_name)


	def n_rows_after_merge(self):

		patients_df        = self.tables_dict[TableNamesHOSP.PATIENTS.value]
		admissions_df      = self.tables_dict[TableNamesHOSP.ADMISSIONS.value]
		diagnoses_icd_df   = self.tables_dict[TableNamesHOSP.DIAGNOSES_ICD.value]
		poe_df             = self.tables_dict[TableNamesHOSP.POE.value]
		d_icd_diagnoses_df = self.tables_dict[TableNamesHOSP.D_ICD_DIAGNOSES.value]
		poe_detail_df      = self.tables_dict[TableNamesHOSP.POE_DETAIL.value]


		df12          = patients_df.merge(admissions_df,           on='subject_id',  how='inner')
		df34          = diagnoses_icd_df.merge(d_icd_diagnoses_df, on=('icd_code',   'icd_version'), how='inner')
		poe_and_details    = poe_df.merge(poe_detail_df,                on=('poe_id',     'poe_seq',      'subject_id'), how='left')
		merged_wo_poe = df12.merge(df34,                           on=('subject_id', 'hadm_id'),     how='inner')
		merged_full_study  = merged_wo_poe.merge(poe_and_details,            on=('subject_id', 'hadm_id'),     how='inner')


		print(f"{'DataFrame':<15} {'Count':<10} {'DataFrame':<15} {'Count':<10} {'DataFrame':<15} {'Count':<10}")
		print("-" * 70)
		print(f"{'df12':<15} {df12.shape[0].compute():<10} {'patients':<15} {patients_df.shape[0].compute():<10} {'admissions':<15} {admissions_df.shape[0].compute():<10}")
		print(f"{'df34':<15} {df34.shape[0].compute():<10} {'diagnoses_icd':<15} {diagnoses_icd_df.shape[0].compute():<10} {'d_icd_diagnoses':<15} {d_icd_diagnoses_df.shape[0].compute():<10}")
		print(f"{'poe_and_details':<15} {poe_and_details.shape[0].compute():<10} {'poe':<15} {poe_df.shape[0].compute():<10} {'poe_detail':<15} {poe_detail_df.shape[0].compute():<10}")
		print(f"{'merged_wo_poe':<15} {merged_wo_poe.shape[0].compute():<10} {'df34':<15} {df34.shape[0].compute():<10} {'df12':<15} {df12.shape[0].compute():<10}")
		print(f"{'merged_full_study':<15} {merged_full_study.shape[0].compute():<10} {'poe_and_details':<15} {poe_and_details.shape[0].compute():<10} {'merged_wo_poe':<15} {merged_wo_poe.shape[0].compute():<10}")


	def load_table(self, table_name: TableNamesHOSP | TableNamesICU):
		return self.tables_dict[table_name.value]


	def load_all_study_tables(self):
		return self.tables_dict


	def load_merged_tables(self):
		return self.data_loader.load_merged_tables()


class ParquetConverter:
	"""Handles conversion of CSV/CSV.GZ files to Parquet format."""

	def __init__(self, data_loader: DataLoader):
		self.data_loader = data_loader


	def _get_csv_file_path(self, table_name: TableNamesHOSP | TableNamesICU) -> Tuple[Path, str]:

		def _fix_source_csv_path(source_path: Path) -> Tuple[Path, str]:
			""" Fixes the source csv path if it is a parquet file. """

			if source_path.name.endswith('.parquet'):

				csv_path = source_path.parent / source_path.name.replace('.parquet', '.csv')
				gz_path  = source_path.parent / source_path.name.replace('.parquet', '.csv.gz')

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


	def save_as_parquet(self, table_name: TableNamesHOSP | TableNamesICU, df: Optional[pd.DataFrame] = None, target_parquet_path: Optional[Path] = None) -> None:
		"""Saves a DataFrame as a Parquet file.

			Args:
				table_name         : Table name to save as parquet
				df                 : Optional DataFrame to save (if None, loads from source_path)
				target_parquet_path: Optional target path for the parquet file
		"""

		if df is None or target_parquet_path is None:

			# Get csv file path
			csv_file_path, suffix = self._get_csv_file_path(table_name)
			print(f"Saving {table_name.value} with suffix {suffix}")

			# Load the CSV file
			if df is None:
				df = self.data_loader.load_csv_table_with_correct_column_datatypes(file_path=csv_file_path)

			# Get parquet directory
			if target_parquet_path is None:
				target_parquet_path = csv_file_path.parent / csv_file_path.name.replace(suffix, '.parquet')

		# Save to parquet
		df.to_parquet(target_parquet_path, engine='pyarrow')


	def save_all_tables_as_parquet(self, tables_list: Optional[List[TableNamesHOSP | TableNamesICU]] = None) -> None:
		"""Save all tables as Parquet files.

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

	examples_partial = ExampleDataLoader(partial_loading=False, num_subjects=10, random_selection=False)

	examples_partial.load_merged_tables()
	# pc = ParquetConverter(data_loader=DataLoader(mimic_path=DEFAULT_MIMIC_PATH))
	# pc.save_as_parquet(table_name=TableNamesHOSP.ADMISSIONS)


	print('done')


	# TODO (next step):
	#   1. fix the dashboard not showing the merged table
	#   2. Save the merged table as a parquet file.
