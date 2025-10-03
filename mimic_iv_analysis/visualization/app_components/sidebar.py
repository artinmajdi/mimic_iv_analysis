# Standard library imports
import os
from pathlib import Path
from typing import Tuple, Optional, List

# Data processing imports
import pandas as pd
import dask.dataframe as dd

# Dask distributed for background computation
from dask.distributed import Client, LocalCluster

# Streamlit import
import streamlit as st


# Local application imports
from mimic_iv_analysis import logger
from mimic_iv_analysis.core import FeatureEngineerUtils, DaskConfigOptimizer
from mimic_iv_analysis.io import DataLoader, ParquetConverter
from mimic_iv_analysis.configurations import TableNames, DEFAULT_MIMIC_PATH, DEFAULT_NUM_SUBJECTS
from mimic_iv_analysis.visualization.app_components import FilteringTab

class SideBar:
	def __init__(self):

		# Initialize core components
		logger.info(f"Initializing DataLoader with path: {DEFAULT_MIMIC_PATH}")
		self.data_handler = DataLoader(mimic_path=Path(DEFAULT_MIMIC_PATH))

		logger.info("Initializing ParquetConverter...")
		self.parquet_converter = ParquetConverter(data_loader=self.data_handler)

	def render(self):
		"""Handles the display and logic of the sidebar components."""

		def _select_sampling_parameters():

			total_unique_subjects = len(self.data_handler.get_unique_subject_ids(table_name=TableNames(st.session_state.selected_table)))

			# Subject-based sampling not available if no subjects found
			if total_unique_subjects == 0 and self.data_handler.tables_info_df is not None:
				st.sidebar.warning(f"Could not load subject IDs from '{TableNames.PATIENTS}'. Ensure it's present and readable.")

			# Subject-based sampling not available if no subjects found
			elif self.data_handler.tables_info_df is None:
				st.sidebar.warning("Scan the directory first to see available subjects.")


			# Number of subjects to load
			max_value = total_unique_subjects if total_unique_subjects > 0 else 1
			st.sidebar.number_input(
				"Number of Subjects to Load",
				min_value = 1,
				max_value = max_value,
				disabled  = self.has_no_subject_id_column,
				key       = "num_subjects_to_load",
				step      = 10,
				value     = min(st.session_state.get('num_subjects_to_load', DEFAULT_NUM_SUBJECTS), max_value),
				help      = f"Number of subjects to load. Max: {total_unique_subjects}."
			)

			st.sidebar.caption(f"Total unique subjects found: {total_unique_subjects if total_unique_subjects > 0 else 'N/A (Scan or check patients.csv)'}")

		def _parquet_conversion():

			def _parquet_update_convert_single_table():

				# Display "Convert" button if the selected table is not already in Parquet but a source CSV exists
				if not self.is_selected_table_parquet and self._source_csv_exists:
					if st.button(
						label    = "Convert to Parquet",
						key      = "convert_to_parquet_button",
						on_click = self._convert_table_to_parquet,
						args     = ([ TableNames(st.session_state.selected_table) ],),
						help     = f"Convert {st.session_state.selected_table} from CSV to Parquet for faster loading." ):

						st.warning('Please refresh the page to see the updated tables.')

				# Display "Update" button if the selected table is already in Parquet and a source CSV exists
				if self.is_selected_table_parquet and self._source_csv_exists:
					st.button(
						label    = "Update Parquet",
						key      = "update_parquet_button",
						on_click = self._convert_table_to_parquet,
						args     = ([ TableNames(st.session_state.selected_table) ],),
						help     = f"Re-convert {st.session_state.selected_table} from CSV to update the Parquet file.",
						type     = 'secondary' )

					st.warning('You only need to update parquet table if you think the current parquet file is corrupted or the corresponding csv file is changed.')

			def parquet_update_convert_merged_table():

				def _get_tables_that_need_conversion(force_update: bool = False) -> List[TableNames]:
					""" Checks which component tables of the merged table need to be converted to Parquet. """

					tables_to_convert = []
					component_tables = self.data_handler.study_tables_list

					if force_update:
						return component_tables

					for table_name in component_tables:
						try:
							file_path = self.data_handler._get_file_path(table_name=TableNames(table_name))
							if file_path.suffix != '.parquet':
								tables_to_convert.append(table_name)
						except (ValueError, IndexError):
							logger.warning(f"Component table {table_name} not found, skipping for conversion check.")
							continue
					return tables_to_convert

				# Check which component tables need to be converted
				tables_to_convert = _get_tables_that_need_conversion(force_update=False)
				if tables_to_convert:
					st.warning(f"{len(tables_to_convert)} base table(s) are not in Parquet format.")
					st.button(
						label    = "Convert Missing Tables to Parquet",
						key      = "convert_merged_to_parquet",
						on_click = self._convert_table_to_parquet,
						args     = (tables_to_convert,),
						help     = "Convert all required CSV tables to Parquet for the merged view." )

				# Button to update all component tables of the merged view
				st.button(
					label    = "Update All Base Parquet Tables",
					key      = "update_merged_parquet",
					on_click = self._convert_table_to_parquet,
					args     = (self.data_handler.study_tables_list,),
					help     = "Re-convert all base tables from CSV to update their Parquet files."
				)

				st.warning('You only need to update parquet table if you think the current parquet file is corrupted or the corresponding csv file is changed.')

			st.markdown("---")
			with st.sidebar.expander("Parquet Conversion"):

				if st.session_state.selected_table: # Ensure a table is selected
					if st.session_state.selected_table != "merged_table":
						_parquet_update_convert_single_table()
					else:
						parquet_update_convert_merged_table()

			if 'conversion_status' in st.session_state:
				status = st.session_state.conversion_status
				message_type = status.get('type')
				message_text = status.get('message')

				if message_type == 'success':
					st.sidebar.success(message_text)
				elif message_type == 'error':
					st.sidebar.error(message_text)
				elif message_type == 'warning':
					st.sidebar.warning(message_text)
				elif message_type == 'exception':
					st.sidebar.exception(message_text)

		def _load_configuration():

			st.sidebar.markdown("---")

			# if st.session_state.selected_table and not self.is_selected_table_parquet and st.session_state.selected_table != "merged_table":
			# 	# Disable loading options since conversion is required
			# 	st.sidebar.caption("Loading is disabled until the table is converted to Parquet.")
			# 	st.sidebar.checkbox("Load Full Table", value=True, disabled=True, key="load_full_disabled")
			# 	st.sidebar.number_input("Number of Subjects to Load", value=1, disabled=True, key="num_subjects_disabled")
			# 	st.sidebar.checkbox("Apply Filtering", value=True, disabled=True, key="apply_filtering_disabled")
			# 	st.sidebar.checkbox("Use Dask", value=True, disabled=True, key="use_dask_disabled")
			# else:
			# Sampling options
			st.sidebar.checkbox(
				label   = "Load Full Table",
				value   = st.session_state.get('load_full', False) if not self.has_no_subject_id_column else True,
				key     = "load_full",
				disabled=self.has_no_subject_id_column )

			if not st.session_state.load_full:
				_select_sampling_parameters()

			st.sidebar.checkbox("Apply Filtering", value=st.session_state.get('apply_filtering', True), key="apply_filtering", on_change=self._callback_reload_dataloader, help="Apply cohort filtering to the table before loading.")
			st.sidebar.checkbox("Use Dask"		 , value=st.session_state.get('use_dask', True)		  , key="use_dask"		 , help="Enable Dask for distributed computing and memory-efficient processing")

		def _select_table_module():

			def _select_module():

				module_options = list(st.session_state.available_tables.keys())

				module = st.sidebar.selectbox(
					label   = "Select Module",
					options = module_options,
					index   = module_options.index('hosp') if st.session_state.selected_module == 'hosp' else 0,
					key     = "module_select" ,
					help    = "Select which MIMIC-IV module to explore (e.g., hosp, icu)"
				)
				# Update selected module if changed
				if module != st.session_state.selected_module:
					st.session_state.selected_module = module
					st.session_state.selected_table = None # Reset table selection when module changes

				return module

			def _select_table(module: str):
				"""Display table selection dropdown and handle selection logic."""

				def _get_table_options_list():
					# Get sorted table options for the selected module
					table_options = sorted(st.session_state.available_tables[module])

					# Create display options list with the special merged_table option first
					tables_list_w_size_info = ["merged_table"]

					# Create display-to-table mapping for reverse lookup
					display_to_table_map = {'merged_table': 'merged_table'}

					# Format each table with size information
					for table in table_options:

						# Get display name from session state
						display_name = st.session_state.table_display_names.get((module, table), table)

						# Add display name to list
						tables_list_w_size_info.append(display_name)

						# Map display name to table name
						display_to_table_map[display_name] = table

					return tables_list_w_size_info, display_to_table_map

				def _display_table_info(table: str) -> None:
					"""Display table description information in sidebar."""

					logger.info(f"Displaying table info for {module}.{table}")

					table_info = TableNames(table).description

					if table_info:
						st.sidebar.markdown( f"**Description:** {table_info}", help="Table description from MIMIC-IV documentation." )

				# Get sorted table options for the selected module
				tables_list_w_size_info, display_to_table_map = _get_table_options_list()

				# Display the table selection dropdown
				st.sidebar.selectbox(
					label   = "Select Table",
					options = tables_list_w_size_info,
					index   = 0,
					key     = "selected_table_name_w_size",
					help    = "Select which table to load (file size shown in parentheses)" )

				# Get the actual table name from the selected display
				table = display_to_table_map[st.session_state.selected_table_name_w_size]

				# Update session state if table selection changed
				if table != st.session_state.selected_table:
					st.session_state.selected_table = table
					st.session_state.df = None  # Clear dataframe when table changes

				# Show table filters
				with st.sidebar.expander("Inclusion/Exclusion Criteria", expanded=False):
					FilteringTab(table_name=TableNames(table))

				# Show table description if a regular table is selected
				if st.session_state.selected_table != "merged_table":
					_display_table_info(st.session_state.selected_table)

			module = _select_module()

			if module in st.session_state.available_tables:
				_select_table(module=module)

			else:
				st.session_state.selected_table_name_w_size = None

		st.sidebar.title("MIMIC-IV Navigator")
		self._dask_configuration()
		self._dataset_configuration()

		# Module and table selection
		if not st.session_state.available_tables:
			st.sidebar.info("Scan a MIMIC-IV directory to select and load tables.")
			return

		_select_table_module()
		_parquet_conversion()

		_load_configuration()

		if st.session_state.selected_table == "merged_table":
			button_name = "Re calculate subject IDs for all tables"
		else:
			button_name = "Re calculate subject IDs"

		if st.sidebar.button(button_name, key="re_calculate_subject_ids", type="secondary"):
			self._callback_reload_dataloader()
			self.data_handler.get_unique_subject_ids(table_name=TableNames(st.session_state.selected_table), recalculate_subject_ids=True)

		# Only show load button if table is Parquet or it is the merged view
		if st.session_state.get('selected_table') == 'merged_table' or self.is_selected_table_parquet:
			self._load_table(selected_table_name_w_size=st.session_state.selected_table_name_w_size)

		if st.session_state.selected_table == "merged_table":
			with st.sidebar.expander(label="Export Loaded Data", expanded=True):
				if st.session_state.df is not None:
					self._export_options()


	def _dask_configuration(self):
		"""Display Dask configuration options in the sidebar."""

		def _optimize_button():

			if st.button(
				"Find Optimum Parameters",
				help="Automatically optimize Dask configuration based on your system resources for MIMIC-IV workloads",
				use_container_width=True,
				type="primary",
				icon="🔧" ):
				try:
					# Get optimized configuration
					optimized_config = DaskConfigOptimizer.get_optimized_config_for_streamlit()

					# Update session state with optimized values
					st.session_state.dask_n_workers          = optimized_config['n_workers']
					st.session_state.dask_threads_per_worker = optimized_config['threads_per_worker']
					st.session_state.dask_memory_limit       = optimized_config['memory_limit']

					# Show success message with details
					st.success(f"✅ Configuration optimized!\n\n"
								f"**Workers:** {optimized_config['n_workers']}\n"
								f"**Threads per Worker:** {optimized_config['threads_per_worker']}\n"
								f"**Memory Limit:** {optimized_config['memory_limit']}\n\n"
								f"*{optimized_config['description']}*")

					# Force UI refresh to show new values
					st.rerun()

				except Exception as e:
					st.error(f"❌ Failed to optimize configuration: {str(e)}")
					logger.error(f"Dask optimization error: {e}")

		with st.sidebar.expander(label="Dask Configuration", expanded=False):

			_optimize_button()

			# Number of workers
			n_workers = st.number_input(
				label="Number of Workers",
				min_value=1,
				max_value=8,
				value=st.session_state.dask_n_workers,
				help="Number of Dask worker processes. More workers can improve parallel processing but use more memory."
			)

			# Threads per worker
			threads_per_worker = st.number_input(
				label="Threads per Worker",
				min_value=1,
				max_value=32,
				value=st.session_state.dask_threads_per_worker,
				help="Number of threads per worker. Higher values can improve CPU-bound tasks."
			)

			# Memory limit
			memory_limit = st.text_input(
				label="Memory Limit per Worker",
				value=st.session_state.dask_memory_limit,
				help="Memory limit per worker (e.g., '4GB', '8GB', '20GB'). Total memory usage will be this value × number of workers."
			)

			# Dashboard port
			dashboard_port = st.number_input(
				label="Dashboard Port",
				min_value=8000,
				max_value=9999,
				value=st.session_state.dask_dashboard_port,
				help="Port for Dask dashboard. Access at http://localhost:[port] to monitor Dask performance."
			)


		# Check if any values changed and update session state
		config_changed = False
		if n_workers != st.session_state.dask_n_workers:
			st.session_state.dask_n_workers = n_workers
			config_changed = True

		if threads_per_worker != st.session_state.dask_threads_per_worker:
			st.session_state.dask_threads_per_worker = threads_per_worker
			config_changed = True

		if memory_limit != st.session_state.dask_memory_limit:
			st.session_state.dask_memory_limit = memory_limit
			config_changed = True

		if dashboard_port != st.session_state.dask_dashboard_port:
			st.session_state.dask_dashboard_port = dashboard_port
			config_changed = True

		# If configuration changed, reinitialize Dask client
		if config_changed:
			st.sidebar.success("Dask configuration updated! Client will be reinitialized.")
			SideBar.init_dask_client()  # Reinitialize with new settings
			st.rerun()  # Refresh the UI

	def _dataset_configuration(self):

		# st.sidebar.markdown("---") # Separator
		with st.sidebar.expander(label="## Dataset Configuration", expanded=True):

			# MIMIC-IV path input
			mimic_path = st.text_input(label="MIMIC-IV Dataset Path", value=st.session_state.mimic_path, help="Enter the path to your local MIMIC-IV v3.1 dataset directory")

			# Update mimic_path in session state if it changes
			if mimic_path != st.session_state.mimic_path:
				st.session_state.mimic_path = mimic_path
				# Clear previous scan results if path changes
				st.session_state.available_tables = {}
				st.session_state.file_paths = {}
				st.session_state.file_sizes = {}
				st.session_state.table_display_names = {}
				st.session_state.selected_module = None
				st.session_state.selected_table = None
				st.sidebar.info("Path changed. Please re-scan.")

			# Scan button
			if st.button("Scan MIMIC-IV Directory", key="scan_button", type="primary"):

				if not mimic_path or not os.path.isdir(mimic_path):
					st.error("Please enter a valid directory path for the MIMIC-IV dataset")
					return

				with st.spinner("Scanning directory..."):
					self._scan_directory(mimic_path)

	def _load_table(self, selected_table_name_w_size: str = None) -> Tuple[Optional[pd.DataFrame], int]:
		"""Load a specific MIMIC-IV table, handling large files and sampling."""

		def _get_total_subjects(table_name: TableNames) -> int:

			unique_subject_ids = self.data_handler.get_unique_subject_ids(table_name=table_name, recalculate_subject_ids=False)

			st.session_state.total_subjects_count = len(unique_subject_ids)

			return st.session_state.total_subjects_count

		def _load_study_tables_and_merge() -> pd.DataFrame:

			def _merged_df_is_valid(merged_df, total_subjects):

				if isinstance(merged_df, dd.DataFrame) and total_subjects == 0:
					st.sidebar.error("Failed to load connected tables.")
					return False

				if isinstance(merged_df, pd.DataFrame) and merged_df.empty:
					st.sidebar.error("Failed to load connected tables.")
					return False

				return True

			def _dataset_path_is_valid():

				dataset_path = st.session_state.mimic_path

				if not dataset_path or not os.path.exists(dataset_path):
					st.sidebar.error(f"MIMIC-IV directory not found: {dataset_path}. Please set correct path and re-scan.")
					return False
				return True

			def _load_connected_tables():

				def _load_and_return():
					return self.data_handler.load(
						table_name      = TableNames.MERGED,
						tables_dict     = st.session_state.connected_tables,
						partial_loading = False,
						use_dask        = st.session_state.use_dask,
						num_subjects    = st.session_state.get('num_subjects_to_load', DEFAULT_NUM_SUBJECTS) )

				with st.spinner("Loading and merging connected tables..."):

					# If loading full dataset, keep previous behavior
					if st.session_state.load_full:
						st.session_state.connected_tables = self.data_handler.fetch_complete_study_tables(use_dask=st.session_state.use_dask)
						return _load_and_return()

					# Optimized path: 1) choose subject_ids via intersection, 2) load only those rows, 3) merge
					selected_ids = self.data_handler.get_sample_subject_ids(
										table_name   = TableNames.MERGED,
										num_subjects = st.session_state.get('num_subjects_to_load', DEFAULT_NUM_SUBJECTS) )

					# Fallback to full load if no subject_ids found
					if not selected_ids:
						st.session_state.connected_tables = self.data_handler.fetch_complete_study_tables(use_dask=st.session_state.use_dask)
					else:
						st.session_state.connected_tables = self.data_handler.load_filtered_study_tables_by_subjects( subject_ids=selected_ids, use_dask=st.session_state.use_dask )

					return _load_and_return()

			if not _dataset_path_is_valid():
				return

			with st.spinner("Loading and merging connected tables..."):

				merged_df = _load_connected_tables()

				total_subjects = _get_total_subjects(table_name=TableNames.MERGED)

				if _merged_df_is_valid(merged_df=merged_df, total_subjects=total_subjects):

					st.session_state.df                 = merged_df
					st.session_state.current_file_path  = "merged_tables"
					st.session_state.table_display_name = "Merged MIMIC-IV View"

					self._clear_analysis_states()

					st.sidebar.success(f"Successfully merged {len(st.session_state.connected_tables)} tables with {len(merged_df.columns)} columns and {total_subjects} subjects!")

		def _load_single_table():

			def _df_is_valid(df, total_subjects):

				if df is None:
					st.sidebar.error("Failed to load table. Check logs or file format.")
					st.session_state.df = None
					return False

				if total_subjects == 0 and st.session_state.selected_table in TableNames._TABLES_W_SUBJECT_ID_COLUMN:
					st.sidebar.warning("Loaded table is empty.")
					st.session_state.df = None
					return False

				return True

			if not st.session_state.load_full:
				loading_message = f"Loading table for {st.session_state.num_subjects_to_load} subjects..."
			else:
				loading_message = "Loading table using " + ("Dask" if st.session_state.use_dask else "Pandas")


			table_name = TableNames(st.session_state.selected_table)

			file_path = st.session_state.file_paths.get((st.session_state.selected_module, st.session_state.selected_table))

			st.session_state.current_file_path = file_path

			with st.spinner(loading_message):

				df = self.data_handler.load(
					table_name      = table_name,
					partial_loading = not st.session_state.load_full,
					num_subjects    = st.session_state.get('num_subjects_to_load', None),
					use_dask        = st.session_state.use_dask
					)


				# Get total number of rows
				total_subjects = _get_total_subjects(table_name=table_name)

			if _df_is_valid(df, total_subjects):

				st.session_state.df = df
				# Cache DataFrame length to avoid repeated computation
				st.session_state.df_length = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
				st.sidebar.success(f"Loaded {total_subjects} subjects and {st.session_state.df_length} rows from {file_path}")

				# Clear previous analysis results when new data is loaded
				self._clear_analysis_states()

				# Auto-detect columns for feature engineering
				st.session_state.detected_order_cols     = FeatureEngineerUtils.detect_order_columns(df)
				st.session_state.detected_time_cols      = FeatureEngineerUtils.detect_temporal_columns(df)
				st.session_state.detected_patient_id_col = FeatureEngineerUtils.detect_patient_id_column(df)

				st.sidebar.write("Detected Columns (for Feature Eng):")
				st.sidebar.caption(f"Patient ID: {st.session_state.detected_patient_id_col}, Order: {st.session_state.detected_order_cols}, Time: {st.session_state.detected_time_cols}")

		def _load_merged_table():

			st.session_state.current_file_path = self.data_handler.merged_table_parquet_path

			df = self.data_handler.fetch_table(table_name=TableNames.MERGED, use_dask=st.session_state.use_dask, apply_filtering=st.session_state.apply_filtering)

			if not st.session_state.load_full:
				df = self.data_handler.partial_loading(df=df, table_name=TableNames.MERGED, num_subjects=st.session_state.num_subjects_to_load)

			st.session_state.df = df
			# Cache DataFrame length to avoid repeated computation
			st.session_state.df_length = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
			st.sidebar.success(f"Loaded {st.session_state.df_length} rows.")

		def _check_table_selection():
			if selected_table_name_w_size != "merged_table" and (not st.session_state.selected_module or not st.session_state.selected_table):
				st.sidebar.warning("Please select a module and table first.")
				return False
			return True

		def parquet_file_exists_and_not_empty(file_path):
			"""Check if parquet file exists and is not empty"""
			path = Path(file_path)
			return path.exists() and path.stat().st_size > 0

		# Updating self.data_handler
		self._callback_reload_dataloader()

		if selected_table_name_w_size == "merged_table":

			parquet_exist = parquet_file_exists_and_not_empty(self.data_handler.merged_table_parquet_path)

			st.sidebar.checkbox('Load merged table from local parquet file', value=parquet_exist, key="load_merge_from_local_parquet", disabled=not parquet_exist)

		if st.sidebar.button("Load Table", key="load_button", type="primary") and _check_table_selection():

			if selected_table_name_w_size == "merged_table":
				if st.session_state.load_merge_from_local_parquet:
					_load_merged_table()
				else:
					_load_study_tables_and_merge()
			else:
				_load_single_table()

	def _convert_table_to_parquet(self, tables_to_process: Optional[List[TableNames]] = None):
		"""Callback to convert the selected table to Parquet format."""

		selected_table  = st.session_state.selected_table
		selected_module = st.session_state.selected_module

		if tables_to_process is None:
			tables_to_process = [ TableNames(selected_table) ]

		if not tables_to_process:
			st.session_state.conversion_status = {'type': 'warning', 'message': "No tables to process."}
			return

		# try:
		with st.spinner(f"Converting {len(tables_to_process)} table(s) to Parquet..."):
			success_count = 0
			failed_tables = []

			for table_enum in tables_to_process:
				# try:
				logger.info(f"Starting conversion of {table_enum.value}")
				self.parquet_converter.save_as_parquet(table_name=table_enum)
				success_count += 1
				logger.info(f"Successfully converted {table_enum.value}")
				# except Exception as table_error:
				# 	logger.error(f"Failed to convert {table_enum.value}: {str(table_error)}")
				# 	failed_tables.append(table_enum.value)
				# 	continue

		if self._rescan_and_update_state():
			if success_count == len(tables_to_process):
				st.session_state.conversion_status = {'type': 'success', 'message': f"Successfully converted all {success_count} table(s)!"}
			elif success_count > 0:
				st.session_state.conversion_status = {'type': 'warning', 'message': f"Converted {success_count}/{len(tables_to_process)} table(s). Failed: {', '.join(failed_tables)}"}
			else:
				st.session_state.conversion_status = {'type': 'error', 'message': f"Failed to convert any tables. Failed: {', '.join(failed_tables)}"}
		else:
			st.session_state.conversion_status = {'type': 'error', 'message': "Conversion might have failed. Could not find updated tables."}


		# except Exception as e:
		# 	logger.error(f"Parquet conversion job failed: {e}", exc_info=True)
		# 	st.session_state.conversion_status = {'type': 'exception', 'message': f"Critical error during conversion: {str(e)}. Try reducing Dask memory settings or processing smaller tables."}

		st.session_state.selected_table = selected_table
		st.session_state.selected_module = selected_module

	def _scan_directory(self, mimic_path: str):
		try:
			# Update the data handler's path if it changed
			if mimic_path != str(self.data_handler.mimic_path):
				self._callback_reload_dataloader()
				# self.data_handler      = DataLoader(mimic_path=Path(mimic_path))
				# self.parquet_converter = ParquetConverter(data_loader=self.data_handler)

			if self._rescan_and_update_state():
				st.sidebar.success(f"Found {sum(len(tables) for tables in st.session_state.available_tables.values())} tables in {len(st.session_state.available_tables)} modules")

				# Reset selections if scan is successful
				if st.session_state.available_tables:
					st.session_state.selected_module = list(st.session_state.available_tables.keys())[0]

				# Force user to select table after scan
				st.session_state.selected_table = None

			else:
				st.sidebar.error("No MIMIC-IV tables (.csv, .csv.gz, .parquet) found in the specified path or its subdirectories (hosp, icu).")

		except AttributeError:
			st.sidebar.error("Data Handler is not initialized or does not have a 'scan_mimic_directory' method.")
		except Exception as e:
			st.sidebar.error(f"Error scanning directory: {e}")
			logger.exception("Error during directory scan")

	def _callback_reload_dataloader(self):

		self.data_handler = DataLoader(
						mimic_path        = st.session_state.get('mimic_path', Path(DEFAULT_MIMIC_PATH)),
						apply_filtering   = st.session_state.apply_filtering,
						filter_params     = st.session_state.filter_params)

		self.parquet_converter = ParquetConverter(data_loader=self.data_handler)

	def _rescan_and_update_state(self):
		"""Rescans the directory and updates session state with table info."""
		logger.info("Re-scanning directory and updating state...")
		self.data_handler.scan_mimic_directory()
		dataset_info_df = self.data_handler.tables_info_df
		dataset_info    = self.data_handler.tables_info_dict

		if dataset_info_df is not None and not dataset_info_df.empty:
			st.session_state.available_tables    = dataset_info['available_tables']
			st.session_state.file_paths          = dataset_info['file_paths']
			st.session_state.file_sizes          = dataset_info['file_sizes']
			st.session_state.table_display_names = dataset_info['table_display_names']
			return True
		else:
			st.session_state.available_tables = {} # Clear previous results
			return False

	def _clear_analysis_states(self):
		"""Clears session state related to previous analysis when new data is loaded."""
		logger.info("Clearing previous analysis states...")
		# Feature engineering
		st.session_state.freq_matrix = None
		st.session_state.order_sequences = None
		st.session_state.timing_features = None
		st.session_state.order_dist = None
		st.session_state.patient_order_dist = None
		st.session_state.transition_matrix = None
		# Clustering
		st.session_state.clustering_input_data = None
		st.session_state.reduced_data = None
		st.session_state.kmeans_labels = None
		st.session_state.hierarchical_labels = None
		st.session_state.dbscan_labels = None
		st.session_state.lda_results = None
		st.session_state.cluster_metrics = {}
		st.session_state.optimal_k = None
		st.session_state.optimal_eps = None
		# Analysis
		st.session_state.length_of_stay = None

	@property
	def _source_csv_exists(self) -> bool:
		"""Check if a source CSV/GZ file exists for the selected table."""
		if not st.session_state.get('selected_table') or st.session_state.selected_table == "merged_table":
			return False
		try:
			table_name_enum = TableNames(st.session_state.selected_table)

			# This method will raise an error if the source is not found
			self.parquet_converter._get_csv_file_path(table_name=table_name_enum)
			return True
		except (ValueError, IndexError): # _get_csv_file_path might cause IndexError or ValueError
			return False

	@property
	def has_no_subject_id_column(self):
		"""Check if the current table has a subject_id column."""
		tables_that_can_be_sampled = TableNames._TABLES_W_SUBJECT_ID_COLUMN
		return st.session_state.selected_table not in tables_that_can_be_sampled

	@property
	def is_selected_table_parquet(self) -> bool:
		"""Check if the selected table is in Parquet format."""
		if not st.session_state.get('selected_table') or st.session_state.selected_table == "merged_table":
			return False

		file_path = st.session_state.file_paths.get((st.session_state.selected_module, st.session_state.selected_table))
		if file_path and isinstance(file_path, Path) and file_path.suffix == '.parquet':
			return True
		return False

	@staticmethod
	def init_dask_client():
		# ----------------------------------------
		# Initialize (or reuse) a Dask client so heavy
		# computations can run on worker processes and
		# the Streamlit script thread remains responsive
		# ----------------------------------------
		@st.cache_resource(show_spinner=False)
		def _get_dask_client(n_workers, threads_per_worker, memory_limit, dashboard_port):
			cluster = LocalCluster(
								n_workers          = n_workers,
								threads_per_worker = threads_per_worker,
								processes          = True,
								memory_limit       = memory_limit,
								dashboard_address  = f":{dashboard_port}", )
			return Client(cluster)

		# Initialize default values if not in session state with conservative settings
		if 'dask_n_workers' not in st.session_state:
			st.session_state.dask_n_workers = 1  # Reduced from 1 to 2 for better parallelism
		if 'dask_threads_per_worker' not in st.session_state:
			st.session_state.dask_threads_per_worker = 4  # Reduced from 16 to 4 to prevent memory overload
		if 'dask_memory_limit' not in st.session_state:
			st.session_state.dask_memory_limit = '25GB'  # Reduced from 20GB to 8GB for safer memory usage
		if 'dask_dashboard_port' not in st.session_state:
			st.session_state.dask_dashboard_port = 8787


		# Get Dask configuration from session state
		n_workers          = st.session_state.dask_n_workers
		threads_per_worker = st.session_state.dask_threads_per_worker
		memory_limit       = st.session_state.dask_memory_limit
		dashboard_port     = st.session_state.dask_dashboard_port

		# Create a unique key based on configuration to force recreation when settings change
		config_key = f"{n_workers}_{threads_per_worker}_{memory_limit}_{dashboard_port}"

		# Store the client in session_state so that a new one
		# is not spawned on every rerun, but recreate if config changed
		if "dask_client" not in st.session_state or st.session_state.get('dask_config_key') != config_key:
			# Close existing client if it exists
			if "dask_client" in st.session_state:
				st.session_state.dask_client.close()

			st.session_state.dask_client = _get_dask_client(n_workers, threads_per_worker, memory_limit, dashboard_port)
			st.session_state.dask_config_key = config_key
			logger.info("Dask client initialised with config %s: %s", config_key, st.session_state.dask_client)

		# self.dask_client = st.session_state.dask_client

	def _export_options(self):

		# CSV Download button without pre-calculating row count
		if st.button("Prepare CSV for Download", key="download_csv_button"):

			data = self.parquet_converter.prepare_table_for_download_as_csv(df=st.session_state.df)

			if st.download_button(
					label     = f"Click to download CSV file",
					data      = data,
					file_name = f'{st.session_state.selected_table}.csv',
					mime      = "text/csv",
					key       = "download_complete_csv",
					type      = "primary",
					help      = "Download the complete dataset as CSV file (memory optimized with Dask)"):

				st.success(f"CSV export completed!")

		# Export as Parquet button - only for merged tables
		if (st.session_state.load_full and
			st.session_state.selected_table == "merged_table" and
			st.button("Export as Parquet", key="save_as_parquet_button", type="primary")):

			target_path = self.data_handler.merged_table_parquet_path

			self.parquet_converter.save_as_parquet(
										table_name          = TableNames.MERGED,
										target_parquet_path = target_path,
										df                  = st.session_state.df)

			st.success(f"Merged table exported as Parquet: {target_path.name}")