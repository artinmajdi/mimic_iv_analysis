# Standard library imports
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, List

# Dask distributed for background computation
from dask.distributed import Client, LocalCluster

# Data processing imports
import pandas as pd
import dask.dataframe as dd

# Streamlit import
import streamlit as st
import humanize
import pdb

# Local application imports
from mimic_iv_analysis import logger
from mimic_iv_analysis.core import FeatureEngineerUtils
from mimic_iv_analysis.io import DataLoader, ParquetConverter
from mimic_iv_analysis.configurations import TableNames, DEFAULT_MIMIC_PATH, DEFAULT_NUM_SUBJECTS, DEFAULT_STUDY_TABLES_LIST

from mimic_iv_analysis.visualization.app_components import FilteringTab, FeatureEngineeringTab, AnalysisVisualizationTab, ClusteringAnalysisTab

from mimic_iv_analysis.visualization.visualizer_utils import MIMICVisualizerUtils


# TODO: Make sure the UI doesn't reload the table everytime user changes a setting. 
# TODO: fix the Order Timing Analysis tab. I am getting an error when i run "generate timing features" inside the "order timing analysis" tab , which itself is under "feature engineering" tab.
class MIMICDashboardApp:

	def __init__(self):
		logger.info("Initializing MIMICDashboardApp...")
		self.init_session_state()

		# Initialize core components
		logger.info(f"Initializing DataLoader with path: {DEFAULT_MIMIC_PATH}")
		self.data_handler = DataLoader(mimic_path=Path(DEFAULT_MIMIC_PATH))

		logger.info("Initializing ParquetConverter...")
		self.parquet_converter = ParquetConverter(data_loader=self.data_handler)

		logger.info("Initializing FeatureEngineerUtils...")
		self.feature_engineer  = FeatureEngineerUtils()

		self.__init_dask_client()

		# Initialize UI components for tabs
		self.feature_engineering_ui    = None
		self.clustering_analysis_ui    = None
		self.analysis_visualization_ui = None

		# Initialize session state
		self.current_file_path = None

		# self.init_session_state()
		logger.info("MIMICDashboardApp initialized.")


	def __init_dask_client(self):
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
			dashboard_address  = f":{dashboard_port}",
			# I/O optimizations
			# memory_spill_size       = "16GB",     # Spill to disk at 8GB
			# memory_target_fraction = 0.8,  # Target 80% memory usage
			)
			return Client(cluster)

		# Get Dask configuration from session state
		n_workers          = st.session_state.get('dask_n_workers', 1)
		threads_per_worker = st.session_state.get('dask_threads_per_worker', 16)
		memory_limit       = st.session_state.get('dask_memory_limit', '20GB')
		dashboard_port     = st.session_state.get('dask_dashboard_port', 8787)

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

		self.dask_client = st.session_state.dask_client

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

	def _dask_configuration(self):
		"""Display Dask configuration options in the sidebar."""
		
		# Initialize default values if not in session state
		if 'dask_n_workers' not in st.session_state:
			st.session_state.dask_n_workers = 1
		if 'dask_threads_per_worker' not in st.session_state:
			st.session_state.dask_threads_per_worker = 16
		if 'dask_memory_limit' not in st.session_state:
			st.session_state.dask_memory_limit = '20GB'
		if 'dask_dashboard_port' not in st.session_state:
			st.session_state.dask_dashboard_port = 8787

		# st.sidebar.markdown("---")  # Separator

		with st.sidebar.expander(label="Dask Configuration", expanded=False):
			
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
			self.__init_dask_client()  # Reinitialize with new settings
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
			if st.button("Scan MIMIC-IV Directory", key="scan_button"):

				if not mimic_path or not os.path.isdir(mimic_path):
					st.error("Please enter a valid directory path for the MIMIC-IV dataset")
					return

				with st.spinner("Scanning directory..."):
					self._scan_directory(mimic_path)

	def _display_sidebar(self):
		"""Handles the display and logic of the sidebar components."""

		def _select_view():

			# View selection
			st.sidebar.markdown("## Navigation")
			view_options = ["Data Explorer & Analysis", "Cohort Filtering"]

			# Get current index based on session state
			current_view_index = 0 if st.session_state.current_view == 'data_explorer' else 1

			selected_view = st.sidebar.radio("Select View", view_options, index=current_view_index, key="view_selector")

			# Update session state based on selection
			if selected_view == "Data Explorer & Analysis":
				st.session_state.current_view = 'data_explorer'
			else:
				st.session_state.current_view = 'filtering'

			return selected_view

		def _select_sampling_parameters():

			if st.session_state.selected_table == 'merged_table' or self.has_no_subject_id_column:
				table_name = TableNames.ADMISSIONS
			else:
				table_name = TableNames(st.session_state.selected_table)

			total_unique_subjects = len(self.data_handler.all_subject_ids(table_name=table_name))

			help_text_num_subjects = f"Number of subjects to load. Max: {total_unique_subjects}."

			# Subject-based sampling not available if no subjects found
			if total_unique_subjects == 0 and self.data_handler.tables_info_df is not None:
				st.sidebar.warning(f"Could not load subject IDs from '{TableNames.PATIENTS}'. Ensure it's present and readable.")

			# Subject-based sampling not available if no subjects found
			elif self.data_handler.tables_info_df is None:
				st.sidebar.warning("Scan the directory first to see available subjects.")


			# Number of subjects to load
			st.sidebar.number_input(
				"Number of Subjects to Load",
				min_value = 1,
				max_value = total_unique_subjects if total_unique_subjects > 0 else 1,
				disabled  = self.has_no_subject_id_column,
				key       = "num_subjects_to_load",
				step      = 10,
				value     = st.session_state.get('num_subjects_to_load', DEFAULT_NUM_SUBJECTS),
				help      = help_text_num_subjects
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
					component_tables = self.data_handler.merged_table_components

					if force_update:
						return component_tables

					for table_enum in component_tables:
						try:
							file_path = self.data_handler._get_file_path(table_name=table_enum)
							if file_path.suffix != '.parquet':
								tables_to_convert.append(table_enum)
						except (ValueError, IndexError):
							# This can happen if a component table (e.g., transfers) is not present
							logger.warning(f"Component table {table_enum.value} not found, skipping for conversion check.")
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
					args     = (self.data_handler.merged_table_components,),
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

			if st.session_state.selected_table and not self.is_selected_table_parquet and st.session_state.selected_table != "merged_table":
				# Disable loading options since conversion is required
				st.sidebar.caption("Loading is disabled until the table is converted to Parquet.")
				st.sidebar.checkbox("Load Full Table", value=True, disabled=True, key="load_full_disabled")
				st.sidebar.number_input("Number of Subjects to Load", value=1, disabled=True, key="num_subjects_disabled")
				st.sidebar.checkbox("Use Dask", value=True, disabled=True, key="use_dask_disabled")
			else:
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

				# Show table description if a regular table is selected
				if st.session_state.selected_table != "merged_table":
					_display_table_info(st.session_state.selected_table)

			module = _select_module()

			if module in st.session_state.available_tables:
				_select_table(module=module)

				if st.session_state.selected_table == "merged_table":
					st.sidebar.checkbox('Include Transfers Table', value=False, key='include_transfers', on_change=self._callback_reload_dataloader)
			else:
				st.session_state.selected_table_name_w_size = None

		st.sidebar.title("MIMIC-IV Navigator")

		_select_view()

		self._dask_configuration()

		self._dataset_configuration()

		# Module and table selection
		if not st.session_state.available_tables:
			st.sidebar.info("Scan a MIMIC-IV directory to select and load tables.")
			return

		_select_table_module()
		_parquet_conversion()
		_load_configuration()

		# Only show load button if table is Parquet or it is the merged view
		if st.session_state.get('selected_table') == 'merged_table' or self.is_selected_table_parquet:
			self._load_table(selected_table_name_w_size=st.session_state.selected_table_name_w_size)

	def _callback_reload_dataloader(self):

		self.data_handler = DataLoader(
						mimic_path        = st.session_state.get('mimic_path', Path(DEFAULT_MIMIC_PATH)),
						apply_filtering   = st.session_state.apply_filtering,
						include_transfers = st.session_state.include_transfers )

		self.parquet_converter = ParquetConverter(data_loader=self.data_handler)

	def _load_table(self, selected_table_name_w_size: str = None) -> Tuple[Optional[pd.DataFrame], int]:
		"""Load a specific MIMIC-IV table, handling large files and sampling."""

		def _get_total_rows(df):
			if df is None:
				return 0
			if isinstance(df, dd.DataFrame):
				st.session_state.total_row_count = df.shape[0].compute()
			else:
				st.session_state.total_row_count = df.shape[0]

			return st.session_state.total_row_count

		def _load_merged_table() -> pd.DataFrame:

			def _merged_df_is_valid(merged_df, total_rows):

				if isinstance(merged_df, dd.DataFrame) and total_rows == 0:
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
						num_subjects    = st.session_state.get('num_subjects_to_load', DEFAULT_NUM_SUBJECTS)
					) 

				with st.spinner("Loading and merging connected tables..."):

					# If loading full dataset, keep previous behavior
					if st.session_state.load_full:
						st.session_state.connected_tables = self.data_handler.load_all_study_tables_full(use_dask=st.session_state.use_dask)
						# Merge without partial filtering (tables already fully loaded)
						return _load_and_return()

					# Optimized path: 1) choose subject_ids via intersection, 2) load only those rows, 3) merge
					selected_ids = self.data_handler.get_merged_table_subject_id_intersection(
								num_subjects=st.session_state.get('num_subjects_to_load', DEFAULT_NUM_SUBJECTS) )

					# Fallback to full load if no subject_ids found
					if not selected_ids:
						st.session_state.connected_tables = self.data_handler.load_all_study_tables_full(use_dask=st.session_state.use_dask)
					else:
						st.session_state.connected_tables = self.data_handler.load_filtered_study_tables_by_subjects( subject_ids=selected_ids, use_dask=st.session_state.use_dask )

					return _load_and_return()

			if not _dataset_path_is_valid():
				return

			with st.spinner("Loading and merging connected tables..."):

				merged_df = _load_connected_tables()

				total_rows = _get_total_rows(merged_df)

				if _merged_df_is_valid(merged_df=merged_df, total_rows=total_rows):

					st.session_state.df                 = merged_df
					st.session_state.current_file_path  = "merged_tables"
					st.session_state.table_display_name = "Merged MIMIC-IV View"

					self._clear_analysis_states()

					st.sidebar.success(f"Successfully merged {len(st.session_state.connected_tables)} tables with {len(merged_df.columns)} columns and {total_rows} rows!")

		def _load_single_table():

			def _df_is_valid(df, total_rows):
				# Check if DataFrame is not None
				if df is None:
					st.sidebar.error("Failed to load table. Check logs or file format.")
					st.session_state.df = None
					return False

				# check shape
				if (isinstance(df, dd.DataFrame) and total_rows == 0) or (isinstance(df, pd.DataFrame) and df.empty):
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

				total_rows = _get_total_rows(df)

			if _df_is_valid(df, total_rows):

				st.session_state.df = df
				st.sidebar.success(f"Loaded {total_rows} rows.")

				# Clear previous analysis results when new data is loaded
				self._clear_analysis_states()

				# Auto-detect columns for feature engineering
				st.session_state.detected_order_cols     = FeatureEngineerUtils.detect_order_columns(df)
				st.session_state.detected_time_cols      = FeatureEngineerUtils.detect_temporal_columns(df)
				st.session_state.detected_patient_id_col = FeatureEngineerUtils.detect_patient_id_column(df)

				st.sidebar.write("Detected Columns (for Feature Eng):")
				st.sidebar.caption(f"Patient ID: {st.session_state.detected_patient_id_col}, Order: {st.session_state.detected_order_cols}, Time: {st.session_state.detected_time_cols}")

		def _check_table_selection():
			if selected_table_name_w_size != "merged_table" and (not st.session_state.selected_module or not st.session_state.selected_table):
				st.sidebar.warning("Please select a module and table first.")
				return False
			return True

		# Updating self.data_handler
		self._callback_reload_dataloader()

		if st.sidebar.button("Load Selected Table", key="load_button") and _check_table_selection():

			if selected_table_name_w_size == "merged_table":
				_load_merged_table()
			else:
				_load_single_table()

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

	def _export_options(self):
		st.markdown("<h2 class='sub-header'>Export Loaded Data</h2>", unsafe_allow_html=True)
		st.info("Export the currently loaded (and potentially sampled) data shown in the 'Exploration' tab.")
		export_col1, export_col2 = st.columns(2)

		with export_col1:
			export_format        = st.radio("Export Format", ["CSV", "Parquet"], index=0, key="export_main_format")
			export_filename_base = f"mimic_data_{st.session_state.selected_module}_{st.session_state.selected_table}"
			export_filename      = f"{export_filename_base}.{export_format.lower()}"

			if export_format == "CSV":
				try:
					# Check if Dask was used to load the data
					use_dask = st.session_state.get('use_dask', False)

					# Only compute if it's actually a Dask DataFrame
					if use_dask and isinstance(st.session_state.df, dd.DataFrame):
						with st.spinner('Computing data for CSV export...'):
							# Convert Dask DataFrame to pandas for export
							df_export = st.session_state.df.compute()
							csv_data = df_export.to_csv(index=False).encode('utf-8')
							row_count = len(df_export)

					else:
						csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
						row_count = len(st.session_state.df)

					st.download_button(
						label=f"Download as CSV ({row_count} rows)",
						data=csv_data,
						file_name=export_filename,
						mime="text/csv",
						key="download_csv"
					)
				except Exception as e:
					st.error(f"Error preparing CSV for download: {e}")


			elif export_format == "Parquet":
				try:
					# Use BytesIO to create an in-memory parquet file
					buffer = BytesIO()

					# Check if Dask was used to load the data
					use_dask = st.session_state.get('use_dask', False)

					# Only compute if it's actually a Dask DataFrame
					if use_dask and isinstance(st.session_state.df, dd.DataFrame):
						with st.spinner('Computing data for Parquet export...'):
							# Convert Dask DataFrame to pandas for export
							df_export = st.session_state.df.compute()
							df_export.to_parquet(buffer, index=False)
							row_count = len(df_export)

					else:
						st.session_state.df.to_parquet(buffer, index=False)
						row_count = len(st.session_state.df)

					buffer.seek(0)
					st.download_button(
						label=f"Download as Parquet ({row_count} rows)",
						data=buffer,
						file_name=export_filename,
						mime="application/octet-stream", # Generic binary stream
						key="download_parquet"
					)

				except Exception as e:
					st.error(f"Error preparing Parquet for download: {e}")

	def _show_data_explorer_view(self):
		"""Handles the display of the main content area with tabs for data exploration and analysis."""

		def _show_dataset_info():

			# Display Dataset Info if loaded
			st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
			st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)

			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Module", st.session_state.selected_module or "N/A")
				st.metric("Table", st.session_state.selected_table or "N/A")

			with col2:
				# Format file size
				file_size_mb = st.session_state.file_sizes.get((st.session_state.selected_module, st.session_state.selected_table), 0)
				# if file_size_mb < 0.1: size_str = f"{file_size_mb*1024:.0f} KB"
				# elif file_size_mb < 1024: size_str = f"{file_size_mb:.1f} MB"
				# else: size_str = f"{file_size_mb/1024:.1f} GB"
				st.metric("File Size (Full)", humanize.naturalsize(file_size_mb))
				st.metric("Total Rows (Full)", f"{st.session_state.total_row_count:,}")

			with col3:
				st.metric("Rows Loaded", f"{len(st.session_state.df):,}")
				st.metric("Columns Loaded", f"{len(st.session_state.df.columns)}")

			# Display filename
			if st.session_state.current_file_path:
				st.caption(f"Source File: {os.path.basename(st.session_state.current_file_path)}")

			st.markdown("</div>", unsafe_allow_html=True)


		# Welcome message or Data Info
		if st.session_state.df is None:
			# Welcome message when no data is loaded
			st.title("Welcome to the MIMIC-IV Data Explorer & Analyzer")
			st.markdown("""
			<div class='info-box'>
			<p>This tool allows you to load, explore, visualize, and analyze tables from the MIMIC-IV dataset.</p>
			<p>To get started:</p>
			<ol>
				<li>Enter the path to your local MIMIC-IV v3.1 dataset in the sidebar.</li>
				<li>Click "Scan MIMIC-IV Directory" to find available tables.</li>
				<li>Select a module (e.g., 'hosp', 'icu') and a table.</li>
				<li>Choose sampling options if needed.</li>
				<li>Click "Load Selected Table".</li>
			</ol>
			<p>Once data is loaded, you can use the tabs below to explore, engineer features, perform clustering, and analyze the results.</p>
			<p><i>Note: You need access to the MIMIC-IV dataset (v3.1 recommended) downloaded locally.</i></p>
			</div>
			""", unsafe_allow_html=True)

			# About MIMIC-IV Section
			with st.expander("About MIMIC-IV"):
				st.markdown("""
				<p>MIMIC-IV (Medical Information Mart for Intensive Care IV) is a large, freely-available database comprising deidentified health-related data associated with patients who stayed in critical care units at the Beth Israel Deaconess Medical Center between 2008 - 2019.</p>
				<p>The database is organized into modules:</p>
				<ul>
					<li><strong>Hospital (hosp)</strong>: Hospital-wide EHR data (admissions, diagnoses, labs, prescriptions, etc.).</li>
					<li><strong>ICU (icu)</strong>: High-resolution ICU data (vitals, ventilator settings, inputs/outputs, etc.).</li>
					<li><strong>ED (ed)</strong>: Emergency department data.</li>
					<li><strong>CXRN (cxrn)</strong>: Chest X-ray reports (requires separate credentialing).</li>
				</ul>
				<p>For more information, visit the <a href="https://physionet.org/content/mimiciv/3.1/" target="_blank">MIMIC-IV PhysioNet page</a>.</p>
				""", unsafe_allow_html=True)

		else:
			_show_dataset_info()

			# Create tabs for different functionalities
			tab_titles = [
				"📊 Exploration & Viz",
				"🛠️ Feature Engineering",
				"🧩 Clustering Analysis",
				"💡 Cluster Interpretation", # Renamed for clarity
				"💾 Export Options"
			]
			tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

			# Tab 1: Exploration & Visualization
			with tab1:
				st.markdown("<h2 class='sub-header'>Data Exploration & Visualization</h2>", unsafe_allow_html=True)
				# Check if Dask was used to load the data
				use_dask = st.session_state.get('use_dask', False)

				# Pass the use_dask parameter to all visualizer methods
				MIMICVisualizerUtils.display_data_preview(st.session_state.df, use_dask=use_dask)
				MIMICVisualizerUtils.display_dataset_statistics(st.session_state.df, use_dask=use_dask)

				try:
					st.info(f'dataframe size: {st.session_state.df.size / len(st.session_state.df.columns)}')
				except Exception as e:
					st.error(f"Error calculating dataframe size: {e}")

				MIMICVisualizerUtils.display_visualizations(st.session_state.df, use_dask=use_dask)

			with tab2:
				FeatureEngineeringTab().render()

			with tab3:
				ClusteringAnalysisTab().render()

			with tab4:
				AnalysisVisualizationTab().render()

			# Tab 5: Export Options
			with tab5:
				self._export_options()


	def run(self):
		"""Run the main application loop."""

		logger.info("Starting MIMICDashboardApp run...")

		# Set page config (do this only once at the start)
		st.set_page_config( page_title="MIMIC-IV Explorer", page_icon="🏥", layout="wide", initial_sidebar_state="expanded" )

		# Custom CSS for better styling
		st.markdown("""
			<style>
			.main .block-container {padding-top: 2rem; padding-bottom: 2rem; padding-left: 5rem; padding-right: 5rem;}
			.sub-header {margin-top: 20px; margin-bottom: 10px; color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px;}
			h3 {margin-top: 15px; margin-bottom: 10px; color: #333;}
			h4 {margin-top: 10px; margin-bottom: 5px; color: #555;}
			.info-box {
				background-color: #eef2f7; /* Lighter blue */
				border-radius: 5px;
				padding: 15px;
				margin-bottom: 15px;
				border-left: 5px solid #1E88E5; /* Blue left border */
				font-size: 0.95em;
			}
			.stTabs [data-baseweb="tab-list"] {
				gap: 12px; /* Smaller gap between tabs */
			}
			.stTabs [data-baseweb="tab"] {
				height: 45px;
				white-space: pre-wrap;
				background-color: #f0f2f6;
				border-radius: 4px 4px 0px 0px;
				gap: 1px;
				padding: 10px 15px; /* Adjust padding */
				font-size: 0.9em; /* Slightly smaller font */
			}
			.stTabs [aria-selected="true"] {
				background-color: #ffffff; /* White background for selected tab */
				font-weight: bold;
			}
			.stButton>button {
				border-radius: 4px;
				padding: 8px 16px;
			}
			.stMultiSelect > div > div {
				border-radius: 4px;
			}
			.stDataFrame {
				border: 1px solid #eee;
				border-radius: 4px;
			}
			</style>
			""", unsafe_allow_html=True)

		# Display the sidebar
		self._display_sidebar()

		# Display the selected view (Data Explorer or Filtering)
		if st.session_state.current_view == 'data_explorer':
			self._show_data_explorer_view()

		else:
			st.title("Cohort Filtering Configuration")
			# Ensure necessary components are passed if FilteringTab needs them
			# FilteringTab(current_file_path=self.current_file_path).render(data_handler=self.data_handler, feature_engineer=self.feature_engineer)

		logger.info("MIMICDashboardApp run finished.")


	@staticmethod
	def init_session_state():
		""" Function to initialize session state """
		# Check if already initialized (e.g., during Streamlit rerun)
		if 'app_initialized' in st.session_state:
			return

		logger.info("Initializing session state...")
		# Basic App State
		st.session_state.loader = None
		st.session_state.datasets = {}
		st.session_state.selected_module = None
		st.session_state.selected_table = None
		st.session_state.df = None
		st.session_state.available_tables = {}
		st.session_state.file_paths = {}
		st.session_state.file_sizes = {}
		st.session_state.table_display_names = {}
		st.session_state.mimic_path = DEFAULT_MIMIC_PATH
		st.session_state.total_row_count = 0
		st.session_state.use_dask = True
		st.session_state.current_view = 'data_explorer'

		# Feature engineering states
		st.session_state.detected_order_cols = []
		st.session_state.detected_time_cols = []
		st.session_state.detected_patient_id_col = None
		st.session_state.freq_matrix = None
		st.session_state.order_sequences = None
		st.session_state.timing_features = None
		st.session_state.order_dist = None
		st.session_state.patient_order_dist = None
		st.session_state.transition_matrix = None
		st.session_state.include_transfers = False

		# Clustering states
		st.session_state.clustering_input_data = None # Holds the final data used for clustering (post-preprocessing)
		st.session_state.reduced_data = None         # Holds dimensionality-reduced data
		st.session_state.kmeans_labels = None
		st.session_state.hierarchical_labels = None
		st.session_state.dbscan_labels = None
		st.session_state.lda_results = None          # Dictionary to hold LDA outputs
		st.session_state.cluster_metrics = {}        # Store metrics like {'kmeans': {...}, 'dbscan': {...}}
		st.session_state.optimal_k = None
		st.session_state.optimal_eps = None

		# Analysis states (Post-clustering)
		st.session_state.length_of_stay = None

		# Filtering states
		st.session_state.filter_params = {
			'apply_encounter_timeframe' : False, 'encounter_timeframe'            : [],    # Default to off
			'apply_age_range'           : False, 'min_age'                        : 18,    'max_age': 90, # Default to off
			'apply_t2dm_diagnosis'      : False, 'apply_valid_admission_discharge': False,
			'apply_inpatient_stay'      : False, 'admission_types'                : [],
			'require_inpatient_transfer': False, 'required_inpatient_units'       : [],
			'exclude_in_hospital_death' : False
		}

		st.session_state.app_initialized = True # Mark as initialized
		logger.info("Session state initialized.")

	@property
	def is_selected_table_parquet(self) -> bool:
		"""Check if the selected table is in Parquet format."""
		if not st.session_state.get('selected_table') or st.session_state.selected_table == "merged_table":
			return False

		file_path = st.session_state.file_paths.get((st.session_state.selected_module, st.session_state.selected_table))
		if file_path and isinstance(file_path, Path) and file_path.suffix == '.parquet':
			return True
		return False

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
		tables_that_can_be_sampled = [	"merged_table" ] + [table.value for table in self.data_handler._list_of_tables_w_subject_id_column]
		return st.session_state.selected_table not in tables_that_can_be_sampled

	def _convert_table_to_parquet(self, tables_to_process: Optional[List[TableNames]] = None):
		"""Callback to convert the selected table to Parquet format."""

		selected_table = st.session_state.selected_table
		selected_module = st.session_state.selected_module

		if tables_to_process is None:
			tables_to_process = [ TableNames(selected_table) ]

		if not tables_to_process:
			st.session_state.conversion_status = {'type': 'warning', 'message': "No tables to process."}
			return

		try:
			with st.spinner(f"Converting {len(tables_to_process)} table(s) to Parquet..."):
				for table_enum in tables_to_process:
					self.parquet_converter.save_as_parquet(table_name=table_enum)

			if self._rescan_and_update_state():
				st.session_state.conversion_status = {'type': 'success', 'message': f"Successfully converted {len(tables_to_process)} table(s)!"}
			else:
				st.session_state.conversion_status = {'type': 'error', 'message': "Conversion might have failed. Could not find updated tables."}


		except Exception as e:
			logger.error(f"Parquet conversion job failed: {e}", exc_info=True)
			st.session_state.conversion_status = {'type': 'exception', 'message': e}

		st.session_state.selected_table = selected_table
		st.session_state.selected_module = selected_module



def main():
    app = MIMICDashboardApp()
    app.run()


if __name__ == "__main__":
    main()
