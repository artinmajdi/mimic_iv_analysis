# Standard library imports
import os
import logging
import datetime
from io import BytesIO
from typing import Tuple, Optional

# Data processing imports
import numpy as np
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning imports
from scipy.cluster.hierarchy import dendrogram
from scipy import stats

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

# Streamlit import
import streamlit as st

# Local application imports
from mimic_iv_analysis.core import FeatureEngineerUtils, DataLoader
from mimic_iv_analysis.visualization.app_components import FilteringTab, FeatureEngineeringTab, AnalysisVisualizationTab, ClusteringAnalysisTab

from mimic_iv_analysis.visualization.visualizer_utils import MIMICVisualizerUtils


# Constants
DEFAULT_MIMIC_PATH      = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1" # Adjust if necessary
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MIMICDashboardApp:

	def __init__(self):
		logging.info("Initializing MIMICDashboardApp...")
		# Initialize core components
		self.data_handler     = DataLoader()
		self.feature_engineer = FeatureEngineerUtils()

		# Initialize UI components for tabs
		self.feature_engineering_ui    = None
		self.clustering_analysis_ui    = None
		self.analysis_visualization_ui = None

		# Initialize session state
		self.current_file_path = None

		self.init_session_state()
		logging.info("MIMICDashboardApp initialized.")


	@staticmethod
	def init_session_state():
		""" Function to initialize session state """
		# Check if already initialized (e.g., during Streamlit rerun)
		if 'app_initialized' in st.session_state:
			return

		logging.info("Initializing session state...")
		# Basic App State
		st.session_state.loader = None
		st.session_state.datasets = {}
		st.session_state.selected_module = None
		st.session_state.selected_table = None
		st.session_state.df = None
		st.session_state.sample_size = DEFAULT_SAMPLE_SIZE
		st.session_state.available_tables = {}
		st.session_state.file_paths = {}
		st.session_state.file_sizes = {}
		st.session_state.table_display_names = {}
		st.session_state.mimic_path = DEFAULT_MIMIC_PATH
		st.session_state.total_row_count = 0
		st.session_state.use_dask = False
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
		logging.info("Session state initialized.")

	def run(self):
		"""Run the main application loop."""
		logging.info("Starting MIMICDashboardApp run...")
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
			FilteringTab(current_file_path=self.current_file_path).render() # Assuming FilteringTab uses session_state or needs no args

		logging.info("MIMICDashboardApp run finished.")

	def _scan_dataset(self):

		st.sidebar.markdown("---") # Separator
		st.sidebar.markdown("## Dataset Configuration")

		# MIMIC-IV path input
		mimic_path = st.sidebar.text_input(
			"MIMIC-IV Dataset Path",
			value=st.session_state.mimic_path,
			help="Enter the path to your local MIMIC-IV v3.1 dataset directory"
		)

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
		if st.sidebar.button("Scan MIMIC-IV Directory", key="scan_button"):
			if not mimic_path or not os.path.isdir(mimic_path):
				st.sidebar.error("Please enter a valid directory path for the MIMIC-IV dataset")
			else:
				with st.spinner("Scanning directory..."):
					try:
						# Scan the directory structure using the data handler
						available_tables, file_paths, file_sizes, table_display_names = self.data_handler.scan_mimic_directory(mimic_path)

						if available_tables:
							st.session_state.available_tables = available_tables
							st.session_state.file_paths = file_paths
							st.session_state.file_sizes = file_sizes
							st.session_state.table_display_names = table_display_names
							st.sidebar.success(f"Found {sum(len(tables) for tables in available_tables.values())} tables in {len(available_tables)} modules")
							# Reset selections if scan is successful
							st.session_state.selected_module = list(available_tables.keys())[0] if available_tables else None
							st.session_state.selected_table = None # Force user to select table after scan
						else:
							st.sidebar.error("No MIMIC-IV tables (.csv, .csv.gz, .parquet) found in the specified path or its subdirectories (hosp, icu).")
							st.session_state.available_tables = {} # Clear previous results

					except AttributeError:
						st.sidebar.error("Data Handler is not initialized or does not have a 'scan_mimic_directory' method.")
					except Exception as e:
						st.sidebar.error(f"Error scanning directory: {e}")
						logging.exception("Error during directory scan")

	def _display_sidebar(self):
		"""Handles the display and logic of the sidebar components."""

		# Logo and Title
		st.sidebar.image("https://physionet.org/static/images/mimic-logo.png", width=150) # Add logo maybe
		st.sidebar.title("MIMIC-IV Navigator")

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

		self._scan_dataset()


		# Module and table selection (only show if available_tables is populated)
		if st.session_state.available_tables:
			# Module selection
			# Use index=0 safely as we check available_tables isn't empty
			module_options = list(st.session_state.available_tables.keys())
			selected_module_index = module_options.index(st.session_state.selected_module) if st.session_state.selected_module in module_options else 0
			module = st.sidebar.selectbox(
				"Select Module",
				module_options,
				index=selected_module_index,
				key="module_select",
				help="Select which MIMIC-IV module to explore (e.g., hosp, icu)"
			)
			# Update selected module if changed
			if module != st.session_state.selected_module:
				st.session_state.selected_module = module
				st.session_state.selected_table = None # Reset table selection when module changes


			# Table selection based on selected module
			if module in st.session_state.available_tables:

				# Select Table
				selected_display = self._select_table(module=module)

				# Load Table(s)
				st.sidebar.markdown("---")

				# Sampling options
				load_full = st.sidebar.checkbox("Load Full Table (Ignore Sampling)", value=False, key="load_full")

				st.session_state.sample_size = st.sidebar.number_input(
					"Sample Size (Rows)",
					min_value = 100,
					max_value = 6000000,
					disabled  = load_full,
					key       = "sample_size_input",
					step      = 100,
					value     = st.session_state.sample_size,
					help      = "Number of rows to load if 'Load Full Table' is unchecked. Applied randomly."
				)

				# Encoding (less critical for parquet, but keep for CSV)
				encoding = st.sidebar.selectbox("Encoding (for CSV)", ["utf-8", "latin-1", "iso-8859-1"], index=0, key="encoding_select")

				# Dask option (only relevant if dask is installed and intended)
				st.session_state.use_dask = st.sidebar.checkbox("Use Dask for large files", value=st.session_state.use_dask, help="Enable Dask for potentially faster processing of very large files (requires Dask installation)")

			self._load_table(encoding=encoding, load_full=load_full, selected_display=selected_display)

		else:
			st.sidebar.info("Scan a MIMIC-IV directory to select and load tables.")

	def _select_table(self, module: str) -> str:
		"""Display table selection dropdown and handle selection logic.

		Args:
			module: The currently selected module (e.g., 'hosp', 'icu')

		Returns:
			str: The selected display name for the table
		"""
		# TODO: need to fix the merged table mode. currently it is not working.

		# Get sorted table options for the selected module
		table_options = sorted(st.session_state.available_tables[module])

		# Create display options list with the special merged_table option first
		tables_list_w_size_info = ["merged_table"]

		# Create display-to-table mapping for reverse lookup
		display_to_table_map = {}

		# Format each table with size information
		for table in table_options:
			display_name = st.session_state.table_display_names.get((module, table), table)
			size_mb      = st.session_state.file_sizes.get((module, table), 0)
			# Format size as MB or KB based on value
			size_str     = f"{size_mb:.1f} MB" if size_mb > 0.1 else f"{int(size_mb*1024)} KB"

			# Create the display string and add to options
			display_string = f"{display_name} ({size_str})"
			tables_list_w_size_info.append(display_string)

			# Map display string back to table name
			display_to_table_map[display_string] = table

		# Determine default selection index (prefer 'poe' table if available)
		default_index = table_options.index('poe') if 'poe' in table_options else 0

		# Display the table selection dropdown
		selected_table_w_size_info = st.sidebar.selectbox(
			label   = "Select Table",
			options = tables_list_w_size_info,
			index   = default_index,
			key     = "table_select",
			help    = "Select which table to load (file size shown in parentheses)"
		)

		# Determine the actual table name from the selected display
		table = None if selected_table_w_size_info == "merged_table" else display_to_table_map[selected_table_w_size_info]

		# Update session state if table selection changed
		if table != st.session_state.selected_table:
			st.session_state.selected_table = table
			st.session_state.df = None  # Clear dataframe when table changes

		# Show table description if a regular table is selected
		if st.session_state.selected_table:
			self._display_table_info(module, st.session_state.selected_table)

		return selected_table_w_size_info

	def _display_table_info(self, module: str, table: str) -> None:
		"""Display table description information in sidebar.

		Args:
			module: The selected module
			table: The selected table name
		"""
		try:
			table_info = self.data_handler.get_table_info(module, table)
			if table_info:
				st.sidebar.markdown(
					f"**Description:** {table_info}",
					help="Table description from MIMIC-IV documentation."
				)
		except AttributeError:
			st.sidebar.warning("Could not retrieve table info (get_table_info method missing).")
		except Exception as e:
			st.sidebar.warning(f"Could not retrieve table info: {e}")

	def _load_table(self, encoding: str = 'latin-1', load_full: bool = False, selected_display: str = None) -> Tuple[Optional[pd.DataFrame], int]:
		"""Load a specific MIMIC-IV table, handling large files and sampling."""

		if st.sidebar.button("Load Selected Table", key="load_button"):

			# Normal case for regular table loading
			if not st.session_state.selected_module or not st.session_state.selected_table:
				st.sidebar.warning("Please select a module and table first.")
				return

			# Determine sample size to use
			load_sample_size = None if load_full else st.session_state.sample_size

			# Framework info (currently just Pandas, add Dask logic if implemented)
			framework       = "Dask" if st.session_state.use_dask else "Pandas"
			loading_message = f"Loading { 'full table' if load_full else f'{load_sample_size} rows (sampled)' } using {framework}..."

			# TODO: see if I can merge this two into one. so that I can be sure that the merged table is shown everywhere.

			# Special case for "merged_table" option
			if selected_display == "merged_table":

				dataset_path = st.session_state.mimic_path
				if not dataset_path or not os.path.exists(dataset_path):
					st.sidebar.error(f"MIMIC-IV directory not found: {dataset_path}. Please set correct path and re-scan.")
					return

				with st.spinner(loading_message):

					# Load connected tables with merged view
					with st.spinner("Loading and merging connected tables..."):
						result = self.data_handler.load_connected_tables( mimic_path=dataset_path, sample_size=load_sample_size, encoding=encoding, use_dask=st.session_state.use_dask, merged_view=True )

					# Unpack the result (tables dict and merged dataframe)
					if len(result) == 2:  # Successful with merged view
						connected_tables, merged_df = result

						# Store all tables in session state for later access
						st.session_state.connected_tables = connected_tables

						# Check if we have a valid merged dataframe
						if merged_df.empty:
							st.sidebar.error("Failed to load connected tables.")
							return

						# Set the merged dataframe for display
						st.session_state.df                 = merged_df
						st.session_state.total_rows         = len(merged_df)
						st.session_state.loaded_sample_size = len(merged_df)
						st.session_state.current_file_path  = "merged_tables"
						st.session_state.table_display_name = "Merged MIMIC-IV View"

						# Clear previous analysis states
						self._clear_analysis_states()

						# Show success message with table info
						tables_loaded = [name for name, df in connected_tables.items() if not df.empty]
						total_tables = len(tables_loaded)
						st.sidebar.success(f"Successfully merged {total_tables} tables with {len(merged_df.columns)} columns and {len(merged_df)} rows!")

			else:
				file_path = st.session_state.file_paths.get((st.session_state.selected_module, st.session_state.selected_table))

				if not file_path or not os.path.exists(file_path):
					st.sidebar.error(f"File path not found for {st.session_state.selected_module}/{st.session_state.selected_table}. Please re-scan.")
					return

				st.session_state.current_file_path = file_path

				with st.spinner(loading_message):

					df, total_rows = self.data_handler.load_mimic_table( file_path=file_path, sample_size=load_sample_size, encoding=encoding, use_dask=st.session_state.use_dask )

					st.session_state.total_row_count = total_rows

					if df is not None and not df.empty:
						st.session_state.df = df
						st.sidebar.success(f"Loaded {len(df)} rows out of {total_rows}.")

						# Clear previous analysis results when new data is loaded
						self._clear_analysis_states()

						# Auto-detect columns for feature engineering
						st.session_state.detected_order_cols     = FeatureEngineerUtils.detect_order_columns(df)
						st.session_state.detected_time_cols      = FeatureEngineerUtils.detect_temporal_columns(df)
						st.session_state.detected_patient_id_col = FeatureEngineerUtils.detect_patient_id_column(df)

						st.sidebar.write("Detected Columns (for Feature Eng):")
						st.sidebar.caption(f"Patient ID: {st.session_state.detected_patient_id_col}, Order: {st.session_state.detected_order_cols}, Time: {st.session_state.detected_time_cols}")


					elif df is not None and df.empty:
						st.sidebar.warning("Loaded table is empty.")
						st.session_state.df = None # Set to None if empty

					else: # df is None
						st.sidebar.error("Failed to load table. Check logs or file format.")
						st.session_state.df = None

	def _clear_analysis_states(self):
		"""Clears session state related to previous analysis when new data is loaded."""
		logging.info("Clearing previous analysis states...")
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

	def _show_data_explorer_view(self):
		"""Handles the display of the main content area with tabs for data exploration and analysis."""

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
				if file_size_mb < 0.1: size_str = f"{file_size_mb*1024:.0f} KB"
				elif file_size_mb < 1024: size_str = f"{file_size_mb:.1f} MB"
				else: size_str = f"{file_size_mb/1024:.1f} GB"
				st.metric("File Size (Full)", size_str)
				st.metric("Total Rows (Full)", f"{st.session_state.total_row_count:,}")
			with col3:
				st.metric("Rows Loaded", f"{len(st.session_state.df):,}")
				st.metric("Columns Loaded", f"{len(st.session_state.df.columns)}")

			# Display filename
			if st.session_state.current_file_path:
				st.caption(f"Source File: {os.path.basename(st.session_state.current_file_path)}")
			st.markdown("</div>", unsafe_allow_html=True)


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
				MIMICVisualizerUtils.display_visualizations(st.session_state.df, use_dask=use_dask)


			# Tab 2: Feature Engineering
			with tab2:
				FeatureEngineeringTab().render()

			# Tab 3: Clustering Analysis
			with tab3:
				ClusteringAnalysisTab().render()

			# Tab 4: Analysis & Visualization (Cluster Interpretation)
			with tab4:
				AnalysisVisualizationTab().render()

			# Tab 5: Export Options
			with tab5:
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
							if use_dask and hasattr(st.session_state.df, 'compute'):
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
							if use_dask and hasattr(st.session_state.df, 'compute'):
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


def main():
	app = MIMICDashboardApp()
	app.run()

if __name__ == "__main__":
	main()
