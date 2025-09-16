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


# Local application imports
from mimic_iv_analysis import logger
from mimic_iv_analysis.core import FeatureEngineerUtils
from mimic_iv_analysis.core.dask_config_optimizer import DaskConfigOptimizer
from mimic_iv_analysis.io import DataLoader, ParquetConverter
from mimic_iv_analysis.configurations import TableNames, DEFAULT_MIMIC_PATH, DEFAULT_NUM_SUBJECTS, DEFAULT_STUDY_TABLES_LIST

from mimic_iv_analysis.visualization.app_components import FilteringTab, FeatureEngineeringTab, AnalysisVisualizationTab, ClusteringAnalysisTab, SideBar

from mimic_iv_analysis.visualization.app_components.exploration_and_viz import ExplorationAndViz

# TODO: Generate a sphinx documentation for this.
# TODO: Can i show the dask dashboard inside streamlit UI?
# TODO: Add the option to save the Full merged table and load it when available instead of re-merging the tables. Use a hash system using the table names that are used in that merge
# TODO: the partial loading is not working for poe table.
# TODO: also need to check the convert to parquet again to see if it still works.
class MIMICDashboardApp:

	def __init__(self):
		logger.info("Initializing MIMICDashboardApp...")
		self.init_session_state()



		logger.info("Initializing FeatureEngineerUtils...")
		# self.feature_engineer  = FeatureEngineerUtils()

		# Initialize session state
		self.current_file_path = None

		# self.init_session_state()
		logger.info("MIMICDashboardApp initialized.")

		self.__init_dask_client()

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
								dashboard_address  = f":{dashboard_port}", )
			return Client(cluster)

		# Initialize default values if not in session state with conservative settings
		if 'dask_n_workers' not in st.session_state:
			st.session_state.dask_n_workers = 2  # Reduced from 1 to 2 for better parallelism
		if 'dask_threads_per_worker' not in st.session_state:
			st.session_state.dask_threads_per_worker = 4  # Reduced from 16 to 4 to prevent memory overload
		if 'dask_memory_limit' not in st.session_state:
			st.session_state.dask_memory_limit = '8GB'  # Reduced from 20GB to 8GB for safer memory usage
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

		self.dask_client = st.session_state.dask_client

	def _prepare_csv_download(self):
		"""Prepare CSV data on-demand with progress tracking."""
		try:
			use_dask = st.session_state.get('use_dask', False)

			if use_dask and isinstance(st.session_state.df, dd.DataFrame):
				# Create progress bar
				progress_bar = st.progress(0)
				status_text = st.empty()

				status_text.text('Initializing CSV export...')
				progress_bar.progress(10)

				# Calculate row count during preparation
				status_text.text('Calculating data size...')
				progress_bar.progress(20)
				try:
					row_count = len(st.session_state.df)
				except:
					row_count = "Unknown"

				status_text.text(f'Preparing {row_count} rows for export...')
				progress_bar.progress(30)

				# Use Dask's native to_csv with temporary file
				import tempfile
				with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
					tmp_path = tmp_file.name

				status_text.text('Writing data to temporary file...')
				progress_bar.progress(50)

				# Dask writes directly to file without computing entire DataFrame
				st.session_state.df.to_csv(tmp_path, index=False, single_file=True)

				status_text.text('Reading file content...')
				progress_bar.progress(80)

				# Read the file content and clean up
				with open(tmp_path, 'r', encoding='utf-8') as f:
					csv_data = f.read().encode('utf-8')

				status_text.text('Cleaning up temporary files...')
				progress_bar.progress(90)

				# Clean up temporary file
				Path(tmp_path).unlink(missing_ok=True)

				status_text.text(f'CSV export completed! ({row_count} rows)')
				progress_bar.progress(100)

				# Clear progress indicators after a short delay
				import time
				time.sleep(2)
				status_text.empty()
				progress_bar.empty()

				return csv_data
			else:
				# For pandas DataFrames, show simple progress
				progress_bar = st.progress(0)
				status_text = st.empty()

				status_text.text('Calculating data size...')
				progress_bar.progress(20)

				row_count = len(st.session_state.df)

				status_text.text(f'Preparing CSV export for {row_count} rows...')
				progress_bar.progress(50)

				csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')

				status_text.text(f'CSV export completed! ({row_count} rows)')
				progress_bar.progress(100)

				# Clear progress indicators
				import time
				time.sleep(2)
				status_text.empty()
				progress_bar.empty()

				return csv_data

		except Exception as e:
			st.error(f"Error preparing CSV download: {e}")
			return b""  # Return empty bytes on error

	def _export_options(self):
		st.markdown("<h2 class='sub-header'>Export Loaded Data</h2>", unsafe_allow_html=True)
		st.info("Export the currently loaded (and potentially sampled) data shown in the 'Exploration' tab.")

		export_filename = f"mimic_data_{st.session_state.get('selected_table', 'table')}.csv"

		# CSV Download button without pre-calculating row count
		if st.button("Prepare CSV Download", key="download_csv_button"):
			if st.download_button(
					label=f"Click to download CSV file",
					data=self._prepare_csv_download(),
					file_name=export_filename,
					mime="text/csv",
					key="download_complete_csv",
					help="Download the complete dataset as CSV file (memory optimized with Dask)" ):

				st.success(f"CSV export completed!")

	def _show_tabs(self):
		"""Handles the display of the main content area with tabs for data exploration and analysis."""

		def _show_dataset_info():

			# Display Dataset Info if loaded
			st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
			st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)

			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Module", st.session_state.selected_module or "N/A")

				# Format file size
				file_size_mb = st.session_state.file_sizes.get((st.session_state.selected_module, st.session_state.selected_table), 0)
				st.metric("File Size (Full)", humanize.naturalsize(file_size_mb))

			with col2:
				st.metric("Total Subjects", f"{st.session_state.total_subjects_count:,}")

				loaded_subjects = st.session_state.df.subject_id.nunique().compute() if isinstance(st.session_state.df, dd.DataFrame) else len(st.session_state.df.subject_id.unique()) if st.session_state.df is not None else 0
				st.metric("Subjects Loaded", f"{loaded_subjects:,}")

			with col3:
				st.metric("Rows Loaded", f"{len(st.session_state.df):,}")
				st.metric("Columns Loaded", f"{len(st.session_state.df.columns)}")

			# Display filename
			if st.session_state.current_file_path:
				st.caption(f"Source File: {Path(st.session_state.current_file_path).name}")

			st.markdown("</div>", unsafe_allow_html=True)

		# Display the sidebar
		SideBar().render()

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
				ExplorationAndViz().render()

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

		# Display the selected view (Data Explorer or Filtering)
		self._show_tabs()
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
		st.session_state.use_dask = True
		st.session_state.apply_filtering = True

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

			TableNames.POE.value: {
				'selected_columns'      : ["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime", "order_type"],
				'apply_order_type'      : False,
				'order_type'            : [],
				'apply_transaction_type': False,
				'transaction_type'      : []},


			TableNames.ADMISSIONS.value: {
				'selected_columns'         : ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admit_provider_id", "admission_location", "discharge_location", "hospital_expire_flag"],
				'valid_admission_discharge': True,
				'exclude_in_hospital_death': True,
				'discharge_after_admission': True,
				'apply_admission_type'     : False,
				'admission_type'           : [],
				'apply_admission_location' : False,
				'admission_location'       : []}
		}

		st.session_state.app_initialized = True # Mark as initialized
		logger.info("Session state initialized.")


def main():
	app = MIMICDashboardApp()
	app.run()


if __name__ == "__main__":
	main()
