# Standard library imports
from pathlib import Path

# Data processing imports
import dask.dataframe as dd

# Streamlit import
import streamlit as st
import humanize


# Local application imports
from mimic_iv_analysis import logger
from mimic_iv_analysis.configurations import TableNames, DEFAULT_MIMIC_PATH

from mimic_iv_analysis.visualization.app_components import FeatureEngineeringTab, AnalysisVisualizationTab, ClusteringAnalysisTab, SideBar

from mimic_iv_analysis.visualization.app_components.exploration_and_viz import ExplorationAndViz

# TODO: fix the merge table loading and then update the local parquet file.
# TODO: see why it saves the parquet in only one partition and if i increase it, will it improve the performance?
# TODO: Add the prescriptions , labevents, and d_labitems into the study tables and merging scheme
# TODO: Generate a sphinx documentation for this.

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

		SideBar.init_dask_client()

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
				if st.session_state.selected_table != TableNames.MERGED.value:
					file_size_mb = st.session_state.file_sizes.get((st.session_state.selected_module, st.session_state.selected_table), 0)
					st.metric("File Size (Full)", humanize.naturalsize(file_size_mb))
				else:
					st.metric("File Size (Merged)", "N/A")

			with col2:
				if st.session_state.get('total_subjects_count', None) is None:
					unique_subject_ids = self.sidebar.data_handler.get_unique_subject_ids(table_name=TableNames(st.session_state.selected_table), recalculate_subject_ids=False)
					st.session_state.total_subjects_count =  len(unique_subject_ids)

				st.metric("Total Subjects", f"{st.session_state.total_subjects_count:,}")

				if st.session_state.selected_table in TableNames._TABLES_W_SUBJECT_ID_COLUMN:
					# df2 = st.session_state.df.compute()
					loaded_subjects = st.session_state.df.subject_id.nunique().compute() if isinstance(st.session_state.df, dd.DataFrame) else len(st.session_state.df.subject_id.unique()) if st.session_state.df is not None else 0
					st.metric("Subjects Loaded", f"{loaded_subjects:,}")

			with col3:
				st.metric("Rows Loaded", f"{st.session_state.df_length:,}")
				st.metric("Columns Loaded", f"{len(st.session_state.df.columns)}")

			# Display filename
			if st.session_state.current_file_path:
				st.caption(f"Source File: {Path(st.session_state.current_file_path).name}")

			st.markdown("</div>", unsafe_allow_html=True)

		# Display the sidebar
		self.sidebar = SideBar()
		self.sidebar.render()

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
				"💡 Cluster Interpretation"
			]
			tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

			# Tab 1: Exploration & Visualization
			with tab1:
				ExplorationAndViz().render()

			with tab2:
				FeatureEngineeringTab().render()

			with tab3:
				ClusteringAnalysisTab().render()

			with tab4:
				AnalysisVisualizationTab().render()


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
		st.session_state.df_length = 0  # Cached DataFrame length to avoid repeated computation
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
				'selected_columns'      : ["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime", "order_type", "order_subtype"],
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
