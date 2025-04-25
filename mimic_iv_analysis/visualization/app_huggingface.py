# Standard library imports
import os
import logging
import datetime
import pickle
from io import BytesIO

# Data processing imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning imports
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation

# Streamlit import
import streamlit as st

from mimic_iv_analysis.core import MIMICClusteringAnalysis, MIMICClusterAnalyzer, MIMICFeatureEngineer, MIMICDataLoader, MIMICVisualizer
from mimic_iv_analysis.visualization.app_components import FilteringTab


# TODO: Remove the _display_analysis_visualization_tab() function from here as well as the MIMICClusterAnalyzer class from the clustering.py. thn run the following command in claude with the remaining code:

# Please enhance my MIMIC-IV Streamlit dashboard (src/visualization/app_huggingface.py) by adding the following advanced analytics modules. Implement each step sequentially, ensuring the code is well-structured and documented:
# 	3. Analysis & Visualization
# 	- Implement length of stay comparison across identified clusters
# 	- Create interactive visualizations of order patterns using Plotly
# 	- Add statistical significance testing between clusters
# 	- Generate downloadable cluster characterization reports
# 	- Include heatmaps of feature importance for each cluster
# old claude link: https://claude.ai/chat/fce25354-341f-4228-aa56-fe71e406a08f


# Constants
DEFAULT_MIMIC_PATH      = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MIMICDashboardApp:
	def __init__(self):
		logging.info("Initializing MIMICDashboardApp...")
		self.data_handler        = MIMICDataLoader()
		self.visualizer          = MIMICVisualizer()
		self.feature_engineer    = MIMICFeatureEngineer()
		self.clustering_analyzer = MIMICClusteringAnalysis()
		self.cluster_analyzer    = MIMICClusterAnalyzer()
		self.filtering_tab       = FilteringTab()
		self.init_session_state()
		logging.info("MIMICDashboardApp initialized.")


	@staticmethod
	def init_session_state():
		""" Function to initialize session state """
		logging.info("Initializing session state...")
		if 'loader' not in st.session_state:
			st.session_state.loader = None
		if 'datasets' not in st.session_state:
			st.session_state.datasets = {}
		if 'selected_module' not in st.session_state:
			st.session_state.selected_module = None
		if 'selected_table' not in st.session_state:
			st.session_state.selected_table = None
		if 'df' not in st.session_state:
			st.session_state.df = None
		if 'sample_size' not in st.session_state:
			st.session_state.sample_size = DEFAULT_SAMPLE_SIZE
		if 'available_tables' not in st.session_state:
			st.session_state.available_tables = {}
		if 'file_paths' not in st.session_state:
			st.session_state.file_paths = {}
		if 'file_sizes' not in st.session_state:
			st.session_state.file_sizes = {}
		if 'table_display_names' not in st.session_state:
			st.session_state.table_display_names = {}
		if 'current_file_path' not in st.session_state:
			st.session_state.current_file_path = None
		if 'mimic_path' not in st.session_state:
			st.session_state.mimic_path = DEFAULT_MIMIC_PATH

		# Feature engineering states
		if 'feature_eng_tab' not in st.session_state:
			st.session_state.feature_eng_tab = 0
		if 'detected_order_cols' not in st.session_state:
			st.session_state.detected_order_cols = []
		if 'detected_time_cols' not in st.session_state:
			st.session_state.detected_time_cols = []
		if 'detected_patient_id_col' not in st.session_state:
			st.session_state.detected_patient_id_col = None
		if 'freq_matrix' not in st.session_state:
			st.session_state.freq_matrix = None
		if 'order_sequences' not in st.session_state:
			st.session_state.order_sequences = None
		if 'timing_features' not in st.session_state:
			st.session_state.timing_features = None
		if 'order_dist' not in st.session_state:
			st.session_state.order_dist = None
		if 'patient_order_dist' not in st.session_state:
			st.session_state.patient_order_dist = None
		if 'transition_matrix' not in st.session_state:
			st.session_state.transition_matrix = None

		# Clustering states
		if 'clustering_input_data' not in st.session_state:
			st.session_state.clustering_input_data = None
		if 'reduced_data' not in st.session_state:
			st.session_state.reduced_data = None
		if 'kmeans_labels' not in st.session_state:
			st.session_state.kmeans_labels = None
		if 'hierarchical_labels' not in st.session_state:
			st.session_state.hierarchical_labels = None
		if 'dbscan_labels' not in st.session_state:
			st.session_state.dbscan_labels = None
		if 'lda_results' not in st.session_state:
			st.session_state.lda_results = None
		if 'cluster_metrics' not in st.session_state:
			st.session_state.cluster_metrics = {}
		if 'optimal_k' not in st.session_state:
			st.session_state.optimal_k = None
		if 'optimal_eps' not in st.session_state:
			st.session_state.optimal_eps = None

		# Analysis states
		if 'length_of_stay' not in st.session_state:
			st.session_state.length_of_stay = None

		# Filtering states
		if 'filter_params' not in st.session_state:
			st.session_state.filter_params = {
				# Inclusion criteria
				'apply_encounter_timeframe': True,
				'encounter_timeframe': ['2017-2019'],
				'apply_age_range': True,
				'min_age': 18,
				'max_age': 75,
				'apply_t2dm_diagnosis': True,
				'apply_valid_admission_discharge': True,
				'apply_inpatient_stay': True,
				'admission_types': ['EMERGENCY', 'URGENT', 'ELECTIVE'],
				'require_inpatient_transfer': True,
				'required_inpatient_units': [],

				# Exclusion criteria
				'exclude_in_hospital_death': True
			}
		if 'current_view' not in st.session_state:
			st.session_state.current_view = 'data_explorer'

		logging.info("Session state initialized.")


	def run(self):
		"""Run the main application loop."""
		logging.info("Starting MIMICDashboardApp run...")
		# Set page config
		# st.set_page_config(
		# 	page_title="MIMIC-IV Explorer",
		# 	page_icon="🏥",
		# 	layout="wide"
		# )

		# Custom CSS for better styling
		st.markdown("""
		<style>
		.main .block-container {padding-top: 2rem;}
		.sub-header {margin-top: 20px; margin-bottom: 10px; color: #1E88E5;}
		.info-box {
			background-color: #f0f2f6;
			border-radius: 5px;
			padding: 10px;
			margin-bottom: 10px;
		}
		.stTabs [data-baseweb="tab-list"] {
			gap: 24px;
		}
		.stTabs [data-baseweb="tab"] {
			height: 50px;
			white-space: pre-wrap;
			background-color: #f0f2f6;
			border-radius: 4px 4px 0px 0px;
			gap: 1px;
			padding-left: 10px;
			padding-right: 10px;
		}
		</style>
		""", unsafe_allow_html=True)

		# Display the sidebar
		self._display_sidebar()

		if st.session_state.current_view == 'data_explorer':
			self._show_all_tabs()
		else:
			self.filtering_tab.render( data_handler=self.data_handler, feature_engineer=self.feature_engineer )

		logging.info("MIMICDashboardApp run finished.")


	def _display_sidebar(self):
		"""Handles the display and logic of the sidebar components."""
		# View selection
		st.sidebar.markdown("## Navigation")
		view_options = ["Data Explorer", "Filtering"]
		selected_view = st.sidebar.radio("Select View", view_options, index=0 if st.session_state.current_view == 'data_explorer' else 1)
		st.session_state.current_view = 'data_explorer' if selected_view == "Data Explorer" else 'filtering'

		st.sidebar.markdown("## Dataset Configuration")

		# MIMIC-IV path input
		mimic_path = st.sidebar.text_input( "MIMIC-IV Dataset Path",
			value=st.session_state.mimic_path,
			help="Enter the path to your local MIMIC-IV v3.1 dataset" )

		# Update mimic_path in session state
		st.session_state.mimic_path = mimic_path

		# Scan button
		if st.sidebar.button("Scan MIMIC-IV Directory"):
			if not mimic_path or mimic_path == "/path/to/mimic-iv-3.1":
				st.sidebar.error("Please enter a valid MIMIC-IV dataset path")
			else:
				# Scan the directory structure
				available_tables, file_paths, file_sizes, table_display_names = self.data_handler.scan_mimic_directory(mimic_path)

				if available_tables:
					st.session_state.available_tables = available_tables
					st.session_state.file_paths = file_paths
					st.session_state.file_sizes = file_sizes
					st.session_state.table_display_names = table_display_names
					st.sidebar.success(f"Found {sum(len(tables) for tables in available_tables.values())} tables in {len(available_tables)} modules")
				else:
					st.sidebar.error("No MIMIC-IV data found in the specified path")

		# Module and table selection (only show if available_tables is populated)
		if st.session_state.available_tables:
			# Module selection
			module = st.sidebar.selectbox(
				"Select Module",
				list(st.session_state.available_tables.keys()),
				help="Select which MIMIC-IV module to explore"
			)

			# Update selected module
			st.session_state.selected_module = module

			# Table selection based on selected module
			if module in st.session_state.available_tables:
				# Create a list of table display names for the dropdown
				table_options = st.session_state.available_tables[module]
				table_display_options = [st.session_state.table_display_names.get((module, table), table) for table in table_options]

				# Create a mapping from display name back to actual table name
				display_to_table = {display: table for table, display in zip(table_options, table_display_options)}

				# Show the dropdown with display names
				selected_display = st.sidebar.selectbox(
					"Select Table",
					table_display_options,
					help="Select which table to load (file size shown in parentheses)"
				)

				# Get the actual table name from the selected display name
				table = display_to_table[selected_display]

				# Update selected table
				st.session_state.selected_table = table

				# Show table info
				table_info = self.data_handler.get_table_info(module, table)
				st.sidebar.info(table_info)

				# Advance options
				with st.sidebar.expander("Advance Options", expanded=True):
					encoding = st.selectbox("Encoding", ["latin-1", "utf-8"], index=0)
					st.session_state.sample_size = st.number_input("Sample Size", min_value=100, max_value=1000000, value=1000, step=100)
					# Add option to use Dask for large files
					st.session_state.use_dask = st.checkbox(
						"Use Dask for large files",
						value=False,
						help="Enable Dask for more efficient processing of very large files"
					)

				# Load button
				if st.sidebar.button("Load Table"):
					file_path = st.session_state.file_paths.get((module, table))
					if file_path:
						st.session_state.current_file_path = file_path

						# Get use_dask value from session state (default to False if not set)
						use_dask = st.session_state.get('use_dask', False)

						# Show loading message with framework info
						framework = "Dask" if use_dask else "Pandas"
						with st.spinner(f"Loading data using {framework}..."):
							df, total_rows = self.data_handler.load_mimic_table(
								file_path=file_path,
								sample_size=st.session_state.sample_size,
								encoding=encoding,
								use_dask=use_dask
							)
						st.session_state.total_row_count = total_rows

						if df is not None:
							st.session_state.df = df

							# Auto-detect columns for feature engineering
							st.session_state.detected_order_cols     = self.feature_engineer.detect_order_columns(df)
							st.session_state.detected_time_cols      = self.feature_engineer.detect_temporal_columns(df)
							st.session_state.detected_patient_id_col = self.feature_engineer.detect_patient_id_column(df)


	def _show_all_tabs(self):
		"""Handles the display of the main content area with tabs."""
		if st.session_state.df is not None:
			# Dataset info
			st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
			st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)
			st.markdown(f"**Module:** {st.session_state.selected_module}")
			st.markdown(f"**Table:** {st.session_state.selected_table}")
			st.markdown(f"**File:** {os.path.basename(st.session_state.current_file_path)}")

			# Get file size and format it
			file_size_mb = st.session_state.file_sizes.get((st.session_state.selected_module, st.session_state.selected_table), 0)
			if file_size_mb < 1:
				size_str = f"{file_size_mb:.2f} KB"
			elif file_size_mb < 1000:
				size_str = f"{file_size_mb:.2f} MB"
			else:
				size_str = f"{file_size_mb/1000:.2f} GB"

			st.markdown(f"**File Size:** {size_str}")
			st.markdown(f"**Sample Size:** {min(len(st.session_state.df), st.session_state.sample_size)} rows out of {st.session_state.total_row_count if 'total_row_count' in st.session_state else len(st.session_state.df)}")
			st.markdown("</div>", unsafe_allow_html=True)

			# Create tabs
			tab1, tab2, tab3, tab4, tab5 = st.tabs([
				"Exploration & Visualization",
				"Feature Engineering",
				"Clustering Analysis",
				"Analysis & Visualization",  # Add this new tab
				"Export Options"
			])

			# Tab 1: Exploration & Visualization
			with tab1:
				# Data preview
				self.visualizer.display_data_preview(st.session_state.df)

				# Dataset statistics
				self.visualizer.display_dataset_statistics(st.session_state.df)

				# Data visualization
				self.visualizer.display_visualizations(st.session_state.df)

			# Tab 2: Feature Engineering
			with tab2:
				self._feature_engineering_tab()

			# Tab 3: Clustering Analysis
			with tab3:
				self._clustering_analysis_tab()

			# Tab 4: Analysis & Visualization
			with tab4:
				self._analysis_visualization_tab()

			# Tab 5: Export Options
			with tab5:
				st.markdown("<h2 class='sub-header'>Export Options</h2>", unsafe_allow_html=True)
				col1, col2 = st.columns(2)

				with col1:
					if st.button("Export to CSV"):
						csv = st.session_state.df.to_csv(index=False)
						st.download_button(
							label="Download CSV",
							data=csv,
							file_name=f"mimic_iv_{st.session_state.selected_module}_{st.session_state.selected_table}.csv",
							mime="text/csv"
						)

				with col2:
					if st.button("Convert to Parquet"):
						try:
							# Create parquet directory if it doesn't exist
							parquet_dir = os.path.join(os.path.dirname(st.session_state.current_file_path), 'parquet_files')
							os.makedirs(parquet_dir, exist_ok=True)

							# Define parquet file path
							parquet_file = os.path.join(parquet_dir, f"{st.session_state.selected_table}.parquet")

							# Convert to parquet
							table = pa.Table.from_pandas(st.session_state.df)
							pq.write_table(table, parquet_file)

							st.success(f"Dataset converted to Parquet format at {parquet_file}")
						except Exception as e:
							st.error(f"Error converting to Parquet: {str(e)}")
		else:
			# Welcome message when no data is loaded
			st.markdown("""
			<div class='info-box'>
			<h2 class='sub-header'>Welcome to the MIMIC-IV Dashboard</h2>
			<p>This dashboard allows you to explore the MIMIC-IV dataset directly from the CSV/CSV.GZ files.</p>
			<p>To get started:</p>
			<ol>
				<li>Enter the path to your local MIMIC-IV v3.1 dataset in the sidebar</li>
				<li>Click "Scan MIMIC-IV Directory" to detect available tables</li>
				<li>Select a module and table to explore</li>
				<li>Configure advanced options if needed</li>
				<li>Click "Load Table" to begin</li>
			</ol>
			<p>Note: You need to have access to the MIMIC-IV dataset and have it downloaded locally.</p>
			</div>
			""", unsafe_allow_html=True)

			# About MIMIC-IV
			st.markdown("""
			<h2 class='sub-header'>About MIMIC-IV</h2>
			<div class='info-box'>
			<p>MIMIC-IV is a large, freely-available database comprising de-identified health-related data associated with patients who stayed in critical care units at the Beth Israel Deaconess Medical Center between 2008 - 2019.</p>
			<p>The database is organized into two main modules:</p>
			<ul>
				<li><strong>Hospital (hosp)</strong>: Contains hospital-wide data including admissions, patients, lab tests, diagnoses, etc.</li>
				<li><strong>ICU (icu)</strong>: Contains ICU-specific data including vital signs, medications, procedures, etc.</li>
			</ul>
			<p>Key tables include:</p>
			<ul>
				<li><strong>patients.csv</strong>: Patient demographic data</li>
				<li><strong>admissions.csv</strong>: Hospital admission information</li>
				<li><strong>labevents.csv</strong>: Laboratory measurements</li>
				<li><strong>chartevents.csv</strong>: Patient charting data (vital signs, etc.)</li>
				<li><strong>icustays.csv</strong>: ICU stay information</li>
			</ul>
			<p>For more information, visit <a href="https://physionet.org/content/mimiciv/3.1/">MIMIC-IV on PhysioNet</a>.</p>
			</div>
			""", unsafe_allow_html=True)


	def _export_tab(self, data, feature_type='order_frequency_matrix'):
		"""Helper function to display export options for engineered features."""
		with st.expander("#### Export Options"):

			save_format = st.radio("Save Format", ["CSV", "Parquet"], horizontal=True)

			if st.button("Save Frequency Matrix"):
				try:
					filepath = self.feature_engineer.save_features( features=data, feature_type=feature_type,
						base_path = os.path.dirname(st.session_state.current_file_path),
						format    = save_format.lower() )
					st.success(f"Saved frequency matrix to {filepath}")
				except Exception as e:
					st.error(f"Error saving frequency matrix: {str(e)}")


	def _feature_engineering_tab(self):
		"""Display the feature engineering tab content."""
		st.markdown("<h2 class='sub-header'>Order Data Feature Engineering</h2>", unsafe_allow_html=True)

		# Show introductory text
		st.info("This section allows you to transform raw MIMIC-IV order data into structured features for analysis and machine learning. Choose one of the feature engineering methods below to get started.")

		# Feature engineering subtabs
		feature_tabs = st.tabs([
			"📊 Order Frequency Matrix",
			"⏱️ Temporal Order Sequences",
			"📈 Order Type Distributions",
			"🕒 Order Timing Analysis"
		])

		# Get available columns
		all_columns = st.session_state.df.columns.tolist()

		# 1. Order Frequency Matrix tab
		with feature_tabs[0]:
			st.markdown("### Create Order Frequency Matrix")
			st.info("This creates a matrix where rows are patients and columns are order types, with cells showing frequency of each order type per patient.")

			# Column selection
			col1, col2 = st.columns(2)
			with col1:
				# Suggest patient ID column but allow selection from all columns
				patient_id_col = st.selectbox(
					"Select Patient ID Column",
					all_columns,
					index=2 if len(all_columns) > 2 else 0,
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column but allow selection from all columns
				order_col = st.selectbox(
					"Select Order Type Column",
					all_columns,
					index=3 if len(all_columns) > 3 else 0,
					help="Column containing order types/names"
				)

			# Options
			col1, col2, col3 = st.columns(3)
			with col1:
				normalize = st.checkbox("Normalize by Patient", value=False, help="Convert frequencies to percentages of total orders per patient")
			with col2:
				top_n = st.number_input("Top N Order Types", min_value=0, max_value=100, value=20, help="Limit to most frequent order types (0 = include all)")

			# Generate button
			if st.button("Generate Order Frequency Matrix"):
				try:
					with st.spinner("Generating order frequency matrix..."):
						freq_matrix = self.feature_engineer.create_order_frequency_matrix(
							df             = st.session_state.df,
							patient_id_col = patient_id_col,
							order_col      = order_col,
							normalize      = normalize,
							top_n          = top_n
						)
						# Store the frequency matrix
						st.session_state.freq_matrix = freq_matrix

						# Store in clustering_input_data for clustering analysis
						st.session_state.clustering_input_data = freq_matrix

				except Exception as e:
					st.error(f"Error generating frequency matrix: {str(e)}")

			# Display result if available
			if st.session_state.freq_matrix is not None:
				st.markdown("<h4>Order Frequency Matrix</h4>", unsafe_allow_html=True)

				# Show preview
				st.dataframe(st.session_state.freq_matrix.head(10), use_container_width=True)

				# Matrix stats
				st.markdown(f"<div class='info-box'>Matrix size: {st.session_state.freq_matrix.shape[0]} patients × {st.session_state.freq_matrix.shape[1]} order types</div>", unsafe_allow_html=True)

				# Heatmap visualization
				st.markdown("<h4>Frequency Matrix Heatmap (Sample)</h4>", unsafe_allow_html=True)

				# Visualization
				st.markdown("#### Frequency Heatmap")
				fig = px.imshow(st.session_state.freq_matrix.T,
								labels=dict(x="Patient ID", y="Order Type", color="Count"),
								aspect="auto")
				st.plotly_chart(fig, use_container_width=True)

				# Save options
				self._export_tab(data=st.session_state.freq_matrix)

		# 2. Temporal Order Sequences tab
		with feature_tabs[1]:
			st.markdown("<h3>Extract Temporal Order Sequences</h3>", unsafe_allow_html=True)
			st.info("This extracts chronological sequences of orders for each patient, preserving the temporal relationships between different orders.")

			# Column selection
			col1, col2, col3 = st.columns(3)
			with col1:
				# Suggest patient ID column
				seq_patient_id_col = st.selectbox(
					"Select Patient ID Column",
					all_columns,
					index=all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
					key="seq_patient_id_col",
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column
				seq_order_col = st.selectbox( "Select Order Type Column", all_columns, index=all_columns.index('order_type') if 'order_type' in all_columns else 0, key="seq_order_col", help="Column containing order types/names" )

			with col3:
				# Suggest time column
				seq_time_col = st.selectbox( "Select Timestamp Column", all_columns, index=all_columns.index('ordertime') if 'ordertime' in all_columns else 0, key="seq_time_col", help="Column containing order timestamps" )

			# Options
			max_seq_length = st.slider("Maximum Sequence Length", min_value=5, max_value=100, value=20, help="Maximum number of orders to include in each sequence")

			# Generate button
			if st.button("Extract Order Sequences"):
				try:
					with st.spinner("Extracting temporal order sequences..."):
						sequences = self.feature_engineer.extract_temporal_order_sequences(
							df                  = st.session_state.df,
							patient_id_col      = seq_patient_id_col,
							order_col           = seq_order_col,
							time_col            = seq_time_col,
							max_sequence_length = max_seq_length
						)
						st.session_state.order_sequences = sequences

						# Also generate transition matrix automatically
						transition_matrix = self.feature_engineer.calculate_order_transition_matrix( sequences=sequences, top_n=15 )
						st.session_state.transition_matrix = transition_matrix
				except Exception as e:
					st.error(f"Error extracting order sequences: {str(e)}")

			# Display results if available
			if st.session_state.order_sequences is not None:
				# Show sequence stats
				num_patients = len(st.session_state.order_sequences)
				avg_sequence_length = np.mean([len(seq) for seq in st.session_state.order_sequences.values()])

				st.markdown("<h4>Sequence Statistics</h4>", unsafe_allow_html=True)
				st.markdown(f"""
				<div class='info-box'>
				<p><strong>Number of patients:</strong> {num_patients}</p>
				<p><strong>Average sequence length:</strong> {avg_sequence_length:.2f} orders</p>
				</div>
				""", unsafe_allow_html=True)

				# Show sample sequences
				st.markdown("<h4>Sample Order Sequences</h4>", unsafe_allow_html=True)

				# Get a few sample patients
				sample_patients = list(st.session_state.order_sequences.keys())[:5]
				for patient in sample_patients:
					sequence = st.session_state.order_sequences[patient]
					sequence_str = " → ".join([str(order) for order in sequence])

					st.markdown(f"<strong>Patient {patient}:</strong> {sequence_str}", unsafe_allow_html=True)
					st.markdown("<hr>", unsafe_allow_html=True)

				# Transition matrix visualization
				if st.session_state.transition_matrix is not None:
					st.markdown("<h4>Order Transition Matrix</h4>", unsafe_allow_html=True)
					st.info("This matrix shows the probability of transitioning from one order type (rows) to another (columns).")

					fig = px.imshow(
						st.session_state.transition_matrix,
						labels=dict(x="Next Order", y="Current Order", color="Transition Probability"),
						x=st.session_state.transition_matrix.columns,
						y=st.session_state.transition_matrix.index,
						color_continuous_scale='Blues'
					)
					fig.update_layout(height=700)
					st.plotly_chart(fig, use_container_width=True)

				# Save options
				save_col1, save_col2 = st.columns(2)
				with save_col1:
					seq_save_format = st.radio("Save Format", ["JSON", "CSV"], horizontal=True, key="seq_save_format")
				with save_col2:
					if st.button("Save Order Sequences"):
						try:
							filepath = self.feature_engineer.save_features(
								st.session_state.order_sequences,
								"temporal_order_sequences",
								os.path.dirname(st.session_state.current_file_path),
								seq_save_format.lower()
							)
							st.success(f"Saved order sequences to {filepath}")
						except Exception as e:
							st.error(f"Error saving order sequences: {str(e)}")

		# 3. Order Type Distributions tab
		with feature_tabs[2]:
			st.markdown("<h3>Analyze Order Type Distributions</h3>", unsafe_allow_html=True)
			st.info("This analyzes the distribution of different order types across the dataset and for individual patients.")

			# Column selection
			col1, col2 = st.columns(2)
			with col1:
				# Suggest patient ID column
				dist_patient_id_col = st.selectbox(
					"Select Patient ID Column",
					all_columns,
					index=all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
					key="dist_patient_id_col",
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column
				dist_order_col = st.selectbox(
					"Select Order Type Column",
					all_columns,
					index=all_columns.index('order_type') if 'order_type' in all_columns else 0,
					key="dist_order_col",
					help="Column containing order types/names"
				)

			# Generate button
			if st.button("Analyze Order Distributions"):
				try:
					with st.spinner("Analyzing order type distributions..."):
						overall_dist, patient_dist = self.feature_engineer.get_order_type_distributions(
							st.session_state.df,
							dist_patient_id_col,
							dist_order_col
						)
						st.session_state.order_dist = overall_dist
						st.session_state.patient_order_dist = patient_dist
				except Exception as e:
					st.error(f"Error analyzing order distributions: {str(e)}")

			# Display results if available
			if st.session_state.order_dist is not None:
				# Show overall distribution
				st.markdown("<h4>Overall Order Type Distribution</h4>", unsafe_allow_html=True)

				# Create pie chart for overall distribution
				top_n_orders = 15  # Show top 15 for pie chart
				top_orders = st.session_state.order_dist.head(top_n_orders)

				# Create "Other" category for remaining orders
				if len(st.session_state.order_dist) > top_n_orders:
					others_sum = st.session_state.order_dist.iloc[top_n_orders:]['frequency'].sum()
					other_row = pd.DataFrame({
						dist_order_col: ['Other'],
						'frequency': [others_sum]
					})
					pie_data = pd.concat([top_orders, other_row], ignore_index=True)
				else:
					pie_data = top_orders

				fig = px.pie(
					pie_data,
					values='frequency',
					names=dist_order_col,
					title=f"Overall Distribution of {dist_order_col} (Top {top_n_orders})"
				)
				st.plotly_chart(fig, use_container_width=True)

				# Show bar chart of top 20
				top_20 = st.session_state.order_dist.head(20)
				bar_fig = px.bar(
					top_20,
					x=dist_order_col,
					y='frequency',
					title=f"Top 20 {dist_order_col} by Frequency"
				)
				st.plotly_chart(bar_fig, use_container_width=True)

				# Patient-level distribution (sample)
				if st.session_state.patient_order_dist is not None and not st.session_state.patient_order_dist.empty:
					st.markdown("<h4>Patient-Level Order Type Distribution</h4>", unsafe_allow_html=True)

					# Get unique patients
					patients = st.session_state.patient_order_dist['patient_id'].unique()

					# Sample patients for visualization if there are too many
					if len(patients) > 5:
						sample_patients = patients[:5]
					else:
						sample_patients = patients

					# Create subplots for each patient
					fig = make_subplots(
						rows=len(sample_patients),
						cols=1,
						subplot_titles=[f"Patient {patient}" for patient in sample_patients]
					)

					# Add traces for each patient
					for i, patient in enumerate(sample_patients):
						patient_data = st.session_state.patient_order_dist[
							st.session_state.patient_order_dist['patient_id'] == patient
						].head(10)  # Top 10 orders for this patient

						fig.add_trace(
							go.Bar(
								x=patient_data[dist_order_col],
								y=patient_data['frequency'],
								name=f"Patient {patient}"
							),
							row=i+1, col=1
						)

					fig.update_layout(height=200*len(sample_patients), showlegend=False)
					st.plotly_chart(fig, use_container_width=True)

				# Save options
				save_col1, save_col2 = st.columns(2)
				with save_col1:
					dist_save_format = st.radio("Save Format", ["CSV", "Parquet"], horizontal=True, key="dist_save_format")
				with save_col2:
					if st.button("Save Distribution Data"):
						try:
							# Save overall distribution
							filepath1 = self.feature_engineer.save_features(
								st.session_state.order_dist,
								"overall_order_distribution",
								os.path.dirname(st.session_state.current_file_path),
								dist_save_format.lower()
							)

							# Save patient-level distribution
							filepath2 = self.feature_engineer.save_features(
								st.session_state.patient_order_dist,
								"patient_order_distribution",
								os.path.dirname(st.session_state.current_file_path),
								dist_save_format.lower()
							)

							st.success(f"Saved distribution data to:\n- {filepath1}\n- {filepath2}")
						except Exception as e:
							st.error(f"Error saving distribution data: {str(e)}")

		# 4. Order Timing Analysis tab
		with feature_tabs[3]:
			st.markdown("<h3>Analyze Order Timing</h3>", unsafe_allow_html=True)
			st.markdown("""
			<div class='info-box'>
			This analyzes the timing of orders relative to admission, providing features about when orders occur during a patient's stay.
			</div>
			""", unsafe_allow_html=True)

			# Column selection
			col1, col2 = st.columns(2)
			with col1:
				# Suggest patient ID column
				if st.session_state.detected_patient_id_col in all_columns:
					timing_patient_id_col = st.selectbox(
						"Select Patient ID Column",
						all_columns,
						index=all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
						key="timing_patient_id_col",
						help="Column containing unique patient identifiers"
					)

			with col2:
				# Suggest order column
				if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns:
					timing_order_col = st.selectbox(
						"Select Order Type Column",
						all_columns,
						index=all_columns.index('order_type') if 'order_type' in all_columns else 0,
						key="timing_order_col",
						help="Column containing order types/names"
					)

			# Time columns
			col1, col2 = st.columns(2)
			with col1:
				order_time_col = st.selectbox(
					"Select Order Time Column",
					all_columns,
					index=all_columns.index('ordertime') if 'ordertime' in all_columns else 0,
					key="order_time_col",
					help="Column containing order timestamps"
				)

			with col2:
				# Optional admission time column
				admission_time_col = st.selectbox(
					"Select Admission Time Column (Optional)",
					["None"] + all_columns,
					index=0,
					key="admission_time_col",
					help="Column containing admission timestamps (for relative timing features)"
				)

				if admission_time_col == "None":
					admission_time_col = None

			# Optional discharge time column
			discharge_time_col = st.selectbox(
				"Select Discharge Time Column (Optional)",
				["None"] + all_columns,
				index=0,
				key="discharge_time_col",
				help="Column containing discharge timestamps (for relative timing features)"
			)

			if discharge_time_col == "None":
				discharge_time_col = None

			# Generate button
			if st.button("Generate Timing Features"):
				try:
					with st.spinner("Generating order timing features..."):
						timing_features = self.feature_engineer.create_order_timing_features(
							df                    = st.session_state.df,
							patient_id_col        = timing_patient_id_col,
							order_col             = timing_order_col,
							order_time_col        = order_time_col,
							admission_time_col    = admission_time_col,
							discharge_time_col    = discharge_time_col
						)
						st.session_state.timing_features = timing_features
				except Exception as e:
					st.error(f"Error generating timing features: {str(e)}")

			# Display results if available
			if st.session_state.timing_features is not None:
				st.markdown("<h4>Order Timing Features</h4>", unsafe_allow_html=True)

				# Show preview of features
				st.dataframe(st.session_state.timing_features.head(10), use_container_width=True)

				# Generate visualizations based on available features
				st.markdown("<h4>Order Timing Visualizations</h4>", unsafe_allow_html=True)

				numeric_cols = st.session_state.timing_features.select_dtypes(include=['number']).columns

				# Bar chart of total orders
				if 'total_orders' in st.session_state.timing_features.columns:
					col1, col2 = st.columns(2)

					with col1:
						# Histogram of total orders
						total_orders_fig = px.histogram(
							st.session_state.timing_features,
							x='total_orders',
							title="Distribution of Total Orders per Patient"
						)
						st.plotly_chart(total_orders_fig, use_container_width=True)

					with col2:
						# Histogram of unique order types
						if 'unique_order_types' in st.session_state.timing_features.columns:
							unique_orders_fig = px.histogram(
								st.session_state.timing_features,
								x='unique_order_types',
								title="Distribution of Unique Order Types per Patient"
							)
							st.plotly_chart(unique_orders_fig, use_container_width=True)

				# Time-based analyses
				if admission_time_col and 'time_to_first_order_hours' in st.session_state.timing_features.columns:
					col1, col2 = st.columns(2)

					with col1:
						# Histogram of time to first order
						first_order_fig = px.histogram(
							st.session_state.timing_features,
							x='time_to_first_order_hours',
							title="Time from Admission to First Order (hours)"
						)
						st.plotly_chart(first_order_fig, use_container_width=True)

					with col2:
						# Bar chart of orders in first 24/48/72 hours
						if all(col in st.session_state.timing_features.columns for col in
							  ['orders_in_first_24h', 'orders_in_first_48h', 'orders_in_first_72h']):

							# Prepare data for bar chart
							time_periods = ['First 24h', 'First 48h', 'First 72h']
							avg_orders = [
								st.session_state.timing_features['orders_in_first_24h'].mean(),
								st.session_state.timing_features['orders_in_first_48h'].mean(),
								st.session_state.timing_features['orders_in_first_72h'].mean()
							]

							orders_by_time = pd.DataFrame({
								'Time Period': time_periods,
								'Average Orders': avg_orders
							})

							time_orders_fig = px.bar(
								orders_by_time,
								x='Time Period',
								y='Average Orders',
								title="Average Orders in Time Periods After Admission"
							)
							st.plotly_chart(time_orders_fig, use_container_width=True)

				# Save options
				save_col1, save_col2 = st.columns(2)
				with save_col1:
					timing_save_format = st.radio("Save Format", ["CSV", "Parquet"], horizontal=True, key="timing_save_format")
				with save_col2:
					if st.button("Save Timing Features"):
						try:
							filepath = self.feature_engineer.save_features(
								st.session_state.timing_features,
								"order_timing_features",
								os.path.dirname(st.session_state.current_file_path),
								timing_save_format.lower()
							)
							st.success(f"Saved timing features to {filepath}")
						except Exception as e:
							st.error(f"Error saving timing features: {str(e)}")


	def _clustering_analysis_tab(self):
			"""Display the clustering analysis tab content."""
			st.markdown("<h2 class='sub-header'>Clustering Analysis</h2>", unsafe_allow_html=True)

			# Introductory text
			st.info("This section enables advanced clustering analysis on MIMIC-IV order data to discover patterns and patient groupings. You can apply different clustering algorithms and analyze the resulting clusters to gain insights.")

			# Clustering subtabs
			clustering_tabs = st.tabs([
				"📋 Data Selection",
				"📊 Dimensionality Reduction",
				"🔄 K-Means Clustering",
				"🌴 Hierarchical Clustering",
				"🔍 DBSCAN Clustering",
				"📝 LDA Topic Modeling",
				"📈 Evaluation Metrics"
			])

			# 1. Data Selection Tab
			with clustering_tabs[0]:
				st.markdown("<h3>Select Input Data for Clustering</h3>", unsafe_allow_html=True)

				# Option to use the current DataFrame or a feature matrix
				data_source = st.radio(
					"Select Data Source",
					["Current DataFrame", "Order Frequency Matrix", "Order Timing Features", "Upload Data"],
					horizontal=True
				)

				input_data = None

				if data_source == "Current DataFrame":
					# Let user select columns from the current DataFrame
					if st.session_state.df is not None:
						# Get numeric columns only for clustering
						numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()

						if numeric_cols:
							selected_cols = st.multiselect(
								"Select numeric columns for clustering",
								numeric_cols,
								default=numeric_cols[:min(5, len(numeric_cols))]
							)

							if selected_cols:
								input_data = st.session_state.df[selected_cols].copy()
								st.markdown(f"Data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
								st.dataframe(input_data.head(), use_container_width=True)
						else:
							st.warning("No numeric columns found in the current DataFrame. Please select another data source.")
					else:
						st.warning("No DataFrame is currently loaded. Please load a dataset first.")

				elif data_source == "Order Frequency Matrix":
					# Use order frequency matrix if available
					if st.session_state.freq_matrix is not None:
						input_data = st.session_state.freq_matrix
						st.markdown(f"Using order frequency matrix with shape: {input_data.shape[0]} patients × {input_data.shape[1]} order types")
						st.dataframe(input_data.head(), use_container_width=True)
					else:
						st.warning("Order frequency matrix not found. Please generate it in the Feature Engineering tab first.")

				elif data_source == "Order Timing Features":
					# Use timing features if available
					if st.session_state.timing_features is not None:
						# Get numeric columns only
						numeric_cols = st.session_state.timing_features.select_dtypes(include=['number']).columns.tolist()
						selected_cols = st.multiselect(
							"Select timing features for clustering",
							numeric_cols,
							default=numeric_cols
						)

						if selected_cols:
							input_data = st.session_state.timing_features[selected_cols].copy()
							st.markdown(f"Data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
							st.dataframe(input_data.head(), use_container_width=True)
					else:
						st.warning("Order timing features not found. Please generate them in the Feature Engineering tab first.")

				elif data_source == "Upload Data":
					# Allow user to upload a CSV or Parquet file
					uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=["csv", "parquet"])

					if uploaded_file is not None:
						try:
							if uploaded_file.name.endswith('.csv'):
								input_data = pd.read_csv(uploaded_file)
							elif uploaded_file.name.endswith('.parquet'):
								input_data = pd.read_parquet(uploaded_file)

							st.markdown(f"Uploaded data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
							st.dataframe(input_data.head(), use_container_width=True)
						except Exception as e:
							st.error(f"Error loading file: {str(e)}")

				# Data preprocessing options
				if input_data is not None:
					st.markdown("<h4>Data Preprocessing</h4>", unsafe_allow_html=True)

					preprocess_col1, preprocess_col2 = st.columns(2)

					with preprocess_col1:
						preprocess_method = st.selectbox(
							"Preprocessing Method",
							["None", "Standard Scaling", "Min-Max Scaling", "Normalization"],
							index=1,
							help="Select method to preprocess the data"
						)

					with preprocess_col2:
						handle_missing = st.selectbox(
							"Handle Missing Values",
							["Drop", "Fill with Mean", "Fill with Median", "Fill with Zero"],
							index=0,
							help="Select method to handle missing values"
						)

					# Map selections to parameter values
					preprocess_method_map = {
						"None": None,
						"Standard Scaling": "standard",
						"Min-Max Scaling": "minmax",
						"Normalization": "normalize"
					}

					handle_missing_map = {
						"Drop": "drop",
						"Fill with Mean": "mean",
						"Fill with Median": "median",
						"Fill with Zero": "zero"
					}

					# Button to process and store the data
					if st.button("Prepare Data for Clustering"):
						try:
							if preprocess_method != "None":
								# Apply preprocessing
								processed_data = self.clustering_analyzer.preprocess_data(
									input_data,
									method=preprocess_method_map[preprocess_method],
									handle_missing=handle_missing_map[handle_missing]
								)

								st.session_state.clustering_input_data = processed_data
								st.success(f"Data preprocessed and ready for clustering! Shape: {processed_data.shape}")

								# Show preview of processed data
								st.dataframe(processed_data.head(), use_container_width=True)
							else:
								# Use raw data
								st.session_state.clustering_input_data = input_data
								st.success(f"Raw data ready for clustering! Shape: {input_data.shape}")

						except Exception as e:
							st.error(f"Error preparing data: {str(e)}")

			# 2. Dimensionality Reduction Tab
			with clustering_tabs[1]:
				st.markdown("<h3>Dimensionality Reduction</h3>", unsafe_allow_html=True)

				# Check if input data is available
				if st.session_state.clustering_input_data is not None:
					# Get data shape
					input_shape = st.session_state.clustering_input_data.shape

					st.markdown(f"""
					<div class='info-box'>
					Reduce the dimensionality of your data to visualize and improve clustering performance.
					Current data shape: {input_shape[0]} rows × {input_shape[1]} columns
					</div>
					""", unsafe_allow_html=True)

					# Dimensionality reduction method selection
					reduction_col1, reduction_col2 = st.columns(2)

					with reduction_col1:
						reduction_method = st.selectbox(
							"Dimensionality Reduction Method",
							["PCA", "t-SNE", "UMAP", "SVD"],
							index=0,
							help="Select method to reduce dimensions"
						)

					with reduction_col2:
						n_components = st.number_input(
							"Number of Components",
							min_value=2,
							max_value=min(10, input_shape[1]),
							value=2,
							help="Number of dimensions to reduce to"
						)

					# Method-specific parameters
					if reduction_method == "t-SNE":
						tsne_col1, tsne_col2 = st.columns(2)

						with tsne_col1:
							perplexity = st.slider(
								"Perplexity",
								min_value=5,
								max_value=50,
								value=30,
								help="Balance between preserving local and global structure"
							)

						with tsne_col2:
							learning_rate = st.slider(
								"Learning Rate",
								min_value=10,
								max_value=1000,
								value=200,
								step=10,
								help="Learning rate for t-SNE"
							)

						n_iter = st.slider(
							"Max Iterations",
							min_value=250,
							max_value=2000,
							value=1000,
							step=250,
							help="Maximum number of iterations"
						)

						extra_params = {
							"perplexity": perplexity,
							"learning_rate": learning_rate,
							"n_iter": n_iter
						}

					elif reduction_method == "UMAP":
						umap_col1, umap_col2 = st.columns(2)

						with umap_col1:
							n_neighbors = st.slider(
								"Number of Neighbors",
								min_value=2,
								max_value=100,
								value=15,
								help="Controls how local or global the embedding is"
							)

						with umap_col2:
							min_dist = st.slider(
								"Minimum Distance",
								min_value=0.0,
								max_value=0.99,
								value=0.1,
								step=0.05,
								help="Controls how tightly points are packed together"
							)

						metric = st.selectbox(
							"Distance Metric",
							["euclidean", "manhattan", "cosine", "correlation"],
							index=0,
							help="Metric used to measure distances"
						)

						extra_params = {
							"n_neighbors": n_neighbors,
							"min_dist": min_dist,
							"metric": metric
						}

					else:
						# PCA and SVD don't need extra parameters
						extra_params = {}

					# Button to apply dimensionality reduction
					if st.button("Apply Dimensionality Reduction"):
						try:
							with st.spinner(f"Applying {reduction_method} dimensionality reduction..."):
								# Map method names
								method_map = {
									"PCA": "pca",
									"t-SNE": "tsne",
									"UMAP": "umap",
									"SVD": "svd"
								}

								# Apply reduction
								reduced_data = self.clustering_analyzer.apply_dimensionality_reduction(
									st.session_state.clustering_input_data,
									method=method_map[reduction_method],
									n_components=n_components,
									**extra_params
								)

								# Store reduced data
								st.session_state.reduced_data = reduced_data

								# Show success message
								st.success(f"Dimensionality reduction complete! Reduced from {input_shape[1]} to {n_components} dimensions.")

								# Show preview
								st.dataframe(reduced_data.head(), use_container_width=True)

						except Exception as e:
							st.error(f"Error applying dimensionality reduction: {str(e)}")

					# Visualization of reduced data (if available and 2D/3D)
					if st.session_state.reduced_data is not None:
						reduced_shape = st.session_state.reduced_data.shape

						st.markdown("<h4>Visualization of Reduced Data</h4>", unsafe_allow_html=True)

						if reduced_shape[1] == 2:
							# Create 2D scatter plot
							fig = px.scatter(
								st.session_state.reduced_data,
								x=st.session_state.reduced_data.columns[0],
								y=st.session_state.reduced_data.columns[1],
								title=f"2D Projection using {reduction_method}"
							)
							st.plotly_chart(fig, use_container_width=True)

						elif reduced_shape[1] == 3:
							# Create 3D scatter plot
							fig = px.scatter_3d(
								st.session_state.reduced_data,
								x=st.session_state.reduced_data.columns[0],
								y=st.session_state.reduced_data.columns[1],
								z=st.session_state.reduced_data.columns[2],
								title=f"3D Projection using {reduction_method}"
							)
							st.plotly_chart(fig, use_container_width=True)

						else:
							st.info("Reduced data has more than 3 dimensions. Select 2 or 3 components for visualization.")

						# Option to save reduced data
						save_col1, save_col2 = st.columns(2)

						with save_col1:
							save_format = st.radio(
								"Save Format",
								["CSV", "Parquet"],
								horizontal=True,
								key="dimreduction_save_format"
							)

						with save_col2:
							if st.button("Save Reduced Data"):
								try:
									# Use feature engineer's save method
									filepath = self.feature_engineer.save_features(
										st.session_state.reduced_data,
										f"{reduction_method.lower()}_reduced_data",
										os.path.dirname(st.session_state.current_file_path),
										save_format.lower()
									)
									st.success(f"Saved reduced data to {filepath}")
								except Exception as e:
									st.error(f"Error saving data: {str(e)}")
				else:
					st.warning("No input data available. Please prepare data in the Data Selection tab first.")

			# 3. K-Means Clustering Tab
			with clustering_tabs[2]:
				st.markdown("<h3>K-Means Clustering</h3>", unsafe_allow_html=True)

				# Check if input data is available
				if st.session_state.clustering_input_data is not None:
					st.markdown("""
					<div class='info-box'>
					K-means clustering partitions data into k clusters, where each observation belongs to the cluster with the nearest mean.
					</div>
					""", unsafe_allow_html=True)

					# Option to use reduced data if available
					use_reduced_data = False
					if st.session_state.reduced_data is not None:
						use_reduced_data = st.checkbox(
							"Use dimensionality-reduced data for clustering",
							value=True,
							help="Use the reduced data from the previous tab instead of original data"
						)

					# K-means parameters
					kmeans_params_col1, kmeans_params_col2 = st.columns(2)

					with kmeans_params_col1:
						n_clusters = st.number_input(
							"Number of Clusters (k)",
							min_value=2,
							max_value=20,
							value=5,
							help="Number of clusters to form"
						)

					with kmeans_params_col2:
						max_iter = st.slider(
							"Maximum Iterations",
							min_value=100,
							max_value=1000,
							value=300,
							step=100,
							help="Maximum number of iterations for a single run"
						)

					n_init = st.slider(
						"Number of Initializations",
						min_value=1,
						max_value=20,
						value=10,
						help="Number of times the algorithm will run with different centroid seeds"
					)

					# Button to find optimal k
					st.markdown("<h4>Optimal Number of Clusters</h4>", unsafe_allow_html=True)

					optimal_k_col1, optimal_k_col2 = st.columns(2)

					with optimal_k_col1:
						k_min = st.number_input("Minimum k", min_value=2, max_value=10, value=2)

					with optimal_k_col2:
						k_max = st.number_input("Maximum k", min_value=3, max_value=20, value=10)

					metric = st.selectbox(
						"Optimization Metric",
						["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index", "Inertia (Elbow Method)"],
						index=0,
						help="Metric to optimize when finding the best k"
					)

					# Map to internal names
					metric_map = {
						"Silhouette Score": "silhouette",
						"Davies-Bouldin Index": "davies_bouldin",
						"Calinski-Harabasz Index": "calinski_harabasz",
						"Inertia (Elbow Method)": "inertia"
					}

					# Button to find optimal k
					if st.button("Find Optimal Number of Clusters"):
						try:
							with st.spinner("Finding optimal number of clusters..."):
								# Get the appropriate data
								if use_reduced_data and st.session_state.reduced_data is not None:
									data_for_clustering = st.session_state.reduced_data
								else:
									data_for_clustering = st.session_state.clustering_input_data

								# Find optimal k
								optimal_k, k_metrics = self.clustering_analyzer.find_optimal_k_for_kmeans(
									data_for_clustering,
									k_range=range(k_min, k_max + 1),
									metric=metric_map[metric],
									n_init=n_init,
									max_iter=max_iter
								)

								# Store optimal k
								st.session_state.optimal_k = optimal_k

								# Show result
								st.success(f"Optimal number of clusters (k): {optimal_k}")

								# Plot metric values
								fig = go.Figure()

								if metric == "Inertia (Elbow Method)":
									fig.add_trace(go.Scatter(
										x=k_metrics['k'],
										y=k_metrics['inertia'],
										mode='lines+markers',
										name='Inertia'
									))
									fig.update_layout(
										title=f"Elbow Method for Optimal k (k={optimal_k})",
										xaxis_title="Number of Clusters (k)",
										yaxis_title="Inertia",
										hovermode="x unified"
									)
								elif metric == "Davies-Bouldin Index":
									fig.add_trace(go.Scatter(
										x=k_metrics['k'],
										y=k_metrics['davies_bouldin'],
										mode='lines+markers',
										name='Davies-Bouldin Index'
									))
									fig.update_layout(
										title=f"Davies-Bouldin Index for Different k (k={optimal_k})",
										xaxis_title="Number of Clusters (k)",
										yaxis_title="Davies-Bouldin Index (lower is better)",
										hovermode="x unified"
									)
								elif metric == "Silhouette Score":
									fig.add_trace(go.Scatter(
										x=k_metrics['k'],
										y=k_metrics['silhouette'],
										mode='lines+markers',
										name='Silhouette Score'
									))
									fig.update_layout(
										title=f"Silhouette Score for Different k (k={optimal_k})",
										xaxis_title="Number of Clusters (k)",
										yaxis_title="Silhouette Score (higher is better)",
										hovermode="x unified"
									)
								else:  # Calinski-Harabasz Index
									fig.add_trace(go.Scatter(
										x=k_metrics['k'],
										y=k_metrics['calinski_harabasz'],
										mode='lines+markers',
										name='Calinski-Harabasz Index'
									))
									fig.update_layout(
										title=f"Calinski-Harabasz Index for Different k (k={optimal_k})",
										xaxis_title="Number of Clusters (k)",
										yaxis_title="Calinski-Harabasz Index (higher is better)",
										hovermode="x unified"
									)

								# Add vertical line at optimal k
								fig.add_vline(
									x=optimal_k,
									line_dash="dash",
									line_color="red",
									annotation_text=f"Optimal k = {optimal_k}",
									annotation_position="top right"
								)

								st.plotly_chart(fig, use_container_width=True)

								# Set the optimal k as the default value
								n_clusters = optimal_k

						except Exception as e:
							st.error(f"Error finding optimal k: {str(e)}")

					# Button to run K-means
					st.markdown("<h4>Run K-means Clustering</h4>", unsafe_allow_html=True)

					# Option to use optimal k if available
					if st.session_state.optimal_k is not None:
						use_optimal_k = st.checkbox(
							f"Use optimal k ({st.session_state.optimal_k})",
							value=True,
							help="Use the optimal k found above"
						)

						if use_optimal_k:
							n_clusters = st.session_state.optimal_k

					if st.button("Run K-means Clustering"):
						try:
							with st.spinner(f"Running K-means clustering with k={n_clusters}..."):
								# Get the appropriate data
								if use_reduced_data and st.session_state.reduced_data is not None:
									data_for_clustering = st.session_state.reduced_data
								else:
									data_for_clustering = st.session_state.clustering_input_data

								# Run K-means
								labels, kmeans_model = self.clustering_analyzer.run_kmeans_clustering(
									data_for_clustering,
									n_clusters=n_clusters,
									n_init=n_init,
									max_iter=max_iter
								)

								# Store labels in session state
								st.session_state.kmeans_labels = labels

								# Calculate metrics
								metrics = self.clustering_analyzer.evaluate_clustering(
									data_for_clustering,
									labels,
									"kmeans"
								)

								# Show success message with metrics
								st.success(f"""
								Clustering complete! Results:
								- Number of clusters: {n_clusters}
								- Silhouette score: {metrics['silhouette_score']:.4f} (higher is better)
								- Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f} (lower is better)
								- Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.4f} (higher is better)
								""")

								# Show cluster distribution
								cluster_counts = labels.value_counts().sort_index()

								# Create bar chart of cluster sizes
								fig = px.bar(
									x=cluster_counts.index,
									y=cluster_counts.values,
									labels={'x': 'Cluster', 'y': 'Count'},
									title="Distribution of Clusters"
								)
								st.plotly_chart(fig, use_container_width=True)

								# If we have 2D or 3D data, visualize the clusters
								if data_for_clustering.shape[1] == 2:
									# Create 2D scatter plot with clusters
									vis_data = data_for_clustering.copy()
									vis_data['Cluster'] = labels

									fig = px.scatter(
										vis_data,
										x=vis_data.columns[0],
										y=vis_data.columns[1],
										color='Cluster',
										title="K-means Clustering Results (2D)",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig, use_container_width=True)

								elif data_for_clustering.shape[1] == 3:
									# Create 3D scatter plot with clusters
									vis_data = data_for_clustering.copy()
									vis_data['Cluster'] = labels

									fig = px.scatter_3d(
										vis_data,
										x=vis_data.columns[0],
										y=vis_data.columns[1],
										z=vis_data.columns[2],
										color='Cluster',
										title="K-means Clustering Results (3D)",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig, use_container_width=True)

								# Get cluster centers
								if hasattr(kmeans_model, 'cluster_centers_'):
									st.markdown("<h4>Cluster Centers</h4>", unsafe_allow_html=True)

									centers = pd.DataFrame(
										kmeans_model.cluster_centers_,
										columns=data_for_clustering.columns,
										index=[f"Cluster {i}" for i in range(n_clusters)]
									)

									st.dataframe(centers, use_container_width=True)

						except Exception as e:
							st.error(f"Error running K-means clustering: {str(e)}")

					# Save model and results (if available)
					if st.session_state.kmeans_labels is not None:
						st.markdown("<h4>Save Clustering Results</h4>", unsafe_allow_html=True)

						save_col1, save_col2 = st.columns(2)

						with save_col1:
							if st.button("Save K-means Model"):
								try:
									# Save model
									model_path = self.clustering_analyzer.save_model(
										"kmeans",
										os.path.dirname(st.session_state.current_file_path)
									)
									st.success(f"Saved K-means model to {model_path}")
								except Exception as e:
									st.error(f"Error saving model: {str(e)}")

						with save_col2:
							if st.button("Save Cluster Assignments"):
								try:
									# Create DataFrame with cluster assignments
									if use_reduced_data and st.session_state.reduced_data is not None:
										cluster_df = st.session_state.reduced_data.copy()
									else:
										cluster_df = st.session_state.clustering_input_data.copy()

									cluster_df['cluster'] = st.session_state.kmeans_labels

									# Save
									filepath = self.feature_engineer.save_features(
										cluster_df,
										"kmeans_cluster_assignments",
										os.path.dirname(st.session_state.current_file_path),
										"csv"
									)
									st.success(f"Saved cluster assignments to {filepath}")
								except Exception as e:
									st.error(f"Error saving cluster assignments: {str(e)}")
				else:
					st.warning("No input data available. Please prepare data in the Data Selection tab first.")

			# 4. Hierarchical Clustering Tab
			with clustering_tabs[3]:
				st.markdown("<h3>Hierarchical Clustering</h3>", unsafe_allow_html=True)

				# Check if input data is available
				if st.session_state.clustering_input_data is not None:
					st.markdown("""
					<div class='info-box'>
					Hierarchical clustering creates a tree of clusters by progressively merging or splitting groups.
					It's useful for discovering hierarchical relationships in the data.
					</div>
					""", unsafe_allow_html=True)

					# Option to use reduced data if available
					use_reduced_data = False
					if st.session_state.reduced_data is not None:
						use_reduced_data = st.checkbox(
							"Use dimensionality-reduced data for hierarchical clustering",
							value=True,
							help="Use the reduced data from the dimensionality reduction tab"
						)

					# Hierarchical clustering parameters
					hier_col1, hier_col2 = st.columns(2)

					with hier_col1:
						n_clusters = st.number_input(
							"Number of Clusters",
							min_value=2,
							max_value=20,
							value=5,
							help="Number of clusters to form",
							key="hier_n_clusters"
						)

					with hier_col2:
						linkage_method = st.selectbox(
							"Linkage Method",
							["ward", "complete", "average", "single"],
							index=0,
							help="Method for calculating distances between clusters"
						)

					distance_metric = st.selectbox(
						"Distance Metric",
						["euclidean", "manhattan", "cosine", "correlation"],
						index=0,
						help="Metric for measuring distances between samples"
					)

					# Warning for ward linkage
					if linkage_method == "ward" and distance_metric != "euclidean":
						st.warning("Ward linkage requires Euclidean distance. Switching to Euclidean.")
						distance_metric = "euclidean"

					# Button to run hierarchical clustering
					if st.button("Run Hierarchical Clustering"):
						try:
							with st.spinner(f"Running hierarchical clustering with {linkage_method} linkage..."):
								# Get the appropriate data
								if use_reduced_data and st.session_state.reduced_data is not None:
									data_for_clustering = st.session_state.reduced_data
								else:
									data_for_clustering = st.session_state.clustering_input_data

								# Limit data size if too large (hierarchical clustering can be memory-intensive)
								max_samples = 1000
								if len(data_for_clustering) > max_samples:
									st.warning(f"Limiting to {max_samples} samples for hierarchical clustering to avoid memory issues.")
									data_for_clustering = data_for_clustering.sample(max_samples, random_state=42)

								# Run hierarchical clustering
								labels, linkage_data = self.clustering_analyzer.run_hierarchical_clustering(
									data_for_clustering,
									n_clusters=n_clusters,
									linkage_method=linkage_method,
									distance_metric=distance_metric
								)

								# Store labels in session state
								st.session_state.hierarchical_labels = labels

								# Calculate metrics
								metrics = self.clustering_analyzer.evaluate_clustering(
									data_for_clustering,
									labels,
									"hierarchical"
								)

								# Show success message with metrics
								st.success(f"""
								Hierarchical clustering complete! Results:
								- Number of clusters: {n_clusters}
								- Silhouette score: {metrics['silhouette_score']:.4f} (higher is better)
								- Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f} (lower is better)
								- Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.4f} (higher is better)
								""")

								# Show cluster distribution
								cluster_counts = labels.value_counts().sort_index()

								# Create bar chart of cluster sizes
								fig = px.bar(
									x=cluster_counts.index,
									y=cluster_counts.values,
									labels={'x': 'Cluster', 'y': 'Count'},
									title="Distribution of Hierarchical Clusters"
								)
								st.plotly_chart(fig, use_container_width=True)

								# If we have 2D or 3D data, visualize the clusters
								if data_for_clustering.shape[1] == 2:
									# Create 2D scatter plot with clusters
									vis_data = data_for_clustering.copy()
									vis_data['Cluster'] = labels

									fig = px.scatter(
										vis_data,
										x=vis_data.columns[0],
										y=vis_data.columns[1],
										color='Cluster',
										title="Hierarchical Clustering Results (2D)",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig, use_container_width=True)

								elif data_for_clustering.shape[1] == 3:
									# Create 3D scatter plot with clusters
									vis_data = data_for_clustering.copy()
									vis_data['Cluster'] = labels

									fig = px.scatter_3d(
										vis_data,
										x=vis_data.columns[0],
										y=vis_data.columns[1],
										z=vis_data.columns[2],
										color='Cluster',
										title="Hierarchical Clustering Results (3D)",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig, use_container_width=True)

								# Plot dendrogram
								st.markdown("<h4>Dendrogram</h4>", unsafe_allow_html=True)

								# Create dendrogram figure
								plt.figure(figsize=(10, 7))
								dendrogram(linkage_data['linkage_matrix'])
								plt.title('Hierarchical Clustering Dendrogram')
								plt.xlabel('Sample index')
								plt.ylabel('Distance')

								# Draw a horizontal line at the height where we get n_clusters
								if n_clusters > 1:
									plt.axhline(y=linkage_data['linkage_matrix'][-(n_clusters-1), 2],
											c='k', linestyle='--',
											label=f'Cut for {n_clusters} clusters')
									plt.legend()

								# Display in Streamlit
								st.pyplot(plt.gcf())
								plt.close()

						except Exception as e:
							st.error(f"Error running hierarchical clustering: {str(e)}")

					# Save model and results (if available)
					if st.session_state.hierarchical_labels is not None:
						st.markdown("<h4>Save Clustering Results</h4>", unsafe_allow_html=True)

						save_col1, save_col2 = st.columns(2)

						with save_col1:
							if st.button("Save Hierarchical Model"):
								try:
									# Save model
									model_path = self.clustering_analyzer.save_model(
										"hierarchical",
										os.path.dirname(st.session_state.current_file_path)
									)
									st.success(f"Saved hierarchical clustering model to {model_path}")
								except Exception as e:
									st.error(f"Error saving model: {str(e)}")

						with save_col2:
							if st.button("Save Hierarchical Cluster Assignments"):
								try:
									# Create DataFrame with cluster assignments
									if use_reduced_data and st.session_state.reduced_data is not None:
										cluster_df = st.session_state.reduced_data.copy()
									else:
										cluster_df = st.session_state.clustering_input_data.copy()

									cluster_df['cluster'] = st.session_state.hierarchical_labels

									# Save
									filepath = self.feature_engineer.save_features(
										cluster_df,
										"hierarchical_cluster_assignments",
										os.path.dirname(st.session_state.current_file_path),
										"csv"
									)
									st.success(f"Saved cluster assignments to {filepath}")
								except Exception as e:
									st.error(f"Error saving cluster assignments: {str(e)}")
				else:
					st.warning("No input data available. Please prepare data in the Data Selection tab first.")

			# 5. DBSCAN Clustering Tab
			with clustering_tabs[4]:
				st.markdown("<h3>DBSCAN Clustering</h3>", unsafe_allow_html=True)

				# Check if input data is available
				if st.session_state.clustering_input_data is not None:
					st.markdown("""
					<div class='info-box'>
					DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of arbitrary shapes
					by grouping points that are closely packed together, while marking outlier points as noise.
					</div>
					""", unsafe_allow_html=True)

					# Option to use reduced data if available
					use_reduced_data = False
					if st.session_state.reduced_data is not None:
						use_reduced_data = st.checkbox(
							"Use dimensionality-reduced data for DBSCAN",
							value=True,
							help="Use the reduced data from the dimensionality reduction tab"
						)

					# DBSCAN parameters
					dbscan_col1, dbscan_col2 = st.columns(2)

					with dbscan_col1:
						eps = st.number_input(
							"Epsilon (ε)",
							min_value=0.01,
							max_value=5.0,
							value=0.5,
							step=0.05,
							help="Maximum distance between two samples to be considered neighbors"
						)

					with dbscan_col2:
						min_samples = st.number_input(
							"Minimum Samples",
							min_value=2,
							max_value=100,
							value=5,
							help="Number of samples in a neighborhood for a point to be a core point"
						)

					metric = st.selectbox(
						"Distance Metric",
						["euclidean", "manhattan", "cosine", "l1", "l2"],
						index=0,
						help="Metric for measuring distances between samples",
						key="dbscan_metric"
					)

					# Button to find optimal eps
					st.markdown("<h4>Find Optimal Epsilon (ε)</h4>", unsafe_allow_html=True)

					k_dist = st.slider(
						"k for k-distance graph",
						min_value=2,
						max_value=20,
						value=5,
						help="Number of neighbors to consider for k-distance graph"
					)

					if st.button("Find Optimal Epsilon (ε)"):
						try:
							with st.spinner("Calculating k-distance graph to find optimal epsilon..."):
								# Get the appropriate data
								if use_reduced_data and st.session_state.reduced_data is not None:
									data_for_clustering = st.session_state.reduced_data
								else:
									data_for_clustering = st.session_state.clustering_input_data

								# Find optimal epsilon
								suggested_eps, k_distances = self.clustering_analyzer.find_optimal_eps_for_dbscan(
									data_for_clustering,
									k_dist=k_dist
								)

								# Store the suggested eps
								st.session_state.optimal_eps = suggested_eps

								# Show result
								st.success(f"Suggested epsilon (ε): {suggested_eps:.4f}")

								# Plot k-distance graph
								fig = go.Figure()

								fig.add_trace(go.Scatter(
									x=list(range(len(k_distances))),
									y=k_distances,
									mode='lines',
									name=f'{k_dist}-distance'
								))

								# Add point at the suggested epsilon
								knee_point_idx = np.argmax(np.diff(k_distances)) + 1  # Find the "knee" of the curve

								fig.add_trace(go.Scatter(
									x=[knee_point_idx],
									y=[k_distances[knee_point_idx]],
									mode='markers',
									marker=dict(size=10, color='red'),
									name=f'Suggested ε = {suggested_eps:.4f}'
								))

								fig.update_layout(
									title=f"{k_dist}-Distance Graph for DBSCAN Epsilon Selection",
									xaxis_title="Points (sorted by distance)",
									yaxis_title=f"{k_dist}-distance",
									hovermode="x"
								)

								st.plotly_chart(fig, use_container_width=True)

						except Exception as e:
							st.error(f"Error finding optimal epsilon: {str(e)}")

					# Button to run DBSCAN
					st.markdown("<h4>Run DBSCAN Clustering</h4>", unsafe_allow_html=True)

					# Option to use optimal eps if available
					if st.session_state.optimal_eps is not None:
						use_optimal_eps = st.checkbox(
							f"Use suggested epsilon ({st.session_state.optimal_eps:.4f})",
							value=True,
							help="Use the suggested epsilon value found above"
						)

						if use_optimal_eps:
							eps = st.session_state.optimal_eps

					if st.button("Run DBSCAN Clustering"):
						try:
							with st.spinner(f"Running DBSCAN clustering with ε={eps}..."):
								# Get the appropriate data
								if use_reduced_data and st.session_state.reduced_data is not None:
									data_for_clustering = st.session_state.reduced_data
								else:
									data_for_clustering = st.session_state.clustering_input_data

								# Run DBSCAN
								labels, dbscan_model = self.clustering_analyzer.run_dbscan_clustering(
									data_for_clustering,
									eps=eps,
									min_samples=min_samples,
									metric=metric
								)

								# Store labels in session state
								st.session_state.dbscan_labels = labels

								# Count number of clusters (excluding noise points with label -1)
								n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
								n_noise = list(labels).count(-1)

								# Calculate metrics (if more than one cluster)
								if n_clusters > 1:
									# For metrics, exclude noise points
									if -1 in labels:
										non_noise_mask = labels != -1
										metrics = self.clustering_analyzer.evaluate_clustering(
											data_for_clustering[non_noise_mask],
											labels[non_noise_mask],
											"dbscan"
										)
									else:
										metrics = self.clustering_analyzer.evaluate_clustering(
											data_for_clustering,
											labels,
											"dbscan"
										)

									# Show success message with metrics
									st.success(f"""
									DBSCAN clustering complete! Results:
									- Number of clusters: {n_clusters}
									- Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}% of data)
									- Silhouette score: {metrics['silhouette_score']:.4f} (higher is better)
									- Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f} (lower is better)
									- Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.4f} (higher is better)
									""")
								else:
									st.warning(f"""
									DBSCAN clustering completed, but resulted in only {n_clusters} cluster(s).
									- Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}% of data)

									Try adjusting epsilon (ε) or min_samples parameters.
									""")

								# Show cluster distribution
								cluster_counts = labels.value_counts().sort_index()

								# Create bar chart of cluster sizes
								fig = px.bar(
									x=cluster_counts.index.map(lambda x: "Noise" if x == -1 else f"Cluster {x}"),
									y=cluster_counts.values,
									labels={'x': 'Cluster', 'y': 'Count'},
									title="Distribution of DBSCAN Clusters"
								)
								st.plotly_chart(fig, use_container_width=True)

								# If we have 2D or 3D data, visualize the clusters
								if data_for_clustering.shape[1] == 2:
									# Create 2D scatter plot with clusters
									vis_data = data_for_clustering.copy()
									vis_data['Cluster'] = labels.map(lambda x: "Noise" if x == -1 else f"Cluster {x}")

									fig = px.scatter(
										vis_data,
										x=vis_data.columns[0],
										y=vis_data.columns[1],
										color='Cluster',
										title="DBSCAN Clustering Results (2D)",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig, use_container_width=True)

								elif data_for_clustering.shape[1] == 3:
									# Create 3D scatter plot with clusters
									vis_data = data_for_clustering.copy()
									vis_data['Cluster'] = labels.map(lambda x: "Noise" if x == -1 else f"Cluster {x}")

									fig = px.scatter_3d(
										vis_data,
										x=vis_data.columns[0],
										y=vis_data.columns[1],
										z=vis_data.columns[2],
										color='Cluster',
										title="DBSCAN Clustering Results (3D)",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig, use_container_width=True)

						except Exception as e:
							st.error(f"Error running DBSCAN clustering: {str(e)}")

					# Save model and results (if available)
					if st.session_state.dbscan_labels is not None:
						st.markdown("<h4>Save Clustering Results</h4>", unsafe_allow_html=True)

						save_col1, save_col2 = st.columns(2)

						with save_col1:
							if st.button("Save DBSCAN Model"):
								try:
									# Save model
									model_path = self.clustering_analyzer.save_model(
										"dbscan",
										os.path.dirname(st.session_state.current_file_path)
									)
									st.success(f"Saved DBSCAN model to {model_path}")
								except Exception as e:
									st.error(f"Error saving model: {str(e)}")

						with save_col2:
							if st.button("Save DBSCAN Cluster Assignments"):
								try:
									# Create DataFrame with cluster assignments
									if use_reduced_data and st.session_state.reduced_data is not None:
										cluster_df = st.session_state.reduced_data.copy()
									else:
										cluster_df = st.session_state.clustering_input_data.copy()

									cluster_df['cluster'] = st.session_state.dbscan_labels

									# Save
									filepath = self.feature_engineer.save_features(
										cluster_df,
										"dbscan_cluster_assignments",
										os.path.dirname(st.session_state.current_file_path),
										"csv"
									)
									st.success(f"Saved cluster assignments to {filepath}")
								except Exception as e:
									st.error(f"Error saving cluster assignments: {str(e)}")
				else:
					st.warning("No input data available. Please prepare data in the Data Selection tab first.")

			# 6. LDA Topic Modeling Tab
			with clustering_tabs[5]:
				st.markdown("<h3>LDA Topic Modeling</h3>", unsafe_allow_html=True)

				st.markdown("""
				<div class='info-box'>
				Latent Dirichlet Allocation (LDA) is a probabilistic model that discovers "topics" in text data.
				In medical orders context, LDA can find patterns in order sequences and group similar order types.
				</div>
				""", unsafe_allow_html=True)

				# Check if we have text data
				if st.session_state.df is not None:
					# Option to use original data or order sequences
					data_source = st.radio(
						"Text Data Source",
						["Order Sequences", "Text Column in Dataset"],
						index=0,
						help="Select source for text data to use in LDA"
					)

					if data_source == "Order Sequences":
						# Check if we have order sequences
						if st.session_state.order_sequences is not None:
							st.markdown(f"""
							Using {len(st.session_state.order_sequences)} patient order sequences for topic modeling.
							Each sequence will be treated as a "document" for LDA.
							""")

							# Convert sequences to text documents
							documents = [" ".join(map(str, seq)) for seq in st.session_state.order_sequences.values()]

							# Show sample
							st.markdown("<h4>Sample Order Sequences (as Text)</h4>", unsafe_allow_html=True)
							for i, doc in enumerate(documents[:3]):
								st.text_area(f"Patient {i+1}", doc, height=100, key=f"sample_doc_{i}")
						else:
							st.warning("Order sequences not found. Please generate them in the Feature Engineering tab first.")
							documents = None

					else:  # Text Column in Dataset
						# Select text column from DataFrame
						text_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()

						if text_columns:
							text_col = st.selectbox(
								"Select Text Column",
								text_columns,
								help="Select column containing text data for topic modeling"
							)

							# Convert to list of strings and handle NaNs
							documents = st.session_state.df[text_col].fillna("").astype(str).tolist()

							# Show sample
							st.markdown("<h4>Sample Text Documents</h4>", unsafe_allow_html=True)
							for i, doc in enumerate(documents[:3]):
								st.text_area(f"Document {i+1}", doc, height=100, key=f"sample_doc_{i}")
						else:
							st.warning("No text columns found in the dataset.")
							documents = None

					# LDA parameters (if we have documents)
					if documents:
						st.markdown("<h4>LDA Parameters</h4>", unsafe_allow_html=True)

						lda_col1, lda_col2 = st.columns(2)

						with lda_col1:
							n_topics = st.number_input(
								"Number of Topics",
								min_value=2,
								max_value=20,
								value=5,
								help="Number of topics to extract"
							)

						with lda_col2:
							max_iter = st.slider(
								"Maximum Iterations",
								min_value=10,
								max_value=1000,
								value=100,
								step=10,
								help="Maximum number of iterations for LDA"
							)

						lda_col3, lda_col4 = st.columns(2)

						with lda_col3:
							vectorizer_type = st.selectbox(
								"Vectorizer Type",
								["Count", "TF-IDF"],
								index=0,
								help="Method to convert text to numerical features"
							)

						with lda_col4:
							max_features = st.number_input(
								"Maximum Features",
								min_value=100,
								max_value=10000,
								value=1000,
								step=100,
								help="Maximum number of terms to include in the vocabulary"
							)

						learning_method = st.selectbox(
							"Learning Method",
							["Online", "Batch"],
							index=0,
							help="Method for LDA parameter estimation"
						)

						# Button to run LDA
						if st.button("Run LDA Topic Modeling"):
							try:
								with st.spinner(f"Running LDA topic modeling with {n_topics} topics..."):
									# Map parameters
									vectorizer_map = {
										"Count": "count",
										"TF-IDF": "tfidf"
									}

									learning_map = {
										"Online": "online",
										"Batch": "batch"
									}

									# Run LDA
									lda_model, doc_topic_matrix, topic_term_matrix = self.clustering_analyzer.run_lda_topic_modeling(
										documents,
										n_topics=n_topics,
										vectorizer_type=vectorizer_map[vectorizer_type],
										max_features=max_features,
										max_iter=max_iter,
										learning_method=learning_map[learning_method]
									)

									# Store results in session state
									st.session_state.lda_results = {
										'doc_topic_matrix': doc_topic_matrix,
										'topic_term_matrix': topic_term_matrix
									}

									# Show success message
									st.success(f"LDA topic modeling complete with {n_topics} topics!")

									# Extract top terms per topic
									top_terms = self.clustering_analyzer.get_top_terms_per_topic(
										topic_term_matrix,
										n_terms=10
									)

									# Display top terms per topic
									st.markdown("<h4>Top Terms for Each Topic</h4>", unsafe_allow_html=True)
									st.dataframe(top_terms, use_container_width=True)

									# Display document-topic distribution
									st.markdown("<h4>Document-Topic Distribution</h4>", unsafe_allow_html=True)

									# Create heatmap of document-topic matrix (sample)
									sample_size = min(20, doc_topic_matrix.shape[0])
									doc_topic_sample = doc_topic_matrix.iloc[:sample_size]

									fig = px.imshow(
										doc_topic_sample,
										labels=dict(x="Topic", y="Document", color="Probability"),
										title=f"Document-Topic Distribution (Sample of {sample_size} Documents)",
										color_continuous_scale="Viridis"
									)
									st.plotly_chart(fig, use_container_width=True)

									# Topic distribution visualization
									st.markdown("<h4>Topic Distribution Overview</h4>", unsafe_allow_html=True)

									# Get the dominant topic for each document
									dominant_topics = doc_topic_matrix.idxmax(axis=1).value_counts().sort_index()

									# Create bar chart of topic distribution
									fig = px.bar(
										x=dominant_topics.index,
										y=dominant_topics.values,
										labels={'x': 'Topic', 'y': 'Document Count'},
										title="Documents per Dominant Topic"
									)
									st.plotly_chart(fig, use_container_width=True)

									# Create visualization of topic similarity/distance
									if n_topics > 1:
										st.markdown("<h4>Topic Similarity</h4>", unsafe_allow_html=True)

										# Calculate topic similarity using cosine similarity
										from sklearn.metrics.pairwise import cosine_similarity
										topic_vectors = topic_term_matrix.values
										topic_similarity = cosine_similarity(topic_vectors)

										# Create heatmap of topic similarity
										fig = px.imshow(
											topic_similarity,
											labels=dict(x="Topic", y="Topic", color="Cosine Similarity"),
											x=[f"Topic {i+1}" for i in range(n_topics)],
											y=[f"Topic {i+1}" for i in range(n_topics)],
											title="Topic Similarity Matrix",
											color_continuous_scale="Viridis"
										)
										st.plotly_chart(fig, use_container_width=True)

							except Exception as e:
								st.error(f"Error running LDA topic modeling: {str(e)}")

						# Save model and results (if available)
						if 'lda_results' in st.session_state and st.session_state.lda_results:
							st.markdown("<h4>Save LDA Results</h4>", unsafe_allow_html=True)

							save_col1, save_col2 = st.columns(2)

							with save_col1:
								if st.button("Save LDA Model"):
									try:
										# Save model
										model_path = self.clustering_analyzer.save_model(
											"lda",
											os.path.dirname(st.session_state.current_file_path)
										)
										st.success(f"Saved LDA model to {model_path}")
									except Exception as e:
										st.error(f"Error saving LDA model: {str(e)}")

							with save_col2:
								if st.button("Save Topic Distributions"):
									try:
										# Get document-topic matrix from results
										doc_topic_df = st.session_state.lda_results['doc_topic_matrix']

										# Save
										filepath = self.feature_engineer.save_features(
											doc_topic_df,
											"lda_topic_distributions",
											os.path.dirname(st.session_state.current_file_path),
											"csv"
										)
										st.success(f"Saved topic distributions to {filepath}")
									except Exception as e:
										st.error(f"Error saving topic distributions: {str(e)}")
				else:
					st.warning("No data available. Please load a dataset first.")

			# 7. Evaluation Metrics Tab
			with clustering_tabs[6]:
				st.markdown("<h3>Clustering Evaluation Metrics</h3>", unsafe_allow_html=True)

				st.markdown("""
				<div class='info-box'>
				This section compares the results of different clustering algorithms using various evaluation metrics.
				It helps you determine which algorithm performs best for your data.
				</div>
				""", unsafe_allow_html=True)

				# Check if we have any clustering results
				has_kmeans = st.session_state.kmeans_labels is not None
				has_hierarchical = st.session_state.hierarchical_labels is not None
				has_dbscan = st.session_state.dbscan_labels is not None

				if any([has_kmeans, has_hierarchical, has_dbscan]):
					# Create table of metrics
					metrics_data = {
						'Algorithm': [],
						'Number of Clusters': [],
						'Silhouette Score': [],
						'Davies-Bouldin Index': [],
						'Calinski-Harabasz Index': []
					}

					# Add K-means metrics if available
					if has_kmeans:
						# Get metrics
						kmeans_metrics = self.clustering_analyzer.cluster_metrics.get('kmeans', {})

						# Add to table
						metrics_data['Algorithm'].append('K-means')
						metrics_data['Number of Clusters'].append(len(set(st.session_state.kmeans_labels)))
						metrics_data['Silhouette Score'].append(kmeans_metrics.get('silhouette_score', 'N/A'))
						metrics_data['Davies-Bouldin Index'].append(kmeans_metrics.get('davies_bouldin_score', 'N/A'))
						metrics_data['Calinski-Harabasz Index'].append(kmeans_metrics.get('calinski_harabasz_score', 'N/A'))

					# Add hierarchical metrics if available
					if has_hierarchical:
						# Get metrics
						hierarchical_metrics = self.clustering_analyzer.cluster_metrics.get('hierarchical', {})

						# Add to table
						metrics_data['Algorithm'].append('Hierarchical')
						metrics_data['Number of Clusters'].append(len(set(st.session_state.hierarchical_labels)))
						metrics_data['Silhouette Score'].append(hierarchical_metrics.get('silhouette_score', 'N/A'))
						metrics_data['Davies-Bouldin Index'].append(hierarchical_metrics.get('davies_bouldin_score', 'N/A'))
						metrics_data['Calinski-Harabasz Index'].append(hierarchical_metrics.get('calinski_harabasz_score', 'N/A'))

					# Add DBSCAN metrics if available
					if has_dbscan:
						# Get metrics
						dbscan_metrics = self.clustering_analyzer.cluster_metrics.get('dbscan', {})

						# Add to table
						metrics_data['Algorithm'].append('DBSCAN')
						metrics_data['Number of Clusters'].append(len(set(st.session_state.dbscan_labels)) - (1 if -1 in st.session_state.dbscan_labels else 0))
						metrics_data['Silhouette Score'].append(dbscan_metrics.get('silhouette_score', 'N/A'))
						metrics_data['Davies-Bouldin Index'].append(dbscan_metrics.get('davies_bouldin_score', 'N/A'))
						metrics_data['Calinski-Harabasz Index'].append(dbscan_metrics.get('calinski_harabasz_score', 'N/A'))

					# Convert to DataFrame
					metrics_df = pd.DataFrame(metrics_data)

					# Display metrics table
					st.markdown("<h4>Clustering Performance Metrics</h4>", unsafe_allow_html=True)
					st.dataframe(metrics_df, use_container_width=True)

					# Create visualizations of metrics
					st.markdown("<h4>Metric Comparison</h4>", unsafe_allow_html=True)

					# Create bar chart for each metric
					metrics_to_viz = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']

					for metric in metrics_to_viz:
						# Skip if all values are N/A
						if all(val == 'N/A' for val in metrics_df[metric]):
							continue

						# Filter out N/A values
						temp_df = metrics_df[metrics_df[metric] != 'N/A'].copy()

						if len(temp_df) > 0:
							# Create bar chart
							fig = px.bar(
								temp_df,
								x='Algorithm',
								y=metric,
								title=f"Comparison of {metric}",
								color='Algorithm',
								color_discrete_sequence=px.colors.qualitative.G10
							)

							# Add note about which direction is better
							if metric == 'Davies-Bouldin Index':
								fig.add_annotation(
									text="Lower is better",
									xref="paper", yref="paper",
									x=0.5, y=1.05,
									showarrow=False
								)
							else:
								fig.add_annotation(
									text="Higher is better",
									xref="paper", yref="paper",
									x=0.5, y=1.05,
									showarrow=False
								)

							st.plotly_chart(fig, use_container_width=True)

					# Model Selection Guide
					st.markdown("<h4>Clustering Algorithm Selection Guide</h4>", unsafe_allow_html=True)

					# Determine best algorithm for each metric
					best_silhouette = None
					best_davies_bouldin = None
					best_calinski_harabasz = None

					if 'Silhouette Score' in metrics_df.columns:
						valid_scores = metrics_df[metrics_df['Silhouette Score'] != 'N/A']
						if not valid_scores.empty:
							best_silhouette = valid_scores.loc[valid_scores['Silhouette Score'].astype(float).idxmax()]['Algorithm']

					if 'Davies-Bouldin Index' in metrics_df.columns:
						valid_scores = metrics_df[metrics_df['Davies-Bouldin Index'] != 'N/A']
						if not valid_scores.empty:
							best_davies_bouldin = valid_scores.loc[valid_scores['Davies-Bouldin Index'].astype(float).idxmin()]['Algorithm']

					if 'Calinski-Harabasz Index' in metrics_df.columns:
						valid_scores = metrics_df[metrics_df['Calinski-Harabasz Index'] != 'N/A']
						if not valid_scores.empty:
							best_calinski_harabasz = valid_scores.loc[valid_scores['Calinski-Harabasz Index'].astype(float).idxmax()]['Algorithm']

					# Create a recommendations section
					with st.expander("Algorithm Recommendations", expanded=True):
						if best_silhouette or best_davies_bouldin or best_calinski_harabasz:
							st.markdown("""
							Based on the evaluation metrics, here are the top-performing algorithms:
							""")

							recommendations = []

							if best_silhouette:
								recommendations.append(f"- **Best Silhouette Score**: {best_silhouette}")

							if best_davies_bouldin:
								recommendations.append(f"- **Best Davies-Bouldin Index**: {best_davies_bouldin}")

							if best_calinski_harabasz:
								recommendations.append(f"- **Best Calinski-Harabasz Index**: {best_calinski_harabasz}")

							for rec in recommendations:
								st.markdown(rec)

							# Count algorithm occurrences
							from collections import Counter

							algorithms = [a for a in [best_silhouette, best_davies_bouldin, best_calinski_harabasz] if a]
							algo_count = Counter(algorithms)

							# Make a final recommendation
							if algo_count:
								most_common = algo_count.most_common(1)[0][0]
								st.markdown(f"""
								### Overall Recommendation

								**{most_common}** appears to be the best clustering algorithm for this dataset
								based on the evaluation metrics.
								""")
						else:
							st.markdown("""
							No metrics are available for comparison. Please run at least one clustering algorithm
							to generate evaluation metrics.
							""")

					# Save metrics report
					if st.button("Save Metrics Report"):
						try:
							# Create directory if it doesn't exist
							reports_dir = os.path.join(os.path.dirname(st.session_state.current_file_path), 'reports')
							os.makedirs(reports_dir, exist_ok=True)

							# Create timestamp for filename
							timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

							# Create report file path
							report_path = os.path.join(reports_dir, f"clustering_metrics_report_{timestamp}.csv")

							# Save metrics DataFrame
							metrics_df.to_csv(report_path, index=False)

							# Show success message
							st.success(f"Saved metrics report to {report_path}")
						except Exception as e:
							st.error(f"Error saving metrics report: {str(e)}")

					# Compare cluster assignments
					st.markdown("<h4>Cluster Assignment Comparison</h4>", unsafe_allow_html=True)

					# Check if we have at least two clustering results
					algorithms_with_results = []
					if has_kmeans:
						algorithms_with_results.append("K-means")
					if has_hierarchical:
						algorithms_with_results.append("Hierarchical")
					if has_dbscan:
						algorithms_with_results.append("DBSCAN")

					if len(algorithms_with_results) >= 2:
						# Select two algorithms to compare
						compare_col1, compare_col2 = st.columns(2)

						with compare_col1:
							algo1 = st.selectbox(
								"First Algorithm",
								algorithms_with_results,
								index=0,
								key="compare_algo1"
							)

						with compare_col2:
							# Default to second algorithm in list
							default_idx = 1 if len(algorithms_with_results) > 1 else 0
							algo2 = st.selectbox(
								"Second Algorithm",
								algorithms_with_results,
								index=default_idx,
								key="compare_algo2"
							)

						# Get cluster labels for selected algorithms
						labels1 = None
						labels2 = None

						if algo1 == "K-means":
							labels1 = st.session_state.kmeans_labels
						elif algo1 == "Hierarchical":
							labels1 = st.session_state.hierarchical_labels
						elif algo1 == "DBSCAN":
							labels1 = st.session_state.dbscan_labels

						if algo2 == "K-means":
							labels2 = st.session_state.kmeans_labels
						elif algo2 == "Hierarchical":
							labels2 = st.session_state.hierarchical_labels
						elif algo2 == "DBSCAN":
							labels2 = st.session_state.dbscan_labels

						# Create comparison if we have both labels
						if labels1 is not None and labels2 is not None:
							# Create contingency table
							contingency = pd.crosstab(
								labels1,
								labels2,
								rownames=[f"{algo1} Clusters"],
								colnames=[f"{algo2} Clusters"]
							)

							# Display contingency table
							st.markdown(f"<h5>Agreement between {algo1} and {algo2}</h5>", unsafe_allow_html=True)
							st.dataframe(contingency, use_container_width=True)

							# Create heatmap
							fig = px.imshow(
								contingency,
								labels=dict(x=f"{algo2} Clusters", y=f"{algo1} Clusters", color="Count"),
								title=f"Cluster Assignment Comparison: {algo1} vs {algo2}",
								color_continuous_scale="Viridis"
							)
							st.plotly_chart(fig, use_container_width=True)

							# Calculate agreement metrics
							from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

							# Handle noise points in DBSCAN (exclude from comparison)
							if algo1 == "DBSCAN" or algo2 == "DBSCAN":
								# Create mask for non-noise points
								mask1 = labels1 != -1 if algo1 == "DBSCAN" else pd.Series(True, index=labels1.index)
								mask2 = labels2 != -1 if algo2 == "DBSCAN" else pd.Series(True, index=labels2.index)

								# Combined mask
								combined_mask = mask1 & mask2

								# Filter labels
								compare_labels1 = labels1[combined_mask]
								compare_labels2 = labels2[combined_mask]

								# Note about noise points
								st.info("Noise points (cluster -1) from DBSCAN are excluded from the agreement calculation.")
							else:
								compare_labels1 = labels1
								compare_labels2 = labels2

							# Calculate metrics
							ari = adjusted_rand_score(compare_labels1, compare_labels2)
							ami = adjusted_mutual_info_score(compare_labels1, compare_labels2)

							# Display metrics
							st.markdown(f"""
							**Agreement Metrics:**
							- Adjusted Rand Index (ARI): {ari:.4f} (ranges from -1 to 1, where 1 means perfect agreement)
							- Adjusted Mutual Information (AMI): {ami:.4f} (ranges from 0 to 1, where 1 means perfect agreement)
							""")

							# Interpret agreement
							if ari > 0.8:
								st.success("The two clustering algorithms show very strong agreement in their results.")
							elif ari > 0.5:
								st.info("The two clustering algorithms show moderate agreement in their results.")
							elif ari > 0.2:
								st.warning("The two clustering algorithms show weak agreement in their results.")
							else:
								st.error("The two clustering algorithms show very little agreement in their results.")
					else:
						st.info("Run at least two clustering algorithms to compare their results.")
				else:
					st.warning("No clustering results available. Please run at least one clustering algorithm first.")

				# Load and Compare Models Section
				st.markdown("<h4>Load and Compare Saved Models</h4>", unsafe_allow_html=True)

				# File uploader for saved models
				uploaded_model = st.file_uploader(
					"Load a saved clustering model (PKL file)",
					type=["pkl"]
				)

				if uploaded_model is not None:
					try:
						# Read the uploaded model
						model_bytes = uploaded_model.read()

						# Create a BytesIO object
						bytes_io = BytesIO(model_bytes)

						# Load the model
						model = pickle.load(bytes_io)

						# Get model type
						if isinstance(model, KMeans):
							model_type = "kmeans"
						elif isinstance(model, AgglomerativeClustering):
							model_type = "hierarchical"
						elif isinstance(model, DBSCAN):
							model_type = "dbscan"
						elif 'model' in model and isinstance(model['model'], LatentDirichletAllocation):
							model_type = "lda"
						else:
							model_type = "unknown"

						# Show model info
						st.success(f"Successfully loaded {model_type.upper()} model from {uploaded_model.name}")

						# Store model for comparison
						model_name = f"loaded_{model_type}"

						# Store in clustering analyzer
						self.clustering_analyzer.load_model(bytes_io, model_name)

						# Show option to apply model to current data
						if st.button("Apply Loaded Model to Current Data"):
							# Check if we have input data
							if st.session_state.clustering_input_data is not None:
								try:
									with st.spinner(f"Applying loaded {model_type.upper()} model to current data..."):
										# Apply model based on type
										if model_type == "kmeans":
											# Predict clusters
											labels = pd.Series(
												model.predict(st.session_state.clustering_input_data),
												index=st.session_state.clustering_input_data.index,
												name="cluster"
											)

											# Store labels
											st.session_state.kmeans_labels = labels

											# Calculate metrics
											metrics = self.clustering_analyzer.evaluate_clustering(
												st.session_state.clustering_input_data,
												labels,
												"kmeans"
											)

											# Show success message
											st.success(f"""
											Applied K-means model to current data.
											- Number of clusters: {model.n_clusters}
											- Silhouette score: {metrics['silhouette_score']:.4f}
											- Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f}
											- Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.4f}
											""")

										elif model_type == "hierarchical":
											# Predict clusters
											labels = pd.Series(
												model.fit_predict(st.session_state.clustering_input_data),
												index=st.session_state.clustering_input_data.index,
												name="cluster"
											)

											# Store labels
											st.session_state.hierarchical_labels = labels

											# Calculate metrics
											metrics = self.clustering_analyzer.evaluate_clustering(
												st.session_state.clustering_input_data,
												labels,
												"hierarchical"
											)

											# Show success message
											st.success(f"""
											Applied Hierarchical Clustering model to current data.
											- Number of clusters: {model.n_clusters}
											- Silhouette score: {metrics['silhouette_score']:.4f}
											- Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f}
											- Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.4f}
											""")

										elif model_type == "dbscan":
											# Predict clusters
											labels = pd.Series(
												model.fit_predict(st.session_state.clustering_input_data),
												index=st.session_state.clustering_input_data.index,
												name="cluster"
											)

											# Store labels
											st.session_state.dbscan_labels = labels

											# Count clusters and noise points
											n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
											n_noise = list(labels).count(-1)

											# Calculate metrics
											if n_clusters > 1:
												# Exclude noise points
												non_noise_mask = labels != -1
												metrics = self.clustering_analyzer.evaluate_clustering(
													st.session_state.clustering_input_data[non_noise_mask],
													labels[non_noise_mask],
													"dbscan"
												)

												# Show success message
												st.success(f"""
												Applied DBSCAN model to current data.
												- Number of clusters: {n_clusters}
												- Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)
												- Silhouette score: {metrics['silhouette_score']:.4f}
												- Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f}
												- Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.4f}
												""")
											else:
												st.warning(f"""
												Applied DBSCAN model to current data.
												- Number of clusters: {n_clusters}
												- Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)
												- No metrics available with fewer than 2 clusters.
												""")

										elif model_type == "lda":
											st.warning("""
											LDA models cannot be applied directly to the current data.
											Please run a new LDA model on your current text data.
											""")
								except Exception as e:
									st.error(f"Error applying loaded model to current data: {str(e)}")
							else:
								st.warning("No input data available. Please prepare data in the Data Selection tab first.")
					except Exception as e:
						st.error(f"Error loading model: {str(e)}")


	def _analysis_visualization_tab(self):
		"""Display the analysis and visualization tab content."""
		st.markdown("<h2 class='sub-header'>Analysis & Visualization</h2>", unsafe_allow_html=True)

		# Introductory text
		st.markdown("""
		<div class='info-box'>
		This section provides deeper insights into identified clusters, with statistical analysis,
		feature importance, and comprehensive visualizations of patient patterns.
		</div>
		""", unsafe_allow_html=True)

		# Analysis subtabs
		analysis_tabs = st.tabs([
			"📊 Length of Stay Analysis",
			"📈 Order Pattern Visualization",
			"🔍 Statistical Testing",
			"📋 Cluster Characterization",
			"🔥 Feature Importance"
		])

		# Check if we have clustering results
		has_kmeans = st.session_state.kmeans_labels is not None
		has_hierarchical = st.session_state.hierarchical_labels is not None
		has_dbscan = st.session_state.dbscan_labels is not None

		# Check if we have sufficient data for analysis
		if not any([has_kmeans, has_hierarchical, has_dbscan]):
			st.warning("No clustering results found. Please run clustering algorithms first.")
			return

		# Allow user to select which clustering result to analyze
		cluster_options = []
		if has_kmeans:
			cluster_options.append("K-means")
		if has_hierarchical:
			cluster_options.append("Hierarchical")
		if has_dbscan:
			cluster_options.append("DBSCAN")

		selected_clustering = st.selectbox(
			"Select Clustering Results to Analyze",
			cluster_options
		)

		# Get appropriate cluster labels
		if selected_clustering == "K-means":
			cluster_labels = st.session_state.kmeans_labels
		elif selected_clustering == "Hierarchical":
			cluster_labels = st.session_state.hierarchical_labels
		else:  # DBSCAN
			cluster_labels = st.session_state.dbscan_labels

		# Get appropriate data
		if st.session_state.reduced_data is not None:
			analysis_data = st.session_state.reduced_data.copy()
		else:
			analysis_data = st.session_state.clustering_input_data.copy()

		if analysis_data is not None:
			# Add cluster labels to data
			analysis_data['cluster'] = cluster_labels

		# 1. Length of Stay Analysis tab
		with analysis_tabs[0]:
			st.markdown("<h3>Length of Stay Analysis by Cluster</h3>", unsafe_allow_html=True)

			if st.session_state.df is not None:
				# Select admission and discharge columns
				time_columns = st.session_state.df.select_dtypes(include=['datetime64']).columns.tolist()
				object_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()

				# If no datetime columns, check if we can convert object columns
				potential_date_columns = []
				for col in object_columns:
					# Check first few values for date-like strings
					sample = st.session_state.df[col].dropna().head(5).astype(str)
					date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}']

					if any(sample.str.contains(pattern).any() for pattern in date_patterns):
						potential_date_columns.append(col)

				time_columns = time_columns + potential_date_columns

				if time_columns:
					col1, col2 = st.columns(2)

					with col1:
						admission_col = st.selectbox(
							"Select Admission Time Column",
							time_columns,
							key="los_admission_col"
						)

					with col2:
						# Try to suggest discharge column based on naming
						discharge_idx = 0
						for idx, col in enumerate(time_columns):
							if "discharge" in col.lower() or "end" in col.lower():
								discharge_idx = idx
								break

						discharge_col = st.selectbox(
							"Select Discharge Time Column",
							time_columns,
							index=discharge_idx,
							key="los_discharge_col"
						)

					patient_id_col = st.selectbox(
						"Select Patient ID Column",
						st.session_state.df.columns.tolist(),
						index=0,
						key="los_patient_id_col"
					)

					# Button to calculate LOS
					if st.button("Calculate Length of Stay"):
						try:
							# Copy dataframe to avoid modifying original
							df_copy = st.session_state.df.copy()

							# Convert columns to datetime if needed
							for col in [admission_col, discharge_col]:
								if df_copy[col].dtype != 'datetime64[ns]':
									df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

							# Calculate LOS
							los = self.cluster_analyzer.calculate_length_of_stay(
								df_copy,
								admission_col,
								discharge_col,
								patient_id_col
							)

							# Store in session state
							st.session_state.length_of_stay = los

							# Compare across clusters
							los_comparison = self.cluster_analyzer.compare_los_across_clusters(
								los,
								cluster_labels
							)

							# Display results
							st.success("Length of stay calculated successfully!")
							st.dataframe(los_comparison, use_container_width=True)

							# Create boxplot
							los_data = pd.DataFrame({
								'length_of_stay': los,
								'cluster': cluster_labels
							})

							fig = px.box(
								los_data,
								x='cluster',
								y='length_of_stay',
								title="Length of Stay by Cluster",
								labels={
									'cluster': 'Cluster',
									'length_of_stay': 'Length of Stay (days)'
								},
								color='cluster'
							)

							st.plotly_chart(fig, use_container_width=True)

							# Statistical test for LOS differences
							st.markdown("<h4>Statistical Analysis of LOS Differences</h4>", unsafe_allow_html=True)

							# Use ANOVA if more than 2 clusters, otherwise t-test
							from scipy import stats
							unique_clusters = los_data['cluster'].unique()

							if len(unique_clusters) > 2:
								# Group data by cluster
								groups = [
									los_data[los_data['cluster'] == cluster]['length_of_stay'].dropna().values
									for cluster in unique_clusters if cluster != -1  # Exclude noise points
								]

								# Perform ANOVA if we have valid groups
								if len(groups) >= 2 and all(len(g) > 0 for g in groups):
									f_stat, p_value = stats.f_oneway(*groups)

									st.markdown(f"""
									**One-way ANOVA Test Results:**
									- F-statistic: {f_stat:.4f}
									- P-value: {p_value:.4f}
									- Significant difference: {"Yes" if p_value < 0.05 else "No"}
									""")
							elif len(unique_clusters) == 2:
								# Get the two clusters
								c1, c2 = unique_clusters

								# Get data for each cluster
								group1 = los_data[los_data['cluster'] == c1]['length_of_stay'].dropna().values
								group2 = los_data[los_data['cluster'] == c2]['length_of_stay'].dropna().values

								# Perform t-test
								if len(group1) > 0 and len(group2) > 0:
									t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

									st.markdown(f"""
									**Independent t-test Results:**
									- t-statistic: {t_stat:.4f}
									- P-value: {p_value:.4f}
									- Significant difference: {"Yes" if p_value < 0.05 else "No"}
									""")

						except Exception as e:
							st.error(f"Error calculating length of stay: {str(e)}")
				else:
					st.warning("No suitable datetime columns found for length of stay calculation.")
			else:
				st.warning("No dataset loaded. Please load data first.")

		# 2. Order Pattern Visualization tab
		with analysis_tabs[1]:
			st.markdown("<h3>Order Pattern Visualization</h3>", unsafe_allow_html=True)

			# Check if we have order sequences
			if st.session_state.order_sequences is not None:
				st.markdown("""
				<div class='info-box'>
				Visualize patterns of orders across different clusters to identify characteristic
				ordering behaviors for each patient group.
				</div>
				""", unsafe_allow_html=True)

				# Parameters for visualization
				col1, col2 = st.columns(2)

				with col1:
					max_orders = st.slider(
						"Maximum Order Types",
						min_value=5,
						max_value=50,
						value=20,
						help="Maximum number of order types to include (most frequent)"
					)

				with col2:
					max_patients = st.slider(
						"Maximum Patients per Cluster",
						min_value=10,
						max_value=100,
						value=30,
						help="Maximum number of patients to visualize per cluster"
					)

				# Button to generate visualization
				if st.button("Generate Order Pattern Visualization"):
					try:
						# Create visualization
						fig = self.cluster_analyzer.visualize_order_patterns(
							st.session_state.order_sequences,
							cluster_labels,
							max_orders=max_orders,
							max_patients=max_patients
						)

						# Display figure
						st.plotly_chart(fig, use_container_width=True)

						# Create sankey diagram of order transitions
						st.markdown("<h4>Order Transition Patterns by Cluster</h4>", unsafe_allow_html=True)

						# Create transition data by cluster
						from collections import defaultdict

						# Get top clusters (limit to top 3 for clarity)
						top_clusters = cluster_labels.value_counts().head(3).index.tolist()

						# For each cluster
						for cluster in top_clusters:
							# Get patients in this cluster
							cluster_patients = cluster_labels[cluster_labels == cluster].index
							cluster_patients = list(set(cluster_patients) & set(st.session_state.order_sequences.keys()))

							# Skip if no patients
							if not cluster_patients:
								continue

							# Count transitions
							transitions = defaultdict(int)

							for patient_id in cluster_patients:
								sequence = st.session_state.order_sequences[patient_id]

								# Count transitions between orders
								for i in range(len(sequence) - 1):
									from_order = sequence[i]
									to_order = sequence[i + 1]
									transitions[(from_order, to_order)] += 1

							# Create sankey data (limit to top 20 transitions)
							top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:20]

							# Get unique orders involved in top transitions
							unique_orders = set()
							for (from_order, to_order), _ in top_transitions:
								unique_orders.add(from_order)
								unique_orders.add(to_order)

							# Map orders to indices
							order_to_idx = {order: i for i, order in enumerate(unique_orders)}

							# Create node and link data
							node_labels = list(unique_orders)
							source_indices = []
							target_indices = []
							values = []

							for (from_order, to_order), count in top_transitions:
								source_indices.append(order_to_idx[from_order])
								target_indices.append(order_to_idx[to_order])
								values.append(count)

							# Create sankey diagram
							if node_labels and source_indices and target_indices and values:
								fig = go.Figure(data=[go.Sankey(
									node=dict(
										pad=15,
										thickness=20,
										line=dict(color="black", width=0.5),
										label=node_labels
									),
									link=dict(
										source=source_indices,
										target=target_indices,
										value=values
									)
								)])

								fig.update_layout(
									title_text=f"Top Order Transitions in Cluster {cluster}",
									font_size=10
								)

								st.plotly_chart(fig, use_container_width=True)

					except Exception as e:
						st.error(f"Error generating order pattern visualization: {str(e)}")
			else:
				st.warning("No order sequences available. Please generate them in the Feature Engineering tab first.")

		# 3. Statistical Testing tab
		with analysis_tabs[2]:
			st.markdown("<h3>Statistical Testing Between Clusters</h3>", unsafe_allow_html=True)

			if analysis_data is not None:
				st.markdown("""
				<div class='info-box'>
				Perform statistical tests to identify significant differences in features between clusters.
				This helps understand what characteristics distinguish each patient group.
				</div>
				""", unsafe_allow_html=True)

				# Get feature columns (exclude cluster column)
				feature_cols = [col for col in analysis_data.columns if col != 'cluster']

				# Let user select features
				selected_features = st.multiselect(
					"Select Features for Testing",
					feature_cols,
					default=feature_cols[:min(5, len(feature_cols))]
				)

				# Test method
				test_method = st.radio(
					"Statistical Test Method",
					["ANOVA (parametric)", "Kruskal-Wallis (non-parametric)"],
					horizontal=True
				)

				# Map to method name
				method_map = {
					"ANOVA (parametric)": "anova",
					"Kruskal-Wallis (non-parametric)": "kruskal"
				}

				# Button to run tests
				if st.button("Run Statistical Tests") and selected_features:
					try:
						# Run tests
						results = self.cluster_analyzer.statistical_testing(
							analysis_data,
							selected_features,
							cluster_col='cluster',
							method=method_map[test_method]
						)

						# Display results
						st.success("Statistical tests completed!")

						# Format p-values
						results_display = results.copy()
						for col in ['P-Value', 'Adjusted P-Value']:
							if col in results_display.columns:
								results_display[col] = results_display[col].apply(
									lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
								)

						# Display as table with highlighting
						st.markdown("<h4>Test Results</h4>", unsafe_allow_html=True)

						# Custom styling
						def color_significant(val):
							if val == True:
								return 'background-color: #d4efdf'
							elif val == False:
								return 'background-color: #fadbd8'
							else:
								return ''

						# Apply styling and display
						styled_results = results_display.style.applymap(
							color_significant,
							subset=['Significant', 'Significant (Adjusted)']
						)

						st.dataframe(styled_results, use_container_width=True)

						# Show visualization of p-values
						st.markdown("<h4>Feature Significance Visualization</h4>", unsafe_allow_html=True)

						# Create bar chart of -log(p-value)
						log_p_values = -np.log10(results['P-Value'])
						significance_threshold = -np.log10(0.05)

						# Create color list (red if significant, grey if not)
						colors = ['rgba(255, 99, 132, 0.8)' if p < 0.05 else 'rgba(203, 213, 232, 0.8)'
								for p in results['P-Value']]

						fig = go.Figure()

						fig.add_trace(go.Bar(
							x=results['Feature'],
							y=log_p_values,
							marker_color=colors,
							text=results['P-Value'].apply(lambda x: f"p={x:.4f}"),
							name='Feature Significance'
						))

						# Add significance threshold line
						fig.add_shape(
							type='line',
							x0=-0.5,
							y0=significance_threshold,
							x1=len(results['Feature']) - 0.5,
							y1=significance_threshold,
							line=dict(
								color='red',
								width=2,
								dash='dash'
							)
						)

						fig.update_layout(
							title='Feature Significance by -log10(p-value)',
							xaxis_title='Feature',
							yaxis_title='-log10(p-value)',
							xaxis_tickangle=-45,
							showlegend=False
						)

						# Add annotation for threshold
						fig.add_annotation(
							x=len(results['Feature']) - 1,
							y=significance_threshold,
							text='p=0.05',
							showarrow=False,
							yshift=10
						)

						st.plotly_chart(fig, use_container_width=True)
					except Exception as e:
						st.error(f"Error performing statistical tests: {str(e)}")
			else:
				st.warning("No data available for statistical testing.")

		# 4. Cluster Characterization tab
		with analysis_tabs[3]:
			st.markdown("<h3>Cluster Characterization</h3>", unsafe_allow_html=True)

			if analysis_data is not None:
				st.markdown("""
				<div class='info-box'>
				Generate comprehensive profiles of each cluster to understand their key characteristics
				and differences. This provides insights into the clinical meaning of each patient group.
				</div>
				""", unsafe_allow_html=True)

				# Get original feature columns if available
				if st.session_state.clustering_input_data is not None:
					orig_cols = list(st.session_state.clustering_input_data.columns)
				else:
					orig_cols = list(analysis_data.columns)
					if 'cluster' in orig_cols:
						orig_cols.remove('cluster')

				# Let user select important features
				selected_features = st.multiselect(
					"Select Key Features for Characterization",
					orig_cols,
					default=orig_cols[:min(10, len(orig_cols))]
				)

				# Button to generate characterization
				if st.button("Generate Cluster Characterization") and selected_features:
					try:
						# Generate characterization
						characterization = self.cluster_analyzer.generate_cluster_characterization(
							analysis_data,
							cluster_col='cluster',
							important_features=selected_features
						)

						# Display characterization
						st.success("Cluster characterization generated!")

						# Create tabs for each cluster
						cluster_tabs = st.tabs([f"Cluster {c}" for c in characterization.keys()])

						# For each cluster
						for i, (cluster_name, stats) in enumerate(characterization.items()):
							with cluster_tabs[i]:
								# Basic info
								st.markdown(f"""
								**Size:** {stats['size']} patients ({stats['percentage']:.2f}% of dataset)
								""")

								# Feature statistics table
								st.markdown("### Key Feature Statistics")

								# Create dataframe of feature stats
								feature_stats = []
								for feature, f_stats in stats['features'].items():
									feature_stats.append({
										'Feature': feature,
										'Mean': f_stats['mean'],
										'Median': f_stats['median'],
										'Std Dev': f_stats['std'],
										'Min': f_stats['min'],
										'Max': f_stats['max'],
										'Diff from Mean (%)': f_stats.get('diff_from_mean', 0)
									})

								# Convert to dataframe and sort by difference from mean
								feature_df = pd.DataFrame(feature_stats)
								feature_df = feature_df.sort_values('Diff from Mean (%)', key=abs, ascending=False)

								# Format numeric columns
								numeric_cols = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Diff from Mean (%)']
								for col in numeric_cols:
									feature_df[col] = feature_df[col].round(2)

								# Apply styling to highlight distinguishing features
								def color_diff(val):
									if pd.isna(val):
										return ''
									if abs(val) > 50:
										return 'background-color: rgba(255, 99, 132, 0.6)'
									elif abs(val) > 25:
										return 'background-color: rgba(255, 205, 86, 0.6)'
									elif abs(val) > 10:
										return 'background-color: rgba(75, 192, 192, 0.4)'
									else:
										return ''

								# Apply styling
								styled_df = feature_df.style.applymap(
									color_diff,
									subset=['Diff from Mean (%)']
								)

								# Display table
								st.dataframe(styled_df, use_container_width=True)

								# Radar chart of key features
								st.markdown("### Feature Profile")

								# Get top N differentiating features
								top_n = min(8, len(feature_df))
								top_features = feature_df.head(top_n)

								# Create radar chart
								categories = top_features['Feature'].tolist()
								values = top_features['Diff from Mean (%)'].tolist()

								fig = go.Figure()

								fig.add_trace(go.Scatterpolar(
									r=values,
									theta=categories,
									fill='toself',
									name=cluster_name
								))

								fig.update_layout(
									polar=dict(
										radialaxis=dict(
											visible=True,
											range=[-100, 100]
										)
									),
									showlegend=False,
									title=f"Feature Profile for {cluster_name}"
								)

								st.plotly_chart(fig, use_container_width=True)

						# Generate report button
						st.markdown("### Generate PDF Report")
						st.markdown("""
						Generate a comprehensive PDF report with all cluster characteristics and visualizations
						for sharing or documentation.
						""")

						report_title = st.text_input(
							"Report Title",
							value=f"{selected_clustering} Cluster Analysis Report"
						)

						include_plots = st.checkbox("Include Visualizations", value=True)

						if st.button("Generate HTML Report"):
							try:
								# Generate HTML report
								html_report = self.cluster_analyzer.generate_html_report(
									title=report_title,
									include_plots=include_plots
								)

								# Provide download link
								st.success("Report generated! Click below to download.")

								st.download_button(
									label="Download HTML Report",
									data=html_report,
									file_name=f"cluster_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
									mime="text/html"
								)
							except Exception as e:
								st.error(f"Error generating report: {str(e)}")

					except Exception as e:
						st.error(f"Error generating cluster characterization: {str(e)}")
			else:
				st.warning("No data available for cluster characterization.")

		# 5. Feature Importance tab
		with analysis_tabs[4]:
			st.markdown("<h3>Feature Importance Analysis</h3>", unsafe_allow_html=True)

			if analysis_data is not None:
				st.markdown("""
				<div class='info-box'>
				Analyze the importance of different features in distinguishing between clusters.
				This helps identify which characteristics are most influential in defining patient groups.
				</div>
				""", unsafe_allow_html=True)

				# Button to calculate feature importance
				if st.button("Calculate Feature Importance"):
					try:
						# Prepare data (remove non-numeric columns)
						numeric_data = analysis_data.select_dtypes(include=['number'])

						# Skip cluster column if it exists
						if 'cluster' in numeric_data.columns:
							feature_data = numeric_data.drop(columns=['cluster'])
							labels = numeric_data['cluster']
						else:
							feature_data = numeric_data
							labels = cluster_labels

						# Skip if empty
						if feature_data.empty:
							st.error("No numeric features available for importance calculation.")
							return

						# Create combined DataFrame
						combined_data = feature_data.copy()
						combined_data['cluster'] = labels

						# Calculate feature importance
						importance_df = self.cluster_analyzer.calculate_feature_importance(
							combined_data,
							cluster_col='cluster'
						)

						# Display results
						st.success("Feature importance calculated!")

						# Show table
						st.markdown("### Feature Importance Ranking")
						st.dataframe(importance_df, use_container_width=True)

						# Create bar chart
						fig = px.bar(
							importance_df,
							y='Feature',
							x='Importance',
							orientation='h',
							title="Feature Importance for Cluster Differentiation",
							labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
							color='Importance',
							color_continuous_scale='Viridis'
						)

						fig.update_layout(yaxis={'categoryorder': 'total ascending'})

						st.plotly_chart(fig, use_container_width=True)

						# Create heatmap of feature values across clusters
						st.markdown("### Feature Values Across Clusters")

						# Get top 10 important features
						top_features = importance_df.head(10)['Feature'].tolist()

						# Calculate mean values by cluster
						cluster_means = combined_data.groupby('cluster')[top_features].mean()

						# Create heatmap
						fig = px.imshow(
							cluster_means,
							labels=dict(x="Feature", y="Cluster", color="Value"),
							x=top_features,
							y=cluster_means.index,
							title="Feature Values Across Clusters (Mean)",
							color_continuous_scale='Viridis',
							aspect="auto"
						)

						st.plotly_chart(fig, use_container_width=True)

						# Feature correlation analysis
						st.markdown("### Feature Correlation Analysis")

						# Calculate correlation matrix of top features
						corr_matrix = combined_data[top_features].corr()

						# Create heatmap
						fig = px.imshow(
							corr_matrix,
							x=corr_matrix.columns,
							y=corr_matrix.columns,
							title="Feature Correlation Matrix",
							color_continuous_scale='RdBu_r',
							range_color=[-1, 1]
						)

						st.plotly_chart(fig, use_container_width=True)

					except Exception as e:
						st.error(f"Error calculating feature importance: {str(e)}")
			else:
				st.warning("No data available for feature importance analysis.")


if __name__ == "__main__":
	app = MIMICDashboardApp()
	app.run()
