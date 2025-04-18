import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add the parent directory to the path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Constants
DEFAULT_MIMIC_PATH      = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Handling Class ---
class MIMICDataHandler:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""
	def __init__(self):
		pass

	def _format_size(self, size_mb: float) -> str:
		"""Format file size into KB, MB, or GB string."""
		if size_mb < 0.001:
			 return "(< 1 KB)"
		elif size_mb < 1:
			return f"({size_mb * 1024:.1f} KB)"
		elif size_mb < 1000:
			return f"({size_mb:.1f} MB)"
		else:
			return f"({size_mb / 1000:.1f} GB)"

	def scan_mimic_directory(self, mimic_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
		"""Scans the MIMIC-IV directory structure and returns table info."""
		available_tables = {}
		file_paths = {}
		file_sizes = {}
		table_display_names = {}

		if not os.path.exists(mimic_path):
			return available_tables, file_paths, file_sizes, table_display_names

		modules = ['hosp', 'icu']
		for module in modules:
			module_path = os.path.join(mimic_path, module)
			if os.path.exists(module_path):
				available_tables[module] = []
				current_module_tables = {}

				csv_files = glob.glob(os.path.join(module_path, '*.csv'))
				csv_gz_files = glob.glob(os.path.join(module_path, '*.csv.gz'))

				all_files = csv_files + csv_gz_files

				for file_path in all_files:
					is_gz = file_path.endswith('.csv.gz')
					table_name = os.path.basename(file_path).replace('.csv.gz' if is_gz else '.csv', '')

					if table_name in current_module_tables and not is_gz:
						current_module_tables[table_name] = file_path
					elif table_name not in current_module_tables:
						current_module_tables[table_name] = file_path

				for table_name, file_path in current_module_tables.items():
					available_tables[module].append(table_name)
					file_paths[(module, table_name)] = file_path
					try:
						file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
					except OSError:
						file_size_mb = 0
					file_sizes[(module, table_name)] = file_size_mb
					size_str = self._format_size(file_size_mb)
					table_display_names[(module, table_name)] = f"{table_name} {size_str}"

				available_tables[module].sort()

		return available_tables, file_paths, file_sizes, table_display_names

	def load_mimic_table(self, file_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1') -> Optional[pd.DataFrame]:
		"""Loads a specific MIMIC-IV table, handling large files and sampling."""
		try:
			file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
			is_compressed = file_path.endswith('.gz')
			compression = 'gzip' if is_compressed else None

			read_params = {
				'encoding': encoding,
				'compression': compression,
				'low_memory': False
			}

			if file_size_mb > LARGE_FILE_THRESHOLD_MB:
				df = pd.read_csv(file_path, nrows=sample_size, **read_params)
			else:
				df = pd.read_csv(file_path, **read_params)
				if len(df) > sample_size:
					df = df.sample(sample_size, random_state=RANDOM_STATE)
			return df
		except Exception as e:
			logging.error(f"Error loading data from {os.path.basename(file_path)}: {str(e)}")
			return None

	def get_table_info(self, module: str, table_name: str) -> str:
		"""Get descriptive information about a specific MIMIC-IV table."""
		table_info = {
			('hosp', 'admissions'): "Patient hospital admissions information",
			('hosp', 'patients'): "Patient demographic data",
			('hosp', 'labevents'): "Laboratory measurements (large file)",
			('hosp', 'microbiologyevents'): "Microbiology test results",
			('hosp', 'pharmacy'): "Pharmacy orders",
			('hosp', 'prescriptions'): "Medication prescriptions",
			('hosp', 'procedures_icd'): "Patient procedures",
			('hosp', 'diagnoses_icd'): "Patient diagnoses",
			('hosp', 'emar'): "Electronic medication administration records",
			('hosp', 'emar_detail'): "Detailed medication administration data",
			('hosp', 'poe'): "Provider order entries",
			('hosp', 'poe_detail'): "Detailed order information",
			('hosp', 'd_hcpcs'): "HCPCS code definitions",
			('hosp', 'd_icd_diagnoses'): "ICD diagnosis code definitions",
			('hosp', 'd_icd_procedures'): "ICD procedure code definitions",
			('hosp', 'd_labitems'): "Laboratory test definitions",
			('hosp', 'hcpcsevents'): "HCPCS events",
			('hosp', 'drgcodes'): "Diagnosis-related group codes",
			('hosp', 'services'): "Hospital services",
			('hosp', 'transfers'): "Patient transfers",
			('hosp', 'provider'): "Provider information",
			('hosp', 'omr'): "Order monitoring results",
			('icu', 'chartevents'): "Patient charting data (vital signs, etc.)",
			('icu', 'datetimeevents'): "Date/time-based events",
			('icu', 'inputevents'): "Patient intake data",
			('icu', 'outputevents'): "Patient output data",
			('icu', 'procedureevents'): "ICU procedures",
			('icu', 'ingredientevents'): "Detailed medication ingredients",
			('icu', 'd_items'): "Dictionary of ICU items",
			('icu', 'icustays'): "ICU stay information",
			('icu', 'caregiver'): "Caregiver information"
		}
		return table_info.get((module, table_name), "No description available")

	def convert_to_parquet(self, df: pd.DataFrame, table_name: str, current_file_path: str) -> Optional[str]:
		"""Converts the loaded DataFrame to Parquet format. Returns path or None."""
		try:
			parquet_dir = os.path.join(os.path.dirname(current_file_path), 'parquet_files')
			os.makedirs(parquet_dir, exist_ok=True)
			parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")
			table = pa.Table.from_pandas(df)
			pq.write_table(table, parquet_file)
			return parquet_file
		except Exception as e:
			logging.error(f"Error converting {table_name} to Parquet: {str(e)}")
			return None

# --- Visualization Class (Placeholder) ---
class MIMICVisualizer:
	def __init__(self):
		pass

	def display_dataset_statistics(self, df: Optional[pd.DataFrame]):
		"""Displays key statistics about the loaded DataFrame."""
		if df is not None:
			st.markdown("<h2 class='sub-header'>Dataset Statistics</h2>", unsafe_allow_html=True)

			col1, col2 = st.columns(2)
			with col1:
				st.markdown("<div class='info-box'>", unsafe_allow_html=True)
				st.markdown(f"**Number of rows:** {len(df)}")
				st.markdown(f"**Number of columns:** {len(df.columns)}")
				st.markdown("</div>", unsafe_allow_html=True)

			with col2:
				st.markdown("<div class='info-box'>", unsafe_allow_html=True)
				st.markdown(f"**Memory usage:** {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
				st.markdown(f"**Missing values:** {df.isna().sum().sum()}")
				st.markdown("</div>", unsafe_allow_html=True)

			# Display column information
			st.markdown("<h3>Column Information</h3>", unsafe_allow_html=True)
			try:
				col_info = pd.DataFrame({
					'Column': df.columns,
					'Type': df.dtypes.values,
					'Non-Null Count': df.count().values,
					'Missing Values (%)': (df.isna().sum() / len(df) * 100).values.round(2),
					'Unique Values': [df[col].nunique() for col in df.columns]
				})
				st.dataframe(col_info, use_container_width=True)
			except Exception as e:
				st.error(f"Error generating column info: {e}")
		else:
			st.info("No data loaded to display statistics.")

	# Function to display data preview (Original)
	def display_data_preview(self, df: Optional[pd.DataFrame]):
		if df is not None:
			st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
			st.dataframe(df.head(10), use_container_width=True)

	# Function to display visualizations (Original)
	def display_visualizations(self, df: Optional[pd.DataFrame]):
		if df is not None:
			st.markdown("<h2 class='sub-header'>Data Visualization</h2>", unsafe_allow_html=True)

			# Select columns for visualization
			numeric_cols    : List[str] = df.select_dtypes(include=['number']).columns.tolist()
			categorical_cols: List[str] = df.select_dtypes(include=['object', 'category']).columns.tolist()

			if len(numeric_cols) > 0:
				st.markdown("<h3>Numeric Data Visualization</h3>", unsafe_allow_html=True)

				# Histogram
				selected_num_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
				if selected_num_col:
					fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
					st.plotly_chart(fig, use_container_width=True)

				# Scatter plot (if at least 2 numeric columns)
				if len(numeric_cols) >= 2:
					st.markdown("<h3>Scatter Plot</h3>", unsafe_allow_html=True)
					col1, col2 = st.columns(2)
					with col1:
						x_col = st.selectbox("Select X-axis", numeric_cols)
					with col2:
						y_col = st.selectbox("Select Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))

					if x_col and y_col:
						fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
						st.plotly_chart(fig, use_container_width=True)

			if len(categorical_cols) > 0:
				st.markdown("<h3>Categorical Data Visualization</h3>", unsafe_allow_html=True)

				# Bar chart
				selected_cat_col = st.selectbox("Select a categorical column for bar chart", categorical_cols)
				if selected_cat_col:
					value_counts = df[selected_cat_col].value_counts().reset_index()
					value_counts.columns = [selected_cat_col, 'Count']

					# Limit to top 20 categories if there are too many
					if len(value_counts) > 20:
						value_counts = value_counts.head(20)
						title = f"Top 20 values in {selected_cat_col}"
					else:
						title = f"Distribution of {selected_cat_col}"

					fig = px.bar(value_counts, x=selected_cat_col, y='Count', title=title)
					st.plotly_chart(fig, use_container_width=True)


# --- Main Application Class (Placeholder) ---
class MIMICDashboardApp:
	def __init__(self):
		logging.info("Initializing MIMICDashboardApp...")
		self.data_handler = MIMICDataHandler()
		self.visualizer = MIMICVisualizer()
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
		logging.info("Session state initialized.")

	def run(self):
		"""Run the main application loop."""
		logging.info("Starting MIMICDashboardApp run...")
		# Display the sidebar
		self._display_sidebar()

		# Call the method to display the main content
		self._display_main_content()
		logging.info("MIMICDashboardApp run finished.")

	def _display_sidebar(self):
		"""Handles the display and logic of the sidebar components."""
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

				# Advanced options
				with st.sidebar.expander("Advanced Options"):
					encoding = st.selectbox("Encoding", ["latin-1", "utf-8"], index=0)
					st.session_state.sample_size = st.number_input("Sample Size", 100, 10000, 1000, 100)

				# Load button
				if st.sidebar.button("Load Table"):
					file_path = st.session_state.file_paths.get((module, table))
					if file_path:
						st.session_state.current_file_path = file_path
						df = self.data_handler.load_mimic_table( file_path=file_path, sample_size=st.session_state.sample_size, encoding=encoding )

						if df is not None:
							st.session_state.df = df

	def _display_main_content(self):
		"""Handles the display of the main content area based on the loaded dataset."""
		# Main content area
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
			st.markdown(f"**Sample Size:** {min(len(st.session_state.df), st.session_state.sample_size)} rows out of {len(st.session_state.df)}") # TODO: Need total row count
			st.markdown("</div>", unsafe_allow_html=True)

			# Data preview
			self.visualizer.display_data_preview(st.session_state.df)

			# Dataset statistics
			self.visualizer.display_dataset_statistics(st.session_state.df)

			# Data visualization
			self.visualizer.display_visualizations(st.session_state.df)

			# Export options
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


if __name__ == "__main__":
    app = MIMICDashboardApp()
    app.run()
