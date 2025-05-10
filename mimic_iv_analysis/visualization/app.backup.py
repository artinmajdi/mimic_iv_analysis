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

# Local application imports (assuming these exist in the specified structure)
# If these imports cause errors, ensure the paths and class names are correct.
from mimic_iv_analysis.core import (
	ClusteringAnalyzer,
	ClusterInterpreter,
	FeatureEngineerUtils,
	DataLoader,
	# MIMICVisualizer
)
from mimic_iv_analysis.visualization.app_components import FilteringTab, FeatureEngineeringTab, ClusteringAnalysisTab


# Constants
DEFAULT_MIMIC_PATH      = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1" # Adjust if necessary
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================
# Refactored Tab Classes
# =============================================

class FeatureEngineeringTab:
	""" Handles the UI and logic for the Feature Engineering tab. """

	def _display_export_options(self, data, feature_engineer, feature_type='engineered_feature'):
		"""Helper function to display export options for engineered features."""
		with st.expander("#### Export Options"):
			save_format = st.radio(f"Save Format for {feature_type}", ["CSV", "Parquet"], horizontal=True, key=f"save_format_{feature_type}")

			if st.button(f"Save {feature_type.replace('_', ' ').title()}"):
				try:
					# Ensure base_path exists, handle potential errors
					base_path = "." # Default to current directory if path not available
					if 'current_file_path' in st.session_state and st.session_state.current_file_path:
						potential_path = os.path.dirname(st.session_state.current_file_path)
						if os.path.isdir(potential_path):
							base_path = potential_path
						else:
							st.warning(f"Directory not found: {potential_path}. Saving to current directory.")

					filepath = feature_engineer.save_features(
						features=data,
						feature_type=feature_type,
						base_path=base_path,
						format=save_format.lower()
					)
					st.success(f"Saved {feature_type.replace('_', ' ').title()} to {filepath}")
				except AttributeError:
					st.error("Feature Engineer is not properly initialized or does not have a 'save_features' method.")
				except Exception as e:
					st.error(f"Error saving {feature_type.replace('_', ' ').title()}: {str(e)}")

	def render(self, feature_engineer):
		""" Renders the content of the Feature Engineering tab. """
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

		# Check if DataFrame exists
		if st.session_state.df is None:
			st.warning("Please load a table first to enable feature engineering.")
			return

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
				patient_id_col_index = 0
				if st.session_state.get('detected_patient_id_col') in all_columns:
					patient_id_col_index = all_columns.index(st.session_state['detected_patient_id_col'])
				patient_id_col = st.selectbox(
					"Select Patient ID Column",
					all_columns,
					index=patient_id_col_index,
					key="freq_patient_id_col",
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column but allow selection from all columns

				order_col = st.selectbox(
					"Select Order Type Column",
					all_columns,
					index=all_columns.index('order_type') if 'order_type' in all_columns else 0,
					key="freq_order_col",
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
					with st.spinner("Creating order frequency matrix..."):
						# Check if Dask was used to load the data
						use_dask = st.session_state.get('use_dask', False)

						freq_matrix = feature_engineer.create_order_frequency_matrix(
							st.session_state.df,
							patient_id_col=patient_id_col,
							order_col=order_col,
							normalize=normalize,
							top_n=top_n,
							use_dask=use_dask
						)
						# Store the frequency matrix
						st.session_state.freq_matrix = freq_matrix

						# Store in clustering_input_data for clustering analysis
						st.session_state.clustering_input_data = freq_matrix
						st.success(f"Order Frequency Matrix generated ({freq_matrix.shape[0]}x{freq_matrix.shape[1]}) and set as input for clustering.")

				except AttributeError:
					st.error("Feature Engineer is not properly initialized or does not have a 'create_order_frequency_matrix' method.")
				except KeyError as e:
					st.error(f"Column '{e}' not found in the DataFrame. Please check your selections.")
				except Exception as e:
					st.error(f"Error generating frequency matrix: {str(e)}")
					logging.exception("Error in Generate Order Frequency Matrix")


			# Display result if available
			if 'freq_matrix' in st.session_state and st.session_state.freq_matrix is not None:
				st.markdown("<h4>Order Frequency Matrix</h4>", unsafe_allow_html=True)

				# Show preview
				st.dataframe(st.session_state.freq_matrix.head(10), use_container_width=True)

				# Matrix stats
				st.markdown(f"<div class='info-box'>Matrix size: {st.session_state.freq_matrix.shape[0]} patients × {st.session_state.freq_matrix.shape[1]} order types</div>", unsafe_allow_html=True)

				# Heatmap visualization
				st.markdown("<h4>Frequency Matrix Heatmap (Sample)</h4>", unsafe_allow_html=True)
				try:
					# Sample data for heatmap if too large
					heatmap_data = st.session_state.freq_matrix
					if heatmap_data.shape[0] > 50 or heatmap_data.shape[1] > 50:
						st.info("Displaying a sample of the heatmap due to large size.")
						sample_rows = min(50, heatmap_data.shape[0])
						sample_cols = min(50, heatmap_data.shape[1])
						heatmap_data = heatmap_data.iloc[:sample_rows, :sample_cols]

					fig = px.imshow(heatmap_data.T,
									labels=dict(x="Patient ID (Index)", y="Order Type", color="Frequency/Count"),
									aspect="auto")
					st.plotly_chart(fig, use_container_width=True)
				except Exception as e:
					st.error(f"Could not generate heatmap: {e}")

				# Save options
				self._display_export_options(data=st.session_state.freq_matrix, feature_engineer=feature_engineer, feature_type='order_frequency_matrix')

		# 2. Temporal Order Sequences tab
		with feature_tabs[1]:
			st.markdown("<h3>Extract Temporal Order Sequences</h3>", unsafe_allow_html=True)
			st.info("This extracts chronological sequences of orders for each patient, preserving the temporal relationships between different orders.")

			# Column selection
			col1, col2, col3 = st.columns(3)
			with col1:
				# Suggest patient ID column
				patient_id_col_index = 0
				if st.session_state.get('detected_patient_id_col') in all_columns:
					patient_id_col_index = all_columns.index(st.session_state['detected_patient_id_col'])
				seq_patient_id_col = st.selectbox(
					"Select Patient ID Column ", # Added space to differentiate key
					all_columns,
					index=patient_id_col_index,
					key="seq_patient_id_col",
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column
				order_col_index = 0
				if st.session_state.get('detected_order_cols') and st.session_state['detected_order_cols'][0] in all_columns:
					order_col_index = all_columns.index(st.session_state['detected_order_cols'][0])
				seq_order_col = st.selectbox( "Select Order Type Column ", all_columns, index=order_col_index, key="seq_order_col", help="Column containing order types/names" )

			with col3:
				# Suggest time column
				time_col_index = 0
				if st.session_state.get('detected_time_cols') and st.session_state['detected_time_cols'][0] in all_columns:
					time_col_index = all_columns.index(st.session_state['detected_time_cols'][0])
				seq_time_col = st.selectbox( "Select Timestamp Column ", all_columns, index=time_col_index, key="seq_time_col", help="Column containing order timestamps" )

			# Options
			max_seq_length = st.slider("Maximum Sequence Length", min_value=5, max_value=100, value=20, help="Maximum number of orders to include in each sequence")

			# Generate button
			if st.button("Extract Order Sequences"):
				try:
					with st.spinner("Extracting temporal order sequences..."):
						# Check if Dask was used to load the data
						use_dask = st.session_state.get('use_dask', False)

						sequences = feature_engineer.extract_temporal_order_sequences(
							df                  = st.session_state.df,
							patient_id_col      = seq_patient_id_col,
							order_col           = seq_order_col,
							time_col            = seq_time_col,
							max_sequence_length = max_seq_length,
							use_dask            = use_dask
						)
						st.session_state.order_sequences = sequences
						st.success(f"Extracted sequences for {len(sequences)} patients.")

						# Also generate transition matrix automatically
						st.info("Calculating order transition matrix...")
						transition_matrix = feature_engineer.calculate_order_transition_matrix( sequences=sequences, top_n=15 )
						st.session_state.transition_matrix = transition_matrix
						st.success("Order transition matrix calculated.")

				except AttributeError:
					st.error("Feature Engineer is not properly initialized or does not have the required methods.")
				except KeyError as e:
					st.error(f"Column '{e}' not found in the DataFrame. Please check your selections.")
				except Exception as e:
					st.error(f"Error extracting order sequences: {str(e)}")
					logging.exception("Error in Extract Order Sequences")


			# Display results if available
			if 'order_sequences' in st.session_state and st.session_state.order_sequences is not None:
				# Show sequence stats
				num_patients = len(st.session_state.order_sequences)
				if num_patients > 0:
					avg_sequence_length = np.mean([len(seq) for seq in st.session_state.order_sequences.values()])
				else:
					avg_sequence_length = 0

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
				if sample_patients:
					for patient in sample_patients:
						sequence = st.session_state.order_sequences[patient]
						sequence_str = " → ".join([str(order) for order in sequence])

						st.markdown(f"<strong>Patient {patient}:</strong> {sequence_str}", unsafe_allow_html=True)
						st.markdown("<hr>", unsafe_allow_html=True)
				else:
					st.info("No sequences generated to display samples.")


				# Transition matrix visualization
				if 'transition_matrix' in st.session_state and st.session_state.transition_matrix is not None:
					st.markdown("<h4>Order Transition Matrix</h4>", unsafe_allow_html=True)
					st.info("This matrix shows the probability of transitioning from one order type (rows) to another (columns). Based on top 15 orders.")
					try:
						fig = px.imshow(
							st.session_state.transition_matrix,
							labels=dict(x="Next Order", y="Current Order", color="Transition Probability"),
							x=st.session_state.transition_matrix.columns,
							y=st.session_state.transition_matrix.index,
							color_continuous_scale='Blues'
						)
						fig.update_layout(height=700)
						st.plotly_chart(fig, use_container_width=True)
					except Exception as e:
						st.error(f"Could not generate transition matrix heatmap: {e}")


				# Save options for sequences (transition matrix is derived, not saved directly here)
				self._display_export_options(data=st.session_state.order_sequences, feature_engineer=feature_engineer, feature_type='temporal_order_sequences')


		# 3. Order Type Distributions tab
		with feature_tabs[2]:
			st.markdown("<h3>Analyze Order Type Distributions</h3>", unsafe_allow_html=True)
			st.info("This analyzes the distribution of different order types across the dataset and for individual patients.")

			# Column selection
			col1, col2 = st.columns(2)
			with col1:
				# Suggest patient ID column
				patient_id_col_index = 0
				if st.session_state.get('detected_patient_id_col') in all_columns:
					patient_id_col_index = all_columns.index(st.session_state['detected_patient_id_col'])
				dist_patient_id_col = st.selectbox(
					"Select Patient ID Column  ", # Added spaces to differentiate key
					all_columns,
					index=patient_id_col_index,
					key="dist_patient_id_col",
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column
				order_col_index = 0
				if st.session_state.get('detected_order_cols') and st.session_state['detected_order_cols'][0] in all_columns:
					order_col_index = all_columns.index(st.session_state['detected_order_cols'][0])
				dist_order_col = st.selectbox(
					"Select Order Type Column  ",
					all_columns,
					index=order_col_index,
					key="dist_order_col",
					help="Column containing order types/names"
				)

			# Generate button
			if st.button("Analyze Order Distributions"):
				try:
					with st.spinner("Analyzing order type distributions..."):
						overall_dist, patient_dist = feature_engineer.get_order_type_distributions(
							st.session_state.df,
							dist_patient_id_col,
							dist_order_col
						)
						st.session_state.order_dist = overall_dist
						st.session_state.patient_order_dist = patient_dist
						st.success("Order distributions analyzed.")
				except AttributeError:
					st.error("Feature Engineer is not properly initialized or does not have a 'get_order_type_distributions' method.")
				except KeyError as e:
					st.error(f"Column '{e}' not found in the DataFrame. Please check your selections.")
				except Exception as e:
					st.error(f"Error analyzing order distributions: {str(e)}")
					logging.exception("Error in Analyze Order Distributions")


			# Display results if available
			if 'order_dist' in st.session_state and st.session_state.order_dist is not None:
				# Show overall distribution
				st.markdown("<h4>Overall Order Type Distribution</h4>", unsafe_allow_html=True)
				try:
					# Create pie chart for overall distribution
					top_n_orders = 15  # Show top 15 for pie chart
					if not st.session_state.order_dist.empty:
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

						fig_pie = px.pie(
							pie_data,
							values='frequency',
							names=dist_order_col,
							title=f"Overall Distribution of {dist_order_col} (Top {top_n_orders})"
						)
						st.plotly_chart(fig_pie, use_container_width=True)

						# Show bar chart of top 20
						top_20 = st.session_state.order_dist.head(20)
						fig_bar = px.bar(
							top_20,
							x=dist_order_col,
							y='frequency',
							title=f"Top 20 {dist_order_col} by Frequency"
						)
						st.plotly_chart(fig_bar, use_container_width=True)
					else:
						st.info("Overall distribution data is empty.")

				except Exception as e:
					st.error(f"Error visualizing overall distribution: {e}")


				# Patient-level distribution (sample)
				if 'patient_order_dist' in st.session_state and st.session_state.patient_order_dist is not None and not st.session_state.patient_order_dist.empty:
					st.markdown("<h4>Patient-Level Order Type Distribution (Sample)</h4>", unsafe_allow_html=True)
					try:
						# Get unique patients
						patients = st.session_state.patient_order_dist[dist_patient_id_col].unique() # Use selected patient ID col

						# Sample patients for visualization if there are too many
						num_samples = min(len(patients), 5)
						if num_samples > 0:
							sample_patients = np.random.choice(patients, num_samples, replace=False)

							# Create subplots for each patient
							fig_patient = make_subplots(
								rows=len(sample_patients),
								cols=1,
								subplot_titles=[f"Patient {patient}" for patient in sample_patients]
							)

							# Add traces for each patient
							for i, patient in enumerate(sample_patients):
								patient_data = st.session_state.patient_order_dist[
									st.session_state.patient_order_dist[dist_patient_id_col] == patient
								].head(10)  # Top 10 orders for this patient

								fig_patient.add_trace(
									go.Bar(
										x=patient_data[dist_order_col],
										y=patient_data['frequency'],
										name=f"Patient {patient}"
									),
									row=i+1, col=1
								)

							fig_patient.update_layout(height=200*len(sample_patients), showlegend=False)
							st.plotly_chart(fig_patient, use_container_width=True)
						else:
							st.info("No patient-level distribution data available.")
					except Exception as e:
						st.error(f"Error visualizing patient-level distribution: {e}")
				else:
					st.info("Patient-level distribution data not generated or is empty.")


				# Save options for both distributions
				self._display_export_options(data=st.session_state.order_dist, feature_engineer=feature_engineer, feature_type='overall_order_distribution')
				if 'patient_order_dist' in st.session_state and st.session_state.patient_order_dist is not None:
					self._display_export_options(data=st.session_state.patient_order_dist, feature_engineer=feature_engineer, feature_type='patient_order_distribution')


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
				patient_id_col_index = 0
				if st.session_state.get('detected_patient_id_col') in all_columns:
					patient_id_col_index = all_columns.index(st.session_state['detected_patient_id_col'])
				timing_patient_id_col = st.selectbox(
					"Select Patient ID Column   ", # Added spaces to differentiate key
					all_columns,
					index=patient_id_col_index,
					key="timing_patient_id_col",
					help="Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column
				order_col_index = 0
				if st.session_state.get('detected_order_cols') and st.session_state['detected_order_cols'][0] in all_columns:
					order_col_index = all_columns.index(st.session_state['detected_order_cols'][0])
				timing_order_col = st.selectbox(
					"Select Order Type Column   ",
					all_columns,
					index=order_col_index,
					key="timing_order_col",
					help="Column containing order types/names"
				)

			# Time columns
			col1, col2 = st.columns(2)
			with col1:
				# Suggest time column
				time_col_index = 0
				if st.session_state.get('detected_time_cols') and st.session_state['detected_time_cols'][0] in all_columns:
					time_col_index = all_columns.index(st.session_state['detected_time_cols'][0])

				order_time_col = st.selectbox(
					"Select Order Time Column",
					all_columns,
					index=time_col_index,
					key="order_time_col",
					help="Column containing order timestamps"
				)

			with col2:
				# Optional admission time column - try to find 'admittime' or similar
				admission_time_index = 0
				potential_admission_cols = [c for c in all_columns if 'admit' in c.lower()]
				if potential_admission_cols:
					admission_time_index = all_columns.index(potential_admission_cols[0]) + 1 # +1 for "None" option

				admission_time_col = st.selectbox(
					"Select Admission Time Column (Optional)",
					["None"] + all_columns,
					index=admission_time_index,
					key="admission_time_col",
					help="Column containing admission timestamps (for relative timing features)"
				)
				admission_time_col = None if admission_time_col == "None" else admission_time_col


			# Optional discharge time column - try to find 'dischtime' or similar
			discharge_time_index = 0
			potential_discharge_cols = [c for c in all_columns if 'disch' in c.lower()]
			if potential_discharge_cols:
				discharge_time_index = all_columns.index(potential_discharge_cols[0]) + 1 # +1 for "None" option

			discharge_time_col = st.selectbox(
				"Select Discharge Time Column (Optional)",
				["None"] + all_columns,
				index=discharge_time_index,
				key="discharge_time_col",
				help="Column containing discharge timestamps (for relative timing features)"
			)
			discharge_time_col = None if discharge_time_col == "None" else discharge_time_col


			# Generate button
			if st.button("Generate Timing Features"):
				try:
					with st.spinner("Generating order timing features..."):
						timing_features = feature_engineer.create_order_timing_features(
							df                    = st.session_state.df,
							patient_id_col        = timing_patient_id_col,
							order_col             = timing_order_col,
							order_time_col        = order_time_col,
							admission_time_col    = admission_time_col,
							discharge_time_col    = discharge_time_col
						)
						st.session_state.timing_features = timing_features
						st.success("Order timing features generated.")
				except AttributeError:
					st.error("Feature Engineer is not properly initialized or does not have a 'create_order_timing_features' method.")
				except KeyError as e:
					st.error(f"Column '{e}' not found in the DataFrame. Please check your selections.")
				except ValueError as e:
					st.error(f"Data type error: {e}. Ensure time columns are in a recognizable format.")
				except Exception as e:
					st.error(f"Error generating timing features: {str(e)}")
					logging.exception("Error in Generate Timing Features")


			# Display results if available
			if 'timing_features' in st.session_state and st.session_state.timing_features is not None:
				st.markdown("<h4>Order Timing Features</h4>", unsafe_allow_html=True)

				# Show preview of features
				st.dataframe(st.session_state.timing_features.head(10), use_container_width=True)

				# Generate visualizations based on available features
				st.markdown("<h4>Order Timing Visualizations</h4>", unsafe_allow_html=True)

				timing_df = st.session_state.timing_features
				numeric_cols = timing_df.select_dtypes(include=['number']).columns

				try:
					# Bar chart of total orders and unique orders
					if 'total_orders' in timing_df.columns and 'unique_order_types' in timing_df.columns:
						col1, col2 = st.columns(2)
						with col1:
							fig_total = px.histogram(
								timing_df, x='total_orders', nbins=30,
								title="Distribution of Total Orders per Patient"
							)
							st.plotly_chart(fig_total, use_container_width=True)
						with col2:
							fig_unique = px.histogram(
								timing_df, x='unique_order_types', nbins=30,
								title="Distribution of Unique Order Types per Patient"
							)
							st.plotly_chart(fig_unique, use_container_width=True)

					# Time-based analyses (if admission time was provided)
					relative_time_cols = ['time_to_first_order_hours', 'orders_in_first_24h', 'orders_in_first_48h', 'orders_in_first_72h']
					if admission_time_col and any(col in timing_df.columns for col in relative_time_cols):
						col1, col2 = st.columns(2)
						with col1:
							if 'time_to_first_order_hours' in timing_df.columns:
								fig_first_order = px.histogram(
									timing_df, x='time_to_first_order_hours', nbins=30,
									title="Time from Admission to First Order (hours)"
								)
								st.plotly_chart(fig_first_order, use_container_width=True)

						with col2:
							if all(col in timing_df.columns for col in ['orders_in_first_24h', 'orders_in_first_48h', 'orders_in_first_72h']):
								time_periods = ['First 24h', 'First 48h', 'First 72h']
								avg_orders = [
									timing_df['orders_in_first_24h'].mean(),
									timing_df['orders_in_first_48h'].mean(),
									timing_df['orders_in_first_72h'].mean()
								]
								orders_by_time = pd.DataFrame({'Time Period': time_periods, 'Average Orders': avg_orders})
								fig_time_orders = px.bar(
									orders_by_time, x='Time Period', y='Average Orders',
									title="Average Orders in Time Periods After Admission"
								)
								st.plotly_chart(fig_time_orders, use_container_width=True)
				except Exception as e:
					st.error(f"Error generating timing visualizations: {e}")

				# Save options
				self._display_export_options(data=st.session_state.timing_features, feature_engineer=feature_engineer, feature_type='order_timing_features')


class ClusteringAnalysisTab:
	""" Handles the UI and logic for the Clustering Analysis tab. """

	def render(self, clustering_analyzer, feature_engineer):
		""" Renders the content of the Clustering Analysis tab. """
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
			data_source_options = ["Current DataFrame", "Order Frequency Matrix", "Order Timing Features", "Upload Data"]
			# Determine default index based on available features
			default_data_source_index = 0
			if 'freq_matrix' in st.session_state and st.session_state.freq_matrix is not None:
				default_data_source_index = 1
			elif 'timing_features' in st.session_state and st.session_state.timing_features is not None:
				default_data_source_index = 2

			data_source = st.radio( "Select Data Source", data_source_options, index=default_data_source_index, horizontal=True )

			input_data = None
			input_data_ready = False # Flag to track if data is loaded and ready

			if data_source == "Current DataFrame":
				# Let user select columns from the current DataFrame
				if st.session_state.df is not None:
					# Get numeric columns only for clustering
					numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()

					if numeric_cols:
						default_selection = numeric_cols[:min(5, len(numeric_cols))]
						selected_cols = st.multiselect(
							"Select numeric columns for clustering",
							numeric_cols,
							default=default_selection
						)

						if selected_cols:
							input_data = st.session_state.df[selected_cols].copy()
							st.markdown(f"Selected data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
							st.dataframe(input_data.head(), use_container_width=True)
							input_data_ready = True
						else:
							st.warning("Please select at least one numeric column.")
					else:
						st.warning("No numeric columns found in the current DataFrame. Please select another data source or load a table with numeric data.")
				else:
					st.warning("No DataFrame is currently loaded. Please load a dataset first.")

			elif data_source == "Order Frequency Matrix":
				# Use order frequency matrix if available
				if 'freq_matrix' in st.session_state and st.session_state.freq_matrix is not None:
					input_data = st.session_state.freq_matrix.copy() # Use copy
					st.markdown(f"Using order frequency matrix with shape: {input_data.shape[0]} patients × {input_data.shape[1]} order types")
					st.dataframe(input_data.head(), use_container_width=True)
					input_data_ready = True
				else:
					st.warning("Order frequency matrix not found. Please generate it in the Feature Engineering tab first.")

			elif data_source == "Order Timing Features":
				# Use timing features if available
				if 'timing_features' in st.session_state and st.session_state.timing_features is not None:
					# Get numeric columns only
					numeric_cols = st.session_state.timing_features.select_dtypes(include=np.number).columns.tolist()

					if numeric_cols:
						selected_cols = st.multiselect(
							"Select timing features for clustering",
							numeric_cols,
							default=numeric_cols # Default to all numeric timing features
						)

						if selected_cols:
							input_data = st.session_state.timing_features[selected_cols].copy()
							st.markdown(f"Selected data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
							st.dataframe(input_data.head(), use_container_width=True)
							input_data_ready = True
						else:
							st.warning("Please select at least one timing feature.")
					else:
						st.warning("No numeric columns found in the Order Timing Features. Please generate them first.")
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

						# Basic validation: check for numeric data
						if input_data.select_dtypes(include=np.number).empty:
							st.error("Uploaded file contains no numeric columns suitable for clustering.")
							input_data = None
						else:
							st.markdown(f"Uploaded data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
							st.dataframe(input_data.head(), use_container_width=True)
							# Select only numeric columns
							input_data = input_data.select_dtypes(include=np.number)
							st.info(f"Using {input_data.shape[1]} numeric columns for clustering.")
							input_data_ready = True
					except Exception as e:
						st.error(f"Error loading file: {str(e)}")
						logging.exception("Error loading uploaded clustering data")


			# Data preprocessing options (only if data is ready)
			if input_data_ready:
				st.markdown("<h4>Data Preprocessing</h4>", unsafe_allow_html=True)

				preprocess_col1, preprocess_col2 = st.columns(2)

				with preprocess_col1:
					preprocess_method = st.selectbox(
						"Preprocessing Method",
						["None", "Standard Scaling", "Min-Max Scaling", "Normalization"],
						index=1, # Default to Standard Scaling
						help="Select method to preprocess the data"
					)

				with preprocess_col2:
					handle_missing = st.selectbox(
						"Handle Missing Values",
						["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Zero"],
						index=1, # Default to Fill with Mean
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
					"Drop Rows": "drop",
					"Fill with Mean": "mean",
					"Fill with Median": "median",
					"Fill with Zero": "zero"
				}

				# Button to process and store the data
				if st.button("Prepare Data for Clustering"):
					try:
						with st.spinner("Preprocessing data..."):
							# Check if Dask was used to load the data
							use_dask = st.session_state.get('use_dask', False)

							# Apply preprocessing
							processed_data = clustering_analyzer.preprocess_data(
								input_data, # Pass the loaded data
								method=preprocess_method_map[preprocess_method],
								handle_missing=handle_missing_map[handle_missing],
								use_dask=use_dask
							)

							# Check if data remains after preprocessing
							if processed_data.empty:
								st.error("Preprocessing resulted in an empty dataset. Check your missing value handling strategy (e.g., 'Drop Rows' might remove all data).")
							else:
								st.session_state.clustering_input_data = processed_data
								# Clear any previous reduced data
								st.session_state.reduced_data = None
								st.success(f"Data preprocessed and ready for clustering! Shape: {processed_data.shape}")

								# Show preview of processed data
								st.dataframe(processed_data.head(), use_container_width=True)

					except AttributeError:
						st.error("Clustering Analyzer is not properly initialized or does not have a 'preprocess_data' method.")
					except Exception as e:
						st.error(f"Error preparing data: {str(e)}")
						logging.exception("Error in Prepare Data for Clustering")

			elif not input_data_ready and data_source != "Upload Data":
				# Show message if no data source could be loaded (excluding upload state)
				st.info("Select a data source and ensure it's available or upload a file.")


		# 2. Dimensionality Reduction Tab
		with clustering_tabs[1]:
			st.markdown("<h3>Dimensionality Reduction</h3>", unsafe_allow_html=True)

			# Check if input data is available
			if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
				input_data_for_reduction = st.session_state.clustering_input_data
				input_shape = input_data_for_reduction.shape

				st.markdown(f"""
				<div class='info-box'>
				Reduce the dimensionality of your data ({input_shape[0]} rows × {input_shape[1]} columns) to visualize and potentially improve clustering performance.
				</div>
				""", unsafe_allow_html=True)

				# Check if data has enough features for reduction
				if input_shape[1] <= 2:
					st.info("Data already has 2 or fewer dimensions. Dimensionality reduction is not applicable.")
				else:
					# Dimensionality reduction method selection
					reduction_col1, reduction_col2 = st.columns(2)

					with reduction_col1:
						reduction_method = st.selectbox(
							"Dimensionality Reduction Method",
							["PCA", "t-SNE", "UMAP"], # SVD often used differently, removed for clustering context
							index=0,
							help="Select method to reduce dimensions"
						)

					with reduction_col2:
						# Ensure n_components is less than original dimensions
						max_components = min(10, input_shape[1] -1) # At least reduce by 1
						n_components = st.number_input(
							"Number of Components",
							min_value=2,
							max_value=max_components,
							value=2,
							help=f"Number of dimensions to reduce to (max {max_components})"
						)

					# Method-specific parameters
					extra_params = {}
					if reduction_method == "t-SNE":
						tsne_col1, tsne_col2 = st.columns(2)
						with tsne_col1:
							perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=min(30, input_shape[0]-1), help="Balance between local/global structure (must be < n_samples)")
						with tsne_col2:
							learning_rate = st.slider("Learning Rate", min_value=10, max_value=1000, value=200, step=10, help="Learning rate for t-SNE")
						n_iter = st.slider("Max Iterations", min_value=250, max_value=2000, value=1000, step=250, help="Maximum number of iterations")
						extra_params = {"perplexity": perplexity, "learning_rate": learning_rate, "n_iter": n_iter}

					elif reduction_method == "UMAP":
						try:
							umap_col1, umap_col2 = st.columns(2)
							with umap_col1:
								n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=min(100, input_shape[0]-1), value=min(15, input_shape[0]-1), help="Controls local/global embedding (must be < n_samples)")
							with umap_col2:
								min_dist = st.slider("Minimum Distance", min_value=0.0, max_value=0.99, value=0.1, step=0.05, help="Controls how tightly points are packed")
							metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine", "correlation"], index=0, help="Metric used for distances")
							extra_params = {"n_neighbors": n_neighbors, "min_dist": min_dist, "metric": metric}
						except ImportError:
							st.warning("UMAP is not installed. Please install it (`pip install umap-learn`) to use this option.")
							reduction_method = "PCA" # Fallback
							st.info("Falling back to PCA.")
							extra_params = {}


					# Button to apply dimensionality reduction
					if st.button("Apply Dimensionality Reduction"):
						try:
							with st.spinner(f"Applying {reduction_method} dimensionality reduction..."):
								# Map method names
								method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}

								# Check if Dask was used to load the data
								use_dask = st.session_state.get('use_dask', False)

								# Apply reduction
								reduced_data = clustering_analyzer.apply_dimensionality_reduction(
									input_data_for_reduction,
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

						except AttributeError:
							st.error("Clustering Analyzer is not properly initialized or does not have an 'apply_dimensionality_reduction' method.")
						except ValueError as e:
							st.error(f"Value Error during reduction: {e}. Check parameters (e.g., perplexity/n_neighbors vs sample size).")
						except Exception as e:
							st.error(f"Error applying dimensionality reduction: {str(e)}")
							logging.exception("Error in Apply Dimensionality Reduction")


					# Visualization of reduced data (if available and 2D/3D)
					if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
						reduced_df = st.session_state.reduced_data
						reduced_shape = reduced_df.shape

						st.markdown("<h4>Visualization of Reduced Data</h4>", unsafe_allow_html=True)
						try:
							if reduced_shape[1] == 2:
								fig = px.scatter(
									reduced_df, x=reduced_df.columns[0], y=reduced_df.columns[1],
									title=f"2D Projection using {reduction_method}",
									opacity=0.7
								)
								st.plotly_chart(fig, use_container_width=True)
							elif reduced_shape[1] == 3:
								fig = px.scatter_3d(
									reduced_df, x=reduced_df.columns[0], y=reduced_df.columns[1], z=reduced_df.columns[2],
									title=f"3D Projection using {reduction_method}",
									opacity=0.7
								)
								st.plotly_chart(fig, use_container_width=True)
							else:
								st.info("Reduced data has more than 3 dimensions. Select 2 or 3 components for visualization.")
						except Exception as e:
							st.error(f"Error visualizing reduced data: {e}")


						# Option to save reduced data using FeatureEngineer's save method
						with st.expander("Save Reduced Data"):
							save_format_reduced = st.radio(
								"Save Format ", # Key differentiation
								["CSV", "Parquet"],
								horizontal=True,
								key="dimreduction_save_format"
							)
							if st.button("Save Reduced Data"):
								try:
									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:
										potential_path_save = os.path.dirname(st.session_state.current_file_path)
										if os.path.isdir(potential_path_save):
											base_path_save = potential_path_save

									filepath = feature_engineer.save_features(
										st.session_state.reduced_data,
										f"{reduction_method.lower()}_reduced_data",
										base_path_save,
										save_format_reduced.lower()
									)
									st.success(f"Saved reduced data to {filepath}")
								except AttributeError:
									st.error("Feature Engineer is not properly initialized or does not have a 'save_features' method.")
								except Exception as e:
									st.error(f"Error saving reduced data: {str(e)}")
									logging.exception("Error saving reduced data")
			else:
				st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")


		# 3. K-Means Clustering Tab
		with clustering_tabs[2]:
			st.markdown("<h3>K-Means Clustering</h3>", unsafe_allow_html=True)

			# Check if input data is available
			if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
				st.markdown("""
				<div class='info-box'>
				K-means clustering partitions data into k clusters, where each observation belongs to the cluster with the nearest mean.
				</div>
				""", unsafe_allow_html=True)

				# Determine which data to use
				data_for_clustering = st.session_state.clustering_input_data
				use_reduced_data = False
				if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
					use_reduced_data = st.checkbox(
						"Use dimensionality-reduced data for clustering",
						value=True, # Default to using reduced if available
						help="Use the reduced data instead of the original preprocessed data"
					)
					if use_reduced_data:
						data_for_clustering = st.session_state.reduced_data
						st.info(f"Using reduced data with shape: {data_for_clustering.shape}")
					else:
						st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")
				else:
					st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")


				# K-means parameters
				# Make sure n_clusters_default is an integer, not None
				optimal_k = st.session_state.get('optimal_k')
				n_clusters_default = 5 if optimal_k is None else optimal_k

				n_clusters = st.number_input(
					"Number of Clusters (k)",
					min_value=2,
					max_value=max(20, n_clusters_default + 5), # Dynamic max based on optimal k
					value=n_clusters_default,
					help="Number of clusters to form"
				)

				kmeans_params_col1, kmeans_params_col2 = st.columns(2)
				with kmeans_params_col1:
					max_iter = st.slider("Maximum Iterations", min_value=100, max_value=1000, value=300, step=100, help="Max iterations per run")
				with kmeans_params_col2:
					n_init = st.slider("Number of Initializations", min_value=1, max_value=20, value=10, help="Number of runs with different seeds")


				# --- Optimal k Section ---
				st.markdown("<h4>Find Optimal Number of Clusters (Elbow/Silhouette)</h4>", unsafe_allow_html=True)
				optimal_k_col1, optimal_k_col2 = st.columns(2)
				with optimal_k_col1:
					k_min = st.number_input("Minimum k", min_value=2, max_value=10, value=2)
				with optimal_k_col2:
					k_max = st.number_input("Maximum k", min_value=k_min + 1, max_value=20, value=10)

				# Use optimal_k button
				if st.button("Find Optimal k (using Elbow & Silhouette)"):
					if k_max <= k_min:
						st.error("Maximum k must be greater than Minimum k.")
					else:
						try:
							with st.spinner(f"Calculating Elbow and Silhouette scores for k={k_min} to {k_max}..."):
								k_range = range(k_min, k_max + 1)
								inertia_scores, silhouette_scores = clustering_analyzer.find_optimal_k_kmeans_elbow_silhouette(
									data_for_clustering,
									k_range=k_range,
									n_init=n_init,
									max_iter=max_iter
								)

								# Store results temporarily for plotting
								k_metrics = pd.DataFrame({
									'k': list(k_range),
									'inertia': inertia_scores,
									'silhouette': silhouette_scores
								})

								# Suggest optimal k based on silhouette (usually more reliable than elbow visually)
								optimal_k_silhouette = k_metrics.loc[k_metrics['silhouette'].idxmax()]['k'] if not k_metrics['silhouette'].empty else k_min
								st.session_state.optimal_k = int(optimal_k_silhouette) # Store best k

								st.success(f"Optimal k suggested by Silhouette Score: {st.session_state.optimal_k}")

								# Plot Elbow Method
								fig_elbow = go.Figure()
								fig_elbow.add_trace(go.Scatter(x=k_metrics['k'], y=k_metrics['inertia'], mode='lines+markers', name='Inertia'))
								fig_elbow.update_layout(title="Elbow Method for Optimal k", xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (Within-cluster sum of squares)")
								fig_elbow.add_vline(x=st.session_state.optimal_k, line_dash="dash", line_color="red", annotation_text=f"Suggested k = {st.session_state.optimal_k}")
								st.plotly_chart(fig_elbow, use_container_width=True)

								# Plot Silhouette Scores
								fig_sil = go.Figure()
								fig_sil.add_trace(go.Scatter(x=k_metrics['k'], y=k_metrics['silhouette'], mode='lines+markers', name='Silhouette Score'))
								fig_sil.update_layout(title="Silhouette Score for Different k", xaxis_title="Number of Clusters (k)", yaxis_title="Average Silhouette Score")
								fig_sil.add_vline(x=st.session_state.optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal k = {st.session_state.optimal_k}")
								st.plotly_chart(fig_sil, use_container_width=True)

								# Update n_clusters input to the found optimal k
								st.rerun() # Rerun to update the number input widget value

						except AttributeError:
							st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
						except Exception as e:
							st.error(f"Error finding optimal k: {str(e)}")
							logging.exception("Error in Find Optimal K")


				# --- Run K-means Section ---
				st.markdown("<h4>Run K-means Clustering</h4>", unsafe_allow_html=True)
				if st.button("Run K-means Clustering"):
					try:
						with st.spinner(f"Running K-means clustering with k={n_clusters}..."):
							# Run K-means
							labels, kmeans_model = clustering_analyzer.run_kmeans_clustering(
								data_for_clustering,
								n_clusters=n_clusters,
								n_init=n_init,
								max_iter=max_iter
							)

							# Store labels in session state
							st.session_state.kmeans_labels = pd.Series(labels, index=data_for_clustering.index, name="kmeans_cluster")

							# Calculate metrics
							metrics = clustering_analyzer.evaluate_clustering(
								data_for_clustering,
								st.session_state.kmeans_labels,
								"kmeans" # Store metrics under 'kmeans' key
							)
							# Store metrics centrally
							st.session_state.cluster_metrics['kmeans'] = metrics

							# Show success message with metrics
							metrics_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()])
							st.success(f"K-means clustering complete!\n{metrics_text}")

							# --- Visualization ---
							# Show cluster distribution
							cluster_counts = st.session_state.kmeans_labels.value_counts().sort_index()
							fig_dist = px.bar(
								x=cluster_counts.index, y=cluster_counts.values,
								labels={'x': 'Cluster', 'y': 'Number of Points'},
								title="Distribution of Points per Cluster (K-means)"
							)
							st.plotly_chart(fig_dist, use_container_width=True)

							# Visualize clusters if data is 2D or 3D
							if data_for_clustering.shape[1] in [2, 3]:
								vis_data = data_for_clustering.copy()
								vis_data['Cluster'] = st.session_state.kmeans_labels.astype(str) # Color needs categorical

								if data_for_clustering.shape[1] == 2:
									fig_scatter = px.scatter(
										vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster',
										title="K-means Clustering Results (2D)", color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig_scatter, use_container_width=True)
								else: # 3D
									fig_scatter = px.scatter_3d(
										vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster',
										title="K-means Clustering Results (3D)", color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig_scatter, use_container_width=True)

							# Show cluster centers
							if hasattr(kmeans_model, 'cluster_centers_'):
								st.markdown("<h4>Cluster Centers</h4>", unsafe_allow_html=True)
								centers = pd.DataFrame(
									kmeans_model.cluster_centers_,
									columns=data_for_clustering.columns,
									index=[f"Cluster {i}" for i in range(n_clusters)]
								)
								st.dataframe(centers.style.format("{:.3f}"), use_container_width=True)

					except AttributeError:
						st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
					except Exception as e:
						st.error(f"Error running K-means clustering: {str(e)}")
						logging.exception("Error in Run K-means")


				# --- Save Results Section ---
				if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:
					with st.expander("Save K-means Results"):
						save_col1, save_col2 = st.columns(2)
						with save_col1:
							if st.button("Save K-means Model"):
								try:
									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:
										potential_path_save = os.path.dirname(st.session_state.current_file_path)
										if os.path.isdir(potential_path_save):
											base_path_save = potential_path_save

									model_path = clustering_analyzer.save_model("kmeans", base_path_save)
									st.success(f"Saved K-means model to {model_path}")
								except AttributeError:
									st.error("Clustering Analyzer does not have a 'save_model' method or model is not available.")
								except Exception as e:
									st.error(f"Error saving K-means model: {str(e)}")
						with save_col2:
							if st.button("Save K-means Cluster Assignments"):
								try:
									# Create DataFrame with original index and cluster assignments
									assignments_df = pd.DataFrame({'cluster': st.session_state.kmeans_labels}, index=data_for_clustering.index)

									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:
										potential_path_save = os.path.dirname(st.session_state.current_file_path)
										if os.path.isdir(potential_path_save):
											base_path_save = potential_path_save

									filepath = feature_engineer.save_features(
										assignments_df, "kmeans_cluster_assignments", base_path_save, "csv"
									)
									st.success(f"Saved cluster assignments to {filepath}")
								except AttributeError:
									st.error("Feature Engineer does not have a 'save_features' method.")
								except Exception as e:
									st.error(f"Error saving K-means assignments: {str(e)}")
			else:
				st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")


		# 4. Hierarchical Clustering Tab
		with clustering_tabs[3]:
			st.markdown("<h3>Hierarchical Clustering</h3>", unsafe_allow_html=True)

			# Check if input data is available
			if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
				st.markdown("""
				<div class='info-box'>
				Hierarchical clustering creates a tree of clusters (dendrogram) by progressively merging or splitting groups. It doesn't require specifying k beforehand but can be computationally intensive.
				</div>
				""", unsafe_allow_html=True)

				# Determine which data to use
				data_for_clustering = st.session_state.clustering_input_data
				use_reduced_data = False
				if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
					use_reduced_data = st.checkbox(
						"Use dimensionality-reduced data for hierarchical clustering",
						value=True, # Default to using reduced if available
						help="Use the reduced data instead of the original preprocessed data"
					)
					if use_reduced_data:
						data_for_clustering = st.session_state.reduced_data
						st.info(f"Using reduced data with shape: {data_for_clustering.shape}")
					else:
						st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")
				else:
					st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")


				# Hierarchical clustering parameters
				hier_col1, hier_col2 = st.columns(2)
				with hier_col1:
					n_clusters_hier = st.number_input("Number of Clusters (for cutting dendrogram)", min_value=2, max_value=20, value=5, help="Number of clusters to extract after building the tree", key="hier_n_clusters")
				with hier_col2:
					linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], index=0, help="Method for calculating distances between clusters")

				# Distance metric (affinity)
				distance_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], index=0, help="Metric for measuring distances between samples") # Removed correlation as less common for AgglomerativeClustering affinity

				# Ward linkage requires euclidean distance
				if linkage_method == "ward" and distance_metric != "euclidean":
					st.warning("Ward linkage requires Euclidean distance. Switching distance metric to 'euclidean'.")
					distance_metric = "euclidean"

				# Limit data size due to memory intensity
				max_samples_hier = 2000
				if len(data_for_clustering) > max_samples_hier:
					st.warning(f"Dataset size ({len(data_for_clustering)}) is large for hierarchical clustering. Using a random sample of {max_samples_hier} points to avoid memory issues.")
					data_for_clustering_hier = data_for_clustering.sample(max_samples_hier, random_state=RANDOM_STATE)
				else:
					data_for_clustering_hier = data_for_clustering


				# Button to run hierarchical clustering
				if st.button("Run Hierarchical Clustering"):
					if data_for_clustering_hier.empty:
						st.error("Cannot run clustering on empty data sample.")
					else:
						try:
							with st.spinner(f"Running hierarchical clustering ({linkage_method} linkage, {distance_metric} metric)..."):
								# Run hierarchical clustering
								labels, linkage_data = clustering_analyzer.run_hierarchical_clustering(
									data_for_clustering_hier,
									n_clusters=n_clusters_hier,
									linkage_method=linkage_method,
									distance_metric=distance_metric # Pass the selected metric
								)

								# Store labels in session state (using the index from the sampled data)
								st.session_state.hierarchical_labels = pd.Series(labels, index=data_for_clustering_hier.index, name="hierarchical_cluster")

								# Calculate metrics using the sampled data and labels
								metrics = clustering_analyzer.evaluate_clustering(
									data_for_clustering_hier,
									st.session_state.hierarchical_labels,
									"hierarchical" # Store metrics under 'hierarchical' key
								)
								# Store metrics centrally
								st.session_state.cluster_metrics['hierarchical'] = metrics

								# Show success message with metrics
								metrics_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()])
								st.success(f"Hierarchical clustering complete!\n{metrics_text}")

								# --- Visualization ---
								# Show cluster distribution (based on the sample)
								cluster_counts = st.session_state.hierarchical_labels.value_counts().sort_index()
								fig_dist = px.bar(
									x=cluster_counts.index, y=cluster_counts.values,
									labels={'x': 'Cluster', 'y': 'Number of Points'},
									title=f"Distribution of Points per Cluster (Hierarchical, Sample Size={len(data_for_clustering_hier)})"
								)
								st.plotly_chart(fig_dist, use_container_width=True)

								# Visualize clusters if data is 2D or 3D (using the sample)
								if data_for_clustering_hier.shape[1] in [2, 3]:
									vis_data = data_for_clustering_hier.copy()
									vis_data['Cluster'] = st.session_state.hierarchical_labels.astype(str)

									if data_for_clustering_hier.shape[1] == 2:
										fig_scatter = px.scatter(
											vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster',
											title=f"Hierarchical Clustering Results (2D, Sample Size={len(data_for_clustering_hier)})",
											color_discrete_sequence=px.colors.qualitative.G10
										)
										st.plotly_chart(fig_scatter, use_container_width=True)
									else: # 3D
										fig_scatter = px.scatter_3d(
											vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster',
											title=f"Hierarchical Clustering Results (3D, Sample Size={len(data_for_clustering_hier)})",
											color_discrete_sequence=px.colors.qualitative.G10
										)
										st.plotly_chart(fig_scatter, use_container_width=True)

								# Plot dendrogram
								st.markdown("<h4>Dendrogram</h4>", unsafe_allow_html=True)
								if linkage_data and 'linkage_matrix' in linkage_data:
									try:
										fig_dendro, ax = plt.subplots(figsize=(12, 7))
										dendrogram(
											linkage_data['linkage_matrix'],
											ax               = ax,
											truncate_mode    = 'lastp', # Show only the last p merged clusters
											p                = 12,      # Number of clusters to show at bottom
											show_leaf_counts = True,
											show_contracted  = True,
										)
										ax.set_title('Hierarchical Clustering Dendrogram (Truncated)')
										ax.set_xlabel('Cluster size (or sample index if leaf)')
										ax.set_ylabel('Distance')
										# Add cut line if n_clusters > 1
										if n_clusters_hier > 1 and len(linkage_data['linkage_matrix']) >= n_clusters_hier -1 :
											cut_distance = linkage_data['linkage_matrix'][-(n_clusters_hier-1), 2]
											ax.axhline(y=cut_distance, c='k', linestyle='--', label=f'Cut for {n_clusters_hier} clusters')
											ax.legend()
										st.pyplot(fig_dendro)
										plt.close(fig_dendro) # Close plot to free memory
									except Exception as e:
										st.error(f"Error plotting dendrogram: {e}")
								else:
									st.warning("Linkage data for dendrogram not available.")

						except AttributeError:
							st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
						except Exception as e:
							st.error(f"Error running hierarchical clustering: {str(e)}")
							logging.exception("Error in Run Hierarchical Clustering")


				# --- Save Results Section ---
				if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:
					with st.expander("Save Hierarchical Clustering Results"):
						st.info("Note: Saved assignments are based on the sampled data used for clustering.")
						save_col1, save_col2 = st.columns(2)
						# Hierarchical doesn't save a "model" in the same way as K-means/DBSCAN, usually just labels/linkage.
						# Skipping model save button here.
						with save_col2:
							if st.button("Save Hierarchical Cluster Assignments"):
								try:
									# Create DataFrame with sampled index and cluster assignments
									assignments_df = pd.DataFrame({'cluster': st.session_state.hierarchical_labels}, index=st.session_state.hierarchical_labels.index)

									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:

										potential_path_save = os.path.dirname(st.session_state.current_file_path)
										if os.path.isdir(potential_path_save):
											base_path_save = potential_path_save

									filepath = feature_engineer.save_features( assignments_df, "hierarchical_cluster_assignments", base_path_save, "csv" )
									st.success(f"Saved cluster assignments (for sample) to {filepath}")
								except AttributeError:
									st.error("Feature Engineer does not have a 'save_features' method.")
								except Exception as e:
									st.error(f"Error saving hierarchical assignments: {str(e)}")
			else:
				st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")


		# 5. DBSCAN Clustering Tab
		with clustering_tabs[4]:
			st.markdown("<h3>DBSCAN Clustering</h3>", unsafe_allow_html=True)

			# Check if input data is available
			if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
				st.markdown("""
				<div class='info-box'>
				DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of arbitrary shapes by grouping points that are closely packed together, marking outliers as noise (-1). It requires tuning `epsilon` (neighborhood distance) and `min_samples`.
				</div>
				""", unsafe_allow_html=True)

				# Determine which data to use
				data_for_clustering = st.session_state.clustering_input_data
				use_reduced_data = False
				if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
					use_reduced_data = st.checkbox(
						"Use dimensionality-reduced data for DBSCAN",
						value=True, # Default to using reduced if available
						help="Use the reduced data instead of the original preprocessed data"
					)
					if use_reduced_data:
						data_for_clustering = st.session_state.reduced_data
						st.info(f"Using reduced data with shape: {data_for_clustering.shape}")
					else:
						st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")


				# DBSCAN parameters
				eps_default = st.session_state.get('optimal_eps', 0.5) # Use optimal_eps if found, else 0.5
				eps = st.number_input("Epsilon (ε)", min_value=0.01, max_value=10.0, value=eps_default, step=0.05, help="Maximum distance between samples to be neighbors", format="%.4f")

				dbscan_col1, dbscan_col2 = st.columns(2)
				with dbscan_col1:
					min_samples = st.number_input("Minimum Samples", min_value=2, max_value=100, value=max(5, int(0.01*len(data_for_clustering))), help="Number of samples in a neighborhood for a core point") # Dynamic default suggestion
				with dbscan_col2:
					metric_dbscan = st.selectbox("Distance Metric ", ["euclidean", "manhattan", "cosine", "l1", "l2"], index=0, help="Metric for distances", key="dbscan_metric")


				# --- Optimal Epsilon (k-distance plot) ---
				st.markdown("<h4>Find Optimal Epsilon (ε) using k-distance plot</h4>", unsafe_allow_html=True)
				# Suggest k based on data dimensionality or min_samples
				k_dist_default = min_samples if data_for_clustering.shape[1] <= 2 else min(max(min_samples, 2 * data_for_clustering.shape[1] - 1), len(data_for_clustering)-1) # Heuristic based on literature or min_samples
				k_dist = st.slider("k for k-distance graph", min_value=2, max_value=min(50, len(data_for_clustering)-1), value=k_dist_default, help="Number of neighbors to consider (k = MinPts is common)")

				if st.button("Calculate k-distance plot to find 'knee'"):
					if k_dist >= len(data_for_clustering):
						st.error(f"k ({k_dist}) must be smaller than the number of samples ({len(data_for_clustering)}).")
					else:
						try:
							with st.spinner(f"Calculating {k_dist}-distance graph..."):
								# Find optimal epsilon using the analyzer method
								suggested_eps, k_distances_sorted = clustering_analyzer.find_optimal_eps_for_dbscan(
									data_for_clustering,
									k_dist=k_dist,
									metric=metric_dbscan
								)

								# Store the suggested eps
								st.session_state.optimal_eps = suggested_eps

								# Show result
								st.success(f"Suggested epsilon (ε) based on the 'knee': {suggested_eps:.4f}")

								# Plot k-distance graph
								fig_kdist = go.Figure()
								fig_kdist.add_trace(go.Scatter(
									x=list(range(len(k_distances_sorted))),
									y=k_distances_sorted,
									mode='lines', name=f'{k_dist}-distance'
								))

								# Try to find the knee point mathematically (e.g., using Kneedle algorithm or max difference)
								# Simple max difference approach:
								diffs = np.diff(k_distances_sorted, 1)
								knee_point_idx = np.argmax(diffs) + 1 # Index in the original sorted array
								knee_eps = k_distances_sorted[knee_point_idx]

								fig_kdist.add_trace(go.Scatter(
									x=[knee_point_idx], y=[knee_eps],
									mode='markers', marker=dict(size=10, color='red'),
									name=f'Suggested ε ≈ {knee_eps:.4f}'
								))
								fig_kdist.update_layout(
									title=f"{k_dist}-Distance Graph (Sorted)",
									xaxis_title="Points (sorted by distance)",
									yaxis_title=f"{k_dist}-th Nearest Neighbor Distance",
									hovermode="x"
								)
								st.plotly_chart(fig_kdist, use_container_width=True)

								# Update eps input to the found optimal eps
								st.rerun() # Rerun to update the number input

						except AttributeError:
							st.error("Clustering Analyzer is not properly initialized or does not have a 'find_optimal_eps_for_dbscan' method.")
						except Exception as e:
							st.error(f"Error finding optimal epsilon: {str(e)}")
							logging.exception("Error in Find Optimal Epsilon (DBSCAN)")


				# --- Run DBSCAN Section ---
				st.markdown("<h4>Run DBSCAN Clustering</h4>", unsafe_allow_html=True)
				if st.button("Run DBSCAN Clustering"):
					try:
						with st.spinner(f"Running DBSCAN clustering with ε={eps:.4f}, MinPts={min_samples}..."):
							# Run DBSCAN
							labels, dbscan_model = clustering_analyzer.run_dbscan_clustering(
								data_for_clustering,
								eps=eps,
								min_samples=min_samples,
								metric=metric_dbscan
							)

							# Store labels in session state
							st.session_state.dbscan_labels = pd.Series(labels, index=data_for_clustering.index, name="dbscan_cluster")

							# Count number of clusters and noise points
							unique_labels = set(labels)
							n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
							n_noise_ = list(labels).count(-1)

							# Calculate metrics (only if more than one cluster found, excluding noise)
							metrics = {}
							if n_clusters_ > 1:
								# Filter out noise points for metric calculation
								non_noise_mask = st.session_state.dbscan_labels != -1
								if non_noise_mask.sum() > 1: # Need at least 2 points in clusters
									metrics = clustering_analyzer.evaluate_clustering(
										data_for_clustering[non_noise_mask],
										st.session_state.dbscan_labels[non_noise_mask],
										"dbscan" # Store metrics under 'dbscan' key
									)
									# Store metrics centrally
									st.session_state.cluster_metrics['dbscan'] = metrics
								else:
									st.warning("Not enough points in clusters (excluding noise) to calculate evaluation metrics.")
							else:
								st.warning("DBSCAN resulted in 0 or 1 cluster (excluding noise). Evaluation metrics require at least 2 clusters.")
								st.session_state.cluster_metrics.pop('dbscan', None) # Remove old metrics if they exist


							# Show success message
							st.success(f"""
							DBSCAN clustering complete!
							- Number of clusters found: {n_clusters_}
							- Number of noise points: {n_noise_} ({n_noise_/len(labels)*100:.2f}%)
							""")
							if metrics:
								metrics_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()])
								st.markdown(f"**Evaluation Metrics (excluding noise):**\n{metrics_text}")


							# --- Visualization ---
							# Show cluster distribution
							cluster_counts = st.session_state.dbscan_labels.value_counts().sort_index()
							# Map -1 to "Noise" for display
							display_labels = cluster_counts.index.map(lambda x: "Noise" if x == -1 else f"Cluster {x}")
							fig_dist = px.bar(
								x=display_labels, y=cluster_counts.values,
								labels={'x': 'Cluster / Noise', 'y': 'Number of Points'},
								title="Distribution of Points per Cluster (DBSCAN)"
							)
							st.plotly_chart(fig_dist, use_container_width=True)

							# Visualize clusters if data is 2D or 3D
							if data_for_clustering.shape[1] in [2, 3]:
								vis_data = data_for_clustering.copy()
								# Map labels to strings for coloring, handle noise
								vis_data['Cluster'] = st.session_state.dbscan_labels.apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}').astype(str)

								if data_for_clustering.shape[1] == 2:
									fig_scatter = px.scatter(
										vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster',
										title="DBSCAN Clustering Results (2D)",
										color_discrete_map={"Noise": "grey"}, # Explicitly color noise
										category_orders={"Cluster": sorted(vis_data['Cluster'].unique(), key=lambda x: int(x.split()[-1]) if x != 'Noise' else -1)} # Sort clusters correctly
									)
									st.plotly_chart(fig_scatter, use_container_width=True)
								else: # 3D
									fig_scatter = px.scatter_3d(
										vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster',
										title="DBSCAN Clustering Results (3D)",
										color_discrete_map={"Noise": "grey"},
										category_orders={"Cluster": sorted(vis_data['Cluster'].unique(), key=lambda x: int(x.split()[-1]) if x != 'Noise' else -1)}
									)
									st.plotly_chart(fig_scatter, use_container_width=True)

					except AttributeError:
						st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
					except Exception as e:
						st.error(f"Error running DBSCAN clustering: {str(e)}")
						logging.exception("Error in Run DBSCAN")


				# --- Save Results Section ---
				if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None:
					with st.expander("Save DBSCAN Results"):
						save_col1, save_col2 = st.columns(2)
						with save_col1:
							if st.button("Save DBSCAN Model"):
								try:
									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:
										potential_path_save = os.path.dirname(st.session_state.current_file_path)
										if os.path.isdir(potential_path_save):
											base_path_save = potential_path_save

									model_path = clustering_analyzer.save_model("dbscan", base_path_save)
									st.success(f"Saved DBSCAN model to {model_path}")
								except AttributeError:
									st.error("Clustering Analyzer does not have a 'save_model' method or model is not available.")
								except Exception as e:
									st.error(f"Error saving DBSCAN model: {str(e)}")
						with save_col2:
							if st.button("Save DBSCAN Cluster Assignments"):
								try:
									# Create DataFrame with original index and cluster assignments
									assignments_df = pd.DataFrame({'cluster': st.session_state.dbscan_labels}, index=data_for_clustering.index)

									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:
										potential_path_save = os.path.dirname(st.session_state.current_file_path)
										if os.path.isdir(potential_path_save):
											base_path_save = potential_path_save

									filepath = feature_engineer.save_features(
										assignments_df, "dbscan_cluster_assignments", base_path_save, "csv"
									)
									st.success(f"Saved cluster assignments to {filepath}")
								except AttributeError:
									st.error("Feature Engineer does not have a 'save_features' method.")
								except Exception as e:
									st.error(f"Error saving DBSCAN assignments: {str(e)}")
			else:
				st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")


		# 6. LDA Topic Modeling Tab
		with clustering_tabs[5]:
			st.markdown("<h3>LDA Topic Modeling</h3>", unsafe_allow_html=True)

			st.markdown("""
			<div class='info-box'>
			Latent Dirichlet Allocation (LDA) is a generative probabilistic model often used for topic modeling on text data. Here, it can identify underlying themes or patterns in order sequences or text columns.
			</div>
			""", unsafe_allow_html=True)

			# --- Data Selection for LDA ---
			lda_data_source = st.radio(
				"Select Data Source for LDA",
				["Order Sequences", "Text Column in Current Table"],
				index=0,
				help="Choose the input data for LDA."
			)

			documents = None
			doc_ids = None # To store corresponding patient IDs or indices

			if lda_data_source == "Order Sequences":
				if 'order_sequences' in st.session_state and st.session_state.order_sequences:
					# Convert sequences to space-separated strings
					documents = [" ".join(map(str, seq)) for seq in st.session_state.order_sequences.values()]
					doc_ids = list(st.session_state.order_sequences.keys())
					st.info(f"Using {len(documents)} patient order sequences as documents.")
					st.text_area("Sample Document (Sequence):", documents[0] if documents else "", height=100)
				else:
					st.warning("Order sequences not found or empty. Please generate them in the 'Feature Engineering' tab first.")

			elif lda_data_source == "Text Column in Current Table":
				if st.session_state.df is not None:
					text_columns = st.session_state.df.select_dtypes(include=['object', 'string']).columns.tolist()
					if text_columns:
						text_col = st.selectbox("Select Text Column", text_columns, help="Choose the column containing text documents.")
						# Convert to list of strings, handle NaNs
						documents = st.session_state.df[text_col].fillna("").astype(str).tolist()
						doc_ids = st.session_state.df.index # Use DataFrame index
						st.info(f"Using text from column '{text_col}' ({len(documents)} documents).")
						st.text_area("Sample Document (Text Column):", documents[0] if documents else "", height=100)
					else:
						st.warning("No text (object/string) columns found in the current table.")
				else:
					st.warning("No table loaded. Please load data first.")

			# --- LDA Parameters and Execution ---
			if documents:
				st.markdown("<h4>LDA Parameters</h4>", unsafe_allow_html=True)
				lda_col1, lda_col2 = st.columns(2)
				with lda_col1:
					n_topics = st.number_input("Number of Topics (k)", min_value=2, max_value=30, value=5, help="Number of topics to extract")
				with lda_col2:
					max_iter_lda = st.slider("Maximum Iterations (LDA)", min_value=10, max_value=100, value=20, step=5, help="Max iterations for LDA fitting") # Reduced default for faster online learning

				lda_col3, lda_col4 = st.columns(2)
				with lda_col3:
					vectorizer_type = st.selectbox("Vectorizer", ["CountVectorizer", "TfidfVectorizer"], index=0, help="Method to convert text to features")
				with lda_col4:
					max_features_lda = st.number_input("Max Features (Vocabulary Size)", min_value=100, max_value=10000, value=1000, step=100, help="Limit vocabulary size")

				# LDA implementation often uses batch learning by default in sklearn
				# learning_method = st.selectbox("Learning Method", ["batch", "online"], index=0, help="LDA parameter estimation method")

				if st.button("Run LDA Topic Modeling"):
					try:
						with st.spinner(f"Running LDA with {n_topics} topics..."):
							# Map vectorizer type
							vectorizer_map = {"CountVectorizer": "count", "TfidfVectorizer": "tfidf"}

							# Run LDA using the analyzer
							lda_model, doc_topic_matrix, topic_term_matrix, feature_names = clustering_analyzer.run_lda_topic_modeling(
								documents,
								n_topics=n_topics,
								vectorizer_type=vectorizer_map[vectorizer_type],
								max_features=max_features_lda,
								max_iter=max_iter_lda,
								# learning_method=learning_method # Sklearn LDA defaults usually fine
							)

							# Store results in session state
							st.session_state.lda_results = {
								'doc_topic_matrix': pd.DataFrame(doc_topic_matrix, index=doc_ids, columns=[f"Topic_{i}" for i in range(n_topics)]),
								'topic_term_matrix': pd.DataFrame(topic_term_matrix, index=[f"Topic_{i}" for i in range(n_topics)], columns=feature_names),
								'model': lda_model # Store the model itself if needed later
							}
							st.success(f"LDA topic modeling complete with {n_topics} topics!")

					except AttributeError:
						st.error("Clustering Analyzer is not properly initialized or does not have an 'run_lda_topic_modeling' method.")
					except Exception as e:
						st.error(f"Error running LDA topic modeling: {str(e)}")
						logging.exception("Error in Run LDA")


				# --- Display LDA Results ---
				if 'lda_results' in st.session_state and st.session_state.lda_results:
					st.markdown("<h4>LDA Results Visualization</h4>", unsafe_allow_html=True)
					try:
						lda_res = st.session_state.lda_results
						doc_topic_df = lda_res['doc_topic_matrix']
						topic_term_df = lda_res['topic_term_matrix']

						# Display top terms per topic
						st.markdown("<h5>Top Terms per Topic</h5>", unsafe_allow_html=True)
						top_terms = clustering_analyzer.get_top_terms_per_topic(topic_term_df, n_terms=10)
						st.dataframe(top_terms, use_container_width=True)

						# Display document-topic distribution heatmap (sample)
						st.markdown("<h5>Document-Topic Distribution (Sample Heatmap)</h5>", unsafe_allow_html=True)
						sample_size_lda = min(30, doc_topic_df.shape[0])
						doc_topic_sample = doc_topic_df.iloc[:sample_size_lda]
						fig_heatmap = px.imshow(
							doc_topic_sample, aspect="auto",
							labels=dict(x="Topic", y="Document Index/ID", color="Probability"),
							title=f"Document-Topic Probabilities (Sample of {sample_size_lda})",
							color_continuous_scale="Viridis"
						)
						st.plotly_chart(fig_heatmap, use_container_width=True)

						# Topic distribution overview (dominant topic per document)
						st.markdown("<h5>Overall Topic Distribution (Dominant Topic per Document)</h5>", unsafe_allow_html=True)
						dominant_topics = doc_topic_df.idxmax(axis=1).value_counts().sort_index()
						fig_dist = px.bar(
							x=dominant_topics.index, y=dominant_topics.values,
							labels={'x': 'Topic', 'y': 'Number of Documents'},
							title="Number of Documents Primarily Assigned to Each Topic"
						)
						st.plotly_chart(fig_dist, use_container_width=True)

						# Optional: Topic Similarity (if useful)
						# st.markdown("<h5>Topic Similarity (Cosine Similarity of Topic-Term Vectors)</h5>", unsafe_allow_html=True)
						# topic_similarity = cosine_similarity(topic_term_df.values)
						# fig_sim = px.imshow(topic_similarity, x=topic_term_df.index, y=topic_term_df.index,
						#                     labels=dict(color="Cosine Similarity"), title="Topic Similarity Matrix",
						#                     color_continuous_scale="Blues")
						# st.plotly_chart(fig_sim, use_container_width=True)

					except AttributeError:
						st.error("Clustering Analyzer does not have a 'get_top_terms_per_topic' method.")
					except Exception as e:
						st.error(f"Error displaying LDA results: {e}")
						logging.exception("Error displaying LDA results")


					# --- Save LDA Results ---
					with st.expander("Save LDA Results"):
						save_col1, save_col2 = st.columns(2)
						with save_col1:
							# LDA model saving can be tricky (requires vectorizer too). Often results are saved.
							# Skipping direct model save button for simplicity.
							pass
						with save_col2:
							if st.button("Save LDA Document-Topic Distributions"):
								try:
									assignments_df = st.session_state.lda_results['doc_topic_matrix']
									base_path_save = "."
									if 'current_file_path' in st.session_state and st.session_state.current_file_path:
											potential_path_save = os.path.dirname(st.session_state.current_file_path)
											if os.path.isdir(potential_path_save):
												base_path_save = potential_path_save

									filepath = feature_engineer.save_features(
											assignments_df, "lda_doc_topic_distributions", base_path_save, "csv"
									)
									st.success(f"Saved document-topic distributions to {filepath}")
								except AttributeError:
									st.error("Feature Engineer does not have a 'save_features' method.")
								except Exception as e:
									st.error(f"Error saving LDA distributions: {str(e)}")


		# 7. Evaluation Metrics Tab
		with clustering_tabs[6]:
			st.markdown("<h3>Clustering Evaluation Metrics Comparison</h3>", unsafe_allow_html=True)

			st.markdown("""
			<div class='info-box'>
			Compare the performance of different clustering algorithms run in this session using standard internal evaluation metrics (which do not require ground truth labels).
			</div>
			""", unsafe_allow_html=True)

			# Check which results are available
			available_results = {}
			data_used_for_metrics = {} # Store the data used for each algorithm's metrics

			if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:
				available_results['K-means'] = st.session_state.kmeans_labels
				# Try to determine data used based on checkbox state during run - this is fragile
				# A better approach would be to store the data alongside the labels when run
				if st.session_state.get('kmeans_used_reduced', False) and 'reduced_data' in st.session_state:
					data_used_for_metrics['K-means'] = st.session_state.reduced_data
				elif 'clustering_input_data' in st.session_state:
					data_used_for_metrics['K-means'] = st.session_state.clustering_input_data

			if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:
				available_results['Hierarchical'] = st.session_state.hierarchical_labels
				# Hierarchical often uses sampled data - need the sample used
				# This assumes the sample is stored correctly, which might not be the case
				# If not stored, we cannot reliably recalculate metrics here
				if 'hierarchical_data_sample' in st.session_state: # Need to ensure this is saved during run
					data_used_for_metrics['Hierarchical'] = st.session_state.hierarchical_data_sample
				# Fallback - cannot reliably evaluate if sample not stored

			if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None:
				available_results['DBSCAN'] = st.session_state.dbscan_labels
				if st.session_state.get('dbscan_used_reduced', False) and 'reduced_data' in st.session_state:
					data_used_for_metrics['DBSCAN'] = st.session_state.reduced_data
				elif 'clustering_input_data' in st.session_state:
					data_used_for_metrics['DBSCAN'] = st.session_state.clustering_input_data


			if not available_results:
				st.warning("No clustering results available in the current session. Run at least one clustering algorithm first.")
			else:
				st.markdown("#### Summary of Metrics")
				metrics_summary = []

				for name, labels in available_results.items():
					# Retrieve stored metrics if available
					metrics = st.session_state.cluster_metrics.get(name.lower(), {})
					n_clusters = len(set(labels)) - (1 if name == 'DBSCAN' and -1 in set(labels) else 0)
					n_noise = list(labels).count(-1) if name == 'DBSCAN' else 0

					metrics_summary.append({
						'Algorithm': name,
						'Num Clusters': n_clusters,
						'Noise Points (%)': f"{n_noise_ / len(labels) * 100:.1f}%" if name == 'DBSCAN' else "N/A",
						'Silhouette Score': metrics.get('silhouette_score', None),
						'Davies-Bouldin Index': metrics.get('davies_bouldin_score', None),
						'Calinski-Harabasz Index': metrics.get('calinski_harabasz_score', None),
					})

				metrics_df = pd.DataFrame(metrics_summary)

				# Format numeric columns nicely
				float_cols = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
				format_dict = {col: "{:.4f}" for col in float_cols}
				st.dataframe(metrics_df.style.format(format_dict, na_rep="N/A"), use_container_width=True)

				st.markdown("""
				*   **Silhouette Score:** Higher is better (range -1 to 1). Measures how similar an object is to its own cluster compared to other clusters.
				*   **Davies-Bouldin Index:** Lower is better (min 0). Measures the average similarity ratio of each cluster with its most similar cluster.
				*   **Calinski-Harabasz Index:** Higher is better. Ratio of between-cluster dispersion to within-cluster dispersion.
				""")

				# --- Metrics Comparison Plot ---
				st.markdown("<h4>Metrics Comparison Visualization</h4>", unsafe_allow_html=True)
				metrics_to_plot = [col for col in float_cols if metrics_df[col].notna().any()] # Only plot metrics with values

				if metrics_to_plot:
					# Use melt for easier plotting with Plotly
					plot_df = metrics_df.melt(id_vars=['Algorithm'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
					plot_df = plot_df.dropna() # Remove rows where metrics couldn't be calculated

					if not plot_df.empty:
						fig_comp = px.bar(plot_df, x='Metric', y='Score', color='Algorithm', barmode='group',
										title="Comparison of Clustering Evaluation Metrics")
						st.plotly_chart(fig_comp, use_container_width=True)
					else:
						st.info("No valid metrics available to plot.")
				else:
					st.info("No evaluation metrics were calculated or available for comparison.")


				# --- Cluster Agreement (if >1 result) ---
				if len(available_results) >= 2:
					st.markdown("<h4>Cluster Assignment Agreement (Adjusted Rand Index)</h4>", unsafe_allow_html=True)
					st.info("Compares the similarity of cluster assignments between pairs of algorithms (ignores noise points). Score close to 1 means high agreement, close to 0 means random agreement.")

					algo_names = list(available_results.keys())
					agreement_scores = pd.DataFrame(index=algo_names, columns=algo_names, dtype=float)

					for i in range(len(algo_names)):
						for j in range(i, len(algo_names)):
							algo1_name = algo_names[i]
							algo2_name = algo_names[j]

							if i == j:
								agreement_scores.loc[algo1_name, algo2_name] = 1.0
							else:
								labels1 = available_results[algo1_name]
								labels2 = available_results[algo2_name]

								# Ensure labels are aligned by index (important if sampling occurred)
								common_index = labels1.index.intersection(labels2.index)
								if len(common_index) < 2:
									ari = np.nan # Cannot compare if indices don't overlap sufficiently
								else:
									l1_common = labels1.loc[common_index]
									l2_common = labels2.loc[common_index]

									# Filter noise points (-1) for ARI calculation
									mask1 = l1_common != -1
									mask2 = l2_common != -1
									valid_mask = mask1 & mask2

									if valid_mask.sum() < 2:
										ari = np.nan # Not enough non-noise points to compare
									else:
										ari = adjusted_rand_score(l1_common[valid_mask], l2_common[valid_mask])

								agreement_scores.loc[algo1_name, algo2_name] = ari
								agreement_scores.loc[algo2_name, algo1_name] = ari # Symmetric matrix

					# Display heatmap
					fig_ari = px.imshow(agreement_scores,
										labels=dict(color="Adjusted Rand Index"),
										title="Pairwise Cluster Agreement (ARI)",
										color_continuous_scale='Blues', range_color=[0,1], # ARI typically 0-1, can be negative
										text_auto=".3f") # Show scores on heatmap
					st.plotly_chart(fig_ari, use_container_width=True)

				# --- Load/Compare Models (Placeholder/Optional) ---
				# This section is complex as it requires applying saved models to *current* data,
				# which might need the exact same preprocessing steps.
				# st.markdown("<h4>Load and Compare Saved Models</h4>", unsafe_allow_html=True)
				# st.info("Functionality to load previously saved models and compare them is under development.")


class AnalysisVisualizationTab:
	""" Handles the UI and logic for the post-clustering Analysis & Visualization tab. """

	def render(self, cluster_analyzer):
		""" Renders the content of the Analysis & Visualization tab. """
		st.markdown("<h2 class='sub-header'>Cluster Analysis & Interpretation</h2>", unsafe_allow_html=True)

		# Introductory text
		st.markdown("""
		<div class='info-box'>
		Explore and interpret the identified clusters. Analyze differences in patient characteristics, visualize patterns, and generate reports to understand the meaning behind the groupings.
		</div>
		""", unsafe_allow_html=True)

		# --- Select Clustering Result ---
		available_labels = {}
		if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:
			available_labels['K-means'] = st.session_state.kmeans_labels
		if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:
			available_labels['Hierarchical (Sampled)'] = st.session_state.hierarchical_labels # Note it's sampled
		if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None:
			available_labels['DBSCAN'] = st.session_state.dbscan_labels
		# Add LDA dominant topic if available
		if 'lda_results' in st.session_state and st.session_state.lda_results:
			doc_topic_df = st.session_state.lda_results['doc_topic_matrix']
			if not doc_topic_df.empty:
				available_labels['LDA Dominant Topic'] = doc_topic_df.idxmax(axis=1)


		if not available_labels:
			st.warning("No clustering or topic modeling results found in the current session. Please run an algorithm in the 'Clustering Analysis' tab first.")
			return

		selected_clustering_name = st.selectbox(
			"Select Clustering/Topic Result to Analyze",
			list(available_labels.keys())
		)
		cluster_labels = available_labels[selected_clustering_name]

		# --- Get Data for Analysis ---
		# Prefer original preprocessed data for interpretation if possible
		analysis_data = None
		if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
			analysis_data = st.session_state.clustering_input_data.copy()
			# Align data with labels (important if sampling occurred, e.g., hierarchical)
			common_index = analysis_data.index.intersection(cluster_labels.index)
			if len(common_index) != len(cluster_labels):
				st.warning(f"Analysis data index ({len(analysis_data)}) does not fully match cluster label index ({len(cluster_labels)}). Analyzing subset with {len(common_index)} common points.")
			analysis_data = analysis_data.loc[common_index]
			cluster_labels = cluster_labels.loc[common_index]

		elif 'df' in st.session_state and st.session_state.df is not None:
			# Fallback to original df if preprocessed is missing, but warn user
			analysis_data = st.session_state.df.copy()
			common_index = analysis_data.index.intersection(cluster_labels.index)
			analysis_data = analysis_data.loc[common_index]
			cluster_labels = cluster_labels.loc[common_index]
			st.warning("Using original loaded table data for analysis as preprocessed clustering input is unavailable. Results might be less meaningful if data wasn't scaled/numeric.")
			# Try to select only numeric columns for some analyses
			analysis_data_numeric = analysis_data.select_dtypes(include=np.number)
			if not analysis_data_numeric.empty:
				analysis_data = analysis_data_numeric
			else:
				st.error("No numeric data available in the original table for analysis.")
				analysis_data = None # Cannot proceed

		else:
			st.error("No suitable data found for cluster analysis (neither preprocessed input nor original table).")
			return # Cannot proceed


		if analysis_data is not None and not analysis_data.empty:
			# Add cluster labels to the analysis data
			analysis_data['cluster'] = cluster_labels.astype(str) # Use string for categorical coloring/grouping
			# Handle DBSCAN noise label
			if selected_clustering_name == 'DBSCAN':
				analysis_data['cluster'] = analysis_data['cluster'].replace('-1', 'Noise')

			# Filter out noise for some analyses if needed
			analysis_data_no_noise = analysis_data[analysis_data['cluster'] != 'Noise'] if 'Noise' in analysis_data['cluster'].unique() else analysis_data

			# --- Analysis Tabs ---
			analysis_tabs = st.tabs([
				"📊 Cluster Profiles",
				"🔍 Statistical Differences",
				"🔥 Feature Importance",
				"📈 LOS / Outcome Analysis",
				"📋 Generate Report"
			])

			# 1. Cluster Profiles Tab
			with analysis_tabs[0]:
				st.markdown("<h3>Cluster Profiles & Characteristics</h3>", unsafe_allow_html=True)
				st.info("Explore the average characteristics of each cluster based on the input features.")

				try:
					# Calculate summary statistics per cluster (using data without noise for means)
					cluster_summary = analysis_data_no_noise.groupby('cluster').agg(['mean', 'median', 'std', 'count'])

					# Display summary table
					st.dataframe(cluster_summary.style.format("{:.3f}", na_rep="-"), use_container_width=True)

					# Select features for radar plot
					profile_features = st.multiselect(
						"Select features for Radar Plot Profile",
						[col for col in analysis_data_no_noise.columns if col != 'cluster'],
						default=[col for col in analysis_data_no_noise.columns if col != 'cluster'][:min(8, analysis_data_no_noise.shape[1]-1)] # Default up to 8 features
					)

					if profile_features:
						# Normalize data for radar plot (e.g., Min-Max scaling across all data)
						scaler = MinMaxScaler()
						radar_data = analysis_data_no_noise[profile_features].copy()
						# Handle potential NaNs before scaling
						radar_data = radar_data.fillna(radar_data.median()) # Impute with median
						scaled_values = scaler.fit_transform(radar_data)
						scaled_df = pd.DataFrame(scaled_values, columns=profile_features, index=radar_data.index)
						scaled_df['cluster'] = analysis_data_no_noise['cluster']

						# Calculate mean scaled values per cluster
						radar_means = scaled_df.groupby('cluster')[profile_features].mean()

						# Create radar plot
						fig_radar = go.Figure()
						categories = radar_means.columns.tolist()

						for i, cluster_id in enumerate(radar_means.index):
							fig_radar.add_trace(go.Scatterpolar(
								r=radar_means.loc[cluster_id].values.flatten(),
								theta=categories,
								fill='toself',
								name=f"Cluster {cluster_id}",
								line=dict(color=px.colors.qualitative.G10[i % len(px.colors.qualitative.G10)])
							))

						fig_radar.update_layout(
							polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
							showlegend=True,
							title="Cluster Profiles (Scaled Feature Means)"
						)
						st.plotly_chart(fig_radar, use_container_width=True)
					else:
						st.info("Select features to generate the radar plot.")

				except Exception as e:
					st.error(f"Error generating cluster profiles: {e}")
					logging.exception("Error in Cluster Profiles tab")


			# 2. Statistical Differences Tab
			with analysis_tabs[1]:
				st.markdown("<h3>Statistical Differences Between Clusters</h3>", unsafe_allow_html=True)
				st.info("Identify features that show statistically significant differences across the clusters (excluding noise points).")

				if len(analysis_data_no_noise['cluster'].unique()) < 2:
					st.warning("Need at least two non-noise clusters to perform statistical tests.")
				else:
					# Get feature columns (exclude cluster column)
					feature_cols = [col for col in analysis_data_no_noise.columns if col != 'cluster']

					# Let user select features
					selected_features_test = st.multiselect(
						"Select Features for Statistical Testing",
						feature_cols,
						default=feature_cols[:min(10, len(feature_cols))], # Default up to 10
						key="stat_test_features"
					)

					# Test method based on number of clusters
					num_clusters_no_noise = len(analysis_data_no_noise['cluster'].unique())
					if num_clusters_no_noise == 2:
						test_method = st.radio("Statistical Test Method", ["t-test (parametric)", "Mann-Whitney U (non-parametric)"], horizontal=True, key="stat_test_method_2")
						method_map = {"t-test (parametric)": "ttest", "Mann-Whitney U (non-parametric)": "mannwhitneyu"}
					else: # > 2 clusters
						test_method = st.radio("Statistical Test Method", ["ANOVA (parametric)", "Kruskal-Wallis (non-parametric)"], horizontal=True, key="stat_test_method_multi")
						method_map = {"ANOVA (parametric)": "anova", "Kruskal-Wallis (non-parametric)": "kruskal"}


					# Button to run tests
					if st.button("Run Statistical Tests") and selected_features_test:
						try:
							with st.spinner("Performing statistical tests..."):
								# Run tests using the analyzer
								test_results = cluster_analyzer.statistical_testing(
									analysis_data_no_noise, # Use data without noise
									selected_features_test,
									cluster_col='cluster',
									method=method_map[test_method]
								)

							st.success("Statistical tests completed!")
							st.dataframe(test_results.style.format({
								'Statistic': '{:.3f}',
								'P-Value': '{:.4g}', # General format for p-values
								'Adjusted P-Value': '{:.4g}'
							}).applymap(
								lambda v: 'background-color: lightcoral' if isinstance(v, bool) and v else '', subset=['Significant (Adjusted)']
							), use_container_width=True)

							# --- P-value Visualization ---
							st.markdown("<h4>Feature Significance Visualization (-log10 Adjusted P-Value)</h4>", unsafe_allow_html=True)
							results_vis = test_results.dropna(subset=['Adjusted P-Value']).copy() # Drop features where test failed
							if not results_vis.empty:
								# Avoid log(0) errors
								results_vis['log_p'] = -np.log10(results_vis['Adjusted P-Value'] + 1e-10) # Add small epsilon
								significance_threshold = -np.log10(0.05)

								fig_pvals = px.bar(
									results_vis.sort_values('log_p', ascending=False),
									x='Feature', y='log_p',
									color='Significant (Adjusted)',
									color_discrete_map={True: 'red', False: 'grey'},
									labels={'log_p': '-log10(Adjusted P-Value)'},
									title='Feature Significance by Adjusted P-Value'
								)
								fig_pvals.add_hline(y=significance_threshold, line_dash="dash", annotation_text="p=0.05 Threshold")
								st.plotly_chart(fig_pvals, use_container_width=True)
							else:
								st.info("No valid adjusted p-values to visualize.")

						except AttributeError:
							st.error("Cluster Analyzer is not properly initialized or does not have a 'statistical_testing' method.")
						except Exception as e:
							st.error(f"Error performing statistical tests: {str(e)}")
							logging.exception("Error in Statistical Testing tab")
					elif not selected_features_test:
						st.warning("Please select features to run statistical tests.")


			# 3. Feature Importance Tab
			with analysis_tabs[2]:
				st.markdown("<h3>Feature Importance for Cluster Separation</h3>", unsafe_allow_html=True)
				st.info("Identify features that are most important for distinguishing between the clusters using a Random Forest classifier (trained on clusters vs features).")

				if len(analysis_data_no_noise['cluster'].unique()) < 2:
					st.warning("Need at least two non-noise clusters to calculate feature importance.")
				else:
					# Button to calculate importance
					if st.button("Calculate Feature Importance (using RandomForest)"):
						try:
							with st.spinner("Calculating feature importance..."):
								# Calculate importance using the analyzer
								importance_df = cluster_analyzer.calculate_feature_importance(
									analysis_data_no_noise, # Use data without noise
									cluster_col='cluster'
								)

							st.success("Feature importance calculated!")
							st.dataframe(importance_df, use_container_width=True)

							# --- Importance Visualization ---
							fig_imp = px.bar(
								importance_df.head(20).sort_values('Importance', ascending=True), # Show top 20
								x='Importance', y='Feature', orientation='h',
								title='Top 20 Most Important Features for Cluster Separation'
							)
							st.plotly_chart(fig_imp, use_container_width=True)

						except ImportError as e:
							st.error(f"Missing dependency for feature importance: {e}. Please install scikit-learn (`pip install scikit-learn`).")
						except AttributeError:
							st.error("Cluster Analyzer is not properly initialized or does not have a 'calculate_feature_importance' method.")
						except Exception as e:
							st.error(f"Error calculating feature importance: {str(e)}")
							logging.exception("Error in Feature Importance tab")


			# 4. LOS / Outcome Analysis Tab
			with analysis_tabs[3]:
				st.markdown("<h3>Length of Stay (LOS) or Outcome Analysis by Cluster</h3>", unsafe_allow_html=True)
				st.info("Compare clinical outcomes like Length of Stay across the identified clusters. Requires appropriate columns in the *original* loaded table.")

				if st.session_state.df is None:
					st.warning("Original table data not loaded. Cannot perform outcome analysis.")
				else:
					original_df = st.session_state.df
					# Ensure original_df index aligns with cluster_labels index
					common_index_outcome = original_df.index.intersection(cluster_labels.index)
					if len(common_index_outcome) == 0:
						st.error("Index mismatch between original data and cluster labels. Cannot perform outcome analysis.")
					else:
						original_df_aligned = original_df.loc[common_index_outcome]
						cluster_labels_aligned = cluster_labels.loc[common_index_outcome]

						# --- LOS Calculation ---
						st.markdown("#### Length of Stay (LOS)")
						time_columns = original_df_aligned.select_dtypes(include=['datetime64[ns]', 'datetime64[us]', 'datetime64[ms]']).columns.tolist()
						# Try to find potential date columns from objects/strings if no datetime found
						if not time_columns:
							object_cols = original_df_aligned.select_dtypes(include=['object', 'string']).columns
							for col in object_cols:
								try:
									# Attempt conversion on a sample - very basic check
									pd.to_datetime(original_df_aligned[col].dropna().iloc[:5], errors='raise')
									time_columns.append(col)
								except (ValueError, TypeError, AttributeError, IndexError):
									continue # Cannot convert reliably

						if not time_columns:
							st.warning("No datetime columns found or detected in the original table for LOS calculation.")
						else:
							col1, col2, col3 = st.columns(3)
							with col1:
								# Try to guess admission column
								admit_guess = [c for c in time_columns if 'admit' in c.lower()]
								admit_idx = time_columns.index(admit_guess[0]) if admit_guess else 0
								admission_col = st.selectbox("Admission Time Column", time_columns, index=admit_idx, key="los_admit")
							with col2:
								# Try to guess discharge column
								disch_guess = [c for c in time_columns if 'disch' in c.lower()]
								disch_idx = time_columns.index(disch_guess[0]) if disch_guess else (1 if len(time_columns)>1 else 0)
								discharge_col = st.selectbox("Discharge Time Column", time_columns, index=disch_idx, key="los_disch")
							with col3:
								# Try to guess patient ID
								id_guess = [c for c in original_df_aligned.columns if 'subject_id' in c.lower() or 'patient_id' in c.lower()]
								id_idx = original_df_aligned.columns.tolist().index(id_guess[0]) if id_guess else 0
								patient_id_col = st.selectbox("Patient ID Column (for grouping)", original_df_aligned.columns.tolist(), index=id_idx, key="los_id")


							if st.button("Analyze Length of Stay by Cluster"):
								try:
									with st.spinner("Calculating LOS and comparing across clusters..."):
										# Calculate LOS using analyzer
										los_data = cluster_analyzer.calculate_length_of_stay(
											original_df_aligned, # Use aligned data
											admission_col,
											discharge_col,
											patient_id_col
										)

										# Add cluster labels (aligned)
										los_data_clustered = los_data.to_frame(name='los_days').join(cluster_labels_aligned.to_frame(name='cluster'))
										los_data_clustered = los_data_clustered.dropna() # Drop patients where LOS or cluster is missing

										# Add DBSCAN noise handling
										if selected_clustering_name == 'DBSCAN':
											los_data_clustered['cluster'] = los_data_clustered['cluster'].replace('-1', 'Noise')

										# Display summary stats
										st.markdown("##### LOS Summary Statistics by Cluster")
										los_summary = los_data_clustered.groupby('cluster')['los_days'].agg(['mean', 'median', 'std', 'count'])
										st.dataframe(los_summary.style.format("{:.2f}"), use_container_width=True)

										# --- LOS Visualization ---
										st.markdown("##### LOS Distribution by Cluster")
										fig_los = px.box(los_data_clustered, x='cluster', y='los_days',
														title="Length of Stay Distribution by Cluster",
														labels={'los_days': 'Length of Stay (Days)'},
														points='outliers', color='cluster')
										st.plotly_chart(fig_los, use_container_width=True)

										# --- Statistical Test for LOS ---
										st.markdown("##### Statistical Test for LOS Differences")
										los_no_noise = los_data_clustered[los_data_clustered['cluster'] != 'Noise']
										unique_clusters = los_no_noise['cluster'].unique()

										if len(unique_clusters) < 2:
											st.info("Need at least two non-noise clusters for statistical comparison.")
										elif len(unique_clusters) == 2:
											# Mann-Whitney U test (non-parametric often safer for LOS)
											group1 = los_no_noise[los_no_noise['cluster'] == unique_clusters[0]]['los_days']
											group2 = los_no_noise[los_no_noise['cluster'] == unique_clusters[1]]['los_days']
											stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
											st.markdown(f"**Mann-Whitney U Test:** Statistic={stat:.3f}, p-value={p_val:.4g}")

											if p_val < 0.05:
												st.success("Significant difference in LOS found (p < 0.05).")
											else:
												st.info("No significant difference in LOS found (p >= 0.05).")
										else:
											# Kruskal-Wallis test (non-parametric ANOVA)
											groups = [los_no_noise[los_no_noise['cluster'] == c]['los_days'] for c in unique_clusters]
											stat, p_val = stats.kruskal(*groups)
											st.markdown(f"**Kruskal-Wallis Test:** Statistic={stat:.3f}, p-value={p_val:.4g}")
											if p_val < 0.05:
												st.success("Significant difference in LOS found across clusters (p < 0.05).")
											else:
												st.info("No significant difference in LOS found across clusters (p >= 0.05).")

								except AttributeError:
									st.error("Cluster Analyzer is not properly initialized or does not have a 'calculate_length_of_stay' method.")
								except KeyError as e:
									st.error(f"Column Error: Could not find column '{e}'. Check selections.")
								except ValueError as e:
									st.error(f"Data Error: {e}. Ensure time columns are in a parsable format and admit time is before discharge time.")
								except Exception as e:
									st.error(f"Error analyzing LOS: {str(e)}")
									logging.exception("Error in LOS Analysis")


						# --- Other Outcome Analysis (Example: Mortality) ---
						st.markdown("#### Other Outcome Analysis (e.g., Mortality)")
						# Find potential mortality columns
						mortality_cols = [c for c in original_df_aligned.columns if 'mortality' in c.lower() or 'death' in c.lower() or 'expire_flag' in c.lower()]
						if mortality_cols:
							outcome_col = st.selectbox("Select Outcome Column (e.g., Mortality Flag)", mortality_cols + ["None"], index=0)
							if outcome_col != "None":
								if st.button(f"Analyze {outcome_col} by Cluster"):
									try:
										# Ensure outcome is binary (0/1) or boolean
										outcome_data = original_df_aligned[[outcome_col]].join(cluster_labels_aligned.to_frame(name='cluster'))
										outcome_data = outcome_data.dropna()
										# Attempt conversion to numeric/boolean
										try:
											outcome_data[outcome_col] = pd.to_numeric(outcome_data[outcome_col])
											# Check if mostly 0s and 1s
											if not outcome_data[outcome_col].isin([0, 1]).all():
												st.warning(f"Outcome column '{outcome_col}' contains values other than 0/1. Analysis might be invalid. Trying anyway...")
										except ValueError:
											st.warning(f"Could not convert outcome column '{outcome_col}' to numeric. Trying boolean conversion.")
											outcome_data[outcome_col] = outcome_data[outcome_col].astype(bool)


										# Add DBSCAN noise handling
										if selected_clustering_name == 'DBSCAN':
											outcome_data['cluster'] = outcome_data['cluster'].replace('-1', 'Noise')

										st.markdown(f"##### {outcome_col} Rate by Cluster")
										# Calculate rate (assuming 1 = event, 0 = no event)
										outcome_summary = outcome_data.groupby('cluster')[outcome_col].agg(['mean', 'count'])
										outcome_summary.rename(columns={'mean': 'Event Rate'}, inplace=True)
										st.dataframe(outcome_summary.style.format({'Event Rate': '{:.1%}'}), use_container_width=True) # Format as percentage

										# --- Chi-squared Test ---
										st.markdown(f"##### Statistical Test for {outcome_col} Differences")
										outcome_no_noise = outcome_data[outcome_data['cluster'] != 'Noise']
										if len(outcome_no_noise['cluster'].unique()) >= 2:
											contingency_table = pd.crosstab(outcome_no_noise['cluster'], outcome_no_noise[outcome_col])
											st.write("Contingency Table (Cluster vs Outcome):")
											st.dataframe(contingency_table)
											chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
											st.markdown(f"**Chi-squared Test:** Chi2={chi2:.3f}, p-value={p_val:.4g}")
											if p_val < 0.05: st.success(f"Significant difference in {outcome_col} found across clusters (p < 0.05).")
											else: st.info(f"No significant difference in {outcome_col} found across clusters (p >= 0.05).")
										else:
											st.info("Need at least two non-noise clusters for Chi-squared test.")

									except Exception as e:
										st.error(f"Error analyzing outcome '{outcome_col}': {e}")
										logging.exception(f"Error analyzing outcome {outcome_col}")
						else:
							st.info("No columns matching typical mortality indicators found in the original table.")


			# 5. Generate Report Tab
			with analysis_tabs[4]:
				st.markdown("<h3>Generate Cluster Analysis Report</h3>", unsafe_allow_html=True)
				st.info("Create a downloadable HTML report summarizing the cluster analysis findings, including profiles, statistics, and visualizations.")

				report_title = st.text_input("Report Title", value=f"{selected_clustering_name} Analysis Report")
				include_plots_report = st.checkbox("Include Visualizations in Report", value=True)

				# Add option to select which sections to include
				st.markdown("Select sections to include:")
				include_profile = st.checkbox("Cluster Profiles", value=True)
				include_stats = st.checkbox("Statistical Differences", value=True)
				include_importance = st.checkbox("Feature Importance", value=True)
				include_outcome = st.checkbox("LOS/Outcome Analysis", value=True)
				# Add more sections as needed

				if st.button("Generate HTML Report"):
					try:
						with st.spinner("Generating report..."):
							# Gather data for the report (this might involve re-running parts of the analysis or retrieving from session state)
							report_data = {
								'title': report_title,
								'clustering_method': selected_clustering_name,
								'analysis_data': analysis_data, # Pass the data used
								'cluster_labels': cluster_labels, # Pass the labels
								'include_plots': include_plots_report,
								'sections': {
									'profile': include_profile,
									'stats': include_stats,
									'importance': include_importance,
									'outcome': include_outcome
								},
								# Add other necessary data like test results, importance scores, LOS data etc.
								# These might need to be explicitly passed or retrieved from session state
								# Example:
								# 'stat_test_results': st.session_state.get('last_stat_test_results'),
								# 'feature_importance': st.session_state.get('last_feature_importance'),
								# 'los_data': st.session_state.get('last_los_data_clustered'),
							}

							# Generate HTML using the analyzer's method
							# This method needs to be implemented in MIMICClusterAnalyzer
							html_content = cluster_analyzer.generate_html_report(report_data)

						st.success("Report generated successfully!")
						st.download_button(
							label="Download HTML Report",
							data=html_content,
							file_name=f"{selected_clustering_name}_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
							mime="text/html",
						)
					except NotImplementedError:
						st.error("Report generation functionality is not yet implemented in the Cluster Analyzer.")
					except AttributeError:
						st.error("Cluster Analyzer is not properly initialized or does not have a 'generate_html_report' method.")
					except Exception as e:
						st.error(f"Error generating report: {str(e)}")
						logging.exception("Error generating report")


# =============================================
# Main Application Class
# =============================================
class MIMICDashboardApp:

	def __init__(self):
		logging.info("Initializing MIMICDashboardApp...")
		# Initialize core components
		self.data_handler        = DataLoader()
		# self.visualizer          = MIMICVisualizer()
		self.feature_engineer    = FeatureEngineerUtils()
		self.clustering_analyzer = ClusteringAnalyzer() # Handles running algorithms, evaluation
		self.cluster_analyzer    = ClusterInterpreter()    # Handles post-cluster analysis, interpretation
		# Initialize UI components for tabs
		self.feature_engineering_ui    = FeatureEngineeringTab()
		self.clustering_analysis_ui    = ClusteringAnalysisTab()
		self.analysis_visualization_ui = AnalysisVisualizationTab()

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
						st.session_state.detected_order_cols     = self.feature_engineer.detect_order_columns(df)
						st.session_state.detected_time_cols      = self.feature_engineer.detect_temporal_columns(df)
						st.session_state.detected_patient_id_col = self.feature_engineer.detect_patient_id_column(df)
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
				try:
					# Check if Dask was used to load the data
					use_dask = st.session_state.get('use_dask', False)

					# Pass the use_dask parameter to all visualizer methods
					self.visualizer.display_data_preview(st.session_state.df, use_dask=use_dask)
					self.visualizer.display_dataset_statistics(st.session_state.df, use_dask=use_dask)
					self.visualizer.display_visualizations(st.session_state.df, use_dask=use_dask)
				except AttributeError:
					st.error("Visualizer component is not properly initialized or lacks required methods.")
				except Exception as e:
					st.error(f"An error occurred in the Exploration tab: {e}")
					logging.exception("Error in Exploration Tab")


			# Tab 2: Feature Engineering
			with tab2:
				try:
					self.feature_engineering_ui.render(self.feature_engineer)
				except AttributeError:
					st.error("Feature Engineering UI component is not properly initialized.")
				except Exception as e:
					st.error(f"An error occurred in the Feature Engineering tab: {e}")
					logging.exception("Error in Feature Engineering Tab")


			# Tab 3: Clustering Analysis
			with tab3:
				try:
					# Pass both analyzers as needed by different parts of the tab
					self.clustering_analysis_ui.render(self.clustering_analyzer, self.feature_engineer)
				except AttributeError:
					st.error("Clustering Analysis UI component is not properly initialized.")
				except Exception as e:
					st.error(f"An error occurred in the Clustering Analysis tab: {e}")
					logging.exception("Error in Clustering Analysis Tab")


			# Tab 4: Analysis & Visualization (Cluster Interpretation)
			with tab4:
				try:
					self.analysis_visualization_ui.render(self.cluster_analyzer)
				except AttributeError:
					st.error("Analysis & Visualization UI component is not properly initialized.")
				except Exception as e:
					st.error(f"An error occurred in the Cluster Interpretation tab: {e}")
					logging.exception("Error in Cluster Interpretation Tab")


			# Tab 5: Export Options
			with tab5:
				st.markdown("<h2 class='sub-header'>Export Loaded Data</h2>", unsafe_allow_html=True)
				st.info("Export the currently loaded (and potentially sampled) data shown in the 'Exploration' tab.")
				export_col1, export_col2 = st.columns(2)

				with export_col1:
					export_format = st.radio("Export Format", ["CSV", "Parquet"], index=0, key="export_main_format")
					export_filename_base = f"mimic_data_{st.session_state.selected_module}_{st.session_state.selected_table}"
					export_filename = f"{export_filename_base}.{export_format.lower()}"

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

# =============================================
# Entry Point
# =============================================

def main():
	app = MIMICDashboardApp()
	app.run()

if __name__ == "__main__":
	main()
