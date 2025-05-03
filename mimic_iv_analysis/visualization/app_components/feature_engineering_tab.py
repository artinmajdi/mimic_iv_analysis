# Standard library imports
import os
import logging
import datetime
from typing import Tuple, Optional

# Data processing imports
import numpy as np
import pandas as pd

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit import
import streamlit as st

# Local application imports
from mimic_iv_analysis.core.feature_engineering import FeatureEngineerUtils


class FeatureEngineeringTab:
	""" Handles the UI and logic for the Feature Engineering tab. """

	@staticmethod
	def _display_export_options(data, feature_type='engineered_feature'):
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

					filepath = FeatureEngineerUtils.save_features(
						features     = data,
						feature_type = feature_type,
						base_path    = base_path,
						format       = save_format.lower()
					)
					st.success(f"Saved {feature_type.replace('_', ' ').title()} to {filepath}")
				except AttributeError:
					st.error("Feature Engineer is not properly initialized or does not have a 'save_features' method.")
				except Exception as e:
					st.error(f"Error saving {feature_type.replace('_', ' ').title()}: {str(e)}")

	def render(self):
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
					label   = "Select Patient ID Column",
					options = all_columns,
					index   = patient_id_col_index,
					key     = "freq_patient_id_col",
					help    = "Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column but allow selection from all columns

				order_col = st.selectbox(
					label   = "Select Order Type Column",
					options = all_columns,
					index   = all_columns.index('order_type') if 'order_type' in all_columns else 0,
					key     = "freq_order_col",
					help    = "Column containing order types/names"
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

						freq_matrix = FeatureEngineerUtils.create_order_frequency_matrix(
							df             = st.session_state.df,
							patient_id_col = patient_id_col,
							order_col      = order_col,
							normalize      = normalize,
							top_n          = top_n,
							use_dask       = use_dask
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

						sample_rows  = min(50, heatmap_data.shape[0])
						sample_cols  = min(50, heatmap_data.shape[1])
						heatmap_data = heatmap_data.iloc[:sample_rows, :sample_cols]

					fig = px.imshow(img    = heatmap_data.T,
									labels = dict(x="Patient ID (Index)", y="Order Type", color="Frequency/Count"),
									aspect = "auto")

					st.plotly_chart(fig, use_container_width=True)
				except Exception as e:
					st.error(f"Could not generate heatmap: {e}")

				# Save options
				self._display_export_options(data=st.session_state.freq_matrix, feature_type='order_frequency_matrix')

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
					label   = "Select Patient ID Column",
					options = all_columns,
					index   = patient_id_col_index,
					key     = "seq_patient_id_col",
					help    = "Column containing unique patient identifiers"
				)

			with col2:
				# Suggest order column
				order_col_index = 0
				if st.session_state.get('detected_order_cols') and st.session_state['detected_order_cols'][0] in all_columns:
					order_col_index = all_columns.index(st.session_state['detected_order_cols'][0])

				seq_order_col = st.selectbox(
					label   = "Select Order Type Column",
					options = all_columns,
					index   = order_col_index,
					key     = "seq_order_col",
					help    = "Column containing order types/names"
				)

			with col3:
				# Suggest time column
				time_col_index = 0
				if st.session_state.get('detected_time_cols') and st.session_state['detected_time_cols'][0] in all_columns:
					time_col_index = all_columns.index(st.session_state['detected_time_cols'][0])

				seq_time_col = st.selectbox(
					label   = "Select Timestamp Column",
					options = all_columns,
					index   = time_col_index,
					key     = "seq_time_col",
					help    = "Column containing order timestamps"
				)

			# Options
			max_seq_length = st.slider("Maximum Sequence Length", min_value=5, max_value=100, value=20, help="Maximum number of orders to include in each sequence")

			# Generate button
			if st.button("Extract Order Sequences"):
				try:
					with st.spinner("Extracting temporal order sequences..."):
						# Check if Dask was used to load the data
						use_dask = st.session_state.get('use_dask', False)

						sequences = FeatureEngineerUtils.extract_temporal_order_sequences(
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

						transition_matrix = FeatureEngineerUtils.calculate_order_transition_matrix( sequences=sequences, top_n=15 )

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
							img    = st.session_state.transition_matrix,
							labels = dict(x="Next Order", y="Current Order", color="Transition Probability"),
							x      = st.session_state.transition_matrix.columns,
							y      = st.session_state.transition_matrix.index,
							color_continuous_scale = 'Blues'
						)
						fig.update_layout(height=700)
						st.plotly_chart(fig, use_container_width=True)

					except Exception as e:
						st.error(f"Could not generate transition matrix heatmap: {e}")


				# Save options for sequences (transition matrix is derived, not saved directly here)
				self._display_export_options(data=st.session_state.order_sequences, feature_type='temporal_order_sequences')


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
					label   = "Select Patient ID Column",
					options = all_columns,
					index   = patient_id_col_index,
					key     = "dist_patient_id_col",
					help    = "Column containing unique patient identifiers"
				)

			with col2:

				# Suggest order column
				order_col_index = 0
				if st.session_state.get('detected_order_cols') and st.session_state['detected_order_cols'][0] in all_columns:
					order_col_index = all_columns.index(st.session_state['detected_order_cols'][0])

				dist_order_col = st.selectbox(
					label   = "Select Order Type Column",
					options = all_columns,
					index   = order_col_index,
					key     = "dist_order_col",
					help    = "Column containing order types/names"
				)

			# Generate button
			if st.button("Analyze Order Distributions"):
				try:
					with st.spinner("Analyzing order type distributions..."):
						overall_dist, patient_dist = FeatureEngineerUtils.get_order_type_distributions(
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
				self._display_export_options(data=st.session_state.order_dist, feature_type='overall_order_distribution')

				if 'patient_order_dist' in st.session_state and st.session_state.patient_order_dist is not None:
					self._display_export_options(data=st.session_state.patient_order_dist, feature_type='patient_order_distribution')


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
						timing_features = FeatureEngineerUtils.create_order_timing_features(
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
				self._display_export_options(data=st.session_state.timing_features, feature_type='order_timing_features')
