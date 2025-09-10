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
from mimic_iv_analysis.visualization.visualizer_utils import display_dataframe_head


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

	def _order_frequency_matrix(self):

		# Get available columns
		all_columns = st.session_state.df.columns.tolist()

		st.markdown("### Create Order Frequency Matrix")
		st.info("This creates a matrix where rows are patients and columns are order types, with cells showing frequency of each order type per patient.")

		# Column selection
		cols = st.columns(3)
		with cols[0]:
			# Suggest patient ID column but allow selection from all columns
			patient_id_col = st.selectbox(
				label   = "Select Patient ID Column",
				options = all_columns,
				index   = all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
				key     = "freq_patient_id_col",
				help    = "Column containing unique patient identifiers" )

		with cols[1]:
			# Suggest order column but allow selection from all columns
			order_col = st.selectbox(
				label   = "Select Order Type Column",
				options = all_columns,
				index   = all_columns.index('order_type') if 'order_type' in all_columns else 0,
				key     = "freq_order_col",
				help    = "Column containing order types/names" )

		with cols[2]:
			top_n = st.number_input("Top N Order Types", min_value=0, max_value=100, value=20, help="Limit to most frequent order types (0 = include all)")

		cols = st.columns(3)
		with cols[0]:
			normalize = st.checkbox("Normalize by Patient", value=False, help="Convert frequencies to percentages of total orders per patient")

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
			display_dataframe_head(st.session_state.freq_matrix)

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

	def _temporal_order_sequences(self):

		# Get available columns
		all_columns = st.session_state.df.columns.tolist()


		st.markdown("<h3>Extract Temporal Order Sequences</h3>", unsafe_allow_html=True)
		st.info("This extracts chronological sequences of orders for each patient, preserving the temporal relationships between different orders.")

		# Column selection
		col1, col2, col3 = st.columns(3)
		with col1:
			seq_patient_id_col = st.selectbox(
				label   = "Select Patient ID Column",
				options = all_columns,
				index   = all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
				key     = "seq_patient_id_col",
				help    = "Column containing unique patient identifiers"
			)

		with col2:
			seq_order_col = st.selectbox(
				label   = "Select Order Type Column",
				options = all_columns,
				index   = all_columns.index('order_type') if 'order_type' in all_columns else 0,
				key     = "seq_order_col",
				help    = "Column containing order types/names"
			)

		with col3:
			seq_time_col = st.selectbox(
				label   = "Select Timestamp Column",
				options = all_columns,
				index   = all_columns.index('ordertime') if 'ordertime' in all_columns else 0,
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

			patient_id = st.selectbox(label="Select Patient ID", options=st.session_state.order_sequences.keys(), index=0, key="seq_patient_id_col2")

			sequence = st.session_state.order_sequences[patient_id]
			sequence_str = " → ".join([str(order) for order in sequence])
	
			st.markdown(f"<strong>Patient {patient_id}:</strong> {sequence_str}", unsafe_allow_html=True)
			st.markdown("<hr>", unsafe_allow_html=True)


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

	def _order_type_distributions(self):

		# Get available columns
		all_columns = st.session_state.df.columns.tolist()


		st.markdown("<h3>Analyze Order Type Distributions</h3>", unsafe_allow_html=True)
		st.info("This analyzes the distribution of different order types across the dataset and for individual patients.")

		# Column selection
		col1, col2 = st.columns(2)
		with col1:
			dist_patient_id_col = st.selectbox(
				label   = "Select Patient ID Column",
				options = all_columns,
				index   = all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
				key     = "dist_patient_id_col",
				help    = "Column containing unique patient identifiers"
			)

		with col2:
			dist_order_col = st.selectbox(
				label   = "Select Order Type Column",
				options = all_columns,
				index   = all_columns.index('order_type') if 'order_type' in all_columns else 0,
				key     = "dist_order_col",
				help    = "Column containing order types/names"
			)

		# Generate button
		if st.button("Analyze Order Distributions"):
			with st.spinner("Analyzing order type distributions..."):
				overall_dist, patient_dist = FeatureEngineerUtils.get_order_type_distributions( df=st.session_state.df, patient_id_col=dist_patient_id_col, order_col=dist_order_col)
				st.session_state.order_dist         = overall_dist
				st.session_state.patient_order_dist = patient_dist
				st.success("Order distributions analyzed.")

		# Display results if available
		if 'order_dist' in st.session_state and st.session_state.order_dist is not None:

			st.markdown("<h4>Overall Order Type Distribution</h4>", unsafe_allow_html=True)
			
			# Create pie chart for overall distribution
			top_n_orders = 15  # Show top 15 for pie chart
			if not st.session_state.order_dist.empty:
				top_orders = st.session_state.order_dist.head(top_n_orders)

				fig_pie = px.pie(
					data_frame = top_orders.to_frame('frequency'),
					values     = 'frequency',
					title      = f"Overall Distribution of {dist_order_col} (Top {top_n_orders})"
				)
				st.plotly_chart(fig_pie, use_container_width=True)

				fig_bar = px.bar( data_frame = top_orders.to_frame('frequency'), title = f"Top 20 {dist_order_col} by Frequency" )
				st.plotly_chart(fig_bar, use_container_width=True)
			else:
				st.info("Overall distribution data is empty.")

			self._display_export_options(data=st.session_state.order_dist, feature_type='overall_order_distribution')
   
   
			# Patient-level distribution
			if 'patient_order_dist' in st.session_state and st.session_state.patient_order_dist is not None and not st.session_state.patient_order_dist.empty:
				st.markdown("<h4>Patient-Level Order Type Distribution</h4>", unsafe_allow_html=True)

				patients_df = st.session_state.patient_order_dist

				subject_id = st.selectbox(label="Select Patient ID", options=patients_df.index, index=0)

				patient_data = patients_df.loc[subject_id].to_frame('proportion')

				fig_patient = px.bar(data_frame=patient_data, title=f"Patient {subject_id} Order Type Distribution", x=patient_data.index, y='proportion')
				st.plotly_chart(fig_patient, use_container_width=True)


				self._display_export_options(data=st.session_state.patient_order_dist, feature_type='patient_order_distribution')

	def _order_timing_analysis(self):

		# Get available columns
		all_columns = st.session_state.df.columns.tolist()

		st.markdown("<h3>Analyze Order Timing</h3>", unsafe_allow_html=True)
		st.markdown("""
		<div class='info-box'>
		This analyzes the timing of orders relative to admission, providing features about when orders occur during a patient's stay.
		</div>
		""", unsafe_allow_html=True)

		# Column selection
		col1, col2 = st.columns(2)
		with col1:
			timing_patient_id_col = st.selectbox(
				"Select Patient ID Column",
				all_columns,
				index=all_columns.index('subject_id') if 'subject_id' in all_columns else 0,
				key="timing_patient_id_col",
				help="Column containing unique patient identifiers"
			)

		with col2:
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
			admission_time_col = st.selectbox(
				label   = "Select Admission Time Column (Optional)",
				options = [None] + all_columns,
				index   = ([None] + all_columns).index('admittime') if 'admittime' in all_columns else 0,
				key     = "admission_time_col",
				help    = "Column containing admission timestamps (for relative timing features)",
			)
			# admission_time_col = None if admission_time_col == "None" else admission_time_col


		# Optional discharge time column - try to find 'dischtime' or similar
		discharge_time_col = st.selectbox(
			label   = "Select Discharge Time Column (Optional)",
			options = [None] + all_columns,
			index   = ([None] + all_columns).index('dischtime') if 'dischtime' in all_columns else 0,
			key     = "discharge_time_col",
			help    = "Column containing discharge timestamps (for relative timing features)",
		)
		# discharge_time_col = None if discharge_time_col == "None" else discharge_time_col


		# Generate button
		if st.button("Generate Timing Features"):
			with st.spinner("Generating order timing features..."):
				timing_features = FeatureEngineerUtils.create_order_timing_features(
					df                 = st.session_state.df,
					patient_id_col     = timing_patient_id_col,
					order_col          = timing_order_col,
					order_time_col     = order_time_col,
					admission_time_col = admission_time_col,
					discharge_time_col = discharge_time_col
				)
				st.session_state.timing_features = timing_features
				st.success("Order timing features generated.")



		# Display results if available
		if 'timing_features' in st.session_state and st.session_state.timing_features is not None:
			st.markdown("<h4>Order Timing Features</h4>", unsafe_allow_html=True)

			# Show preview of features
			display_dataframe_head(st.session_state.timing_features)

			# Generate visualizations based on available features
			st.markdown("<h4>Order Timing Visualizations</h4>", unsafe_allow_html=True)

			timing_df = st.session_state.timing_features
			numeric_cols = timing_df.select_dtypes(include=['number']).columns

			# try:
			# Bar chart of total orders and unique orders
			if 'total_orders' in timing_df.columns and 'unique_order_types' in timing_df.columns:
				col1, col2 = st.columns(2)
				with col1:
					fig_total = px.histogram(
						data_frame = timing_df['total_orders'],
						# x          = 'total_orders',
						nbins      = 30,
						title      = "Distribution of Total Orders per Patient" )
					st.plotly_chart(fig_total, use_container_width=True)
				with col2:
					fig_unique = px.histogram(
						data_frame = timing_df['unique_order_types'],
						# x          = 'unique_order_types',
						nbins      = 30,
						title      = "Distribution of Unique Order Types per Patient" )
					st.plotly_chart(fig_unique, use_container_width=True)

			# Time-based analyses (if admission time was provided)
			relative_time_cols = ['time_to_first_order_hours', 'orders_in_first_24h', 'orders_in_first_48h', 'orders_in_first_72h']
			if admission_time_col and any(col in timing_df.columns for col in relative_time_cols):
				col1, col2 = st.columns(2)
				with col1:
					if 'time_to_first_order_hours' in timing_df.columns:
						fig_first_order = px.histogram(
							data_frame = timing_df['time_to_first_order_hours'],
							# x          = 'time_to_first_order_hours',
							nbins      = 30,
							title      = "Time from Admission to First Order (hours)" )
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
							data_frame = orders_by_time,
							x          = 'Time Period',
							y          = 'Average Orders',
							title      = "Average Orders in Time Periods After Admission" )
						st.plotly_chart(fig_time_orders, use_container_width=True)
			# except Exception as e:
			# 	st.error(f"Error generating timing visualizations: {e}")

			# Save options
			self._display_export_options(data=st.session_state.timing_features, feature_type='order_timing_features')

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


		with feature_tabs[0]:
			self._order_frequency_matrix()

		with feature_tabs[1]:
			self._temporal_order_sequences()

		with feature_tabs[2]:
			self._order_type_distributions()

		with feature_tabs[3]:
			self._order_timing_analysis()

