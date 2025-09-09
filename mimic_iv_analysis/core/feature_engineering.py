import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import dask.dataframe as dd
import streamlit as st

class FeatureEngineerUtils:
	"""Handles feature engineering for MIMIC-IV data."""
	
	def __init__(self):
		"""Initialize with persisted resources tracking."""
		self._persisted_resources = {}
	
	def _cleanup_persisted_resources(self, resources_dict: Dict[str, dd.DataFrame] = None):
		"""Clean up persisted Dask DataFrames to free memory."""
		try:
			if resources_dict is None:
				resources_dict = self._persisted_resources
			
			for name, df in resources_dict.items():
				if isinstance(df, dd.DataFrame):
					try:
						# Clear the persisted data from memory
						df.clear_divisions()
					except Exception as cleanup_error:
						pass  # Silently handle cleanup errors
			
			# Clear the tracking dictionary
			if resources_dict is self._persisted_resources:
				self._persisted_resources.clear()
				
		except Exception:
			pass  # Silently handle cleanup errors

	@staticmethod
	def detect_order_columns(df: pd.DataFrame) -> List[str]:
		"""Detect columns likely to contain order information."""
		order_columns = []

		# Check column names that might represent orders
		order_related_terms = [ 'order', 'medication', 'drug', 'procedure', 'treatment', 'item', 'event', 'action', 'prescription', 'poe' ]

		for col in df.columns:
			col_lower = col.lower()

			# Check if any order-related term is in column name
			if any(term in col_lower for term in order_related_terms):
				order_columns.append(col)

			# Or if column has common order-related suffixes/prefixes
			elif col_lower.endswith('_id') or col_lower.endswith('_type') or col_lower.endswith('_name') or col_lower.startswith('order_'):
				order_columns.append(col)

		return order_columns

	@staticmethod
	def detect_temporal_columns(df: pd.DataFrame) -> List[str]:
		"""Detect columns containing temporal information."""
		time_columns = []

		# Check column names
		time_related_terms = [ 'time', 'date', 'datetime', 'timestamp', 'start', 'end', 'created', 'updated', 'admission', 'discharge' ]

		for col in df.columns:
			col_lower = col.lower()
			if any(term in col_lower for term in time_related_terms):
				# Check if column contains datetime-like values
				if df[col].dtype == 'datetime64[ns]':
					time_columns.append(col)
				elif df[col].dtype == 'object':
					# Try to detect if string column contains dates
					sample = df[col].dropna().head(10).astype(str)
					date_patterns = [
						r'\d{4}-\d{2}-\d{2}',  # yyyy-mm-dd
						r'\d{2}/\d{2}/\d{4}',  # mm/dd/yyyy
						r'\d{4}/\d{2}/\d{2}',  # yyyy/mm/dd
						r'\d{2}-\d{2}-\d{4}',  # mm-dd-yyyy
					]

					if any(sample.str.contains(pattern).any() for pattern in date_patterns):
						time_columns.append(col)

		return time_columns

	@staticmethod
	def detect_patient_id_column(df: pd.DataFrame) -> Optional[str]:
		"""Detect column likely to contain patient identifiers."""
		# Common patient ID column names in MIMIC-IV
		patient_id_candidates = [
			'subject_id', 'patient_id', 'patientid', 'pat_id', 'patient'
		]

		for candidate in patient_id_candidates:
			if candidate in df.columns:
				return candidate

		# If no exact match, look for columns with 'id' that might be patient IDs
		id_columns = [col for col in df.columns if 'id' in col.lower()]
		if id_columns:
			# Choose the one that looks most like a patient ID based on cardinality and naming
			for col in id_columns:
				if isinstance(df, dd.DataFrame):
					if (df[col].nunique().compute() > df.shape[0].compute() * 0.1):  # High cardinality
						return col
				else:
					if (df[col].nunique() > len(df) * 0.1):  # High cardinality
						return col

		return None

	@staticmethod
	def create_order_frequency_matrix(df: Union[pd.DataFrame, dd.DataFrame], patient_id_col: str, order_col: str, normalize: bool = False, top_n: int = 20, use_dask: bool = False) -> pd.DataFrame:
		"""
		Creates a matrix of order frequencies by patient.

		Args:
			df: DataFrame with order data
			patient_id_col: Column containing patient IDs
			order_col: Column containing order types/names
			normalize: If True, normalize counts by patient
			top_n: Maximum number of order types to include (for dimensionality reduction)

		Returns:
			DataFrame with patients as rows and order types as columns
		"""
		# Validate columns exist
		if patient_id_col not in df.columns or order_col not in df.columns:
			raise ValueError(f"Columns {patient_id_col} or {order_col} not found in DataFrame")

		persisted_intermediates = {}  # Track intermediate results
		
		try:
			# Convert Dask DataFrame to pandas if necessary
			if use_dask and isinstance(df, dd.DataFrame):
				with st.spinner('Computing data for order frequency matrix...'):
					if top_n > 0:
						# Get top N order types first
						value_counts = df[order_col].value_counts().compute()
						top_orders = value_counts.head(top_n).index.tolist()
						
						# Persist the DataFrame before filtering
						df_persisted = df.persist()
						persisted_intermediates['input_df'] = df_persisted
						
						# Filter and persist before computing
						filtered_dask = df_persisted[df_persisted[order_col].isin(top_orders)]
						filtered_dask_persisted = filtered_dask.persist()
						persisted_intermediates['filtered_df'] = filtered_dask_persisted
						
						# Final compute
						filtered_df = filtered_dask_persisted.compute()
					else:
						# Persist entire DataFrame before computing
						df_persisted = df.persist()
						persisted_intermediates['input_df'] = df_persisted
						filtered_df = df_persisted.compute()
			elif isinstance(df, dd.DataFrame):
				# Dask DataFrame but use_dask=False, compute directly
				if top_n > 0:
					value_counts = df[order_col].value_counts().compute()
					top_orders = value_counts.head(top_n).index.tolist()
					filtered_df = df[df[order_col].isin(top_orders)].compute()
				else:
					filtered_df = df.compute()
			else:
				# Regular pandas processing
				# Get the most common order types for dimensionality reduction
				if top_n > 0:
					top_orders = df[order_col].value_counts().head(top_n).index.tolist()
					filtered_df = df[df[order_col].isin(top_orders)].copy()
				else:
					filtered_df = df.copy()

			# Create a crosstab of patient IDs and order types
			freq_matrix = pd.crosstab(
				filtered_df[patient_id_col],
				filtered_df[order_col]
			)

			# Normalize if requested
			if normalize:
				freq_matrix = freq_matrix.div(freq_matrix.sum(axis=1), axis=0)

			return freq_matrix
			
		except Exception as e:
			# Cleanup persisted resources on error
			for name, persisted_df in persisted_intermediates.items():
				if isinstance(persisted_df, dd.DataFrame):
					try:
						persisted_df.clear_divisions()
					except Exception:
						pass
			raise

	@staticmethod
	def extract_temporal_order_sequences(df: Union[pd.DataFrame, dd.DataFrame], patient_id_col: str, order_col: str, time_col: str, max_sequence_length: int = 20, use_dask: bool = False) -> Dict[Any, List[str]]:
		"""
		Extracts temporal sequences of orders for each patient.

		Args:
			df: DataFrame with order data
			patient_id_col: Column containing patient IDs
			order_col: Column containing order types
			time_col: Column containing timestamps
			max_sequence_length: Maximum number of orders to include in each sequence

		Returns:
			Dictionary mapping patient IDs to lists of order sequences
		"""
		# Validate columns exist
		if not all(col in df.columns for col in [patient_id_col, order_col, time_col]):
			missing = [col for col in [patient_id_col, order_col, time_col] if col not in df.columns]
			raise ValueError(f"Columns {missing} not found in DataFrame")

		persisted_intermediates = {}  # Track intermediate results
		
		try:
			# Convert Dask DataFrame to pandas if necessary
			if use_dask and hasattr(df, 'compute'):
				with st.spinner('Computing data for temporal order sequences...'):
					# For sequences, we need the complete DataFrame
					# Persist before computing to optimize memory usage
					if isinstance(df, dd.DataFrame):
						# Select only needed columns and persist
						needed_cols  = [patient_id_col, order_col, time_col]
						df_subset    = df[needed_cols]
						df_persisted = df_subset.persist()
						persisted_intermediates['input_df'] = df_persisted
						
						# Compute the persisted DataFrame
						df = df_persisted.compute()
					else:
						df = df.compute()
			else:
				# Make a copy of the DataFrame
				df = df.copy()

			# Ensure time column is datetime
			if df[time_col].dtype != 'datetime64[ns]':
				try:
					df[time_col] = pd.to_datetime(df[time_col])
				except:
					raise ValueError(f"Could not convert {time_col} to datetime format")

			# Sort by patient ID and timestamp
			sorted_df = df.sort_values([patient_id_col, time_col])

			# Extract sequences
			sequences = {}
			for patient_id, group in sorted_df.groupby(patient_id_col):
				# Get ordered sequence of orders
				patient_sequence = group[order_col].tolist()

				# Limit sequence length if needed
				if max_sequence_length > 0 and len(patient_sequence) > max_sequence_length:
					patient_sequence = patient_sequence[:max_sequence_length]

				sequences[patient_id] = patient_sequence

			return sequences
			
		except Exception as e:
			# Cleanup persisted resources on error
			for name, persisted_df in persisted_intermediates.items():
				if isinstance(persisted_df, dd.DataFrame):
					try:
						persisted_df.clear_divisions()
					except Exception:
						pass
			raise

	@staticmethod
	def create_order_timing_features(
			df: "pd.DataFrame | dd.DataFrame",
			patient_id_col: str,
			order_col: str,
			order_time_col: str,
			admission_time_col: Optional[str] = None,
			discharge_time_col: Optional[str] = None
			) -> "pd.DataFrame | dd.DataFrame":

		"""
		Create timing features from order data for each patient.

		This function calculates various timing-related features based on order events,
		such as total orders, time span of orders, and counts of orders within
		specific time windows relative to admission and discharge.

		It supports both pandas and Dask DataFrames.
		"""	
		
		persisted_intermediates = {}  # Track intermediate results
		
		try:
			# Select needed columns and optimize for Dask processing
			needed_cols = [patient_id_col, order_col, order_time_col]
			if admission_time_col:
				needed_cols.append(admission_time_col)
			if discharge_time_col:
				needed_cols.append(discharge_time_col)
			
			# Persist intermediate DataFrame if using Dask
			if hasattr(df, 'compute') and isinstance(df, dd.DataFrame):
				df_subset    = df[needed_cols]
				df_persisted = df_subset.persist()
				persisted_intermediates['input_df'] = df_persisted
				
				# Convert Dask DataFrame (only the needed portion) to pandas
				df = df_persisted.compute()
			else:
				# For pandas DataFrames, just select columns
				df = df[needed_cols]

			# Ensure datetime columns are properly typed
			df[order_time_col] = pd.to_datetime(df[order_time_col])
			if admission_time_col and admission_time_col in df.columns:
				df[admission_time_col] = pd.to_datetime(df[admission_time_col])
			if discharge_time_col and discharge_time_col in df.columns:
				df[discharge_time_col] = pd.to_datetime(df[discharge_time_col])

			# Calculate unique order counts separately
			unique_orders = df.groupby(patient_id_col)[order_col].nunique().to_frame(name='unique_order_types')

			# Define aggregations
			aggs = {
				order_time_col: ['min', 'max'],
				order_col: ['count']
			}

			# Add admission-related aggregations if admission time is available
			if admission_time_col and admission_time_col in df.columns:
				df['orders_in_first_24h'] = (df[order_time_col] <= df[admission_time_col] + pd.Timedelta(hours=24)).astype(int)
				df['orders_in_first_48h'] = (df[order_time_col] <= df[admission_time_col] + pd.Timedelta(hours=48)).astype(int)
				df['orders_in_first_72h'] = (df[order_time_col] <= df[admission_time_col] + pd.Timedelta(hours=72)).astype(int)
				aggs[admission_time_col] = ['first']
				aggs['orders_in_first_24h'] = ['sum']
				aggs['orders_in_first_48h'] = ['sum']
				aggs['orders_in_first_72h'] = ['sum']

			# Add discharge-related aggregations if discharge time is available
			if discharge_time_col and discharge_time_col in df.columns:
				df['orders_in_last_24h'] = (df[order_time_col] >= df[discharge_time_col] - pd.Timedelta(hours=24)).astype(int)
				df['orders_in_last_48h'] = (df[order_time_col] >= df[discharge_time_col] - pd.Timedelta(hours=48)).astype(int)
				aggs[discharge_time_col] = ['first']
				aggs['orders_in_last_24h'] = ['sum']
				aggs['orders_in_last_48h'] = ['sum']

			# Group by patient and aggregate
			timing_features_agg = df.groupby(patient_id_col).agg(aggs)

			# Flatten multi-index columns
			timing_features_agg.columns = ['_'.join(col).strip('_') for col in timing_features_agg.columns.values]

			# Merge the aggregated features with unique order counts
			timing_features = pd.merge(timing_features_agg, unique_orders, left_index=True, right_index=True)
			
			# Rename columns to match expected output
			rename_map = {
				f"{order_time_col}_min": "first_order_time",
				f"{order_time_col}_max": "last_order_time",
				f"{order_col}_count": "total_orders",
			}
			if admission_time_col and admission_time_col in df.columns:
				rename_map[f"{admission_time_col}_first"] = admission_time_col
				rename_map[f"orders_in_first_24h_sum"] = "orders_in_first_24h"
				rename_map[f"orders_in_first_48h_sum"] = "orders_in_first_48h"
				rename_map[f"orders_in_first_72h_sum"] = "orders_in_first_72h"

			if discharge_time_col and discharge_time_col in df.columns:
				rename_map[f"{discharge_time_col}_first"] = discharge_time_col
				rename_map[f"orders_in_last_24h_sum"] = "orders_in_last_24h"
				rename_map[f"orders_in_last_48h_sum"] = "orders_in_last_48h"
			
			timing_features = timing_features.rename(columns=rename_map)

			# --- Post-aggregation calculations ---
			timing_features['order_span_hours'] = (timing_features['last_order_time'] - timing_features['first_order_time']).dt.total_seconds() / 3600

			if admission_time_col and admission_time_col in df.columns:
				timing_features['time_to_first_order_hours'] = (timing_features['first_order_time'] - timing_features[admission_time_col]).dt.total_seconds() / 3600
				timing_features = timing_features.drop(columns=[admission_time_col])

			if discharge_time_col and discharge_time_col in df.columns:
				timing_features['time_from_last_order_to_discharge_hours'] = (timing_features[discharge_time_col] - timing_features['last_order_time']).dt.total_seconds() / 3600
				timing_features = timing_features.drop(columns=[discharge_time_col])

			return timing_features.reset_index()
			
		except Exception as e:
			# Cleanup persisted resources on error
			for name, persisted_df in persisted_intermediates.items():
				if isinstance(persisted_df, dd.DataFrame):
					try:
						persisted_df.clear_divisions()
					except Exception:
						pass
			raise

	@staticmethod
	def get_order_type_distributions(df: pd.DataFrame, patient_id_col: str='subject_id', order_col: str='order_type') -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Calculates the distribution of order types.

		Args:
			df: DataFrame with patient data.
			patient_id_col: Column name for patient ID.
			order_col: Column name for the order type.

		Returns:
			A tuple containing:
				- Overall distribution of order types.
				- Distribution of order types per patient.
		"""
		# Calculate overall distribution
		if isinstance(df, dd.DataFrame):
			overall_dist = df[order_col].value_counts(normalize=True).compute()
		else:
			overall_dist = df[order_col].value_counts(normalize=True)

		# Calculate patient-level distribution
		if isinstance(df, dd.DataFrame):
      
			# Dask-compatible way to get patient-level distributions
			def calculate_patient_dist_df(patient_data):
				"""Calculates value counts and returns a DataFrame."""
				return patient_data[order_col].value_counts(normalize=True).to_frame().reset_index()

			# Define the output metadata for the apply function
			meta = pd.DataFrame({order_col: pd.Series([], dtype='object'), 'proportion': pd.Series([], dtype='float64')})

			# Apply the function with the specified metadata
			patient_dist_dd = df.groupby(patient_id_col).apply(calculate_patient_dist_df, meta=meta)

			# Compute the result
			patient_dist = patient_dist_dd.compute()

			patient_dist = patient_dist.reset_index(level=1, drop=True).pivot(columns='order_type', values='proportion').fillna(0)

		else:
			# Pandas-native way
			patient_dist = df.groupby(patient_id_col)[order_col].value_counts(normalize=True).unstack(fill_value=0)

		return overall_dist, patient_dist

	@staticmethod
	def calculate_order_transition_matrix(sequences: Dict[Any, List[str]], top_n: int = 20) -> pd.DataFrame:
		"""
		Calculate transition probabilities between different order types.

		Args:
			sequences: Dictionary of order sequences by patient
			top_n: Limit to most common n order types

		Returns:
			DataFrame with transition probabilities
		"""
		# Collect all order types and their counts
		all_orders = []
		for sequence in sequences.values():
			all_orders.extend(sequence)

		# Get most common order types if needed
		order_counts = pd.Series(all_orders).value_counts()
		if top_n > 0 and len(order_counts) > top_n:
			common_orders = order_counts.head(top_n).index.tolist()
		else:
			common_orders = order_counts.index.tolist()

		# Initialize transition count matrix
		transition_counts = pd.DataFrame(0, index=common_orders, columns=common_orders)

		# Count transitions
		for sequence in sequences.values():
			# Filter to common orders
			filtered_sequence = [order for order in sequence if order in common_orders]

			# Count transitions
			for i in range(len(filtered_sequence) - 1):
				from_order = filtered_sequence[i]
				to_order = filtered_sequence[i + 1]
				transition_counts.loc[from_order, to_order] += 1

		# Convert to probabilities
		row_sums = transition_counts.sum(axis=1)
		transition_probs = transition_counts.div(row_sums, axis=0).fillna(0)

		return transition_probs

	@staticmethod
	def save_features(features: Any, feature_type: str, base_path: str, format: str = 'csv') -> str:
		"""
		Save engineered features to file.

		Args:
			features: DataFrame or other data structure to save
			feature_type: String identifier for the feature type
			base_path: Directory to save in
			format: File format ('csv', 'parquet', or 'json')

		Returns:
			Path to saved file
		"""
		# Create directory if it doesn't exist
		features_dir = os.path.join(base_path, 'engineered_features')
		os.makedirs(features_dir, exist_ok=True)

		# Create timestamp for filename
		timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

		# Base filename
		filename = f"{feature_type}_{timestamp}"

		# Save based on format
		if format == 'csv':
			# For DataFrames
			if isinstance(features, pd.DataFrame):
				filepath = os.path.join(features_dir, f"{filename}.csv")
				features.to_csv(filepath, index=True)
			else:
				raise ValueError(f"Cannot save {type(features)} as CSV")

		elif format == 'parquet':
			# For DataFrames
			if isinstance(features, pd.DataFrame):
				filepath = os.path.join(features_dir, f"{filename}.parquet")
				features.to_parquet(filepath, index=True)
			else:
				raise ValueError(f"Cannot save {type(features)} as Parquet")

		elif format == 'json':
			# For dictionaries or DataFrames
			filepath = os.path.join(features_dir, f"{filename}.json")

			if isinstance(features, pd.DataFrame):
				# Convert DataFrame to JSON-compatible format
				json_data = features.to_json(orient='records')
				with open(filepath, 'w') as f:
					f.write(json_data)
			elif isinstance(features, dict):
				# Save dict directly
				with open(filepath, 'w') as f:
					json.dump(features, f)
			else:
				raise ValueError(f"Cannot save {type(features)} as JSON")
		else:
			raise ValueError(f"Unsupported format: {format}")

		return filepath

