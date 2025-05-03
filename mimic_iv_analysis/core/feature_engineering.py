import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import dask.dataframe as dd
import streamlit as st

class FeatureEngineer:
	"""Handles feature engineering for MIMIC-IV data."""

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
				if df[col].nunique() > len(df) * 0.1:  # High cardinality
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

		# Convert Dask DataFrame to pandas if necessary
		if use_dask and hasattr(df, 'compute'):
			with st.spinner('Computing data for order frequency matrix...'):
				if top_n > 0:
					# For Dask, compute value counts to find top orders
					value_counts = df[order_col].value_counts().compute()
					top_orders = value_counts.head(top_n).index.tolist()
					# Filter and compute
					filtered_df = df[df[order_col].isin(top_orders)].compute()
				else:
					# Compute the entire DataFrame
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

		# Convert Dask DataFrame to pandas if necessary
		if use_dask and hasattr(df, 'compute'):
			with st.spinner('Computing data for temporal order sequences...'):
				# For sequences, we need the complete DataFrame
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

	@staticmethod
	def create_order_timing_features(df: Union[pd.DataFrame, dd.DataFrame], patient_id_col: str, order_col: str, order_time_col: str, admission_time_col: str = None, discharge_time_col: str = None, use_dask: bool = False) -> pd.DataFrame:
		"""
		Creates features related to order timing.

		Args:
			df: DataFrame with order data
			patient_id_col: Column containing patient IDs
			order_col: Column containing order types
			order_time_col: Column containing order timestamps
			admission_time_col: Column containing admission timestamps (optional)
			discharge_time_col: Column containing discharge timestamps (optional)

		Returns:
			DataFrame with timing features
		"""
		# Validate columns exist
		required_cols = [patient_id_col, order_col, order_time_col]
		if not all(col in df.columns for col in required_cols):
			missing = [col for col in required_cols if col not in df.columns]
			raise ValueError(f"Columns {missing} not found in DataFrame")

		# Convert Dask DataFrame to pandas if necessary
		if use_dask and hasattr(df, 'compute'):
			with st.spinner('Computing data for order timing features...'):
				# For timing features, we need the complete DataFrame
				df = df.compute()
		else:
			# Make a copy of the DataFrame
			df = df.copy()

		# Ensure time columns are datetime
		time_cols = [order_time_col]
		if admission_time_col:
			time_cols.append(admission_time_col)
		if discharge_time_col:
			time_cols.append(discharge_time_col)

		for col in time_cols:
			if col in df.columns and df[col].dtype != 'datetime64[ns]':
				try:
					df[col] = pd.to_datetime(df[col])
				except:
					raise ValueError(f"Could not convert {col} to datetime format")

		# Initialize results DataFrame
		timing_features = pd.DataFrame()

		# Process by patient
		grouped = df.groupby(patient_id_col)

		# Create base features list
		features = {
			'patient_id'        : [],
			'total_orders'      : [],
			'unique_order_types': [],
			'first_order_time'  : [],
			'last_order_time'   : [],
			'order_span_hours'  : []
		}

		# Add admission-relative features if admission time is available
		if admission_time_col and admission_time_col in df.columns:
			features.update({
				'time_to_first_order_hours': [],
				'orders_in_first_24h'      : [],
				'orders_in_first_48h'      : [],
				'orders_in_first_72h'      : []
			})

		# Add discharge-relative features if discharge time is available
		if discharge_time_col and discharge_time_col in df.columns:
			features.update({
				'time_from_last_order_to_discharge_hours': [],
				'orders_in_last_24h': [],
				'orders_in_last_48h': []
			})

		# Calculate features for each patient
		for patient_id, patient_data in grouped:
			features['patient_id'].append(patient_id)
			features['total_orders'].append(len(patient_data))
			features['unique_order_types'].append(patient_data[order_col].nunique())

			# Sort orders by time
			patient_data = patient_data.sort_values(order_time_col)

			# Get first and last order times
			first_order_time = patient_data[order_time_col].min()
			last_order_time = patient_data[order_time_col].max()

			features['first_order_time'].append(first_order_time)
			features['last_order_time'].append(last_order_time)

			# Calculate order span in hours
			order_span_hours = (last_order_time - first_order_time).total_seconds() / 3600
			features['order_span_hours'].append(order_span_hours)

			# Calculate admission-related features
			if admission_time_col and admission_time_col in df.columns:
				# Get admission time for this patient (should be the same for all rows)
				admission_time = patient_data[admission_time_col].iloc[0]

				# Time from admission to first order
				time_to_first_order = (first_order_time - admission_time).total_seconds() / 3600
				features['time_to_first_order_hours'].append(time_to_first_order)

				# Count orders in first 24/48/72 hours
				orders_24h = patient_data[
					patient_data[order_time_col] <= admission_time + pd.Timedelta(hours=24)
				].shape[0]
				features['orders_in_first_24h'].append(orders_24h)

				orders_48h = patient_data[
					patient_data[order_time_col] <= admission_time + pd.Timedelta(hours=48)
				].shape[0]
				features['orders_in_first_48h'].append(orders_48h)

				orders_72h = patient_data[
					patient_data[order_time_col] <= admission_time + pd.Timedelta(hours=72)
				].shape[0]
				features['orders_in_first_72h'].append(orders_72h)

			# Calculate discharge-related features
			if discharge_time_col and discharge_time_col in df.columns:
				# Get discharge time for this patient
				discharge_time = patient_data[discharge_time_col].iloc[0]

				# Time from last order to discharge
				time_to_discharge = (discharge_time - last_order_time).total_seconds() / 3600
				features['time_from_last_order_to_discharge_hours'].append(time_to_discharge)

				# Count orders in last 24/48 hours
				orders_last_24h = patient_data[
					patient_data[order_time_col] >= discharge_time - pd.Timedelta(hours=24)
				].shape[0]
				features['orders_in_last_24h'].append(orders_last_24h)

				orders_last_48h = patient_data[
					patient_data[order_time_col] >= discharge_time - pd.Timedelta(hours=48)
				].shape[0]
				features['orders_in_last_48h'].append(orders_last_48h)

		# Create DataFrame from features
		timing_features = pd.DataFrame(features)

		return timing_features

	@staticmethod
	def get_order_type_distributions(df: pd.DataFrame, patient_id_col: str, order_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Calculate order type distributions overall and by patient.

		Args:
			df: DataFrame with order data
			patient_id_col: Column containing patient IDs
			order_col: Column containing order types

		Returns:
			Tuple of (overall distribution, patient-level distribution)
		"""
		# Validate columns exist
		if not all(col in df.columns for col in [patient_id_col, order_col]):
			missing = [col for col in [patient_id_col, order_col] if col not in df.columns]
			raise ValueError(f"Columns {missing} not found in DataFrame")

		# Calculate overall distribution
		overall_dist = df[order_col].value_counts(normalize=True).reset_index()
		overall_dist.columns = [order_col, 'frequency']

		# Calculate patient-level distribution
		patient_dists = []

		for patient_id, patient_data in df.groupby(patient_id_col):
			# Get this patient's distribution
			patient_dist = patient_data[order_col].value_counts(normalize=True)

			# Convert to DataFrame and add patient ID
			patient_dist_df = patient_dist.reset_index()
			patient_dist_df.columns = [order_col, 'frequency']
			patient_dist_df['patient_id'] = patient_id

			patient_dists.append(patient_dist_df)

		# Combine all patient distributions
		if patient_dists:
			patient_level_dist = pd.concat(patient_dists, ignore_index=True)
		else:
			patient_level_dist = pd.DataFrame(columns=[order_col, 'frequency', 'patient_id'])

		return overall_dist, patient_level_dist

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

