# Standard library imports
from typing import List

# Data processing imports
import pandas as pd
import dask.dataframe as dd

# Visualization imports
import plotly.express as px

# Streamlit import
import streamlit as st
from mimic_iv_analysis.configurations.params import TableNames

class ExplorationAndViz:
	""" Handles the UI and logic for the Exploration & Visualization tab. """

	def __init__(self):
		pass

	def render(self):
		""" Renders the content of the Exploration & Visualization tab. """

		def _poe_preview_exception_handling():
			""" Handles preview exception for POE table. """

			n_rows = st.session_state.df.shape[0].compute() if isinstance(st.session_state.df, dd.DataFrame) else len(st.session_state.df)

			if (st.session_state.selected_table == TableNames.POE.value) and (not st.session_state.load_full):
				if n_rows > 5000:
					st.warning("For POE table preview, please either load the full table or reduce subject selection to have fewer than 5,000 rows. Current preview is limited due to implementation considerations.")
					return False
				return True
			return False


		st.markdown("<h2 class='sub-header'>Exploration & Visualization</h2>", unsafe_allow_html=True)

		# Introductory text
		st.info("This section enables exploration and visualization of the loaded dataset.")
		

		st.markdown("<h2 class='sub-header'>Data Exploration & Visualization</h2>", unsafe_allow_html=True)

		ExplorationAndViz.display_dataset_statistics(st.session_state.df)

		cols = st.columns([1, 2])
		with cols[0]:
			st.session_state.n_rows_for_visualization = st.number_input("Number of Rows for Visualization", min_value=1, max_value=st.session_state.df_length, value=min(30, st.session_state.df_length))

		# Compute a sample from the DataFrame
		preview_df = ExplorationAndViz.compute_dataframe_sample(df=st.session_state.df, exception_flag=_poe_preview_exception_handling() )

		if preview_df is None:
			return

		ExplorationAndViz.display_data_preview(preview_df)

		ExplorationAndViz.display_visualizations(preview_df)

	@staticmethod
	def display_dataset_statistics(df: pd.DataFrame | dd.DataFrame):
		"""Displays key statistics about the loaded DataFrame using Dask-optimized methods.

		Args:
			df: DataFrame to display statistics for (can be pandas DataFrame or Dask DataFrame)
			use_dask: If True, df is treated as a Dask DataFrame and uses lazy evaluation
		"""
		st.markdown("<h2 class='sub-header'>Dataset Statistics</h2>", unsafe_allow_html=True)

		if isinstance(df, dd.DataFrame):
			# Use Dask-optimized statistics with lazy evaluation
			ExplorationAndViz.display_dask_statistics(df)
		else:
			# Handle pandas DataFrame with standard methods
			ExplorationAndViz.display_pandas_statistics(df)

	@staticmethod
	def compute_dataframe_sample(df: pd.DataFrame | dd.DataFrame, exception_flag: bool = False) -> pd.DataFrame:
		"""Helper method to compute a sample from DataFrame (Dask or pandas). """

		if isinstance(df, dd.DataFrame):

			if exception_flag:
				df2 = df.compute()
				return df2.head(st.session_state.n_rows_for_visualization)

			# Get sample using the most reliable Dask method
			n_rows = st.session_state.n_rows_for_visualization
			methods_to_try = [
				lambda: df.head(n_rows, compute=True),
				lambda: df.head(n_rows).compute(),
				lambda: df.iloc[:n_rows].compute()
			]
			
			for method in methods_to_try:
				try:
					result = method()
					if not result.empty:
						return result
				except Exception:
					continue
			
			st.error("Unable to compute DataFrame sample with any available method")
			return pd.DataFrame()

		else:
			return df.head(st.session_state.n_rows_for_visualization)

	@staticmethod
	def display_data_preview(df: pd.DataFrame):
		"""Displays a preview of the loaded DataFrame."""
		
		st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)

		with st.spinner('Computing preview from Dask DataFrame...'):
			try:
				st.dataframe(df, use_container_width=True)
			except Exception as e:
				st.error(f"Error computing DataFrame preview: {str(e)}")
				return

	@staticmethod
	def display_visualizations(viz_df: pd.DataFrame):
		"""Displays visualizations of the loaded DataFrame.
		"""

		st.markdown("<h2 class='sub-header'>Data Visualization</h2>", unsafe_allow_html=True)

		with st.spinner('Computing data for visualization from DataFrame...'):

			# Select columns for visualization
			numeric_cols    : List[str] = viz_df.select_dtypes(include=['number']).columns.tolist()
			categorical_cols: List[str] = viz_df.select_dtypes(include=['object', 'category']).columns.tolist()

			if len(numeric_cols) > 0:
				st.markdown("<h3>Numeric Data Visualization</h3>", unsafe_allow_html=True)

				# Histogram
				selected_num_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
				if selected_num_col:
					fig = px.histogram(viz_df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
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
						fig = px.scatter(viz_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
						st.plotly_chart(fig, use_container_width=True)

			if len(categorical_cols) > 0:
				st.markdown("<h3>Categorical Data Visualization</h3>", unsafe_allow_html=True)

				# Bar chart
				selected_cat_col = st.selectbox("Select a categorical column for bar chart", categorical_cols)
				if selected_cat_col:
					value_counts = viz_df[selected_cat_col].value_counts().reset_index()
					value_counts.columns = [selected_cat_col, 'Count']

					# Limit to top 20 categories if there are too many
					if len(value_counts) > 20:
						value_counts = value_counts.head(20)
						title = f"Top 20 values in {selected_cat_col}"
					else:
						title = f"Distribution of {selected_cat_col}"

					fig = px.bar(value_counts, x=selected_cat_col, y='Count', title=title)
					st.plotly_chart(fig, use_container_width=True)

	@staticmethod
	def display_dask_statistics(df: dd.DataFrame):
		"""Display statistics for Dask DataFrame using lazy evaluation and efficient methods."""
		try:
			# Persist DataFrame for multiple operations to avoid recomputation
			df_persisted = df.persist()
			
			with st.spinner('Computing efficient statistics from Dask DataFrame...'):
				# Basic info using Dask's efficient methods (no full computation)
				num_rows = len(df_persisted)  # Dask can compute this efficiently
				num_cols = len(df_persisted.columns)
				
				# Use Dask's lazy computation for missing values
				missing_values_delayed = df_persisted.isnull().sum().sum()
				
				# Compute only what we need
				missing_total = missing_values_delayed.compute()
				
				# Display basic statistics
				col1, col2 = st.columns(2)
				with col1:
					st.markdown("<div class='info-box'>", unsafe_allow_html=True)
					st.markdown(f"**Number of rows:** {num_rows:,}")
					st.markdown(f"**Number of columns:** {num_cols}")
					st.markdown("</div>", unsafe_allow_html=True)

				with col2:
					st.markdown("<div class='info-box'>", unsafe_allow_html=True)
					# Estimate memory usage from meta without full computation
					estimated_memory = ExplorationAndViz.estimate_dask_memory(df_persisted)
					st.markdown(f"**Estimated memory:** {estimated_memory:.2f} MB")
					st.markdown(f"**Missing values:** {missing_total:,}")
					st.markdown("</div>", unsafe_allow_html=True)

				# Display column information using Dask meta
				ExplorationAndViz.display_dask_column_info(df_persisted)
				
		except Exception as e:
			st.error(f"Error computing Dask statistics: {str(e)}")
			st.info("Falling back to sample-based statistics...")
			# Fallback to sample-based statistics
			sample_df = df.head(1000, compute=True)
			ExplorationAndViz.display_pandas_statistics(sample_df, is_sample=True)

	@staticmethod
	def display_pandas_statistics(df: pd.DataFrame, is_sample: bool = False):
		"""Display statistics for pandas DataFrame."""
		try:
			sample_text = " (Sample)" if is_sample else ""
			
			col1, col2 = st.columns(2)
			with col1:
				st.markdown("<div class='info-box'>", unsafe_allow_html=True)
				st.markdown(f"**Number of rows{sample_text}:** {len(df):,}")
				st.markdown(f"**Number of columns:** {len(df.columns)}")
				st.markdown("</div>", unsafe_allow_html=True)

			with col2:
				st.markdown("<div class='info-box'>", unsafe_allow_html=True)
				memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
				st.markdown(f"**Memory usage{sample_text}:** {memory_mb:.2f} MB")
				st.markdown(f"**Missing values{sample_text}:** {df.isnull().sum().sum():,}")
				st.markdown("</div>", unsafe_allow_html=True)

			# Display column information
			ExplorationAndViz.display_pandas_column_info(df, is_sample)
			
		except Exception as e:
			st.error(f"Error generating pandas statistics: {str(e)}")

	@staticmethod
	def estimate_dask_memory(df: dd.DataFrame) -> float:
		"""Estimate memory usage for Dask DataFrame without full computation."""
		try:
			# Use meta information and sample to estimate
			if df.npartitions <= 0:
				return 0.0

			sample_df = df.get_partition(0).compute()				
			sample_memory = sample_df.memory_usage(deep=True).sum()
			
			# Estimate total memory based on sample
			total_rows      = len(df)
			estimated_total = (sample_memory / st.session_state.n_rows_for_visualization) * total_rows / (1024 * 1024)
			return estimated_total
		except:
			return 0.0

	@staticmethod
	def display_dask_column_info(df: dd.DataFrame):
		"""Display column information for Dask DataFrame using efficient methods."""
		st.markdown("<h3>Column Information</h3>", unsafe_allow_html=True)
		try:
			# Use Dask meta for dtypes (no computation needed)
			dtypes_dict = dict(df.dtypes)
			columns     = list(df.columns)
			
			# Compute missing values efficiently using delayed operations
			import dask
			missing_counts_delayed  = [df[col].isnull().sum() for col in columns]
			non_null_counts_delayed = [df[col].count() for col in columns]
			
			# Compute all at once for efficiency
			with st.spinner('Computing column statistics...'):
				missing_counts, non_null_counts = dask.compute(missing_counts_delayed, non_null_counts_delayed)
				
				total_rows = len(df)
				missing_percentages = [(missing / total_rows * 100) if total_rows > 0 else 0 
										for missing in missing_counts]
				
				# Create column info DataFrame
				col_info = pd.DataFrame({
					'Column'            : columns,
					'Type'              : [str(dtypes_dict[col]) for col in columns],
					'Non-Null Count'    : non_null_counts,
					'Missing Values (%)': [round(pct, 2) for pct in missing_percentages],
				})
				
				st.dataframe(col_info, use_container_width=True)
				st.info("Unique value counts skipped for performance with large Dask DataFrames.")
				
		except Exception as e:
			st.error(f"Error generating Dask column info: {str(e)}")

	@staticmethod
	def display_pandas_column_info(df: pd.DataFrame, is_sample: bool = False):
		"""Display column information for pandas DataFrame."""
		st.markdown("<h3>Column Information</h3>", unsafe_allow_html=True)
		try:
			sample_text = " (Sample)" if is_sample else ""
			
			# Ensure dtype objects are converted to strings
			dtype_strings = pd.Series(df.dtypes, index=df.columns).astype(str).values
			col_info = pd.DataFrame({
				'Column': df.columns,
				'Type': dtype_strings,
				'Non-Null Count': df.count().values,
				'Missing Values (%)': (df.isnull().sum() / len(df) * 100).values.round(2),
			})
			
			if is_sample:
				col_info.columns = [col + sample_text if 'Count' in col or 'Values' in col else col 
									for col in col_info.columns]
				
			st.dataframe(col_info, use_container_width=True)
			
		except Exception as e:
			st.error(f"Error generating pandas column info: {str(e)}")


