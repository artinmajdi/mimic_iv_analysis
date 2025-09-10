# Standard library imports
from typing import List, Optional, Union

# Data processing imports
import pandas as pd
import dask.dataframe as dd

# Visualization imports
import plotly.express as px

# Streamlit import
import streamlit as st


def display_dataframe_head(df):
	MAX_DATAFRAME_ROWS_DISPLAYED = 30
	if isinstance(df, dd.DataFrame):
		df_length = df.shape[0].compute()
	else:
		df_length = df.shape[0]

	n_rows = min(MAX_DATAFRAME_ROWS_DISPLAYED, df_length)
	st.dataframe(df.head(n_rows) , use_container_width=True)
 

class MIMICVisualizerUtils:
	"""Handles the display of dataset statistics and data preview."""

	@staticmethod
	def display_dataset_statistics(df, use_dask: bool = False):
		"""Displays key statistics about the loaded DataFrame.

		Args:
			df: DataFrame to display statistics for (can be pandas DataFrame or Dask DataFrame)
			use_dask: If True, df is treated as a Dask DataFrame and computed when needed
		"""
		if df is not None:
			st.markdown("<h2 class='sub-header'>Dataset Statistics</h2>", unsafe_allow_html=True)

			# Convert to pandas DataFrame if it's a Dask DataFrame
			if use_dask and hasattr(df, 'compute'):
				# For statistics, we need to compute the full DataFrame
				# Show computing message for better UX
				with st.spinner('Computing statistics from Dask DataFrame...'):
					df_stats = df.compute()
			else:
				df_stats = df

			col1, col2 = st.columns(2)
			with col1:
				st.markdown("<div class='info-box'>", unsafe_allow_html=True)
				st.markdown(f"**Number of rows:** {len(df_stats)}")
				st.markdown(f"**Number of columns:** {len(df_stats.columns)}")
				st.markdown("</div>", unsafe_allow_html=True)

			with col2:
				st.markdown("<div class='info-box'>", unsafe_allow_html=True)
				st.markdown(f"**Memory usage:** {df_stats.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
				st.markdown(f"**Missing values:** {df_stats.isna().sum().sum()}")
				st.markdown("</div>", unsafe_allow_html=True)

			# Display column information
			st.markdown("<h3>Column Information</h3>", unsafe_allow_html=True)
			try:
				# Ensure dtype objects are converted to strings to prevent Arrow conversion issues
				dtype_strings = pd.Series(df_stats.dtypes, index=df_stats.columns).astype(str).values
				col_info = pd.DataFrame({
					'Column': df_stats.columns,
					'Type': dtype_strings,
					'Non-Null Count': df_stats.count().values,
					'Missing Values (%)': (df_stats.isna().sum() / len(df_stats) * 100).values.round(2),
					'Unique Values': [df_stats[col].nunique() for col in df_stats.columns]
				})
				st.dataframe(col_info, use_container_width=True)
			except Exception as e:
				st.error(f"Error generating column info: {e}")
		else:
			st.info("No data loaded to display statistics.")


	@staticmethod
	def display_data_preview(df: pd.DataFrame | dd.DataFrame, use_dask: bool = False):
		"""Displays a preview of the loaded DataFrame."""

		st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)

		if use_dask and isinstance(df, dd.DataFrame):

			with st.spinner('Computing preview from Dask DataFrame...'):
				preview_df = df.head(20, compute=True) if isinstance(df, dd.DataFrame) else df
				st.dataframe(preview_df, use_container_width=True)

		else:
			st.dataframe(df, use_container_width=True)


	@staticmethod
	def display_visualizations(df: pd.DataFrame | dd.DataFrame, use_dask: bool = False):
		"""Displays visualizations of the loaded DataFrame.

		Args:
			df: DataFrame to visualize (can be pandas DataFrame or Dask DataFrame)
			use_dask: If True, df is treated as a Dask DataFrame and computed when needed
		"""
		# TODO: add the option in the UI for user to select the number of subjects to visualize instead of top 20 rows.
		if df is not None:
			st.markdown("<h2 class='sub-header'>Data Visualization</h2>", unsafe_allow_html=True)

			# Convert to pandas DataFrame if it's a Dask DataFrame
			if use_dask and hasattr(df, 'compute'):
				with st.spinner('Computing data for visualization from Dask DataFrame...'):
					# For visualizations, we need the full DataFrame or at least a substantial sample
					# Compute with a reasonable sample size for better performance
					# Check if it's a Dask DataFrame by checking for compute method
					if hasattr(df, 'compute'):
						# This is a Dask DataFrame - handle it appropriately
						try:
							# Try with compute parameter (newer Dask versions)
							viz_df = df.head(20, compute=True)
						except TypeError:
							# For older Dask versions
							viz_df = df.head(20).compute()
					else:
						# Regular pandas DataFrame
						viz_df = df.head(20)
					st.info('Visualizations are based on a sample of the data for better performance.')
			else:
				viz_df = df

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


