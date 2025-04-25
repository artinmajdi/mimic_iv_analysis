# Standard library imports
from typing import List, Optional

# Data processing imports
import pandas as pd

# Visualization imports
import plotly.express as px


# Streamlit import
import streamlit as st


class MIMICVisualizer:
	"""Handles the display of dataset statistics and data preview."""

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


	def display_data_preview(self, df: Optional[pd.DataFrame]):
		"""Displays a preview of the loaded DataFrame."""
		if df is not None:
			st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
			st.dataframe(df, use_container_width=True)


	def display_visualizations(self, df: Optional[pd.DataFrame]):
		"""Displays visualizations of the loaded DataFrame."""
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

