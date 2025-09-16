# Standard library imports
import dask.dataframe as dd

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
 
