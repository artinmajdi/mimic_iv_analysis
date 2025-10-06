# Standard library imports
import dask.dataframe as dd

# Streamlit import
import streamlit as st


def display_dataframe_head(df):
	MAX_DATAFRAME_ROWS_DISPLAYED = 30
	if isinstance(df, dd.DataFrame):
		n_rows_loaded = df.shape[0].compute()
	else:
		n_rows_loaded = df.shape[0]

	n_rows = min(MAX_DATAFRAME_ROWS_DISPLAYED, n_rows_loaded)
	st.dataframe(df.head(n_rows) , use_container_width=True)
 
