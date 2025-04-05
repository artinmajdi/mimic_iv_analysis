"""Utility functions for MIMIC-IV Analysis."""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style(style='whitegrid', context='talk', palette='colorblind'):
    """Set consistent plot style for visualizations.

    Args:
        style (str): Seaborn style name
        context (str): Seaborn context name
        palette (str): Seaborn color palette name
    """
    sns.set_style(style)
    sns.set_context(context)
    sns.set_palette(palette)

def format_time_column(df, time_col, format=None):
    """Convert time column to datetime format.

    Args:
        df (pd.DataFrame): DataFrame containing the time column
        time_col (str): Name of the time column
        format (str, optional): Datetime format string

    Returns:
        pd.DataFrame: DataFrame with formatted time column
    """
    if format:
        df[time_col] = pd.to_datetime(df[time_col], format=format)
    else:
        df[time_col] = pd.to_datetime(df[time_col])
    return df

__all__ = [
    'set_plot_style',
    'format_time_column',
]
