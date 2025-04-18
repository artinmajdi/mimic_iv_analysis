import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional, Tuple, Any
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Add the parent directory to the path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Constants
DEFAULT_MIMIC_PATH      = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Handling Class ---
class MIMICDataHandler:
    """Handles scanning, loading, and providing info for MIMIC-IV data."""
    def __init__(self):
        pass

    def _format_size(self, size_mb: float) -> str:
        """Format file size into KB, MB, or GB string."""
        if size_mb < 0.001:
             return "(< 1 KB)"
        elif size_mb < 1:
            return f"({size_mb * 1024:.1f} KB)"
        elif size_mb < 1000:
            return f"({size_mb:.1f} MB)"
        else:
            return f"({size_mb / 1000:.1f} GB)"

    def scan_mimic_directory(self, mimic_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
        """Scans the MIMIC-IV directory structure and returns table info."""
        available_tables = {}
        file_paths = {}
        file_sizes = {}
        table_display_names = {}

        if not os.path.exists(mimic_path):
            return available_tables, file_paths, file_sizes, table_display_names

        modules = ['hosp', 'icu']
        for module in modules:
            module_path = os.path.join(mimic_path, module)
            if os.path.exists(module_path):
                available_tables[module] = []
                current_module_tables = {}

                csv_files = glob.glob(os.path.join(module_path, '*.csv'))
                csv_gz_files = glob.glob(os.path.join(module_path, '*.csv.gz'))

                all_files = csv_files + csv_gz_files

                for file_path in all_files:
                    is_gz = file_path.endswith('.csv.gz')
                    table_name = os.path.basename(file_path).replace('.csv.gz' if is_gz else '.csv', '')

                    if table_name in current_module_tables and not is_gz:
                        current_module_tables[table_name] = file_path
                    elif table_name not in current_module_tables:
                        current_module_tables[table_name] = file_path

                for table_name, file_path in current_module_tables.items():
                    available_tables[module].append(table_name)
                    file_paths[(module, table_name)] = file_path
                    try:
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    except OSError:
                        file_size_mb = 0
                    file_sizes[(module, table_name)] = file_size_mb
                    size_str = self._format_size(file_size_mb)
                    table_display_names[(module, table_name)] = f"{table_name} {size_str}"

                available_tables[module].sort()

        return available_tables, file_paths, file_sizes, table_display_names

    def load_mimic_table(self, file_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1') -> Optional[pd.DataFrame]:
        """Loads a specific MIMIC-IV table, handling large files and sampling."""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            is_compressed = file_path.endswith('.gz')
            compression = 'gzip' if is_compressed else None

            read_params = {
                'encoding': encoding,
                'compression': compression,
                'low_memory': False
            }

            if file_size_mb > LARGE_FILE_THRESHOLD_MB:
                df = pd.read_csv(file_path, nrows=sample_size, **read_params)
            else:
                df = pd.read_csv(file_path, **read_params)
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=RANDOM_STATE)
            return df
        except Exception as e:
            logging.error(f"Error loading data from {os.path.basename(file_path)}: {str(e)}")
            return None

    def get_table_info(self, module: str, table_name: str) -> str:
        """Get descriptive information about a specific MIMIC-IV table."""
        table_info = {
            ('hosp', 'admissions'): "Patient hospital admissions information",
            ('hosp', 'patients'): "Patient demographic data",
            ('hosp', 'labevents'): "Laboratory measurements (large file)",
            ('hosp', 'microbiologyevents'): "Microbiology test results",
            ('hosp', 'pharmacy'): "Pharmacy orders",
            ('hosp', 'prescriptions'): "Medication prescriptions",
            ('hosp', 'procedures_icd'): "Patient procedures",
            ('hosp', 'diagnoses_icd'): "Patient diagnoses",
            ('hosp', 'emar'): "Electronic medication administration records",
            ('hosp', 'emar_detail'): "Detailed medication administration data",
            ('hosp', 'poe'): "Provider order entries",
            ('hosp', 'poe_detail'): "Detailed order information",
            ('hosp', 'd_hcpcs'): "HCPCS code definitions",
            ('hosp', 'd_icd_diagnoses'): "ICD diagnosis code definitions",
            ('hosp', 'd_icd_procedures'): "ICD procedure code definitions",
            ('hosp', 'd_labitems'): "Laboratory test definitions",
            ('hosp', 'hcpcsevents'): "HCPCS events",
            ('hosp', 'drgcodes'): "Diagnosis-related group codes",
            ('hosp', 'services'): "Hospital services",
            ('hosp', 'transfers'): "Patient transfers",
            ('hosp', 'provider'): "Provider information",
            ('hosp', 'omr'): "Order monitoring results",
            ('icu', 'chartevents'): "Patient charting data (vital signs, etc.)",
            ('icu', 'datetimeevents'): "Date/time-based events",
            ('icu', 'inputevents'): "Patient intake data",
            ('icu', 'outputevents'): "Patient output data",
            ('icu', 'procedureevents'): "ICU procedures",
            ('icu', 'ingredientevents'): "Detailed medication ingredients",
            ('icu', 'd_items'): "Dictionary of ICU items",
            ('icu', 'icustays'): "ICU stay information",
            ('icu', 'caregiver'): "Caregiver information"
        }
        return table_info.get((module, table_name), "No description available")

    def convert_to_parquet(self, df: pd.DataFrame, table_name: str, current_file_path: str) -> Optional[str]:
        """Converts the loaded DataFrame to Parquet format. Returns path or None."""
        try:
            parquet_dir = os.path.join(os.path.dirname(current_file_path), 'parquet_files')
            os.makedirs(parquet_dir, exist_ok=True)
            parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_file)
            return parquet_file
        except Exception as e:
            logging.error(f"Error converting {table_name} to Parquet: {str(e)}")
            return None

# --- Visualization Class ---
class MIMICVisualizer:
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

    # Function to display data preview
    def display_data_preview(self, df: Optional[pd.DataFrame]):
        if df is not None:
            st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

    # Function to display visualizations
    def display_visualizations(self, df: Optional[pd.DataFrame]):
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

# --- NEW Feature Engineering Class ---
class MIMICFeatureEngineer:
    """Handles feature engineering for MIMIC-IV data."""

    def __init__(self):
        pass

    def detect_order_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns likely to contain order information."""
        order_columns = []

        # Check column names that might represent orders
        order_related_terms = [
            'order', 'medication', 'drug', 'procedure', 'treatment',
            'item', 'event', 'action', 'prescription', 'poe'
        ]

        for col in df.columns:
            col_lower = col.lower()
            # Check if any order-related term is in column name
            if any(term in col_lower for term in order_related_terms):
                order_columns.append(col)
            # Or if column has common order-related suffixes/prefixes
            elif col_lower.endswith('_id') or col_lower.endswith('_type') or \
                 col_lower.endswith('_name') or col_lower.startswith('order_'):
                order_columns.append(col)

        return order_columns

    def detect_temporal_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns containing temporal information."""
        time_columns = []

        # Check column names
        time_related_terms = [
            'time', 'date', 'datetime', 'timestamp', 'start', 'end',
            'created', 'updated', 'admission', 'discharge'
        ]

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

    def detect_patient_id_column(self, df: pd.DataFrame) -> Optional[str]:
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

    def create_order_frequency_matrix(self,
                                     df: pd.DataFrame,
                                     patient_id_col: str,
                                     order_col: str,
                                     normalize: bool = False,
                                     top_n: int = 20) -> pd.DataFrame:
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

    def extract_temporal_order_sequences(self,
                                       df: pd.DataFrame,
                                       patient_id_col: str,
                                       order_col: str,
                                       time_col: str,
                                       max_sequence_length: int = 20) -> Dict[Any, List[str]]:
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

        # Ensure time column is datetime
        if df[time_col].dtype != 'datetime64[ns]':
            try:
                df = df.copy()
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

    def create_order_timing_features(self,
                                   df: pd.DataFrame,
                                   patient_id_col: str,
                                   order_col: str,
                                   order_time_col: str,
                                   admission_time_col: str = None,
                                   discharge_time_col: str = None) -> pd.DataFrame:
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

        # Ensure time columns are datetime
        df = df.copy()
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
            'patient_id': [],
            'total_orders': [],
            'unique_order_types': [],
            'first_order_time': [],
            'last_order_time': [],
            'order_span_hours': []
        }

        # Add admission-relative features if admission time is available
        if admission_time_col and admission_time_col in df.columns:
            features.update({
                'time_to_first_order_hours': [],
                'orders_in_first_24h': [],
                'orders_in_first_48h': [],
                'orders_in_first_72h': []
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

    def get_order_type_distributions(self,
                                   df: pd.DataFrame,
                                   patient_id_col: str,
                                   order_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    def calculate_order_transition_matrix(self,
                                        sequences: Dict[Any, List[str]],
                                        top_n: int = 20) -> pd.DataFrame:
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
        transition_counts = pd.DataFrame(0,
                                       index=common_orders,
                                       columns=common_orders)

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

    def save_features(self,
                     features: Any,
                     feature_type: str,
                     base_path: str,
                     format: str = 'csv') -> str:
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

# --- Main Application Class ---
class MIMICDashboardApp:
    def __init__(self):
        logging.info("Initializing MIMICDashboardApp...")
        self.data_handler = MIMICDataHandler()
        self.visualizer = MIMICVisualizer()
        self.feature_engineer = MIMICFeatureEngineer()  # Initialize feature engineer
        self.init_session_state()
        logging.info("MIMICDashboardApp initialized.")

    @staticmethod
    def init_session_state():
        """ Function to initialize session state """
        logging.info("Initializing session state...")
        if 'loader' not in st.session_state:
            st.session_state.loader = None
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        if 'selected_module' not in st.session_state:
            st.session_state.selected_module = None
        if 'selected_table' not in st.session_state:
            st.session_state.selected_table = None
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'sample_size' not in st.session_state:
            st.session_state.sample_size = DEFAULT_SAMPLE_SIZE
        if 'available_tables' not in st.session_state:
            st.session_state.available_tables = {}
        if 'file_paths' not in st.session_state:
            st.session_state.file_paths = {}
        if 'file_sizes' not in st.session_state:
            st.session_state.file_sizes = {}
        if 'table_display_names' not in st.session_state:
            st.session_state.table_display_names = {}
        if 'current_file_path' not in st.session_state:
            st.session_state.current_file_path = None
        if 'mimic_path' not in st.session_state:
            st.session_state.mimic_path = DEFAULT_MIMIC_PATH

        # Feature engineering states
        if 'feature_eng_tab' not in st.session_state:
            st.session_state.feature_eng_tab = 0
        if 'detected_order_cols' not in st.session_state:
            st.session_state.detected_order_cols = []
        if 'detected_time_cols' not in st.session_state:
            st.session_state.detected_time_cols = []
        if 'detected_patient_id_col' not in st.session_state:
            st.session_state.detected_patient_id_col = None
        if 'freq_matrix' not in st.session_state:
            st.session_state.freq_matrix = None
        if 'order_sequences' not in st.session_state:
            st.session_state.order_sequences = None
        if 'timing_features' not in st.session_state:
            st.session_state.timing_features = None
        if 'order_dist' not in st.session_state:
            st.session_state.order_dist = None
        if 'patient_order_dist' not in st.session_state:
            st.session_state.patient_order_dist = None
        if 'transition_matrix' not in st.session_state:
            st.session_state.transition_matrix = None

        logging.info("Session state initialized.")

    def run(self):
        """Run the main application loop."""
        logging.info("Starting MIMICDashboardApp run...")
        # Set page config
        st.set_page_config(
            page_title="MIMIC-IV Explorer",
            page_icon="🏥",
            layout="wide"
        )

        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main .block-container {padding-top: 2rem;}
        .sub-header {margin-top: 20px; margin-bottom: 10px; color: #1E88E5;}
        .info-box {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-left: 10px;
            padding-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display the sidebar
        self._display_sidebar()

        # Call the method to display the main content with tabs
        self._display_main_content_with_tabs()
        logging.info("MIMICDashboardApp run finished.")

    def _display_sidebar(self):
        """Handles the display and logic of the sidebar components."""
        st.sidebar.markdown("## Dataset Configuration")

        # MIMIC-IV path input
        mimic_path = st.sidebar.text_input( "MIMIC-IV Dataset Path",
            value=st.session_state.mimic_path,
            help="Enter the path to your local MIMIC-IV v3.1 dataset" )

        # Update mimic_path in session state
        st.session_state.mimic_path = mimic_path

        # Scan button
        if st.sidebar.button("Scan MIMIC-IV Directory"):
            if not mimic_path or mimic_path == "/path/to/mimic-iv-3.1":
                st.sidebar.error("Please enter a valid MIMIC-IV dataset path")
            else:
                # Scan the directory structure
                available_tables, file_paths, file_sizes, table_display_names = self.data_handler.scan_mimic_directory(mimic_path)

                if available_tables:
                    st.session_state.available_tables = available_tables
                    st.session_state.file_paths = file_paths
                    st.session_state.file_sizes = file_sizes
                    st.session_state.table_display_names = table_display_names
                    st.sidebar.success(f"Found {sum(len(tables) for tables in available_tables.values())} tables in {len(available_tables)} modules")
                else:
                    st.sidebar.error("No MIMIC-IV data found in the specified path")

        # Module and table selection (only show if available_tables is populated)
        if st.session_state.available_tables:
            # Module selection
            module = st.sidebar.selectbox(
                "Select Module",
                list(st.session_state.available_tables.keys()),
                help="Select which MIMIC-IV module to explore"
            )

            # Update selected module
            st.session_state.selected_module = module

            # Table selection based on selected module
            if module in st.session_state.available_tables:
                # Create a list of table display names for the dropdown
                table_options = st.session_state.available_tables[module]
                table_display_options = [st.session_state.table_display_names.get((module, table), table) for table in table_options]

                # Create a mapping from display name back to actual table name
                display_to_table = {display: table for table, display in zip(table_options, table_display_options)}

                # Show the dropdown with display names
                selected_display = st.sidebar.selectbox(
                    "Select Table",
                    table_display_options,
                    help="Select which table to load (file size shown in parentheses)"
                )

                # Get the actual table name from the selected display name
                table = display_to_table[selected_display]

                # Update selected table
                st.session_state.selected_table = table

                # Show table info
                table_info = self.data_handler.get_table_info(module, table)
                st.sidebar.info(table_info)

                # Advanced options
                with st.sidebar.expander("Advanced Options"):
                    encoding = st.selectbox("Encoding", ["latin-1", "utf-8"], index=0)
                    st.session_state.sample_size = st.number_input("Sample Size", 100, 10000, 1000, 100)

                # Load button
                if st.sidebar.button("Load Table"):
                    file_path = st.session_state.file_paths.get((module, table))
                    if file_path:
                        st.session_state.current_file_path = file_path
                        df = self.data_handler.load_mimic_table( file_path=file_path, sample_size=st.session_state.sample_size, encoding=encoding )

                        if df is not None:
                            st.session_state.df = df

                            # Auto-detect columns for feature engineering
                            st.session_state.detected_order_cols = self.feature_engineer.detect_order_columns(df)
                            st.session_state.detected_time_cols = self.feature_engineer.detect_temporal_columns(df)
                            st.session_state.detected_patient_id_col = self.feature_engineer.detect_patient_id_column(df)

    def _display_main_content_with_tabs(self):
        """Handles the display of the main content area with tabs."""
        if st.session_state.df is not None:
            # Dataset info
            st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)
            st.markdown(f"**Module:** {st.session_state.selected_module}")
            st.markdown(f"**Table:** {st.session_state.selected_table}")
            st.markdown(f"**File:** {os.path.basename(st.session_state.current_file_path)}")

            # Get file size and format it
            file_size_mb = st.session_state.file_sizes.get((st.session_state.selected_module, st.session_state.selected_table), 0)
            if file_size_mb < 1:
                size_str = f"{file_size_mb:.2f} KB"
            elif file_size_mb < 1000:
                size_str = f"{file_size_mb:.2f} MB"
            else:
                size_str = f"{file_size_mb/1000:.2f} GB"

            st.markdown(f"**File Size:** {size_str}")
            st.markdown(f"**Sample Size:** {min(len(st.session_state.df), st.session_state.sample_size)} rows out of {len(st.session_state.df)}") # TODO: Need total row count
            st.markdown("</div>", unsafe_allow_html=True)

            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Exploration & Visualization", "Feature Engineering", "Export Options"])

            # Tab 1: Exploration & Visualization
            with tab1:
                # Data preview
                self.visualizer.display_data_preview(st.session_state.df)

                # Dataset statistics
                self.visualizer.display_dataset_statistics(st.session_state.df)

                # Data visualization
                self.visualizer.display_visualizations(st.session_state.df)

            # Tab 2: Feature Engineering
            with tab2:
                self._display_feature_engineering_tab()

            # Tab 3: Export Options
            with tab3:
                st.markdown("<h2 class='sub-header'>Export Options</h2>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Export to CSV"):
                        csv = st.session_state.df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"mimic_iv_{st.session_state.selected_module}_{st.session_state.selected_table}.csv",
                            mime="text/csv"
                        )

                with col2:
                    if st.button("Convert to Parquet"):
                        try:
                            # Create parquet directory if it doesn't exist
                            parquet_dir = os.path.join(os.path.dirname(st.session_state.current_file_path), 'parquet_files')
                            os.makedirs(parquet_dir, exist_ok=True)

                            # Define parquet file path
                            parquet_file = os.path.join(parquet_dir, f"{st.session_state.selected_table}.parquet")

                            # Convert to parquet
                            table = pa.Table.from_pandas(st.session_state.df)
                            pq.write_table(table, parquet_file)

                            st.success(f"Dataset converted to Parquet format at {parquet_file}")
                        except Exception as e:
                            st.error(f"Error converting to Parquet: {str(e)}")
        else:
            # Welcome message when no data is loaded
            st.markdown("""
            <div class='info-box'>
            <h2 class='sub-header'>Welcome to the MIMIC-IV Dashboard</h2>
            <p>This dashboard allows you to explore the MIMIC-IV dataset directly from the CSV/CSV.GZ files.</p>
            <p>To get started:</p>
            <ol>
                <li>Enter the path to your local MIMIC-IV v3.1 dataset in the sidebar</li>
                <li>Click "Scan MIMIC-IV Directory" to detect available tables</li>
                <li>Select a module and table to explore</li>
                <li>Configure advanced options if needed</li>
                <li>Click "Load Table" to begin</li>
            </ol>
            <p>Note: You need to have access to the MIMIC-IV dataset and have it downloaded locally.</p>
            </div>
            """, unsafe_allow_html=True)

            # About MIMIC-IV
            st.markdown("""
            <h2 class='sub-header'>About MIMIC-IV</h2>
            <div class='info-box'>
            <p>MIMIC-IV is a large, freely-available database comprising de-identified health-related data associated with patients who stayed in critical care units at the Beth Israel Deaconess Medical Center between 2008 - 2019.</p>
            <p>The database is organized into two main modules:</p>
            <ul>
                <li><strong>Hospital (hosp)</strong>: Contains hospital-wide data including admissions, patients, lab tests, diagnoses, etc.</li>
                <li><strong>ICU (icu)</strong>: Contains ICU-specific data including vital signs, medications, procedures, etc.</li>
            </ul>
            <p>Key tables include:</p>
            <ul>
                <li><strong>patients.csv</strong>: Patient demographic data</li>
                <li><strong>admissions.csv</strong>: Hospital admission information</li>
                <li><strong>labevents.csv</strong>: Laboratory measurements</li>
                <li><strong>chartevents.csv</strong>: Patient charting data (vital signs, etc.)</li>
                <li><strong>icustays.csv</strong>: ICU stay information</li>
            </ul>
            <p>For more information, visit <a href="https://physionet.org/content/mimiciv/3.1/">MIMIC-IV on PhysioNet</a>.</p>
            </div>
            """, unsafe_allow_html=True)

    def _display_feature_engineering_tab(self):
        """Display the feature engineering tab content."""
        st.markdown("<h2 class='sub-header'>Order Data Feature Engineering</h2>", unsafe_allow_html=True)

        # Show introductory text
        st.markdown("""
        <div class='info-box'>
        This section allows you to transform raw MIMIC-IV order data into structured features for analysis and machine learning.
        Choose one of the feature engineering methods below to get started.
        </div>
        """, unsafe_allow_html=True)

        # Feature engineering subtabs
        feature_tabs = st.tabs([
            "📊 Order Frequency Matrix",
            "⏱️ Temporal Order Sequences",
            "📈 Order Type Distributions",
            "🕒 Order Timing Analysis"
        ])

        # Get available columns
        all_columns = st.session_state.df.columns.tolist()

        # 1. Order Frequency Matrix tab
        with feature_tabs[0]:
            st.markdown("<h3>Create Order Frequency Matrix</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            This creates a matrix where rows are patients and columns are order types, with cells showing frequency of each order type per patient.
            </div>
            """, unsafe_allow_html=True)

            # Column selection
            col1, col2 = st.columns(2)
            with col1:
                # Suggest patient ID column but allow selection from all columns
                default_patient_idx = 0
                if st.session_state.detected_patient_id_col in all_columns:
                    default_patient_idx = all_columns.index(st.session_state.detected_patient_id_col)

                patient_id_col = st.selectbox(
                    "Select Patient ID Column",
                    all_columns,
                    index=default_patient_idx,
                    help="Column containing unique patient identifiers"
                )

            with col2:
                # Suggest order column but allow selection from all columns
                default_order_idx = 0
                if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns:
                    default_order_idx = all_columns.index(st.session_state.detected_order_cols[0])

                order_col = st.selectbox(
                    "Select Order Type Column",
                    all_columns,
                    index=default_order_idx,
                    help="Column containing order types/names"
                )

            # Options
            col1, col2, col3 = st.columns(3)
            with col1:
                normalize = st.checkbox("Normalize by Patient", value=False,
                                      help="Convert frequencies to percentages of total orders per patient")
            with col2:
                top_n = st.number_input("Top N Order Types", min_value=0, max_value=100, value=20,
                                      help="Limit to most frequent order types (0 = include all)")

            # Generate button
            if st.button("Generate Order Frequency Matrix"):
                try:
                    with st.spinner("Generating order frequency matrix..."):
                        freq_matrix = self.feature_engineer.create_order_frequency_matrix(
                            st.session_state.df,
                            patient_id_col,
                            order_col,
                            normalize,
                            top_n
                        )
                        st.session_state.freq_matrix = freq_matrix
                except Exception as e:
                    st.error(f"Error generating frequency matrix: {str(e)}")

            # Display result if available
            if st.session_state.freq_matrix is not None:
                st.markdown("<h4>Order Frequency Matrix</h4>", unsafe_allow_html=True)

                # Show preview
                st.dataframe(st.session_state.freq_matrix.head(10), use_container_width=True)

                # Matrix stats
                st.markdown(f"<div class='info-box'>Matrix size: {st.session_state.freq_matrix.shape[0]} patients × {st.session_state.freq_matrix.shape[1]} order types</div>", unsafe_allow_html=True)

                # Heatmap visualization
                st.markdown("<h4>Frequency Matrix Heatmap (Sample)</h4>", unsafe_allow_html=True)

                # Take a sample for visualization (first 20 patients, first 20 order types)
                sample_size = min(20, st.session_state.freq_matrix.shape[0])
                sample_cols = min(20, st.session_state.freq_matrix.shape[1])

                matrix_sample = st.session_state.freq_matrix.iloc[:sample_size, :sample_cols]

                # Generate heatmap
                fig = px.imshow(
                    matrix_sample,
                    labels=dict(x="Order Type", y="Patient ID", color="Frequency"),
                    x=matrix_sample.columns,
                    y=matrix_sample.index,
                    aspect="auto"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Save options
                save_col1, save_col2 = st.columns(2)
                with save_col1:
                    save_format = st.radio("Save Format", ["CSV", "Parquet"], horizontal=True)
                with save_col2:
                    if st.button("Save Frequency Matrix"):
                        try:
                            filepath = self.feature_engineer.save_features(
                                st.session_state.freq_matrix,
                                "order_frequency_matrix",
                                os.path.dirname(st.session_state.current_file_path),
                                save_format.lower()
                            )
                            st.success(f"Saved frequency matrix to {filepath}")
                        except Exception as e:
                            st.error(f"Error saving frequency matrix: {str(e)}")

        # 2. Temporal Order Sequences tab
        with feature_tabs[1]:
            st.markdown("<h3>Extract Temporal Order Sequences</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            This extracts chronological sequences of orders for each patient, preserving the temporal relationships between different orders.
            </div>
            """, unsafe_allow_html=True)

            # Column selection
            col1, col2, col3 = st.columns(3)
            with col1:
                # Suggest patient ID column
                default_patient_idx = 0
                if st.session_state.detected_patient_id_col in all_columns:
                    default_patient_idx = all_columns.index(st.session_state.detected_patient_id_col)

                seq_patient_id_col = st.selectbox(
                    "Select Patient ID Column",
                    all_columns,
                    index=default_patient_idx,
                    key="seq_patient_id_col",
                    help="Column containing unique patient identifiers"
                )

            with col2:
                # Suggest order column
                default_order_idx = 0
                if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns:
                    default_order_idx = all_columns.index(st.session_state.detected_order_cols[0])

                seq_order_col = st.selectbox(
                    "Select Order Type Column",
                    all_columns,
                    index=default_order_idx,
                    key="seq_order_col",
                    help="Column containing order types/names"
                )

            with col3:
                # Suggest time column
                default_time_idx = 0
                if st.session_state.detected_time_cols and st.session_state.detected_time_cols[0] in all_columns:
                    default_time_idx = all_columns.index(st.session_state.detected_time_cols[0])

                seq_time_col = st.selectbox(
                    "Select Timestamp Column",
                    all_columns,
                    index=default_time_idx,
                    key="seq_time_col",
                    help="Column containing order timestamps"
                )

            # Options
            max_seq_length = st.slider("Maximum Sequence Length", min_value=5, max_value=100, value=20,
                                     help="Maximum number of orders to include in each sequence")

            # Generate button
            if st.button("Extract Order Sequences"):
                try:
                    with st.spinner("Extracting temporal order sequences..."):
                        sequences = self.feature_engineer.extract_temporal_order_sequences(
                            st.session_state.df,
                            seq_patient_id_col,
                            seq_order_col,
                            seq_time_col,
                            max_seq_length
                        )
                        st.session_state.order_sequences = sequences

                        # Also generate transition matrix automatically
                        transition_matrix = self.feature_engineer.calculate_order_transition_matrix(
                            sequences,
                            top_n=15  # Limit to top 15 for visualization
                        )
                        st.session_state.transition_matrix = transition_matrix
                except Exception as e:
                    st.error(f"Error extracting order sequences: {str(e)}")

            # Display results if available
            if st.session_state.order_sequences is not None:
                # Show sequence stats
                num_patients = len(st.session_state.order_sequences)
                avg_sequence_length = np.mean([len(seq) for seq in st.session_state.order_sequences.values()])

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
                for patient in sample_patients:
                    sequence = st.session_state.order_sequences[patient]
                    sequence_str = " → ".join([str(order) for order in sequence])

                    st.markdown(f"<strong>Patient {patient}:</strong> {sequence_str}", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)

                # Transition matrix visualization
                if st.session_state.transition_matrix is not None:
                    st.markdown("<h4>Order Transition Matrix</h4>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='info-box'>
                    This matrix shows the probability of transitioning from one order type (rows) to another (columns).
                    </div>
                    """, unsafe_allow_html=True)

                    fig = px.imshow(
                        st.session_state.transition_matrix,
                        labels=dict(x="Next Order", y="Current Order", color="Transition Probability"),
                        x=st.session_state.transition_matrix.columns,
                        y=st.session_state.transition_matrix.index,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)

                # Save options
                save_col1, save_col2 = st.columns(2)
                with save_col1:
                    seq_save_format = st.radio("Save Format", ["JSON", "CSV"], horizontal=True, key="seq_save_format")
                with save_col2:
                    if st.button("Save Order Sequences"):
                        try:
                            filepath = self.feature_engineer.save_features(
                                st.session_state.order_sequences,
                                "temporal_order_sequences",
                                os.path.dirname(st.session_state.current_file_path),
                                seq_save_format.lower()
                            )
                            st.success(f"Saved order sequences to {filepath}")
                        except Exception as e:
                            st.error(f"Error saving order sequences: {str(e)}")

        # 3. Order Type Distributions tab
        with feature_tabs[2]:
            st.markdown("<h3>Analyze Order Type Distributions</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            This analyzes the distribution of different order types across the dataset and for individual patients.
            </div>
            """, unsafe_allow_html=True)

            # Column selection
            col1, col2 = st.columns(2)
            with col1:
                # Suggest patient ID column
                default_patient_idx = 0
                if st.session_state.detected_patient_id_col in all_columns:
                    default_patient_idx = all_columns.index(st.session_state.detected_patient_id_col)

                dist_patient_id_col = st.selectbox(
                    "Select Patient ID Column",
                    all_columns,
                    index=default_patient_idx,
                    key="dist_patient_id_col",
                    help="Column containing unique patient identifiers"
                )

            with col2:
                # Suggest order column
                default_order_idx = 0
                if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns:
                    default_order_idx = all_columns.index(st.session_state.detected_order_cols[0])

                dist_order_col = st.selectbox(
                    "Select Order Type Column",
                    all_columns,
                    index=default_order_idx,
                    key="dist_order_col",
                    help="Column containing order types/names"
                )

            # Generate button
            if st.button("Analyze Order Distributions"):
                try:
                    with st.spinner("Analyzing order type distributions..."):
                        overall_dist, patient_dist = self.feature_engineer.get_order_type_distributions(
                            st.session_state.df,
                            dist_patient_id_col,
                            dist_order_col
                        )
                        st.session_state.order_dist = overall_dist
                        st.session_state.patient_order_dist = patient_dist
                except Exception as e:
                    st.error(f"Error analyzing order distributions: {str(e)}")

            # Display results if available
            if st.session_state.order_dist is not None:
                # Show overall distribution
                st.markdown("<h4>Overall Order Type Distribution</h4>", unsafe_allow_html=True)

                # Create pie chart for overall distribution
                top_n_orders = 15  # Show top 15 for pie chart
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

                fig = px.pie(
                    pie_data,
                    values='frequency',
                    names=dist_order_col,
                    title=f"Overall Distribution of {dist_order_col} (Top {top_n_orders})"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show bar chart of top 20
                top_20 = st.session_state.order_dist.head(20)
                bar_fig = px.bar(
                    top_20,
                    x=dist_order_col,
                    y='frequency',
                    title=f"Top 20 {dist_order_col} by Frequency"
                )
                st.plotly_chart(bar_fig, use_container_width=True)

                # Patient-level distribution (sample)
                if st.session_state.patient_order_dist is not None and not st.session_state.patient_order_dist.empty:
                    st.markdown("<h4>Patient-Level Order Type Distribution</h4>", unsafe_allow_html=True)

                    # Get unique patients
                    patients = st.session_state.patient_order_dist['patient_id'].unique()

                    # Sample patients for visualization if there are too many
                    if len(patients) > 5:
                        sample_patients = patients[:5]
                    else:
                        sample_patients = patients

                    # Create subplots for each patient
                    fig = make_subplots(
                        rows=len(sample_patients),
                        cols=1,
                        subplot_titles=[f"Patient {patient}" for patient in sample_patients]
                    )

                    # Add traces for each patient
                    for i, patient in enumerate(sample_patients):
                        patient_data = st.session_state.patient_order_dist[
                            st.session_state.patient_order_dist['patient_id'] == patient
                        ].head(10)  # Top 10 orders for this patient

                        fig.add_trace(
                            go.Bar(
                                x=patient_data[dist_order_col],
                                y=patient_data['frequency'],
                                name=f"Patient {patient}"
                            ),
                            row=i+1, col=1
                        )

                    fig.update_layout(height=200*len(sample_patients), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                # Save options
                save_col1, save_col2 = st.columns(2)
                with save_col1:
                    dist_save_format = st.radio("Save Format", ["CSV", "Parquet"], horizontal=True, key="dist_save_format")
                with save_col2:
                    if st.button("Save Distribution Data"):
                        try:
                            # Save overall distribution
                            filepath1 = self.feature_engineer.save_features(
                                st.session_state.order_dist,
                                "overall_order_distribution",
                                os.path.dirname(st.session_state.current_file_path),
                                dist_save_format.lower()
                            )

                            # Save patient-level distribution
                            filepath2 = self.feature_engineer.save_features(
                                st.session_state.patient_order_dist,
                                "patient_order_distribution",
                                os.path.dirname(st.session_state.current_file_path),
                                dist_save_format.lower()
                            )

                            st.success(f"Saved distribution data to:\n- {filepath1}\n- {filepath2}")
                        except Exception as e:
                            st.error(f"Error saving distribution data: {str(e)}")

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
                default_patient_idx = 0
                if st.session_state.detected_patient_id_col in all_columns:
                    default_patient_idx = all_columns.index(st.session_state.detected_patient_id_col)

                timing_patient_id_col = st.selectbox(
                    "Select Patient ID Column",
                    all_columns,
                    index=default_patient_idx,
                    key="timing_patient_id_col",
                    help="Column containing unique patient identifiers"
                )

            with col2:
                # Suggest order column
                default_order_idx = 0
                if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns:
                    default_order_idx = all_columns.index(st.session_state.detected_order_cols[0])

                timing_order_col = st.selectbox(
                    "Select Order Type Column",
                    all_columns,
                    index=default_order_idx,
                    key="timing_order_col",
                    help="Column containing order types/names"
                )

            # Time columns
            col1, col2 = st.columns(2)
            with col1:
                default_time_idx = 0
                if st.session_state.detected_time_cols and st.session_state.detected_time_cols[0] in all_columns:
                    default_time_idx = all_columns.index(st.session_state.detected_time_cols[0])

                order_time_col = st.selectbox(
                    "Select Order Time Column",
                    all_columns,
                    index=default_time_idx,
                    key="order_time_col",
                    help="Column containing order timestamps"
                )

            with col2:
                # Optional admission time column
                admission_time_col = st.selectbox(
                    "Select Admission Time Column (Optional)",
                    ["None"] + all_columns,
                    index=0,
                    help="Column containing admission timestamps (for relative timing features)"
                )

                if admission_time_col == "None":
                    admission_time_col = None

            # Optional discharge time column
            discharge_time_col = st.selectbox(
                "Select Discharge Time Column (Optional)",
                ["None"] + all_columns,
                index=0,
                help="Column containing discharge timestamps (for relative timing features)"
            )

            if discharge_time_col == "None":
                discharge_time_col = None

            # Generate button
            if st.button("Generate Timing Features"):
                try:
                    with st.spinner("Generating order timing features..."):
                        timing_features = self.feature_engineer.create_order_timing_features(
                            st.session_state.df,
                            timing_patient_id_col,
                            timing_order_col,
                            order_time_col,
                            admission_time_col,
                            discharge_time_col
                        )
                        st.session_state.timing_features = timing_features
                except Exception as e:
                    st.error(f"Error generating timing features: {str(e)}")

            # Display results if available
            if st.session_state.timing_features is not None:
                st.markdown("<h4>Order Timing Features</h4>", unsafe_allow_html=True)

                # Show preview of features
                st.dataframe(st.session_state.timing_features.head(10), use_container_width=True)

                # Generate visualizations based on available features
                st.markdown("<h4>Order Timing Visualizations</h4>", unsafe_allow_html=True)

                numeric_cols = st.session_state.timing_features.select_dtypes(include=['number']).columns

                # Bar chart of total orders
                if 'total_orders' in st.session_state.timing_features.columns:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Histogram of total orders
                        total_orders_fig = px.histogram(
                            st.session_state.timing_features,
                            x='total_orders',
                            title="Distribution of Total Orders per Patient"
                        )
                        st.plotly_chart(total_orders_fig, use_container_width=True)

                    with col2:
                        # Histogram of unique order types
                        if 'unique_order_types' in st.session_state.timing_features.columns:
                            unique_orders_fig = px.histogram(
                                st.session_state.timing_features,
                                x='unique_order_types',
                                title="Distribution of Unique Order Types per Patient"
                            )
                            st.plotly_chart(unique_orders_fig, use_container_width=True)

                # Time-based analyses
                if admission_time_col and 'time_to_first_order_hours' in st.session_state.timing_features.columns:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Histogram of time to first order
                        first_order_fig = px.histogram(
                            st.session_state.timing_features,
                            x='time_to_first_order_hours',
                            title="Time from Admission to First Order (hours)"
                        )
                        st.plotly_chart(first_order_fig, use_container_width=True)

                    with col2:
                        # Bar chart of orders in first 24/48/72 hours
                        if all(col in st.session_state.timing_features.columns for col in
                              ['orders_in_first_24h', 'orders_in_first_48h', 'orders_in_first_72h']):

                            # Prepare data for bar chart
                            time_periods = ['First 24h', 'First 48h', 'First 72h']
                            avg_orders = [
                                st.session_state.timing_features['orders_in_first_24h'].mean(),
                                st.session_state.timing_features['orders_in_first_48h'].mean(),
                                st.session_state.timing_features['orders_in_first_72h'].mean()
                            ]

                            orders_by_time = pd.DataFrame({
                                'Time Period': time_periods,
                                'Average Orders': avg_orders
                            })

                            time_orders_fig = px.bar(
                                orders_by_time,
                                x='Time Period',
                                y='Average Orders',
                                title="Average Orders in Time Periods After Admission"
                            )
                            st.plotly_chart(time_orders_fig, use_container_width=True)

                # Save options
                save_col1, save_col2 = st.columns(2)
                with save_col1:
                    timing_save_format = st.radio("Save Format", ["CSV", "Parquet"], horizontal=True, key="timing_save_format")
                with save_col2:
                    if st.button("Save Timing Features"):
                        try:
                            filepath = self.feature_engineer.save_features(
                                st.session_state.timing_features,
                                "order_timing_features",
                                os.path.dirname(st.session_state.current_file_path),
                                timing_save_format.lower()
                            )
                            st.success(f"Saved timing features to {filepath}")
                        except Exception as e:
                            st.error(f"Error saving timing features: {str(e)}")


if __name__ == "__main__":
    app = MIMICDashboardApp()
    app.run()
