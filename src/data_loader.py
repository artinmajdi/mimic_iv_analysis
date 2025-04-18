# Standard library imports
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from datasets import load_dataset
import datasets
import dask.dataframe as dd


class MIMICDataLoader:
    """Class for loading and preprocessing MIMIC-IV dataset files."""

    def __init__(self, mimic_path):
        """Initialize the data loader with the path to the MIMIC-IV dataset.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory
        """
        self.mimic_path = mimic_path
        self.data = {}
        self.preprocessed = {}
        self.parquet_dir = os.path.join(os.path.dirname(mimic_path), 'parquet_files')
        os.makedirs(self.parquet_dir, exist_ok=True)

    def convert_csv_to_parquet(self, file_path, parquet_path):
        """Convert a CSV file to Parquet format with optimized settings and data type handling.

        Args:
            file_path (str): Path to the CSV file
            parquet_path (str): Path where the Parquet file will be saved

        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # First read the CSV file with all columns as strings
            df = pd.read_csv(file_path, dtype=str, na_values=['', 'NA', 'NaN', 'null', 'NULL', 'None', '.', '__', '___'])

            # Clean the data
            df = self.clean_dataframe(df)

            # Now attempt to convert columns to appropriate types
            for column in df.columns:
                col_lower = column.lower()

                # Skip conversion for known text columns
                if any(text_pattern in col_lower for text_pattern in ['label', 'name', 'desc', 'type', 'code', 'status', 'text', 'comment', 'unit', 'source', 'value']):
                    continue

                # Skip conversion for specific columns in poe_detail
                if 'poe_detail' in file_path.lower() and col_lower in ['field_name', 'field_value']:
                    continue

                # Skip conversion for specific columns in pharmacy
                if 'pharmacy' in file_path.lower() and col_lower in ['medication', 'route', 'frequency']:
                    continue

                # Skip conversion for specific columns in labevents
                if 'labevents' in file_path.lower() and col_lower in ['flag', 'valueuom', 'ref_range_lower', 'ref_range_upper']:
                    continue

                try:
                    # Try to convert to numeric, but only if the column looks numeric
                    non_na_values = df[column].dropna()
                    if len(non_na_values) > 0:
                        # Check if the column might be numeric
                        sample = non_na_values.iloc[0]
                        if isinstance(sample, str):
                            # Remove any commas and spaces
                            non_na_values = non_na_values.str.replace(',', '').str.replace(' ', '')
                            # Check if it's a numeric column
                            numeric_check = pd.to_numeric(non_na_values.head(), errors='coerce')
                            if not numeric_check.isna().all():
                                # If it looks numeric, convert the whole column
                                df[column] = df[column].str.replace(',', '').str.replace(' ', '')
                                df[column] = pd.to_numeric(df[column], errors='coerce')

                                # Convert to Int64 if no decimals
                                if df[column].notna().any() and df[column].dropna().apply(lambda x: float(x).is_integer()).all():
                                    df[column] = df[column].astype('Int64')

                except Exception as e:
                    # If conversion fails, keep as string
                    continue

            # Handle datetime columns
            for column in df.columns:
                col_lower = column.lower()
                if any(time_pattern in col_lower for time_pattern in ['time', 'date', 'expire', 'admit', 'disch']):
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    except:
                        pass

            # Convert to PyArrow table and write to Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path, compression='snappy')

            return True

        except Exception as e:
            st.error(f"Error converting {file_path} to Parquet: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")
            import traceback
            st.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def clean_dataframe(self, df):
        """Clean and preprocess a DataFrame to handle missing and noisy data.

        Args:
            df (pandas.DataFrame): Input DataFrame to clean

        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()

        # Define missing value patterns
        missing_values = ['', 'NA', 'NaN', 'null', 'NULL', 'None', '.', '__', '___', 'nan', 'NaT']

        for column in df_clean.columns:
            try:
                # Strip whitespace from string columns
                if df_clean[column].dtype == 'object' or df_clean[column].dtype == 'string':
                    df_clean[column] = df_clean[column].astype(str).str.strip()

                    # Replace missing values with None
                    df_clean.loc[df_clean[column].isin(missing_values), column] = None
                    df_clean.loc[df_clean[column].str.lower().isin([x.lower() for x in missing_values]), column] = None

            except Exception as e:
                st.warning(f"Error cleaning column {column}: {str(e)}")
                continue

        # Remove completely empty columns
        empty_cols = [col for col in df_clean.columns if df_clean[col].isna().all()]
        if empty_cols:
            df_clean = df_clean.drop(columns=empty_cols)

        return df_clean

    def ensure_parquet_files(self):
        """Convert all necessary CSV files to Parquet format if not already converted.

        Returns:
            bool: True if all files are ready in Parquet format, False otherwise
        """
        required_files = [
            'patients.csv', 'admissions.csv', 'transfers.csv',
            'poe.csv', 'poe_detail.csv', 'diagnoses_icd.csv',
            'd_icd_diagnoses.csv', 'pharmacy.csv', 'labevents.csv'
        ]

        all_converted = True
        files_to_convert = []

        # Check which files need conversion
        for file in required_files:
            csv_path = os.path.join(self.mimic_path, 'hosp', file)
            parquet_path = os.path.join(self.parquet_dir, file.replace('.csv', '.parquet'))

            if not os.path.exists(parquet_path) and os.path.exists(csv_path):
                files_to_convert.append((csv_path, parquet_path))

        if files_to_convert:
            st.info("Converting CSV files to Parquet format for faster loading...")
            progress_bar = st.progress(0)

            for i, (csv_path, parquet_path) in enumerate(files_to_convert):
                success = self.convert_csv_to_parquet(csv_path, parquet_path)
                if not success:
                    all_converted = False
                progress_bar.progress((i + 1) / len(files_to_convert))

            if all_converted:
                st.success("All files converted to Parquet format successfully!")
            else:
                st.warning("Some files could not be converted to Parquet format.")

        return all_converted

    def load_parquet_file(self, file_name, columns=None):
        """Load data from a Parquet file with optional column selection.

        Args:
            file_name (str): Name of the file (without extension)
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Loaded data or None if file not found
        """
        try:
            parquet_path = os.path.join(self.parquet_dir, f"{file_name}.parquet")
            if os.path.exists(parquet_path):
                if columns:
                    return pd.read_parquet(parquet_path, columns=columns)
                return pd.read_parquet(parquet_path)
            return None
        except Exception as e:
            st.error(f"Error loading {file_name} from Parquet: {e}")
            return None

    def load_csv_chunked(self, file_path, chunk_size=100000, columns=None):
        """Load a CSV file in chunks with optional column selection.

        Args:
            file_path (str): Path to the CSV file
            chunk_size (int): Number of rows to load per chunk
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Concatenated data from all chunks
        """
        try:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=columns):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        except Exception as e:
            st.error(f"Error loading {file_path} in chunks: {e}")
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_patients(self, columns=None):
        """Load patients data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Patients data or None if file not found
        """
        df = self.load_parquet_file('patients', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'patients.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['patients'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_admissions(self, columns=None):
        """Load admissions data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Admissions data or None if file not found
        """
        df = self.load_parquet_file('admissions', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'admissions.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['admissions'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_transfers(self, columns=None):
        """Load transfers data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Transfers data or None if file not found
        """
        df = self.load_parquet_file('transfers', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'transfers.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['transfers'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_poe(self, columns=None):
        """Load provider order entry data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: POE data or None if file not found
        """
        df = self.load_parquet_file('poe', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'poe.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['poe'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_poe_detail(self, columns=None):
        """Load provider order entry detail data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: POE detail data or None if file not found
        """
        df = self.load_parquet_file('poe_detail', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'poe_detail.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['poe_detail'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_diagnoses_icd(self, columns=None):
        """Load diagnoses ICD data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Diagnoses ICD data or None if file not found
        """
        df = self.load_parquet_file('diagnoses_icd', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'diagnoses_icd.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['diagnoses_icd'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_d_icd_diagnoses(self, columns=None):
        """Load ICD diagnoses dictionary from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: ICD diagnoses dictionary or None if file not found
        """
        df = self.load_parquet_file('d_icd_diagnoses', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'd_icd_diagnoses.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['d_icd_diagnoses'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_pharmacy(self, columns=None):
        """Load pharmacy data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Pharmacy data or None if file not found
        """
        df = self.load_parquet_file('pharmacy', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'pharmacy.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['pharmacy'] = df
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_labevents(self, columns=None):
        """Load laboratory events data from MIMIC-IV.

        Args:
            columns (list, optional): List of columns to load

        Returns:
            pandas.DataFrame: Laboratory events data or None if file not found
        """
        df = self.load_parquet_file('labevents', columns)
        if df is None:
            file_path = os.path.join(self.mimic_path, 'hosp', 'labevents.csv')
            if os.path.exists(file_path):
                df = self.load_csv_chunked(file_path, columns=columns)

        if df is not None:
            self.data['labevents'] = df
        return df

    def validate_path(self):
        """Validate that the MIMIC-IV path exists and contains required directories.

        Returns:
            bool: True if path is valid, False otherwise
        """
        if not os.path.exists(self.mimic_path):
            st.error(f"Path not found: {self.mimic_path}")
            return False

        required_dirs = ['hosp', 'icu']
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(self.mimic_path, dir_name)):
                st.warning(f"Missing directory: {dir_name}")
                return False

        # Check for required files
        required_files = [
            'patients.csv', 'admissions.csv', 'transfers.csv',
            'poe.csv', 'poe_detail.csv', 'diagnoses_icd.csv',
            'd_icd_diagnoses.csv', 'pharmacy.csv', 'labevents.csv'
        ]
        for file in required_files:
            if not os.path.exists(os.path.join(self.mimic_path, 'hosp', file)):
                st.warning(f"Missing file: {file}")
                return False
        return True

    def load_all_data(self):
        """Load all required MIMIC-IV data files.

        Returns:
            dict: Dictionary containing all loaded dataframes
        """
        if not self.validate_path():
            return {}

        # First, ensure all files are converted to Parquet format
        self.ensure_parquet_files()

        with st.spinner("Loading MIMIC-IV data..."):
            # Define required columns for each table to minimize memory usage
            patients_cols = ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group']
            admissions_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type', 'admission_location', 'insurance']
            transfers_cols = ['subject_id', 'hadm_id', 'careunit', 'intime', 'outtime']
            poe_cols = ['subject_id', 'hadm_id', 'order_type', 'ordertime']
            poe_detail_cols = ['poe_id', 'field_name', 'field_value']

            # Load data with specific columns
            patients_df = self.load_patients(columns=patients_cols)
            admissions_df = self.load_admissions(columns=admissions_cols)
            transfers_df = self.load_transfers(columns=transfers_cols)
            poe_df = self.load_poe(columns=poe_cols)
            poe_detail_df = self.load_poe_detail(columns=poe_detail_cols)
            diagnoses_icd_df = self.load_diagnoses_icd()
            d_icd_diagnoses_df = self.load_d_icd_diagnoses()
            pharmacy_df = self.load_pharmacy()
            labevents_df = self.load_labevents()

            if all([patients_df is not None, admissions_df is not None, transfers_df is not None,
                    poe_df is not None, poe_detail_df is not None, diagnoses_icd_df is not None,
                    d_icd_diagnoses_df is not None, pharmacy_df is not None, labevents_df is not None]):
                st.success("Data loaded successfully!")
            else:
                st.error("Some data files could not be loaded. Check the path and file structure.")

        return self.data

    def preprocess_patients(self):
        """Preprocess patients data.

        Returns:
            pandas.DataFrame: Preprocessed patients data
        """
        if 'patients' not in self.data or self.data['patients'] is None:
            return None

        df = self.data['patients'].copy()

        # Convert date columns to datetime
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'])

        # Calculate age (if anchor_year and dob are available)
        if 'anchor_year' in df.columns and 'anchor_year_group' in df.columns and 'dob' in df.columns:
            # Extract the first year from anchor_year_group (e.g., '2180 - 2189' -> 2180)
            df['anchor_year_numeric'] = df['anchor_year_group'].apply(lambda x: int(x.split(' - ')[0]) if isinstance(x, str) else None)
            df['age'] = df['anchor_year_numeric'] - pd.DatetimeIndex(df['dob']).year
            # Cap age at 90 for privacy
            df['age'] = df['age'].clip(upper=90)

        # Create age groups
        if 'age' in df.columns:
            bins = [0, 18, 30, 50, 70, 90, 200]
            labels = ['0-18', '19-30', '31-50', '51-70', '71-90', '90+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        self.preprocessed['patients'] = df
        return df

    def preprocess_admissions(self):
        """Preprocess admissions data.

        Returns:
            pandas.DataFrame: Preprocessed admissions data
        """
        if 'admissions' not in self.data or self.data['admissions'] is None:
            return None

        df = self.data['admissions'].copy()

        # Convert date columns to datetime
        datetime_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Calculate length of stay in days
        if 'admittime' in df.columns and 'dischtime' in df.columns:
            df['los_days'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / (24 * 60 * 60)
            # Remove negative LOS (data errors)
            df['los_days'] = df['los_days'].clip(lower=0)

        # Create LOS categories
        if 'los_days' in df.columns:
            bins = [0, 1, 3, 7, 14, 30, 1000]
            labels = ['<1 day', '1-3 days', '3-7 days', '7-14 days', '14-30 days', '>30 days']
            df['los_group'] = pd.cut(df['los_days'], bins=bins, labels=labels, right=False)

        # Extract admission year and month for time-based analysis
        if 'admittime' in df.columns:
            df['admit_year'] = df['admittime'].dt.year
            df['admit_month'] = df['admittime'].dt.month
            df['admit_dow'] = df['admittime'].dt.dayofweek
            df['admit_hour'] = df['admittime'].dt.hour

        if 'dischtime' in df.columns:
            df['disch_year'] = df['dischtime'].dt.year
            df['disch_month'] = df['dischtime'].dt.month
            df['disch_dow'] = df['dischtime'].dt.dayofweek
            df['disch_hour'] = df['dischtime'].dt.hour

        self.preprocessed['admissions'] = df
        return df

    def preprocess_transfers(self):
        """Preprocess transfers data.

        Returns:
            pandas.DataFrame: Preprocessed transfers data
        """
        if 'transfers' not in self.data or self.data['transfers'] is None:
            return None

        df = self.data['transfers'].copy()

        # Convert date columns to datetime
        datetime_cols = ['intime', 'outtime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Calculate length of stay in each unit (hours)
        if 'intime' in df.columns and 'outtime' in df.columns:
            df['unit_los_hours'] = (df['outtime'] - df['intime']).dt.total_seconds() / (60 * 60)
            # Remove negative LOS (data errors)
            df['unit_los_hours'] = df['unit_los_hours'].clip(lower=0)

        # Extract transfer hour for time-based analysis
        if 'intime' in df.columns:
            df['transfer_in_hour'] = df['intime'].dt.hour
            df['transfer_in_dow'] = df['intime'].dt.dayofweek

        if 'outtime' in df.columns:
            df['transfer_out_hour'] = df['outtime'].dt.hour
            df['transfer_out_dow'] = df['outtime'].dt.dayofweek

        self.preprocessed['transfers'] = df
        return df

    def preprocess_poe(self):
        """Preprocess provider order entry data.

        Returns:
            pandas.DataFrame: Preprocessed POE data
        """
        if 'poe' not in self.data or self.data['poe'] is None:
            return None

        df = self.data['poe'].copy()

        # Convert date columns to datetime
        datetime_cols = ['ordertime', 'starttime', 'stoptime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Extract order hour for time-based analysis
        if 'ordertime' in df.columns:
            df['order_hour'] = df['ordertime'].dt.hour
            df['order_dow'] = df['ordertime'].dt.dayofweek

        # Calculate order duration in hours
        if 'starttime' in df.columns and 'stoptime' in df.columns:
            df['order_duration_hours'] = (df['stoptime'] - df['starttime']).dt.total_seconds() / (60 * 60)
            # Remove negative durations (data errors)
            df['order_duration_hours'] = df['order_duration_hours'].clip(lower=0)

        self.preprocessed['poe'] = df
        return df

    def preprocess_poe_detail(self):
        """Preprocess provider order entry detail data.

        Returns:
            pandas.DataFrame: Preprocessed POE detail data
        """
        if 'poe_detail' not in self.data or self.data['poe_detail'] is None:
            return None

        df = self.data['poe_detail'].copy()

        # Process field values based on field name
        if 'field_name' in df.columns and 'field_value' in df.columns:
            # Example: Extract numeric values from dosage fields
            dosage_mask = df['field_name'].str.contains('dose', case=False, na=False)
            df.loc[dosage_mask, 'numeric_value'] = df.loc[dosage_mask, 'field_value'].apply(
                lambda x: float(re.search(r'\d+\.?\d*', str(x)).group()) if isinstance(x, str) and re.search(r'\d+\.?\d*', str(x)) else None
            )

        self.preprocessed['poe_detail'] = df
        return df

    def preprocess_all_data(self):
        """Preprocess all loaded data.

        Returns:
            dict: Dictionary containing all preprocessed dataframes
        """
        if not self.data:
            st.warning("No data loaded. Please load data first.")
            return {}

        with st.spinner("Preprocessing MIMIC-IV data..."):
            self.preprocess_patients()
            self.preprocess_admissions()
            self.preprocess_transfers()
            self.preprocess_poe()
            self.preprocess_poe_detail()

        return self.preprocessed

    def get_patient_cohort(self, filters=None):
        """Get a filtered patient cohort based on specified criteria.

        Args:
            filters (dict): Dictionary of filter criteria
                Example: {
                    'gender': 'M',
                    'age_min': 18,
                    'age_max': 65,
                    'admission_type': 'EMERGENCY',
                    'los_min': 1,
                    'los_max': 30
                }

        Returns:
            tuple: (patient_ids, admission_ids) - Lists of filtered IDs
        """
        if not filters:
            filters = {}

        # Start with all patients
        if 'patients' in self.preprocessed and self.preprocessed['patients'] is not None:
            patients_df = self.preprocessed['patients']
            patient_ids = set(patients_df['subject_id'])
        else:
            st.warning("Patient data not available for cohort filtering.")
            return [], []

        # Filter by gender
        if 'gender' in filters and filters['gender'] and 'gender' in patients_df.columns:
            gender_mask = patients_df['gender'] == filters['gender']
            patient_ids = patient_ids.intersection(set(patients_df.loc[gender_mask, 'subject_id']))

        # Filter by age
        if 'age' in patients_df.columns:
            if 'age_min' in filters and filters['age_min'] is not None:
                age_min_mask = patients_df['age'] >= filters['age_min']
                patient_ids = patient_ids.intersection(set(patients_df.loc[age_min_mask, 'subject_id']))

            if 'age_max' in filters and filters['age_max'] is not None:
                age_max_mask = patients_df['age'] <= filters['age_max']
                patient_ids = patient_ids.intersection(set(patients_df.loc[age_max_mask, 'subject_id']))

        # Get admissions for filtered patients
        if 'admissions' in self.preprocessed and self.preprocessed['admissions'] is not None:
            admissions_df = self.preprocessed['admissions']
            admission_ids = set(admissions_df[admissions_df['subject_id'].isin(patient_ids)]['hadm_id'])
        else:
            st.warning("Admission data not available for cohort filtering.")
            return list(patient_ids), []

        # Filter by admission type
        if 'admission_type' in filters and filters['admission_type'] and 'admission_type' in admissions_df.columns:
            adm_type_mask = admissions_df['admission_type'] == filters['admission_type']
            admission_ids = admission_ids.intersection(set(admissions_df.loc[adm_type_mask, 'hadm_id']))

        # Filter by length of stay
        if 'los_days' in admissions_df.columns:
            if 'los_min' in filters and filters['los_min'] is not None:
                los_min_mask = admissions_df['los_days'] >= filters['los_min']
                admission_ids = admission_ids.intersection(set(admissions_df.loc[los_min_mask, 'hadm_id']))

            if 'los_max' in filters and filters['los_max'] is not None:
                los_max_mask = admissions_df['los_days'] <= filters['los_max']
                admission_ids = admission_ids.intersection(set(admissions_df.loc[los_max_mask, 'hadm_id']))

        # Update patient_ids based on filtered admissions
        patient_ids = set(admissions_df[admissions_df['hadm_id'].isin(admission_ids)]['subject_id'])

        return list(patient_ids), list(admission_ids)

    def get_patient_orders(self, subject_id, hadm_id=None):
        """Get all orders for a specific patient, optionally filtered by admission.

        Args:
            subject_id (int): Patient subject ID
            hadm_id (int, optional): Hospital admission ID

        Returns:
            pandas.DataFrame: Filtered orders for the patient
        """
        if 'poe' not in self.preprocessed or self.preprocessed['poe'] is None:
            st.warning("POE data not available.")
            return None

        poe_df = self.preprocessed['poe']

        # Filter by subject_id
        patient_orders = poe_df[poe_df['subject_id'] == subject_id].copy()

        # Further filter by hadm_id if provided
        if hadm_id is not None:
            patient_orders = patient_orders[patient_orders['hadm_id'] == hadm_id]

        # Sort by ordertime
        if 'ordertime' in patient_orders.columns:
            patient_orders = patient_orders.sort_values('ordertime')

        return patient_orders

    def get_patient_transfers(self, subject_id, hadm_id=None):
        """Get all transfers for a specific patient, optionally filtered by admission.

        Args:
            subject_id (int): Patient subject ID
            hadm_id (int, optional): Hospital admission ID

        Returns:
            pandas.DataFrame: Filtered transfers for the patient
        """
        if 'transfers' not in self.preprocessed or self.preprocessed['transfers'] is None:
            st.warning("Transfers data not available.")
            return None

        transfers_df = self.preprocessed['transfers']

        # Filter by subject_id
        patient_transfers = transfers_df[transfers_df['subject_id'] == subject_id].copy()

        # Further filter by hadm_id if provided
        if hadm_id is not None:
            patient_transfers = patient_transfers[patient_transfers['hadm_id'] == hadm_id]

        # Sort by intime
        if 'intime' in patient_transfers.columns:
            patient_transfers = patient_transfers.sort_values('intime')

        return patient_transfers

    def get_order_details(self, order_id):
        """Get details for a specific order.

        Args:
            order_id (int): Order ID

        Returns:
            pandas.DataFrame: Order details
        """
        if 'poe_detail' not in self.preprocessed or self.preprocessed['poe_detail'] is None:
            st.warning("POE detail data not available.")
            return None

        poe_detail_df = self.preprocessed['poe_detail']

        # Filter by poe_id
        order_details = poe_detail_df[poe_detail_df['poe_id'] == order_id].copy()

        return order_details

    def get_medication_orders(self, subject_id=None, hadm_id=None):
        """Get medication orders, optionally filtered by patient and/or admission.

        Args:
            subject_id (int, optional): Patient subject ID
            hadm_id (int, optional): Hospital admission ID

        Returns:
            pandas.DataFrame: Filtered medication orders
        """
        if 'poe' not in self.preprocessed or self.preprocessed['poe'] is None:
            st.warning("POE data not available.")
            return None

        poe_df = self.preprocessed['poe']

        # Filter for medication orders
        if 'order_type' in poe_df.columns:
            med_orders = poe_df[poe_df['order_type'] == 'Medications'].copy()
        else:
            st.warning("Order type information not available.")
            return None

        # Filter by subject_id if provided
        if subject_id is not None:
            med_orders = med_orders[med_orders['subject_id'] == subject_id]

        # Filter by hadm_id if provided
        if hadm_id is not None:
            med_orders = med_orders[med_orders['hadm_id'] == hadm_id]

        # Sort by ordertime
        if 'ordertime' in med_orders.columns:
            med_orders = med_orders.sort_values('ordertime')

        return med_orders

    def get_iv_to_oral_transitions(self, subject_id=None, hadm_id=None):
        """Identify IV to oral medication transitions.

        Args:
            subject_id (int, optional): Patient subject ID
            hadm_id (int, optional): Hospital admission ID

        Returns:
            pandas.DataFrame: Identified IV to oral transitions
        """
        # This is a placeholder for the IV to oral transition logic
        # The actual implementation would depend on how medications are coded in the dataset
        st.info("IV to oral transition analysis is a placeholder. Implementation depends on medication coding in the dataset.")
        return None

    def get_discharge_orders(self, subject_id=None, hadm_id=None):
        """Get discharge-related orders, optionally filtered by patient and/or admission.

        Args:
            subject_id (int, optional): Patient subject ID
            hadm_id (int, optional): Hospital admission ID

        Returns:
            pandas.DataFrame: Filtered discharge orders
        """
        if 'poe' not in self.preprocessed or self.preprocessed['poe'] is None:
            st.warning("POE data not available.")
            return None

        poe_df = self.preprocessed['poe']

        # Filter for discharge orders
        # This is a placeholder - actual implementation depends on how discharge orders are coded
        discharge_keywords = ['discharge', 'disch', 'dc']

        if 'order_type' in poe_df.columns and 'order_subtype' in poe_df.columns:
            discharge_mask = (
                poe_df['order_type'].str.contains('|'.join(discharge_keywords), case=False, na=False) |
                poe_df['order_subtype'].str.contains('|'.join(discharge_keywords), case=False, na=False)
            )
            discharge_orders = poe_df[discharge_mask].copy()
        else:
            st.warning("Order type information not available.")
            return None

        # Filter by subject_id if provided
        if subject_id is not None:
            discharge_orders = discharge_orders[discharge_orders['subject_id'] == subject_id]

        # Filter by hadm_id if provided
        if hadm_id is not None:
            discharge_orders = discharge_orders[discharge_orders['hadm_id'] == hadm_id]

        # Sort by ordertime
        if 'ordertime' in discharge_orders.columns:
            discharge_orders = discharge_orders.sort_values('ordertime')

        return discharge_orders


class HuggingFaceMIMICLoader:
    """Class for loading and processing MIMIC-IV dataset using Hugging Face datasets library.

    This class provides a comprehensive interface to load the MIMIC-IV dataset using the
    Hugging Face datasets library. It supports various capabilities including loading different
    tasks, converting to parquet format, and efficiently handling large CSV datasets.

    The loader utilizes the 'thbndi/Mimic4Dataset' dataset from Hugging Face which provides
    structured access to the MIMIC-IV database with various preprocessing options.
    """

    def __init__(self,
                 mimic_path: str,
                 config_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """Initialize the Hugging Face MIMIC-IV data loader.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory
            config_path (str, optional): Path to the configuration file for dataset loading
            cache_dir (str, optional): Directory to cache the dataset files
        """
        self.mimic_path = mimic_path
        self.config_path = config_path
        self.cache_dir = cache_dir
        self.datasets = {}
        self.dask_dfs = {}

        # Create parquet directory if it doesn't exist
        self.parquet_dir = os.path.join(os.path.dirname(mimic_path), 'parquet_files')
        os.makedirs(self.parquet_dir, exist_ok=True)

        # Available tasks in the MIMIC-IV dataset
        self.available_tasks = [
            'core', 'hosp', 'icu', 'ed', 'note',
            'mortality_prediction', 'length_of_stay_prediction', 'readmission_prediction'
        ]

    def load_dataset(self,
                    task           : str,
                    encoding       : str = 'latin-1',
                    generate_cohort: bool = False,
                    val_size       : float = 0.2,
                    split          : Optional[str] = None,
                    streaming      : bool = False) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """Load a specific task dataset from MIMIC-IV using Hugging Face datasets.

        Args:
            task (str): The task/module to load ('core', 'hosp', 'icu', 'ed', 'note', etc.)
            encoding (str, optional): Character encoding for the dataset files. Defaults to 'latin-1'.
            generate_cohort (bool, optional): Whether to generate a cohort for prediction tasks. Defaults to False.
            val_size (float, optional): Validation set size for prediction tasks. Defaults to 0.2.
            split (str, optional): Specific split to load ('train', 'test', 'validation'). Defaults to None.
            streaming (bool, optional): Whether to stream the dataset (for very large datasets). Defaults to False.

        Returns:
            Union[datasets.Dataset, datasets.DatasetDict]: The loaded dataset or dataset dictionary

        Raises:
            ValueError: If the specified task is not available
        """
        if task not in self.available_tasks:
            raise ValueError(f"Task '{task}' not available. Available tasks: {', '.join(self.available_tasks)}")

        try:
            # Load the dataset using Hugging Face datasets
            dataset = load_dataset(
                'thbndi/Mimic4Dataset',
                task,
                mimic_path      = self.mimic_path,
                config_path     = self.config_path,
                encoding        = encoding,
                generate_cohort = generate_cohort,
                val_size        = val_size,
                cache_dir       = self.cache_dir,
                streaming       = streaming
            )

            # Store the dataset for later use
            self.datasets[task] = dataset

            # Return the specific split if requested
            if split and not streaming:
                if split in dataset:
                    return dataset[split]
                else:
                    warnings.warn(f"Split '{split}' not found in dataset. Returning full dataset.")

            return dataset

        except Exception as e:
            st.error(f"Error loading dataset for task '{task}': {str(e)}")
            raise

    def convert_to_parquet(self,
                          task: str,
                          split: Optional[str] = None,
                          columns: Optional[List[str]] = None,
                          batch_size: int = 10000) -> str:
        """Convert a loaded dataset to Parquet format for efficient storage and querying.

        Args:
            task (str): The task/module to convert
            split (str, optional): Specific split to convert. Defaults to None (all splits).
            columns (List[str], optional): Specific columns to include. Defaults to None (all columns).
            batch_size (int, optional): Batch size for processing large datasets. Defaults to 10000.

        Returns:
            str: Path to the directory containing the Parquet files

        Raises:
            ValueError: If the dataset for the specified task is not loaded
        """
        if task not in self.datasets:
            raise ValueError(f"Dataset for task '{task}' not loaded. Call load_dataset() first.")

        dataset = self.datasets[task]
        task_parquet_dir = os.path.join(self.parquet_dir, task)
        os.makedirs(task_parquet_dir, exist_ok=True)

        try:
            if isinstance(dataset, datasets.DatasetDict):
                splits_to_convert = [split] if split else dataset.keys()

                for split_name in splits_to_convert:
                    if split_name not in dataset:
                        warnings.warn(f"Split '{split_name}' not found in dataset. Skipping.")
                        continue

                    split_dataset = dataset[split_name]
                    split_parquet_path = os.path.join(task_parquet_dir, f"{split_name}.parquet")

                    # Convert to pandas in batches to handle large datasets
                    for i in range(0, len(split_dataset), batch_size):
                        batch = split_dataset.select(range(i, min(i + batch_size, len(split_dataset))))
                        df_batch = batch.to_pandas()

                        # Select specific columns if requested
                        if columns:
                            available_cols = [col for col in columns if col in df_batch.columns]
                            if len(available_cols) < len(columns):
                                missing_cols = set(columns) - set(available_cols)
                                warnings.warn(f"Columns {missing_cols} not found in dataset.")
                            df_batch = df_batch[available_cols]

                        # Write to Parquet (append mode for batches after the first)
                        write_mode = 'overwrite' if i == 0 else 'append'
                        table = pa.Table.from_pandas(df_batch)
                        pq.write_to_dataset(table, task_parquet_dir, partition_cols=[split_name])

                    st.success(f"Converted {split_name} split to Parquet format at {split_parquet_path}")
            else:
                # Handle single dataset (no splits)
                parquet_path = os.path.join(task_parquet_dir, f"{task}.parquet")

                # Convert to pandas in batches to handle large datasets
                for i in range(0, len(dataset), batch_size):
                    batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
                    df_batch = batch.to_pandas()

                    # Select specific columns if requested
                    if columns:
                        available_cols = [col for col in columns if col in df_batch.columns]
                        if len(available_cols) < len(columns):
                            missing_cols = set(columns) - set(available_cols)
                            warnings.warn(f"Columns {missing_cols} not found in dataset.")
                        df_batch = df_batch[available_cols]

                    # Write to Parquet (append mode for batches after the first)
                    write_mode = 'overwrite' if i == 0 else 'append'
                    table = pa.Table.from_pandas(df_batch)

                    if i == 0:
                        pq.write_table(table, parquet_path)
                    else:
                        # Read existing table and append
                        existing_table = pq.read_table(parquet_path)
                        combined_table = pa.concat_tables([existing_table, table])
                        pq.write_table(combined_table, parquet_path)

                st.success(f"Converted dataset to Parquet format at {parquet_path}")

            return task_parquet_dir

        except Exception as e:
            st.error(f"Error converting dataset to Parquet: {str(e)}")
            raise

    def load_as_dask(self,
                    task: str,
                    split: Optional[str] = None,
                    columns: Optional[List[str]] = None) -> Dict[str, dd.DataFrame]:
        """Load a dataset as Dask DataFrame for efficient processing of large datasets.

        Args:
            task (str): The task/module to load
            split (str, optional): Specific split to load. Defaults to None (all splits).
            columns (List[str], optional): Specific columns to include. Defaults to None (all columns).

        Returns:
            Dict[str, dd.DataFrame]: Dictionary of Dask DataFrames by split

        Raises:
            ValueError: If the dataset for the specified task is not loaded
        """
        if task not in self.datasets:
            raise ValueError(f"Dataset for task '{task}' not loaded. Call load_dataset() first.")

        dataset = self.datasets[task]
        result = {}

        try:
            if isinstance(dataset, datasets.DatasetDict):
                splits_to_load = [split] if split else dataset.keys()

                for split_name in splits_to_load:
                    if split_name not in dataset:
                        warnings.warn(f"Split '{split_name}' not found in dataset. Skipping.")
                        continue

                    # Convert to pandas first (this will load the entire dataset into memory)
                    # For very large datasets, it's better to convert to parquet first and then load with Dask
                    df = dataset[split_name].to_pandas()

                    # Select specific columns if requested
                    if columns:
                        available_cols = [col for col in columns if col in df.columns]
                        if len(available_cols) < len(columns):
                            missing_cols = set(columns) - set(available_cols)
                            warnings.warn(f"Columns {missing_cols} not found in dataset.")
                        df = df[available_cols]

                    # Convert to Dask DataFrame
                    dask_df = dd.from_pandas(df, npartitions=max(1, len(df) // 100000))
                    result[split_name] = dask_df

                    # Store for later use
                    if task not in self.dask_dfs:
                        self.dask_dfs[task] = {}
                    self.dask_dfs[task][split_name] = dask_df
            else:
                # Handle single dataset (no splits)
                df = dataset.to_pandas()

                # Select specific columns if requested
                if columns:
                    available_cols = [col for col in columns if col in df.columns]
                    if len(available_cols) < len(columns):
                        missing_cols = set(columns) - set(available_cols)
                        warnings.warn(f"Columns {missing_cols} not found in dataset.")
                    df = df[available_cols]

                # Convert to Dask DataFrame
                dask_df = dd.from_pandas(df, npartitions=max(1, len(df) // 100000))
                result['default'] = dask_df

                # Store for later use
                self.dask_dfs[task] = {'default': dask_df}

            return result

        except Exception as e:
            st.error(f"Error loading dataset as Dask DataFrame: {str(e)}")
            raise

    def load_from_parquet(self,
                         task: str,
                         split: Optional[str] = None,
                         columns: Optional[List[str]] = None,
                         use_dask: bool = False) -> Union[Dict[str, pd.DataFrame], Dict[str, dd.DataFrame]]:
        """Load a dataset from Parquet files for efficient querying.

        Args:
            task (str): The task/module to load
            split (str, optional): Specific split to load. Defaults to None (all splits).
            columns (List[str], optional): Specific columns to include. Defaults to None (all columns).
            use_dask (bool, optional): Whether to load as Dask DataFrame. Defaults to False.

        Returns:
            Union[Dict[str, pd.DataFrame], Dict[str, dd.DataFrame]]: Dictionary of DataFrames by split

        Raises:
            FileNotFoundError: If the Parquet files for the specified task are not found
        """
        task_parquet_dir = os.path.join(self.parquet_dir, task)

        if not os.path.exists(task_parquet_dir):
            raise FileNotFoundError(f"Parquet directory for task '{task}' not found. Convert to Parquet first.")

        result = {}

        try:
            # Check if we have split-specific files or a single file
            parquet_files = [f for f in os.listdir(task_parquet_dir) if f.endswith('.parquet')]

            if split:
                # Load specific split
                split_file = f"{split}.parquet"
                if split_file in parquet_files:
                    parquet_path = os.path.join(task_parquet_dir, split_file)
                    if use_dask:
                        df = dd.read_parquet(parquet_path, columns=columns)
                    else:
                        df = pd.read_parquet(parquet_path, columns=columns)
                    result[split] = df
                else:
                    warnings.warn(f"Parquet file for split '{split}' not found.")
            else:
                # Load all available splits or the single file
                for parquet_file in parquet_files:
                    split_name = parquet_file.replace('.parquet', '')
                    parquet_path = os.path.join(task_parquet_dir, parquet_file)
                    if use_dask:
                        df = dd.read_parquet(parquet_path, columns=columns)
                    else:
                        df = pd.read_parquet(parquet_path, columns=columns)
                    result[split_name] = df

            return result

        except Exception as e:
            st.error(f"Error loading dataset from Parquet: {str(e)}")
            raise

    def get_dataset_info(self, task: str) -> Dict[str, Any]:
        """Get information about a loaded dataset.

        Args:
            task (str): The task/module to get information for

        Returns:
            Dict[str, Any]: Dictionary containing dataset information

        Raises:
            ValueError: If the dataset for the specified task is not loaded
        """
        if task not in self.datasets:
            raise ValueError(f"Dataset for task '{task}' not loaded. Call load_dataset() first.")

        dataset = self.datasets[task]
        info = {}

        try:
            if isinstance(dataset, datasets.DatasetDict):
                info['splits'] = {}
                for split_name, split_dataset in dataset.items():
                    info['splits'][split_name] = {
                        'num_rows': len(split_dataset),
                        'features': list(split_dataset.features.keys()),
                        'feature_types': {k: str(v) for k, v in split_dataset.features.items()}
                    }

                    # Get a sample row if available
                    if len(split_dataset) > 0:
                        info['splits'][split_name]['sample'] = split_dataset[0]
            else:
                info['num_rows'] = len(dataset)
                info['features'] = list(dataset.features.keys())
                info['feature_types'] = {k: str(v) for k, v in dataset.features.items()}

                # Get a sample row if available
                if len(dataset) > 0:
                    info['sample'] = dataset[0]

            return info

        except Exception as e:
            st.error(f"Error getting dataset information: {str(e)}")
            raise

    def search_dataset(self,
                      task: str,
                      query: Dict[str, Any],
                      split: Optional[str] = None,
                      max_results: int = 100) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Search a dataset for rows matching the query criteria.

        Args:
            task (str): The task/module to search
            query (Dict[str, Any]): Dictionary of column-value pairs to match
            split (str, optional): Specific split to search. Defaults to None (all splits).
            max_results (int, optional): Maximum number of results to return. Defaults to 100.

        Returns:
            Union[pd.DataFrame, Dict[str, pd.DataFrame]]: Search results as DataFrame(s)

        Raises:
            ValueError: If the dataset for the specified task is not loaded
        """
        if task not in self.datasets:
            raise ValueError(f"Dataset for task '{task}' not loaded. Call load_dataset() first.")

        dataset = self.datasets[task]
        result = {}

        try:
            if isinstance(dataset, datasets.DatasetDict):
                splits_to_search = [split] if split else dataset.keys()

                for split_name in splits_to_search:
                    if split_name not in dataset:
                        warnings.warn(f"Split '{split_name}' not found in dataset. Skipping.")
                        continue

                    # Convert to pandas for easier filtering
                    df = dataset[split_name].to_pandas()

                    # Apply filters
                    for column, value in query.items():
                        if column in df.columns:
                            df = df[df[column] == value]

                    # Limit results
                    df = df.head(max_results)
                    result[split_name] = df
            else:
                # Handle single dataset (no splits)
                df = dataset.to_pandas()

                # Apply filters
                for column, value in query.items():
                    if column in df.columns:
                        df = df[df[column] == value]

                # Limit results
                df = df.head(max_results)
                result['default'] = df

            # Return a single DataFrame if only one split was searched
            if len(result) == 1 and split:
                return result[split]

            return result

        except Exception as e:
            st.error(f"Error searching dataset: {str(e)}")
            raise

    def get_prediction_dataset(self,
                             prediction_task: str,
                             generate_cohort: bool = True,
                             val_size: float = 0.2) -> datasets.DatasetDict:
        """Load a specific prediction task dataset from MIMIC-IV.

        Args:
            prediction_task (str): The prediction task to load ('mortality_prediction',
                                   'length_of_stay_prediction', 'readmission_prediction')
            generate_cohort (bool, optional): Whether to generate a cohort. Defaults to True.
            val_size (float, optional): Validation set size. Defaults to 0.2.

        Returns:
            datasets.DatasetDict: The loaded prediction dataset with train/validation/test splits

        Raises:
            ValueError: If the specified prediction task is not available
        """
        prediction_tasks = ['mortality_prediction', 'length_of_stay_prediction', 'readmission_prediction']

        if prediction_task not in prediction_tasks:
            raise ValueError(f"Prediction task '{prediction_task}' not available. Available tasks: {', '.join(prediction_tasks)}")

        try:
            # Load the prediction dataset
            dataset = self.load_dataset(
                prediction_task,
                generate_cohort=generate_cohort,
                val_size=val_size
            )

            return dataset

        except Exception as e:
            st.error(f"Error loading prediction dataset: {str(e)}")
            raise

    def get_feature_statistics(self,
                             task: str,
                             columns: Optional[List[str]] = None,
                             split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get statistical information about dataset features.

        Args:
            task (str): The task/module to analyze
            columns (List[str], optional): Specific columns to analyze. Defaults to None (all columns).
            split (str, optional): Specific split to analyze. Defaults to None (first available split).

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of feature statistics

        Raises:
            ValueError: If the dataset for the specified task is not loaded
        """
        if task not in self.datasets:
            raise ValueError(f"Dataset for task '{task}' not loaded. Call load_dataset() first.")

        dataset = self.datasets[task]

        try:
            # Get the dataset to analyze
            if isinstance(dataset, datasets.DatasetDict):
                if split and split in dataset:
                    df = dataset[split].to_pandas()
                else:
                    # Use the first available split
                    first_split = next(iter(dataset.keys()))
                    df = dataset[first_split].to_pandas()
                    if split:
                        warnings.warn(f"Split '{split}' not found. Using '{first_split}' instead.")
            else:
                df = dataset.to_pandas()

            # Select columns to analyze
            if columns:
                available_cols = [col for col in columns if col in df.columns]
                if len(available_cols) < len(columns):
                    missing_cols = set(columns) - set(available_cols)
                    warnings.warn(f"Columns {missing_cols} not found in dataset.")
                df = df[available_cols]

            # Calculate statistics for each column
            stats = {}
            for column in df.columns:
                col_stats = {}

                # Get data type
                col_stats['dtype'] = str(df[column].dtype)

                # Count missing values
                col_stats['missing_count'] = df[column].isna().sum()
                col_stats['missing_percentage'] = (col_stats['missing_count'] / len(df)) * 100

                # Get unique values count
                col_stats['unique_count'] = df[column].nunique()

                # For numeric columns, get descriptive statistics
                if np.issubdtype(df[column].dtype, np.number):
                    col_stats['min'] = df[column].min()
                    col_stats['max'] = df[column].max()
                    col_stats['mean'] = df[column].mean()
                    col_stats['median'] = df[column].median()
                    col_stats['std'] = df[column].std()

                    # Get quantiles
                    col_stats['quantiles'] = {
                        '25%': df[column].quantile(0.25),
                        '50%': df[column].quantile(0.5),
                        '75%': df[column].quantile(0.75)
                    }

                # For categorical/string columns, get value counts
                elif df[column].dtype == 'object' or df[column].dtype == 'string' or df[column].dtype.name == 'category':
                    if col_stats['unique_count'] <= 20:  # Only for columns with reasonable number of categories
                        value_counts = df[column].value_counts().head(10).to_dict()
                        col_stats['top_values'] = value_counts

                # For datetime columns
                elif np.issubdtype(df[column].dtype, np.datetime64):
                    col_stats['min'] = df[column].min()
                    col_stats['max'] = df[column].max()
                    col_stats['range_days'] = (df[column].max() - df[column].min()).days

                stats[column] = col_stats

            return stats

        except Exception as e:
            st.error(f"Error calculating feature statistics: {str(e)}")
            raise
