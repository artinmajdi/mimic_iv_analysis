import pandas as pd
import numpy as np
import os
import streamlit as st
import datetime
import re
from pathlib import Path


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
        return True

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_patients(self):
        """Load patients data from MIMIC-IV.

        Returns:
            pandas.DataFrame: Patients data or None if file not found
        """
        try:
            file_path = os.path.join(self.mimic_path, 'hosp', 'patients.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                self.data['patients'] = df
                return df
            else:
                st.error(f"File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"Error loading patient data: {e}")
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_admissions(self):
        """Load admissions data from MIMIC-IV.

        Returns:
            pandas.DataFrame: Admissions data or None if file not found
        """
        try:
            file_path = os.path.join(self.mimic_path, 'hosp', 'admissions.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                self.data['admissions'] = df
                return df
            else:
                st.error(f"File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"Error loading admissions data: {e}")
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_transfers(self):
        """Load transfers data from MIMIC-IV.

        Returns:
            pandas.DataFrame: Transfers data or None if file not found
        """
        try:
            file_path = os.path.join(self.mimic_path, 'hosp', 'transfers.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                self.data['transfers'] = df
                return df
            else:
                st.error(f"File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"Error loading transfers data: {e}")
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_poe(self):
        """Load provider order entry (POE) data from MIMIC-IV.

        Returns:
            pandas.DataFrame: POE data or None if file not found
        """
        try:
            file_path = os.path.join(self.mimic_path, 'hosp', 'poe.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                self.data['poe'] = df
                return df
            else:
                st.error(f"File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"Error loading POE data: {e}")
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_poe_detail(self):
        """Load provider order entry detail data from MIMIC-IV.

        Returns:
            pandas.DataFrame: POE detail data or None if file not found
        """
        try:
            file_path = os.path.join(self.mimic_path, 'hosp', 'poe_detail.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                self.data['poe_detail'] = df
                return df
            else:
                st.error(f"File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"Error loading POE detail data: {e}")
            return None

    def load_all_data(self):
        """Load all required MIMIC-IV data files.

        Returns:
            dict: Dictionary containing all loaded dataframes
        """
        if not self.validate_path():
            return {}

        with st.spinner("Loading MIMIC-IV data..."):
            patients_df = self.load_patients()
            admissions_df = self.load_admissions()
            transfers_df = self.load_transfers()
            poe_df = self.load_poe()
            poe_detail_df = self.load_poe_detail()

            if all([patients_df is not None, admissions_df is not None, transfers_df is not None,
                    poe_df is not None, poe_detail_df is not None]):
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
