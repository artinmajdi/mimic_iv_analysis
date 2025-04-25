"""
Filtering module for MIMIC-IV data.

This module provides functionality for filtering MIMIC-IV data based on
inclusion and exclusion criteria from the MIMIC-IV dataset tables.
"""

# Standard library imports
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set

# Data processing imports
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for MIMIC-IV tables and columns
MIMIC_TABLES = {
    'patients': {
        'id_cols': ['subject_id'],
        'filter_cols': ['anchor_age', 'anchor_year', 'anchor_year_group', 'gender', 'dod']
    },
    'admissions': {
        'id_cols': ['subject_id', 'hadm_id'],
        'filter_cols': ['admittime', 'dischtime', 'deathtime', 'admission_type',
                       'admission_location', 'discharge_location', 'insurance',
                       'language', 'marital_status', 'race', 'hospital_expire_flag']
    },
    'transfers': {
        'id_cols': ['subject_id', 'hadm_id', 'transfer_id'],
        'filter_cols': ['eventtype', 'careunit', 'intime', 'outtime']
    },
    'diagnoses_icd': {
        'id_cols': ['subject_id', 'hadm_id'],
        'filter_cols': ['seq_num', 'icd_code', 'icd_version']
    }
}

# Known careunit values in MIMIC-IV
CARE_UNITS = [
    'MICU', 'SICU', 'CSRU', 'CCU', 'TSICU', 'NICU',  # ICU units
    'Med', 'Surg', 'Trauma', 'Cardiac', 'Neuro',     # Specialty units
    'Obstetrics', 'Psych', 'Rehab'                   # Other units
]

# Known admission types in MIMIC-IV
ADMISSION_TYPES = ['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION']

class Filtering:
    """
    Class for applying inclusion and exclusion filters to MIMIC-IV data.

    This class provides methods to filter pandas DataFrames containing MIMIC-IV data
    based on various inclusion and exclusion criteria from the MIMIC-IV dataset tables.
    It handles the relationships between different tables and applies filters efficiently.
    """

    def __init__(self):
        """Initialize the Filtering class."""
        logging.info("Initializing Filtering class...")
        # Track applied filters for reporting
        self.applied_filters = []
        # Cache for patient and admission IDs to avoid redundant filtering
        self.filtered_subject_ids = None
        self.filtered_hadm_ids = None

    def apply_filters(self, df: pd.DataFrame, filter_params: Dict[str, Any], patients_df: Optional[pd.DataFrame] = None, admissions_df: Optional[pd.DataFrame] = None, diagnoses_df: Optional[pd.DataFrame] = None, transfers_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply all filters based on the provided parameters.

        Args:
            df: DataFrame to filter
            filter_params: Dictionary of filter parameters
            patients_df: Optional DataFrame containing patient information
            admissions_df: Optional DataFrame containing admission information
            diagnoses_df: Optional DataFrame containing diagnosis information
            transfers_df: Optional DataFrame containing transfer information

        Returns:
            Filtered DataFrame and a list of applied filters
        """
        logging.info("Applying filters with parameters: %s", filter_params)

        # Reset applied filters list and cached IDs
        self.applied_filters = []
        self.filtered_subject_ids = None
        self.filtered_hadm_ids = None

        # Make a copy of the DataFrame to avoid modifying the original
        filtered_df = df.copy()

        # Track original row count for reporting
        original_row_count = len(filtered_df)

        # Validate DataFrames
        if patients_df is not None:
            self._validate_dataframe(patients_df, 'patients')
        if admissions_df is not None:
            self._validate_dataframe(admissions_df, 'admissions')
        if diagnoses_df is not None:
            self._validate_dataframe(diagnoses_df, 'diagnoses_icd')
        if transfers_df is not None:
            self._validate_dataframe(transfers_df, 'transfers')

        # Apply filters in an optimized order (demographic filters first, then clinical filters)

        # 1. Demographic filters (patients table)
        if patients_df is not None:
            # Filter by timeframe (anchor_year_group)
            if filter_params.get('timeframe'):
                filtered_df = self.filter_by_encounter_timeframe(filtered_df, filter_params['timeframe'], patients_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after timeframe filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

            # Filter by age range
            if filter_params.get('min_age') is not None and filter_params.get('max_age') is not None:
                filtered_df = self.filter_by_age_range(filtered_df, filter_params['min_age'], filter_params['max_age'], patients_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after age range filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

            # Filter by gender
            if filter_params.get('genders'):
                filtered_df = self.filter_by_gender(filtered_df, filter_params['genders'], patients_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after gender filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

        # 2. Clinical filters (diagnoses table)
        if diagnoses_df is not None:
            # Filter by T2DM diagnosis
            if filter_params.get('include_t2dm', False):
                filtered_df = self.filter_by_t2dm_diagnosis(filtered_df, diagnoses_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after T2DM diagnosis filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

            # Filter by custom ICD codes
            if filter_params.get('icd_codes') and filter_params.get('icd_version') and filter_params.get('seq_nums'):
                filtered_df = self.filter_by_custom_icd(
                    filtered_df,
                    filter_params['icd_codes'],
                    filter_params['icd_version'],
                    filter_params['seq_nums'],
                    diagnoses_df
                )
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after custom ICD filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

        # 3. Admission filters (admissions table)
        if admissions_df is not None:
            # Filter by valid admission and discharge times
            if filter_params.get('valid_admission_discharge', False):
                filtered_df = self.filter_by_valid_admission_discharge(filtered_df, admissions_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after valid admission/discharge filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

            # Filter by length of stay
            if filter_params.get('min_los') is not None and filter_params.get('max_los') is not None:
                filtered_df = self.filter_by_length_of_stay(
                    filtered_df,
                    filter_params['min_los'],
                    filter_params['max_los'],
                    admissions_df
                )
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after length of stay filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

            # Filter by inpatient stay
            if filter_params.get('inpatient_only', False):
                filtered_df = self.filter_by_inpatient_stay(filtered_df, admissions_df, transfers_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after inpatient stay filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

            # Filter to exclude in-hospital deaths
            if filter_params.get('exclude_death', False):
                filtered_df = self.filter_by_exclude_death(filtered_df, admissions_df)
                if filtered_df.empty:
                    logging.warning("DataFrame is empty after exclude death filter. Returning empty DataFrame.")
                    return filtered_df, self.applied_filters

        # Calculate reduction percentage
        reduction_pct = ((original_row_count - len(filtered_df)) / original_row_count * 100) if original_row_count > 0 else 0

        # Log the number of rows after filtering
        logging.info(f"Filtered DataFrame has {len(filtered_df)} rows (original had {original_row_count} rows, {reduction_pct:.1f}% reduction)")
        logging.info(f"Applied filters: {self.applied_filters}")

        return filtered_df, self.applied_filters

    def filter_by_valid_admission_discharge(self, df: pd.DataFrame, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to include only admissions with valid admittime and dischtime.

        Args:
            df: DataFrame to filter
            admissions_df: DataFrame containing admission information

        Returns:
            Filtered DataFrame
        """
        logging.info("Filtering by valid admission and discharge times")
        self.applied_filters.append("Valid admission and discharge times")

        if 'admittime' not in admissions_df.columns or 'dischtime' not in admissions_df.columns:
            logging.warning("admittime or dischtime columns not found in admissions DataFrame. Skipping filter.")
            return df

        # Get admission IDs with valid admission and discharge times
        valid_admissions = admissions_df.dropna(subset=['admittime', 'dischtime'])

        if valid_admissions.empty:
            logging.warning("No admissions found with valid admission and discharge times")
            # Return empty DataFrame with same columns
            return df.iloc[0:0]

        # Ensure datetime format for admittime and dischtime
        for col in ['admittime', 'dischtime']:
            if not is_datetime64_any_dtype(valid_admissions[col]):
                try:
                    valid_admissions[col] = pd.to_datetime(valid_admissions[col])
                except Exception as e:
                    logging.error(f"Could not convert {col} to datetime: {str(e)}")
                    return df

        # Additional validation: dischtime should be after admittime
        valid_admissions = valid_admissions[valid_admissions['dischtime'] > valid_admissions['admittime']]

        if valid_admissions.empty:
            logging.warning("No admissions found with valid admission and discharge times (discharge after admission)")
            return df.iloc[0:0]

        # Store filtered admission IDs for potential reuse
        admission_ids = valid_admissions['hadm_id'].unique()
        logging.info(f"Found {len(admission_ids)} admissions with valid admission and discharge times")

        if self.filtered_hadm_ids is None:
            self.filtered_hadm_ids = set(admission_ids)
        else:
            self.filtered_hadm_ids = self.filtered_hadm_ids.intersection(set(admission_ids))

        # Also get the corresponding patient IDs
        patient_ids = valid_admissions['subject_id'].unique()

        if self.filtered_subject_ids is None:
            self.filtered_subject_ids = set(patient_ids)
        else:
            self.filtered_subject_ids = self.filtered_subject_ids.intersection(set(patient_ids))

        # Filter the main DataFrame based on these admission IDs
        if 'hadm_id' in df.columns:
            return df[df['hadm_id'].isin(admission_ids)]
        elif 'subject_id' in df.columns:
            return df[df['subject_id'].isin(patient_ids)]
        else:
            logging.warning("Neither hadm_id nor subject_id columns found in main DataFrame. Skipping filter.")
            return df

    def filter_by_inpatient_stay(self, df: pd.DataFrame, admissions_df: pd.DataFrame, transfers_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Filter DataFrame to include only inpatient stays.

        Args:
            df: DataFrame to filter
            admissions_df: DataFrame containing admission information
            transfers_df: Optional DataFrame containing transfer information

        Returns:
            Filtered DataFrame
        """
        logging.info("Filtering by inpatient stay")
        self.applied_filters.append("Inpatient stay only")

        # First, try to filter by admission_type in admissions table
        if 'admission_type' in admissions_df.columns:
            inpatient_types = self.ADMISSION_TYPES['inpatient']
            inpatient_admissions = admissions_df[admissions_df['admission_type'].isin(inpatient_types)]

            if inpatient_admissions.empty:
                logging.warning(f"No admissions found with inpatient admission types: {inpatient_types}")
                return df.iloc[0:0]

            admission_ids = inpatient_admissions['hadm_id'].unique()
            logging.info(f"Found {len(admission_ids)} admissions with inpatient admission types")

            if self.filtered_hadm_ids is None:
                self.filtered_hadm_ids = set(admission_ids)
            else:
                self.filtered_hadm_ids = self.filtered_hadm_ids.intersection(set(admission_ids))

            # Also get the corresponding patient IDs
            patient_ids = inpatient_admissions['subject_id'].unique()

            if self.filtered_subject_ids is None:
                self.filtered_subject_ids = set(patient_ids)
            else:
                self.filtered_subject_ids = self.filtered_subject_ids.intersection(set(patient_ids))

            # Filter the main DataFrame based on these IDs
            if 'hadm_id' in df.columns:
                df = df[df['hadm_id'].isin(admission_ids)]
            elif 'subject_id' in df.columns:
                df = df[df['subject_id'].isin(patient_ids)]
            else:
                logging.warning("Neither hadm_id nor subject_id columns found in main DataFrame. Skipping filter.")
                return df

        # Additionally, if transfers table is available, filter by care unit
        if transfers_df is not None and 'careunit' in transfers_df.columns:
            inpatient_units = self.CARE_UNITS['inpatient']
            inpatient_transfers = transfers_df[transfers_df['careunit'].isin(inpatient_units)]

            if inpatient_transfers.empty:
                logging.warning(f"No transfers found with inpatient care units: {inpatient_units}")
                # If we already filtered by admission_type, don't return empty DataFrame
                if 'admission_type' in admissions_df.columns:
                    return df
                return df.iloc[0:0]

            transfer_admission_ids = inpatient_transfers['hadm_id'].unique()
            logging.info(f"Found {len(transfer_admission_ids)} admissions with inpatient care units")

            if self.filtered_hadm_ids is None:
                self.filtered_hadm_ids = set(transfer_admission_ids)
            else:
                self.filtered_hadm_ids = self.filtered_hadm_ids.intersection(set(transfer_admission_ids))

            # Also get the corresponding patient IDs
            transfer_patient_ids = inpatient_transfers['subject_id'].unique()

            if self.filtered_subject_ids is None:
                self.filtered_subject_ids = set(transfer_patient_ids)
            else:
                self.filtered_subject_ids = self.filtered_subject_ids.intersection(set(transfer_patient_ids))

            # Filter the main DataFrame based on these IDs
            if 'hadm_id' in df.columns:
                df = df[df['hadm_id'].isin(transfer_admission_ids)]
            elif 'subject_id' in df.columns:
                df = df[df['subject_id'].isin(transfer_patient_ids)]

        return df

    def filter_by_exclude_death(self, df: pd.DataFrame, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to exclude patients who died during hospitalization.

        Args:
            df: DataFrame to filter
            admissions_df: DataFrame containing admission information

        Returns:
            Filtered DataFrame
        """
        logging.info("Filtering to exclude in-hospital deaths")
        self.applied_filters.append("Exclude in-hospital deaths")

        # First check for hospital_expire_flag (preferred method)
        if 'hospital_expire_flag' in admissions_df.columns:
            # Get admission IDs where patient did not die in hospital
            non_death_admissions = admissions_df[admissions_df['hospital_expire_flag'] == 0]

            if non_death_admissions.empty:
                logging.warning("No admissions found where patient did not die in hospital")
                return df.iloc[0:0]

            admission_ids = non_death_admissions['hadm_id'].unique()
            logging.info(f"Found {len(admission_ids)} admissions where patient did not die in hospital")

            if self.filtered_hadm_ids is None:
                self.filtered_hadm_ids = set(admission_ids)
            else:
                self.filtered_hadm_ids = self.filtered_hadm_ids.intersection(set(admission_ids))

            # Also get the corresponding patient IDs
            patient_ids = non_death_admissions['subject_id'].unique()

            if self.filtered_subject_ids is None:
                self.filtered_subject_ids = set(patient_ids)
            else:
                self.filtered_subject_ids = self.filtered_subject_ids.intersection(set(patient_ids))

            # Filter the main DataFrame based on these IDs
            if 'hadm_id' in df.columns:
                return df[df['hadm_id'].isin(admission_ids)]
            elif 'subject_id' in df.columns:
                return df[df['subject_id'].isin(patient_ids)]
            else:
                logging.warning("Neither hadm_id nor subject_id columns found in main DataFrame. Skipping filter.")
                return df
        # Alternate method: check for deathtime in admissions
        elif 'deathtime' in admissions_df.columns:
            non_death_admissions = admissions_df[admissions_df['deathtime'].isna()]

            if non_death_admissions.empty:
                logging.warning("No admissions found where patient did not die in hospital")
                return df.iloc[0:0]

            admission_ids = non_death_admissions['hadm_id'].unique()
            logging.info(f"Found {len(admission_ids)} admissions where patient did not die in hospital (based on deathtime)")

            if self.filtered_hadm_ids is None:
                self.filtered_hadm_ids = set(admission_ids)
            else:
                self.filtered_hadm_ids = self.filtered_hadm_ids.intersection(set(admission_ids))

            # Also get the corresponding patient IDs
            patient_ids = non_death_admissions['subject_id'].unique()

            if self.filtered_subject_ids is None:
                self.filtered_subject_ids = set(patient_ids)
            else:
                self.filtered_subject_ids = self.filtered_subject_ids.intersection(set(patient_ids))

            # Filter the main DataFrame based on these IDs
            if 'hadm_id' in df.columns:
                return df[df['hadm_id'].isin(admission_ids)]
            elif 'subject_id' in df.columns:
                return df[df['subject_id'].isin(patient_ids)]
            else:
                logging.warning("Neither hadm_id nor subject_id columns found in main DataFrame. Skipping filter.")
                return df
        else:
            logging.warning("Neither hospital_expire_flag nor deathtime column found in admissions DataFrame. Skipping filter.")
            return df
