"""
Filtering tab for the MIMIC-IV Dashboard application.

This module provides a Streamlit UI component for filtering MIMIC-IV data based on
inclusion and exclusion criteria.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Any

# Streamlit import
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class FilteringTab:
    """
    Class for rendering the filtering tab in the MIMIC-IV Dashboard application.

    This class provides UI components for specifying inclusion and exclusion criteria
    for filtering MIMIC-IV data.
    """

    def __init__(self, current_file_path: str):
        """Initialize the FilteringTab class."""
        logging.info("Initializing FilteringTab...")
        self._init_session_state()
        self.current_file_path = current_file_path

    def _init_session_state(self):
        """Initialize session state variables for filtering."""
        if 'filter_params' not in st.session_state:
            st.session_state.filter_params = {
                # Inclusion criteria
                'apply_encounter_timeframe': True,
                'encounter_timeframe': ['2017-2019'],
                'apply_age_range': True,
                'min_age': 18,
                'max_age': 75,
                'apply_t2dm_diagnosis': True,
                'apply_valid_admission_discharge': True,
                'apply_inpatient_stay': True,
                'admission_types': ['EMERGENCY', 'URGENT', 'ELECTIVE'],
                'require_inpatient_transfer': True,
                'required_inpatient_units': [],

                # Exclusion criteria
                'exclude_in_hospital_death': True
            }

    def _render_inclusion_criteria(self):
        """Render UI components for inclusion criteria."""
        st.markdown("### Inclusion Criteria")

        # Encounter Timeframe
        st.session_state.filter_params['apply_encounter_timeframe'] = st.checkbox(
            "Filter by Encounter Timeframe",
            value=st.session_state.filter_params['apply_encounter_timeframe'],
            key="apply_encounter_timeframe",
            help="Filter based on anchor_year_group from the patients table"
        )

        if st.session_state.filter_params['apply_encounter_timeframe']:
            st.session_state.filter_params['encounter_timeframe'] = st.multiselect(
                "Encounter Timeframe",
                options=['2008-2010', '2011-2013', '2014-2016', '2017-2019'],
                default=st.session_state.filter_params['encounter_timeframe'],
                key="encounter_timeframe",
                help="Select specific year groups to include"
            )

        # Age Range
        st.session_state.filter_params['apply_age_range'] = st.checkbox(
            "Filter by Age Range",
            value=st.session_state.filter_params['apply_age_range'],
            key="apply_age_range",
            help="Filter based on anchor_age from the patients table"
        )

        if st.session_state.filter_params['apply_age_range']:
            age_col1, age_col2 = st.columns(2)
            with age_col1:
                st.session_state.filter_params['min_age'] = st.number_input(
                    "Minimum Age",
                    min_value=0,
                    max_value=120,
                    value=st.session_state.filter_params['min_age'],
                    key="min_age"
                )
            with age_col2:
                st.session_state.filter_params['max_age'] = st.number_input(
                    "Maximum Age",
                    min_value=0,
                    max_value=120,
                    value=st.session_state.filter_params['max_age'],
                    key="max_age"
                )

        # T2DM Diagnosis
        st.session_state.filter_params['apply_t2dm_diagnosis'] = st.checkbox(
            "Filter by T2DM Diagnosis (ICD-10)",
            value=st.session_state.filter_params['apply_t2dm_diagnosis'],
            key="apply_t2dm_diagnosis",
            help="Include patients with ICD-10 code starting with 'E11' in diagnoses_icd table, where seq_num is 1, 2, or 3"
        )

        # Valid Admission/Discharge Times
        st.session_state.filter_params['apply_valid_admission_discharge'] = st.checkbox(
            "Filter by Valid Admission/Discharge Times",
            value=st.session_state.filter_params['apply_valid_admission_discharge'],
            key="apply_valid_admission_discharge",
            help="Ensure admittime and dischtime in the admissions table are not null"
        )

        # Inpatient Stay
        st.session_state.filter_params['apply_inpatient_stay'] = st.checkbox(
            "Filter by Inpatient Stay",
            value=st.session_state.filter_params['apply_inpatient_stay'],
            key="apply_inpatient_stay",
            help="Filter out non-inpatient encounters"
        )

        if st.session_state.filter_params['apply_inpatient_stay']:
            # Admission Type
            st.session_state.filter_params['admission_types'] = st.multiselect(
                "Admission Types to Include",
                options=['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION'],
                default=st.session_state.filter_params['admission_types'],
                key="admission_types",
                help="Select admission types to include"
            )

            # Inpatient Transfer
            st.session_state.filter_params['require_inpatient_transfer'] = st.checkbox(
                "Require Inpatient Transfer",
                value=st.session_state.filter_params['require_inpatient_transfer'],
                key="require_inpatient_transfer",
                help="Ensure the patient had at least one transfer to an inpatient careunit"
            )

            if st.session_state.filter_params['require_inpatient_transfer']:
                st.session_state.filter_params['required_inpatient_units'] = st.multiselect(
                    "Required Inpatient Units",
                    options=['MICU', 'SICU', 'CSRU', 'CCU', 'TSICU', 'NICU', 'Med', 'Surg'],
                    default=st.session_state.filter_params['required_inpatient_units'],
                    key="required_inpatient_units",
                    help="Select specific inpatient units to require (leave empty to accept any inpatient unit)"
                )

    def _render_exclusion_criteria(self):
        """Render UI components for exclusion criteria."""
        st.markdown("### Exclusion Criteria")

        # In-Hospital Death/Expiry
        st.session_state.filter_params['exclude_in_hospital_death'] = st.checkbox(
            "Exclude In-Hospital Deaths",
            value=st.session_state.filter_params['exclude_in_hospital_death'],
            key="exclude_in_hospital_death",
            help="Exclude admissions where deathtime is not null OR hospital_expire_flag = 1"
        )

        # Optional explicit age exclusion
        st.markdown("#### Optional Explicit Exclusions")
        st.markdown("*These are typically covered by the inclusion criteria above*")

        # Age exclusion (informational only, synced with inclusion criteria)
        if st.session_state.filter_params['apply_age_range']:
            st.info(
                f"Excluding patients with age < {st.session_state.filter_params['min_age']} "
                f"or > {st.session_state.filter_params['max_age']} "
                f"(based on inclusion criteria)"
            )

        # Non-inpatient exclusion (informational only, synced with inclusion criteria)
        if st.session_state.filter_params['apply_inpatient_stay']:
            excluded_types = [
                t for t in ['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION']
                if t not in st.session_state.filter_params['admission_types']
            ]
            if excluded_types:
                st.info(
                    f"Excluding non-inpatient encounters with admission types: {', '.join(excluded_types)} "
                    f"(based on inclusion criteria)"
                )

    def _reset_filters(self):
        """Reset filter parameters to default values."""
        st.session_state.filter_params = {
            # Inclusion criteria
            'apply_encounter_timeframe': True,
            'encounter_timeframe': ['2017-2019'],
            'apply_age_range': True,
            'min_age': 18,
            'max_age': 75,
            'apply_t2dm_diagnosis': True,
            'apply_valid_admission_discharge': True,
            'apply_inpatient_stay': True,
            'admission_types': ['EMERGENCY', 'URGENT', 'ELECTIVE'],
            'require_inpatient_transfer': True,
            'required_inpatient_units': [],

            # Exclusion criteria
            'exclude_in_hospital_death': True
        }

    def get_filter_params(self) -> Dict[str, Any]:
        """
        Get the current filter parameters.

        Returns:
            Dictionary containing the filter parameters
        """
        return st.session_state.filter_params

    def render(self, data_handler=None, feature_engineer=None):
        """
        Display the filtering view with inclusion and exclusion criteria.

        Args:
            data_handler: The MIMICDataHandler instance for loading data
            feature_engineer: The FeatureEngineer instance for detecting columns
        """
        st.markdown("<h1 class='sub-header'>MIMIC-IV Data Filtering</h1>", unsafe_allow_html=True)

        # Display information about filtering
        st.markdown("""
        <div class='info-box'>
        <p>This page allows you to define inclusion and exclusion criteria for filtering the MIMIC-IV dataset.</p>
        <p>These filters will be applied when loading data for analysis in the Data Explorer view.</p>
        <p><strong>Note:</strong> You need to have scanned and loaded a MIMIC-IV table before applying filters.</p>
        </div>
        """, unsafe_allow_html=True)

        # Check if data is loaded
        if 'df' in st.session_state and st.session_state.df is not None:
            st.markdown("<h3 class='sub-header'>Current Dataset</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-box'>
            <p><strong>Module:</strong> {st.session_state.selected_module}</p>
            <p><strong>Table:</strong> {st.session_state.selected_table}</p>
            <p><strong>Rows:</strong> {len(st.session_state.df)} (sample) / {st.session_state.total_row_count if 'total_row_count' in st.session_state else len(st.session_state.df)} (total)</p>
            </div>
            """, unsafe_allow_html=True)

        # Render the filtering tab UI
        st.markdown("## Data Filtering")
        st.markdown("""
        Define inclusion and exclusion criteria to filter the MIMIC-IV dataset.
        These filters will be applied when loading data for analysis.
        """)

        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            self._render_inclusion_criteria()

        with col2:
            self._render_exclusion_criteria()

        # Apply filters button
        if st.button("Apply Filters", key="apply_filters_button"):
            st.success("Filters applied successfully!")

        # Reset filters button
        if st.button("Reset Filters", key="reset_filters_button"):
            self._reset_filters()
            st.success("Filters reset to default values!")

        filter_params = st.session_state.filter_params

        # Apply filters button
        if st.button("Load Data with Filters", key="load_with_filters"):
            if self.current_file_path is not None:
                if data_handler is not None:
                    with st.spinner("Loading and filtering data..."):
                        df, total_rows = data_handler.load_mimic_table(
                            file_path     = self.current_file_path,
                            sample_size   = st.session_state.sample_size,
                            encoding      = "latin-1",
                            use_dask      = st.session_state.get('use_dask', False),
                            filter_params = filter_params
                        )

                    if df is not None:
                        st.session_state.df = df
                        st.session_state.total_row_count = total_rows
                        st.success(f"Data loaded and filtered successfully! {len(df)} rows remain after filtering.")

                        # Auto-detect columns for feature engineering
                        if feature_engineer is not None:
                            st.session_state.detected_order_cols     = feature_engineer.detect_order_columns(df)
                            st.session_state.detected_time_cols      = feature_engineer.detect_temporal_columns(df)
                            st.session_state.detected_patient_id_col = feature_engineer.detect_patient_id_column(df)

                        # Switch to data explorer view
                        st.session_state.current_view = 'data_explorer'
                        # st.experimental_rerun()
                    else:
                        st.error("Error loading data with filters.")
                else:
                    st.error("Data handler not provided. Cannot load data.")
            else:
                st.error("No data file selected. Please load a table first.")
        elif 'df' not in st.session_state or st.session_state.df is None:
            # Welcome message when no data is loaded
            st.markdown("""
            <div class='info-box'>
            <h3>No Data Loaded</h3>
            <p>To use the filtering functionality, you need to first load a MIMIC-IV table:</p>
            <ol>
                <li>Switch to the Data Explorer view using the sidebar</li>
                <li>Enter the path to your local MIMIC-IV dataset</li>
                <li>Click "Scan MIMIC-IV Directory" to detect available tables</li>
                <li>Select a module and table to explore</li>
                <li>Click "Load Table" to load the data</li>
                <li>Return to this Filtering view to apply filters</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
