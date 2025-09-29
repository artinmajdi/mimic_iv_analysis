"""
Filtering tab for the MIMIC-IV Dashboard application.

This module provides a Streamlit UI component for filtering MIMIC-IV data based on
inclusion and exclusion criteria.
"""

# Streamlit import
import streamlit as st

# Local application imports
from mimic_iv_analysis import logger
from mimic_iv_analysis.configurations.params import TableNames# , POE_COLUMNS, POE_ORDER_TYPES, POE_TRANSACTION_TYPES, ADMISSION_COLUMNS, ADMISSION_TYPES, ADMISSION_LOCATIONS



class FilteringTab:
    """
    Filtering tab for the MIMIC-IV Dashboard application.

    This class provides a Streamlit UI component for filtering MIMIC-IV data based on
    inclusion and exclusion criteria.
    """
    def __init__(self, table_name: TableNames):
        pass
        # if table_name == TableNames.POE:
        #     self._poe_filters()

        # elif table_name == TableNames.ADMISSIONS:
        #     self._admission_filters()

        # elif table_name == TableNames.MERGED:
        #     self._poe_filters()
        #     self._admission_filters()

        # else:
        #     logger.error(f"Invalid table name: {table_name}")

    def _render_inclusion_criteria(self):
        """Render UI components for inclusion criteria."""
        st.markdown("### Inclusion Criteria")

        # # Encounter Timeframe
        # st.session_state.filter_params['apply_encounter_timeframe'] = st.checkbox(
        #     "Filter by Encounter Timeframe",
        #     value=st.session_state.filter_params['apply_encounter_timeframe'],
        #     key="apply_encounter_timeframe",
        #     help="Filter based on anchor_year_group from the patients table"
        # )

        # if st.session_state.filter_params['apply_encounter_timeframe']:
        #     st.session_state.filter_params['encounter_timeframe'] = st.multiselect(
        #         "Encounter Timeframe",
        #         options=['2008-2010', '2011-2013', '2014-2016', '2017-2019'],
        #         default=st.session_state.filter_params['encounter_timeframe'],
        #         key="encounter_timeframe",
        #         help="Select specific year groups to include"
        #     )

        # # Age Range
        # st.session_state.filter_params['apply_age_range'] = st.checkbox(
        #     "Filter by Age Range",
        #     value=st.session_state.filter_params['apply_age_range'],
        #     key="apply_age_range",
        #     help="Filter based on anchor_age from the patients table"
        # )

        # if st.session_state.filter_params['apply_age_range']:
        #     age_col1, age_col2 = st.columns(2)
        #     with age_col1:
        #         st.session_state.filter_params['min_age'] = st.number_input(
        #             "Minimum Age",
        #             min_value=0,
        #             max_value=120,
        #             value=st.session_state.filter_params['min_age'],
        #             key="min_age"
        #         )
        #     with age_col2:
        #         st.session_state.filter_params['max_age'] = st.number_input(
        #             "Maximum Age",
        #             min_value=0,
        #             max_value=120,
        #             value=st.session_state.filter_params['max_age'],
        #             key="max_age"
        #         )

        # # T2DM Diagnosis
        # st.session_state.filter_params['apply_t2dm_diagnosis'] = st.checkbox(
        #     "Filter by T2DM Diagnosis (ICD-10)",
        #     value=st.session_state.filter_params['apply_t2dm_diagnosis'],
        #     key="apply_t2dm_diagnosis",
        #     help="Include patients with ICD-10 code starting with 'E11' in diagnoses_icd table, where seq_num is 1, 2, or 3"
        # )


        # # Inpatient Stay
        # st.session_state.filter_params['apply_inpatient_stay'] = st.checkbox(
        #     "Filter by Inpatient Stay",
        #     value=st.session_state.filter_params['apply_inpatient_stay'],
        #     key="apply_inpatient_stay",
        #     help="Filter out non-inpatient encounters"
        # )

        # if st.session_state.filter_params['apply_inpatient_stay']:
        #     # Admission Type
        #     st.session_state.filter_params['admission_types'] = st.multiselect(
        #         "Admission Types to Include",
        #         options=['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION'],
        #         default=st.session_state.filter_params['admission_types'],
        #         key="admission_types",
        #         help="Select admission types to include"
        #     )

        #     # Inpatient Transfer
        #     st.session_state.filter_params['require_inpatient_transfer'] = st.checkbox(
        #         "Require Inpatient Transfer",
        #         value=st.session_state.filter_params['require_inpatient_transfer'],
        #         key="require_inpatient_transfer",
        #         help="Ensure the patient had at least one transfer to an inpatient careunit"
        #     )

        #     if st.session_state.filter_params['require_inpatient_transfer']:
        #         st.session_state.filter_params['required_inpatient_units'] = st.multiselect(
        #             "Required Inpatient Units",
        #             options=['MICU', 'SICU', 'CSRU', 'CCU', 'TSICU', 'NICU', 'Med', 'Surg'],
        #             default=st.session_state.filter_params['required_inpatient_units'],
        #             key="required_inpatient_units",
        #             help="Select specific inpatient units to require (leave empty to accept any inpatient unit)"
        #         )

        st.markdown("### Exclusion Criteria")

        # # Optional explicit age exclusion
        # st.markdown("#### Optional Explicit Exclusions")
        # st.markdown("*These are typically covered by the inclusion criteria above*")

        # # Age exclusion (informational only, synced with inclusion criteria)
        # if st.session_state.filter_params['apply_age_range']:
        #     st.info(
        #         f"Excluding patients with age < {st.session_state.filter_params['min_age']} "
        #         f"or > {st.session_state.filter_params['max_age']} "
        #         f"(based on inclusion criteria)"
        #     )

        # # Non-inpatient exclusion (informational only, synced with inclusion criteria)
        # if st.session_state.filter_params['apply_inpatient_stay']:
        #     excluded_types = [
        #         t for t in ['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION']
        #         if t not in st.session_state.filter_params['admission_types']
        #     ]
        #     if excluded_types:
        #         st.info(
        #             f"Excluding non-inpatient encounters with admission types: {', '.join(excluded_types)} "
        #             f"(based on inclusion criteria)"
        #         )

    # def _diagnoses_icd_filters(self):

    #     diagnoses_icd = TableNames.DIAGNOSES_ICD.value

    #     if diagnoses_icd not in st.session_state.filter_params:
    #         st.session_state.filter_params[diagnoses_icd] = {}

    #     # DIAGNOSES_ICD Columns to Include
    #     st.session_state.filter_params[diagnoses_icd]['icd_version'] = st.multiselect(
    #         "ICD Version",
    #         options=['10', '9'],
    #         default=st.session_state.filter_params[diagnoses_icd]['icd_version']
    #     )

    #     # DIAGNOSES_ICD Sequence Number
    #     st.session_state.filter_params[diagnoses_icd]['seq_num'] = st.multiselect(
    #         "Sequence Number",
    #         options=['1', '2', '3'],
    #         default=st.session_state.filter_params[diagnoses_icd]['seq_num']
    #     )

    #     # DIAGNOSES_ICD ICD Code
    #     st.session_state.filter_params[diagnoses_icd]['icd_code'] = st.text_input(
    #         "ICD Code Starts With",
    #         value=st.session_state.filter_params[diagnoses_icd]['icd_code']
    #     )

    # def _admission_filters(self):

    #     admission = TableNames.ADMISSIONS.value

    #     if admission not in st.session_state.filter_params:
    #         st.session_state.filter_params[admission] = {}

    #     # ADMISSION Columns to Include
    #     st.session_state.filter_params[admission]['columns'] = st.multiselect(
    #         "ADMISSION Columns to Include",
    #         options=ADMISSION_COLUMNS,
    #         default=st.session_state.filter_params[admission]['selected_columns']
    #     )

    #     # Valid Admission/Discharge Times
    #     if 'admittime' in st.session_state.filter_params[admission]['selected_columns'] and 'dischtime' in st.session_state.filter_params[admission]['selected_columns']:

    #         st.session_state.filter_params[admission]['valid_admission_discharge'] = st.checkbox(
    #             "Filter by Valid Admission/Discharge Times",
    #             value=st.session_state.filter_params[admission]['valid_admission_discharge'],
    #             help="Ensure admittime and dischtime in the admissions table are not null"
    #         )

    #         # Discharge After Admission
    #         st.session_state.filter_params[admission]['discharge_after_admission'] = st.checkbox(
    #             "Filter by Discharge After Admission",
    #             value=st.session_state.filter_params[admission]['discharge_after_admission'],
    #             help="Ensure dischtime is after admittime"
    #         )

    #     # In-Hospital Death/Expiry
    #     if 'deathtime' in st.session_state.filter_params[admission]['selected_columns'] or 'hospital_expire_flag' in st.session_state.filter_params[admission]['selected_columns']:
    #         st.session_state.filter_params[admission]['exclude_in_hospital_death'] = st.checkbox(
    #             "Exclude In-Hospital Deaths",
    #             value=st.session_state.filter_params[admission]['exclude_in_hospital_death'],
    #             help="Exclude admissions where deathtime is not null OR hospital_expire_flag = 1"
    #         )



    #     # Admission Type
    #     st.session_state.filter_params[admission]['apply_admission_type'] = st.checkbox( "Filter by Admission Types", value=st.session_state.filter_params[admission]['apply_admission_type'], )

    #     st.session_state.filter_params[admission]['admission_type'] = st.multiselect(
    #         "Admission Types to Include",
    #         options=ADMISSION_TYPES,
    #         default=st.session_state.filter_params[admission]['admission_type'],
    #         disabled=not st.session_state.filter_params[admission]['apply_admission_type']
    #     )

    #     # Admission Location
    #     st.session_state.filter_params[admission]['apply_admission_location'] = st.checkbox( "Filter by Admission Locations", value=st.session_state.filter_params[admission]['apply_admission_location'], )

    #     # Admission Location
    #     st.session_state.filter_params[admission]['admission_location'] = st.multiselect(
    #         "Admission Locations to Include",
    #         options=ADMISSION_LOCATIONS,
    #         default=st.session_state.filter_params[admission]['admission_location'],
    #         disabled=not st.session_state.filter_params[admission]['apply_admission_location']
    #     )

    # def _poe_filters(self):

    #     poe = TableNames.POE.value

    #     if poe not in st.session_state.filter_params:
    #         st.session_state.filter_params[poe] = {}

    #     # POE Columns to Include
    #     st.session_state.filter_params[poe]['selected_columns'] = st.multiselect(
    #         "POE Columns to Include",
    #         options=POE_COLUMNS,
    #         default=st.session_state.filter_params[poe]['selected_columns']
    #     )

    #     # POE Order Types
    #     if 'order_type' in st.session_state.filter_params[poe]['selected_columns']:
    #         st.session_state.filter_params[poe]['apply_order_type'] = st.checkbox( "Filter by Order Types", value=st.session_state.filter_params[poe]['apply_order_type'] )

    #         st.session_state.filter_params[poe]['order_type'] = st.multiselect(
    #             "Order Types to Include",
    #             options=POE_ORDER_TYPES,
    #             default=st.session_state.filter_params[poe]['order_type'],
    #             disabled=not st.session_state.filter_params[poe]['apply_order_type']
    #         )

    #     # POE Transaction Types
    #     if 'transaction_type' in st.session_state.filter_params[poe]['selected_columns']:

    #         st.session_state.filter_params[poe]['apply_transaction_type'] = st.checkbox( "Filter by Transaction Types", value=st.session_state.filter_params[poe]['apply_transaction_type'] )

    #         st.session_state.filter_params[poe]['transaction_type'] = st.multiselect(
    #             "Transaction Types to Include",
    #             options=POE_TRANSACTION_TYPES,
    #             default=st.session_state.filter_params[poe]['transaction_type'],
    #             disabled=not st.session_state.filter_params[poe]['apply_transaction_type']
    #         )
