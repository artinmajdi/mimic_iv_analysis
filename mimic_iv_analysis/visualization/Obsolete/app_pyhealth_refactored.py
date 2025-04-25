"""
MIMIC-IV Analysis Dashboard using PyHealth
========================================

This file implements a Streamlit dashboard for analyzing MIMIC-IV v3.1 data using PyHealth.

Key Components:
-------------
1. ExtendedMIMIC4Dataset: Extends PyHealth's MIMIC4Dataset to handle additional tables
   - Supports both hosp/ and icu/ directory structures
   - Handles custom table parsing for unsupported tables
   - Maintains extra_tables dict for additional data

2. Data Loading & Processing:
   - Handles MIMIC-IV v3.1 directory structure
   - Supports code mapping between medical coding systems
   - Includes development mode for smaller data subsets
   - Verifies table existence and provides warnings

3. Features:
   - Patient demographics analysis
   - Diagnoses and procedures analysis
   - Medications and lab tests analysis
   - ICU stays analysis
   - Predictive tasks (mortality, readmission, etc.)
   - Model training and evaluation

Recent Changes:
-------------
- Fixed event access methods:
  * Changed visit.get_event_names() to list(visit.event_list_dict.keys())
  * Changed visit.get_event_from_type() to visit.get_event_list()

Important Notes:
-------------
- Visit events are accessed through visit.event_list_dict
- The ExtendedMIMIC4Dataset expects root path to point to hosp/ directory
- Supported predictive tasks: drug recommendation, mortality prediction,
  readmission prediction, length of stay prediction
- Available models: Transformer, RETAIN, RNN, MLP, CNN
"""

# Standard library imports
import json
import os
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# PyHealth imports
from pyhealth.data import Event
from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.metrics import (
    binary_metrics_fn,
    multiclass_metrics_fn,
    multilabel_metrics_fn,
)
from pyhealth.models import CNN, MLP, RETAIN, RNN, Transformer
from pyhealth.tasks import (
    drug_recommendation_mimic4_fn,
    length_of_stay_prediction_mimic4_fn,
    mortality_prediction_mimic4_fn,
    readmission_prediction_mimic4_fn,
)
from pyhealth.trainer import Trainer, save_model

# link : https://gemini.google.com/app/5915c2abbb47b866
# --- Data Handling ---

class ExtendedMIMIC4Dataset(MIMIC4Dataset):
    """
    Extends PyHealth's MIMIC4Dataset to handle additional tables from MIMIC-IV v3.1,
    supporting both hosp/ and icu/ directory structures and generic loading.
    """
    def __init__(self, root, tables, code_mapping=None, dev=False, refresh_cache=False):
        """
        Initializes the extended dataset.

        Args:
            root (str): Path to the 'hosp' directory within the MIMIC-IV dataset.
            tables (list): List of table names to load (without extensions).
            code_mapping (dict, optional): Dictionary for code system mapping. Defaults to None.
            dev (bool): If True, loads a smaller subset for development. Defaults to False.
            refresh_cache (bool): If True, refreshes the cache. Defaults to False.
        """
        super().__init__(root=root, tables=tables, code_mapping=code_mapping, dev=dev, refresh_cache=refresh_cache)
        # Store full dataset root (parent of hosp/)
        self.full_root = os.path.dirname(root)
        # Container for extra table DataFrames not directly parsed by PyHealth
        self.extra_tables = {}
        # Store the originally requested tables
        self.requested_tables = tables

    def parse_tables(self):
        """
        Parses the basic patient/admission info and then loads requested tables.
        Uses PyHealth parsers where available, otherwise loads CSVs into extra_tables.
        """
        # First parse basic info (patients & admissions)
        patients = self.parse_basic_info({})

        # Define known table locations (relative to full_root)
        hosp_tables = {
            'admissions', 'patients', 'transfers', 'diagnoses_icd',
            'procedures_icd', 'prescriptions', 'labevents', 'd_labitems',
            'microbiologyevents', 'pharmacy', 'poe', 'poe_detail', 'services'
        }
        icu_tables = {
            'icustays', 'chartevents', 'd_items', 'inputevents',
            'outputevents', 'procedureevents', 'datetimeevents', 'ingredientevents', 'caregiver'
        }

        # Loop through requested tables (using the stored list)
        for t in self.requested_tables:
            base = t.lower() # Already cleaned in DataHandler

            # If a PyHealth parser exists, use it
            # Note: PyHealth's MIMIC4Dataset expects tables like 'diagnoses_icd', not 'diagnoses'
            parse_fn_name = f"parse_{base}"
            parse_fn = getattr(self, parse_fn_name, None)

            if callable(parse_fn):
                print(f"Using PyHealth parser for: {base}")
                try:
                    patients = parse_fn(patients)
                except Exception as e:
                    print(f"Error using PyHealth parser for {base}: {e}")
            else:
                # Generic load into extra_tables if not parsed by PyHealth core/specific parsers
                print(f"Attempting generic load for: {base}")
                path = None
                if base in hosp_tables:
                    path_gz = os.path.join(self.root, f"{base}.csv.gz")
                    path_csv = os.path.join(self.root, f"{base}.csv")
                    path = path_gz if os.path.exists(path_gz) else path_csv if os.path.exists(path_csv) else None
                elif base in icu_tables:
                    icu_dir = os.path.join(self.full_root, 'icu')
                    path_gz = os.path.join(icu_dir, f"{base}.csv.gz")
                    path_csv = os.path.join(icu_dir, f"{base}.csv")
                    path = path_gz if os.path.exists(path_gz) else path_csv if os.path.exists(path_csv) else None
                else:
                    print(f"Warning: Table '{base}' location unknown, skipping generic load.")
                    continue

                if path and os.path.exists(path):
                    try:
                        print(f"Loading extra table {base} from {path}")
                        df = pd.read_csv(path, low_memory=False) # Added low_memory=False for robustness
                        self.extra_tables[base] = df
                    except Exception as e:
                        print(f"Error loading extra table {base} from {path}: {e}")
                elif path:
                    print(f"Warning: File not found for extra table {base} at expected path {path}")
                # If path is None, it means the table wasn't found in expected locations

        return patients

class DataHandler:
    """Handles loading and basic information display for the MIMIC-IV dataset."""

    def __init__(self, session_state):
        """Initializes the DataHandler."""
        self.state = session_state
        if 'mimic_dataset' not in self.state:
            self.state['mimic_dataset'] = None

    @staticmethod
    def _verify_paths(root_path):
        """Checks if hosp/ and icu/ directories exist."""
        hosp_dir = os.path.join(root_path, 'hosp')
        icu_dir = os.path.join(root_path, 'icu')
        if not os.path.exists(hosp_dir) or not os.path.exists(icu_dir):
            st.error(f"❌ Expected MIMIC-IV v3.1 directory structure with hosp/ and icu/ subdirectories in {root_path}")
            return None, None
        return hosp_dir, icu_dir

    @staticmethod
    def _resolve_tables(root_path, tables):
        """
        Cleans table names and verifies their existence in hosp/ or icu/ directories.
        Returns a list of validated, lowercase table names without extensions.
        """
        hosp_dir = os.path.join(root_path, 'hosp')
        icu_dir = os.path.join(root_path, 'icu')

        hosp_files = {f.lower().replace('.csv.gz', '').replace('.csv', '') for f in os.listdir(hosp_dir)}
        icu_files = {f.lower().replace('.csv.gz', '').replace('.csv', '') for f in os.listdir(icu_dir)}

        resolved_tables = []
        for t in tables:
            base = t.lower().replace('.csv.gz', '').replace('.csv', '')
            if base in hosp_files:
                resolved_tables.append(base)
            elif base in icu_files:
                resolved_tables.append(base)
            else:
                st.warning(f"⚠️ Table '{base}' (from '{t}') not found in hosp/ or icu/ directories. Skipping.")

        if not resolved_tables:
            st.error("❌ No valid tables found to load.")
            return None
        return resolved_tables

    def load_data(self, root_path, tables, code_mapping=None, dev=False):
        """Loads the MIMIC-IV dataset using ExtendedMIMIC4Dataset."""
        hosp_dir, icu_dir = self._verify_paths(root_path)
        if not hosp_dir:
            return False # Error already shown

        resolved_tables = self._resolve_tables(root_path, tables)
        if not resolved_tables:
            return False # Error already shown

        try:
            with st.spinner('🔄 Loading MIMIC-IV dataset... This might take some time.'):
                # Pass the 'hosp' directory as root to the Extended class
                # The Extended class knows how to find the full root and 'icu' dir
                dataset = ExtendedMIMIC4Dataset(
                    root=hosp_dir,
                    tables=resolved_tables, # Pass the validated list
                    code_mapping=code_mapping,
                    dev=dev,
                    refresh_cache=st.session_state.get('refresh_cache', False) # Allow cache refresh option
                )
                self.state['mimic_dataset'] = dataset
                st.success("✅ Dataset loaded successfully!")
                # Reset refresh cache flag if it was set
                if 'refresh_cache' in self.state:
                    del self.state['refresh_cache']
                return True
        except Exception as e:
            st.error(f"❌ Error loading MIMIC-IV dataset: {str(e)}")
            self.state['mimic_dataset'] = None
            return False

    def display_info(self):
        """Displays basic statistics and sample data for the loaded dataset."""
        dataset = self.state['mimic_dataset']
        if not dataset:
            st.warning("No dataset loaded.")
            return

        st.subheader("📊 Dataset Statistics")
        try:
            # Calculate stats directly for more control
            num_patients = len(dataset.patient_ids) # Use patient_ids for efficiency
            num_visits = sum(len(patient) for patient_id, patient in dataset.patients.items())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Patients", f"{num_patients:,}")
            with col2:
                st.metric("Number of Visits", f"{num_visits:,}")
            # Event count can be very slow, omitted by default
            # Consider adding a button to calculate if needed

            # Display available tables (parsed by PyHealth)
            available_tables = set()
            if num_visits > 0:
                 # Check the first patient's first visit for available tables
                 first_patient_id = next(iter(dataset.patients))
                 first_patient = dataset.patients[first_patient_id]
                 if first_patient.visits:
                    first_visit_id = next(iter(first_patient.visits))
                    first_visit = first_patient.visits[first_visit_id]
                    available_tables = set(first_visit.event_list_dict.keys())

            st.write("**Tables Parsed by PyHealth:**", ", ".join(sorted(list(available_tables))) or "None")

            # Display extra tables loaded
            if hasattr(dataset, 'extra_tables') and dataset.extra_tables:
                 st.write("**Extra Tables Loaded:**", ", ".join(sorted(dataset.extra_tables.keys())))


        except Exception as e:
            st.error(f"Error calculating dataset statistics: {e}")
            # Fallback to PyHealth's stat() if direct calculation fails
            try:
                stats = dataset.stat()
                st.json(stats) # Display raw stats string if calculation failed
            except Exception as stat_e:
                 st.error(f"Could not get stats using dataset.stat(): {stat_e}")


        st.subheader("Sample Patient Data")
        sample_patient_ids = list(dataset.patients.keys())[:min(5, num_patients)]

        if not sample_patient_ids:
            st.info("No patient data available to display samples.")
            return

        for patient_id in sample_patient_ids:
            with st.expander(f"Patient {patient_id}"):
                try:
                    patient = dataset.patients[patient_id]
                    st.write(f"- Gender: {getattr(patient, 'gender', 'N/A')}")
                    st.write(f"- Age: {getattr(patient, 'age', 'N/A')}") # Assuming age is available
                    st.write(f"- Number of Visits: {len(patient.visits)}")

                    for visit_id, visit in patient.visits.items():
                        st.markdown(f"--- \n **Visit ID: {visit_id}**")
                        st.write(f"- Admission Time: {visit.encounter_time}")
                        st.write(f"- Discharge Time: {visit.discharge_time}")

                        event_types = list(visit.event_list_dict.keys())
                        st.write(f"- Event Types Present: {', '.join(event_types) if event_types else 'None'}")

                        # Show sample events (limit to 5 per type for performance)
                        for event_type in event_types:
                             events = visit.get_event_list(event_type) # Correct method
                             if events:
                                 with st.expander(f"Sample {event_type} events ({len(events)} total)"):
                                     try:
                                         # Create DataFrame safely, handling potential missing attrs
                                         event_dicts = []
                                         for e in events[:5]:
                                             event_dict = {k: getattr(e, k, None) for k in e.__dict__}
                                             event_dicts.append(event_dict)
                                         event_df = pd.DataFrame(event_dicts)
                                         st.dataframe(event_df)
                                     except Exception as df_e:
                                         st.error(f"Error creating DataFrame for {event_type}: {df_e}")
                                         st.write(events[:5]) # Show raw events if DF fails
                except KeyError:
                    st.error(f"Patient ID {patient_id} not found in dataset.patients.")
                except Exception as e:
                    st.error(f"Error displaying patient {patient_id}: {e}")

# --- UI Management ---

class UI:
    """Handles the Streamlit UI setup and layout."""

    def __init__(self):
        """Sets up the page configuration and custom CSS."""
        st.set_page_config(
            page_title="MIMIC-IV Analysis with PyHealth",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._apply_custom_css()
        st.title("🏥 MIMIC-IV v3.1 Analysis Tool with PyHealth")
        st.markdown("Analyze MIMIC-IV data: explore, predict, train models, and view patient details.")

    def _apply_custom_css(self):
        """Applies custom CSS styles."""
        st.markdown("""
        <style>
            /* General */
            .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
            h1, h2, h3 { font-weight: 600; }
            h1 { color: #0e4a80; }
            h2 { color: #1976D2; margin-top: 2em; margin-bottom: 1em; border-bottom: 2px solid #eee; padding-bottom: 0.3em;}
            h3 { color: #2196F3; margin-top: 1.5em; margin-bottom: 0.8em;}

            /* Cards */
            .metric-card {
                background-color: #f8f9fa; /* Lighter gray */
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); /* Softer shadow */
                margin-bottom: 20px;
                border: 1px solid #e9ecef; /* Light border */
                text-align: center;
            }
            .metric-card h3 { margin-top: 0; margin-bottom: 0.5em; color: #495057; font-size: 1.1em;}
            .metric-card h2 { margin-top: 0; color: #1976D2; font-size: 2em; }

            /* Report Sections */
            .report-section {
                background-color: #ffffff; /* White background */
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #dee2e6; /* Slightly darker border */
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
            }
            .report-section h3 { margin-top: 0; }

            /* Progress Bar */
            .stProgress > div > div > div { background-color: #28a745; } /* Green */

            /* Buttons */
            .stButton>button {
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 600;
            }
            /* Primary button style */
             .stButton>button[kind="primary"] {
                background-color: #1976D2;
                color: white;
                border: none;
             }
             .stButton>button[kind="primary"]:hover {
                 background-color: #0e4a80;
             }

            /* Expander */
            .stExpander { border: 1px solid #e9ecef; border-radius: 8px; margin-bottom: 1rem; }
            .stExpander header { background-color: #f8f9fa; border-radius: 8px 8px 0 0; }

            /* Dataframes */
            .stDataFrame { font-size: 0.9em; }

        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Renders the sidebar navigation."""
        st.sidebar.title("⚙️ Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose a section:",
            ["Home", "Data Loading", "Data Exploration", "Predictive Tasks", "Model Training & Evaluation", "Patient Analytics"]
        )
        return app_mode

    def render_home(self):
        """Renders the Home page content."""
        st.header("Welcome!")
        st.markdown("""
        This application provides a comprehensive interface for analyzing MIMIC-IV v3.1 healthcare data using the PyHealth library.

        **Key Features:**
        - **Data Loading**: Connect to your MIMIC-IV v3.1 dataset (requires `hosp/` and `icu/` subdirectories).
        - **Data Exploration**: Explore patient demographics, diagnoses, procedures, medications, labs, ICU stays, and any extra loaded tables.
        - **Predictive Tasks**: Set up data for healthcare predictive tasks like mortality prediction, drug recommendation, readmission prediction, and length of stay prediction.
        - **Model Training & Evaluation**: Train and evaluate various machine learning models (Transformer, RETAIN, RNN, MLP, CNN) on the prepared task data.
        - **Patient Analytics**: Analyze individual patient journeys, view event timelines, generate reports, and make predictions using trained models.

        **Getting Started:**
        1.  Navigate to **Data Loading** in the sidebar to connect to your MIMIC-IV dataset.
        2.  Use **Data Exploration** to understand the loaded data.
        3.  Prepare data for analysis in **Predictive Tasks**.
        4.  Train models in **Model Training & Evaluation**.
        5.  Dive into individual patient details in **Patient Analytics**.

        **About MIMIC-IV:**
        A large, de-identified database from Beth Israel Deaconess Medical Center ICU patients.

        **About PyHealth:**
        A Python library for healthcare AI, simplifying data processing, feature extraction, and model training.
        """)
        st.info("👈 Select 'Data Loading' from the sidebar to begin.")

    def render_data_loading(self, data_handler):
        """Renders the Data Loading page."""
        st.header("💾 Data Loading")
        st.markdown("Connect to your MIMIC-IV v3.1 dataset. Provide the **root path** containing `hosp/` and `icu/` subdirectories.")

        # Use session state to remember the path
        if 'mimic_root_path' not in st.session_state:
            # Provide a sensible default or leave empty
            st.session_state['mimic_root_path'] = "" # Example: "/path/to/mimic-iv-3.1"

        root_path = st.text_input("MIMIC-IV Dataset Root Path", st.session_state['mimic_root_path'])
        st.session_state['mimic_root_path'] = root_path # Update state on change

        st.subheader("Select Tables to Load")
        # Group tables for better organization
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Hospital (hosp) Module")
            # Common/essential tables selected by default
            default_hosp = ["admissions", "patients", "diagnoses_icd", "procedures_icd", "prescriptions", "labevents"]
            hosp_tables = st.multiselect(
                "Select hospital tables",
                ["admissions", "patients", "transfers", "diagnoses_icd",
                 "procedures_icd", "prescriptions", "labevents", "d_labitems",
                 "microbiologyevents", "pharmacy", "poe", "poe_detail", "services"],
                 default=default_hosp
            )
        with col2:
            st.markdown("##### ICU Module")
            default_icu = ["icustays", "chartevents", "d_items"]
            icu_tables = st.multiselect(
                "Select ICU tables",
                ["icustays", "chartevents", "d_items", "inputevents", "outputevents",
                 "procedureevents", "datetimeevents", 'ingredientevents', 'caregiver'],
                 default=default_icu
            )
        selected_tables = hosp_tables + icu_tables

        with st.expander("Advanced Options"):
            # Code Mapping (Optional)
            code_mapping_enabled = st.checkbox("Enable Code Mapping (e.g., NDC to ATC)")
            code_mapping = None
            if code_mapping_enabled:
                # Simplified example - expand as needed
                map_from = st.selectbox("Map From Code", ["NDC"])
                map_to = st.selectbox("Map To Code", ["ATC"])
                level = st.slider("Target Level (if applicable)", 1, 5, 3)
                code_mapping = {map_from: (map_to, {"target_kwargs": {"level": level}})}
                st.caption(f"Current mapping: {code_mapping}")

            # Dev Mode
            dev_mode = st.checkbox("Development Mode (load smaller subset)", True)

            # Cache Refresh
            refresh_cache = st.checkbox("Refresh Cache (force re-parsing)", False)
            if refresh_cache:
                st.session_state['refresh_cache'] = True # Signal to DataHandler

        # Load Button
        if st.button("Load Dataset", type="primary", key="load_data_button"):
            if not root_path:
                st.error("Please enter a valid path to the MIMIC-IV dataset root.")
            elif not selected_tables:
                st.error("Please select at least one table to load.")
            else:
                # Clear previous task/model state if reloading data
                keys_to_clear = ['mimic4_sample', 'task_name', 'task_fn', 'model', 'metrics', 'test_loader']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                # Attempt to load data
                success = data_handler.load_data(root_path, selected_tables, code_mapping, dev_mode)
                if success:
                     st.balloons() # Fun feedback on success

        # Display info about the currently loaded dataset
        if data_handler.state['mimic_dataset']:
            st.markdown("---")
            st.subheader("Current Dataset Information")
            data_handler.display_info()
            st.success("Dataset is loaded. You can proceed to other sections.")
        elif 'mimic_dataset' in st.session_state and st.session_state['mimic_dataset'] is None:
             # Explicitly handle the case where loading failed or no dataset is loaded yet
             st.info("No dataset is currently loaded. Use the options above.")


# --- Exploration Handling ---
class ExplorationHandler:
    """Handles data exploration visualizations."""

    def __init__(self, session_state):
        """Initializes the ExplorationHandler."""
        self.state = session_state

    def render(self):
        """Renders the Data Exploration page."""
        st.header("🔍 Data Exploration")
        dataset = self.state.get('mimic_dataset')

        if not dataset:
            st.warning("⚠️ Please load a dataset first in the 'Data Loading' section.")
            st.stop() # Stop execution if no dataset

        # Dataset Overview
        st.subheader("Dataset Overview")
        try:
            stats = dataset.stat() # Get basic stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Patients", f"{stats.get('patient_num', 'N/A'):,}")
            col2.metric("Visits", f"{stats.get('visit_num', 'N/A'):,}")
            col3.metric("Events", f"{stats.get('event_num', 'N/A'):,}")
        except Exception as e:
            st.error(f"Could not retrieve basic stats: {e}")

        # Exploration Options
        st.subheader("Exploration Options")
        # Get available event types dynamically + extra tables
        available_events = set()
        if dataset.patients:
            first_patient = next(iter(dataset.patients.values()))
            if first_patient.visits:
                first_visit = next(iter(first_patient.visits.values()))
                available_events = set(first_visit.event_list_dict.keys())

        options = ["Patient Demographics"]
        if "diagnoses_icd" in available_events: options.append("Diagnoses")
        if "procedures_icd" in available_events: options.append("Procedures")
        if "prescriptions" in available_events: options.append("Medications")
        if "labevents" in available_events: options.append("Lab Tests")
        if "icustays" in available_events or ("icustays" in getattr(dataset, 'extra_tables', {})): options.append("ICU Stays") # Check both parsed and extra
        if hasattr(dataset, 'extra_tables') and dataset.extra_tables: options.append("Extra Tables")

        if not options:
             st.info("No specific data types found to explore. Check loaded tables.")
             return

        explore_choice = st.radio("Select data to explore:", options, horizontal=True)

        # Display based on choice
        if explore_choice == "Patient Demographics": self._display_demographics(dataset)
        elif explore_choice == "Diagnoses": self._display_events(dataset, "diagnoses_icd", "diagnosis", ["code", "description"])
        elif explore_choice == "Procedures": self._display_events(dataset, "procedures_icd", "procedure", ["code", "description"])
        elif explore_choice == "Medications": self._display_events(dataset, "prescriptions", "medication", ["drug", "dosage", "route"])
        elif explore_choice == "Lab Tests": self._display_events(dataset, "labevents", "lab test", ["test_name", "value", "unit"])
        elif explore_choice == "ICU Stays": self._display_icu_stays(dataset)
        elif explore_choice == "Extra Tables": self._display_extra_tables(dataset)

    def _get_event_data(self, dataset, event_type):
        """Helper to extract all events of a specific type into a list of dicts."""
        event_data = []
        if not dataset or not dataset.patients:
            return pd.DataFrame()

        # Determine if event_type is a PyHealth parsed event or an extra table
        is_pyhealth_event = False
        if dataset.patients:
            first_patient = next(iter(dataset.patients.values()))
            if first_patient.visits:
                first_visit = next(iter(first_patient.visits.values()))
                if event_type in first_visit.event_list_dict:
                    is_pyhealth_event = True

        if is_pyhealth_event:
            for patient_id, patient in dataset.patients.items():
                for visit_id, visit in patient.visits.items():
                    events = visit.get_event_list(event_type)
                    for event in events:
                        event_dict = event.__dict__.copy() # Get attributes
                        event_dict['patient_id'] = patient_id
                        event_dict['visit_id'] = visit_id
                        event_data.append(event_dict)
        elif hasattr(dataset, 'extra_tables') and event_type in dataset.extra_tables:
             # Handle extra tables - they are already DataFrames
             # We might need to add patient/visit info if joining is feasible,
             # but for basic exploration, just return the table.
             # This part might need refinement based on how extra tables are used.
             return dataset.extra_tables[event_type]
        else:
            st.warning(f"Event type '{event_type}' not found in parsed events or extra tables.")
            return pd.DataFrame()

        return pd.DataFrame(event_data)


    def _display_demographics(self, dataset):
        """Displays patient demographics."""
        st.subheader("Patient Demographics")
        patient_data = []
        for patient_id, patient in dataset.patients.items():
            # Basic info directly from Patient object
            patient_info = {
                "patient_id": patient_id,
                "gender": getattr(patient, 'gender', 'Unknown'),
                "age": getattr(patient, 'age', None), # Keep age numeric if possible
                "race": getattr(patient, 'race', 'Unknown'), # Assuming race might be available
                "num_visits": len(patient.visits),
            }
            # Get ethnicity from the first visit if available (example)
            if patient.visits:
                first_visit = next(iter(patient.visits.values()))
                patient_info["ethnicity"] = getattr(first_visit, 'ethnicity', 'Unknown')
            patient_data.append(patient_info)

        if not patient_data:
            st.info("No patient demographic data available.")
            return

        patient_df = pd.DataFrame(patient_data)
        st.dataframe(patient_df.head(10))

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            if 'gender' in patient_df.columns and patient_df['gender'].nunique() > 0:
                st.subheader("Gender Distribution")
                gender_counts = patient_df['gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                fig = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution', hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.caption("Gender data not available.")

            if 'race' in patient_df.columns and patient_df['race'].nunique() > 0:
                 st.subheader("Race Distribution")
                 race_counts = patient_df['race'].value_counts().reset_index()
                 race_counts.columns = ['Race', 'Count']
                 fig = px.bar(race_counts.head(10), x='Race', y='Count', title='Top 10 Race Categories') # Show top 10
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.caption("Race data not available.")


        with col2:
            if 'age' in patient_df.columns and pd.api.types.is_numeric_dtype(patient_df['age']):
                st.subheader("Age Distribution")
                fig = px.histogram(patient_df.dropna(subset=['age']), x='age', nbins=30, title='Age Distribution')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.caption("Age data not available or not numeric.")

            if 'num_visits' in patient_df.columns:
                st.subheader("Visits per Patient")
                fig = px.histogram(patient_df, x='num_visits', title='Number of Visits per Patient')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.caption("Number of visits data not available.")


    def _display_events(self, dataset, event_type, event_name_singular, top_n_cols):
        """Generic function to display event data (diagnoses, procedures, etc.)."""
        st.subheader(f"{event_name_singular.capitalize()} Analysis")
        with st.spinner(f"Loading {event_name_singular} data..."):
            event_df = self._get_event_data(dataset, event_type)

        if event_df.empty:
            st.info(f"No {event_name_singular} data available. Ensure the '{event_type}' table was loaded.")
            return

        st.dataframe(event_df.head(10))

        # Top N analysis
        st.subheader(f"Top 20 {event_name_singular.capitalize()}s")
        for col in top_n_cols:
            if col in event_df.columns:
                # Handle potential high cardinality by showing top 20
                if event_df[col].nunique() > 0:
                    top_items = event_df[col].value_counts().reset_index().head(20)
                    top_items.columns = [col.replace('_', ' ').title(), 'Count']
                    try:
                         # Use categoryorder for better bar chart sorting
                         fig = px.bar(top_items,
                                      x=col.replace('_', ' ').title(),
                                      y='Count',
                                      title=f'Top 20 {col.replace("_", " ").title()}',
                                      text_auto=True)
                         fig.update_layout(xaxis={'categoryorder':'total descending'})
                         st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                         st.error(f"Could not plot top {col}: {e}")
                         st.dataframe(top_items) # Show table as fallback
                else:
                     st.caption(f"No data found for column '{col}'.")
            else:
                st.caption(f"Column '{col}' not found in {event_name_singular} data.")

        # Additional plots specific to event type could be added here
        # Example: ICD version distribution for diagnoses/procedures
        if event_type in ["diagnoses_icd", "procedures_icd"] and 'icd_version' in event_df.columns:
             st.subheader("ICD Version Distribution")
             version_counts = event_df['icd_version'].value_counts().reset_index()
             version_counts.columns = ['ICD Version', 'Count']
             fig = px.pie(version_counts, values='Count', names='ICD Version', title='ICD Version Distribution')
             st.plotly_chart(fig, use_container_width=True)

        # Example: Medication routes
        if event_type == "prescriptions" and 'route' in event_df.columns:
             st.subheader("Medication Routes")
             route_counts = event_df['route'].value_counts().reset_index().head(15) # Top 15 routes
             route_counts.columns = ['Route', 'Count']
             fig = px.bar(route_counts, x='Route', y='Count', title='Top 15 Medication Routes')
             fig.update_layout(xaxis={'categoryorder':'total descending'})
             st.plotly_chart(fig, use_container_width=True)


    def _display_icu_stays(self, dataset):
        """Displays ICU stay information."""
        st.subheader("ICU Stays Analysis")

        # Try getting data from PyHealth parsed events first
        icu_df = self._get_event_data(dataset, "icustays")

        # If empty, check extra_tables (if ExtendedMIMIC4Dataset was used)
        if icu_df.empty and hasattr(dataset, 'extra_tables') and "icustays" in dataset.extra_tables:
             st.info("Loading ICU data from 'extra_tables'.")
             icu_df = dataset.extra_tables["icustays"]
             # May need to rename columns if they differ from PyHealth's Event attributes
             # Example: icu_df = icu_df.rename(columns={'stay_id': 'icu_stay_id', 'first_careunit': 'icu_type'})


        if icu_df.empty:
            st.info("No ICU stay data available. Ensure the 'icustays' table was loaded.")
            return

        # Ensure necessary columns exist before proceeding
        required_cols = ['intime', 'outtime', 'first_careunit'] # Example columns from MIMIC-IV icustays.csv
        if not all(col in icu_df.columns for col in required_cols):
             st.warning(f"ICU data is missing one or more required columns: {required_cols}. Displaying raw data.")
             st.dataframe(icu_df.head(10))
             return

        # Calculate Length of Stay (LOS)
        try:
            # Convert times to datetime, coercing errors
            icu_df['intime'] = pd.to_datetime(icu_df['intime'], errors='coerce')
            icu_df['outtime'] = pd.to_datetime(icu_df['outtime'], errors='coerce')
            # Calculate LOS in days
            icu_df['los_days'] = (icu_df['outtime'] - icu_df['intime']).dt.total_seconds() / (24 * 3600)
            # Filter out negative or zero LOS
            icu_df = icu_df[icu_df['los_days'] > 0].copy()
        except Exception as e:
            st.error(f"Error calculating ICU LOS: {e}")
            icu_df['los_days'] = None # Set to None if calculation fails

        st.dataframe(icu_df.head(10))

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            if 'los_days' in icu_df.columns and icu_df['los_days'].notna().any():
                st.subheader("ICU Length of Stay")
                # Cap LOS for visualization if needed (e.g., at 60 days)
                max_los_display = 60
                los_viz_df = icu_df[icu_df['los_days'] <= max_los_display]
                fig = px.histogram(los_viz_df.dropna(subset=['los_days']), x='los_days', nbins=30,
                                   title=f'ICU LOS Distribution (up to {max_los_display} days)')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                st.write("LOS Statistics (days):", icu_df['los_days'].describe())
            else:
                 st.caption("ICU Length of Stay data not available or calculation failed.")

        with col2:
            # Use 'first_careunit' as ICU type proxy if 'icu_type' isn't directly available
            icu_type_col = 'first_careunit' if 'first_careunit' in icu_df.columns else 'icu_type' if 'icu_type' in icu_df.columns else None
            if icu_type_col and icu_df[icu_type_col].nunique() > 0:
                st.subheader("ICU Type Distribution")
                type_counts = icu_df[icu_type_col].value_counts().reset_index()
                type_counts.columns = ['ICU Type', 'Count']
                fig = px.pie(type_counts, values='Count', names='ICU Type', title='ICU Type Distribution', hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.caption("ICU Type data not available.")


    def _display_extra_tables(self, dataset):
        """Displays data from tables loaded into extra_tables."""
        st.subheader("Extra Tables")
        extra = getattr(dataset, 'extra_tables', {})
        if not extra:
            st.info("No extra tables were loaded. Select tables not directly parsed by PyHealth during Data Loading.")
            return

        table_name = st.selectbox("Select extra table to view:", list(extra.keys()))
        if table_name and table_name in extra:
            df = extra[table_name]
            st.write(f"**Table:** `{table_name}`")
            st.write(f"**Shape:** {df.shape}")
            # Show column names and types
            with st.expander("Column Details"):
                 st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))
            st.dataframe(df.head(20)) # Show first 20 rows
        else:
            st.warning("Selected table not found in extra tables.")


# --- Prediction Handling ---
class PredictionHandler:
    """Handles setting up predictive tasks."""

    TASK_FUNCTIONS = {
        'Drug Recommendation': drug_recommendation_mimic4_fn,
        'Mortality Prediction': mortality_prediction_mimic4_fn,
        'Readmission Prediction': readmission_prediction_mimic4_fn,
        'Length of Stay Prediction': length_of_stay_prediction_mimic4_fn
    }

    TASK_DESCRIPTIONS = {
        "Drug Recommendation": "Predict medications based on patient history/diagnoses.",
        "Mortality Prediction": "Predict in-hospital mortality risk.",
        "Readmission Prediction": "Predict likelihood of readmission.",
        "Length of Stay Prediction": "Predict expected hospital length of stay."
    }

    # Define typical modes for tasks (can be overridden in training UI)
    TASK_MODES = {
        'Drug Recommendation': 'multilabel',
        'Mortality Prediction': 'binary',
        'Readmission Prediction': 'binary',
        'Length of Stay Prediction': 'multiclass' # Or regression, depending on fn output
    }


    def __init__(self, session_state):
        """Initializes the PredictionHandler."""
        self.state = session_state
        # Initialize state variables if they don't exist
        if 'mimic4_sample' not in self.state: self.state['mimic4_sample'] = None
        if 'task_name' not in self.state: self.state['task_name'] = None
        if 'task_fn' not in self.state: self.state['task_fn'] = None
        if 'task_mode' not in self.state: self.state['task_mode'] = None


    def render(self):
        """Renders the Predictive Tasks page."""
        st.header("🎯 Predictive Tasks")
        dataset = self.state.get('mimic_dataset')

        if not dataset:
            st.warning("⚠️ Please load a dataset first in the 'Data Loading' section.")
            st.stop()

        st.subheader("Select a Predictive Task")
        task_display_name = st.selectbox(
            "Choose a task:",
            list(self.TASK_FUNCTIONS.keys())
        )

        st.markdown(f"**Description:** {self.TASK_DESCRIPTIONS.get(task_display_name, 'N/A')}")

        if st.button("Process Task Data", type="primary", key="process_task"):
            task_fn = self.TASK_FUNCTIONS.get(task_display_name)
            if task_fn:
                with st.spinner(f"⚙️ Processing data for {task_display_name}..."):
                    try:
                        # The set_task function modifies the dataset in place
                        # and returns a sample dataset (often the same object)
                        # It filters visits/patients based on task criteria
                        # and adds a 'label' field.
                        sample_dataset = dataset.set_task(task_fn=task_fn)

                        if sample_dataset is None or len(sample_dataset) == 0:
                             st.error(f"❌ No samples generated for task '{task_display_name}'. "
                                      "This might be due to missing required data (e.g., specific event tables) "
                                      "or the task function filtering out all visits/patients.")
                             # Clear previous state if processing failed
                             self.state['mimic4_sample'] = None
                             self.state['task_name'] = None
                             self.state['task_fn'] = None
                             self.state['task_mode'] = None

                        else:
                            self.state['mimic4_sample'] = sample_dataset
                            self.state['task_name'] = task_display_name
                            self.state['task_fn'] = task_fn # Store the function if needed later
                            self.state['task_mode'] = self.TASK_MODES.get(task_display_name) # Store default mode
                            st.success(f"✅ Task data processed! {len(sample_dataset)} samples created.")
                            st.balloons()
                            self._display_task_info() # Show info immediately after processing

                    except Exception as e:
                        st.error(f"❌ Error setting task '{task_display_name}': {e}")
                        # Clear state on error
                        self.state['mimic4_sample'] = None
                        self.state['task_name'] = None
                        self.state['task_fn'] = None
                        self.state['task_mode'] = None

            else:
                st.error("Selected task function not found.")

        # Display info about the currently processed task data
        if self.state.get('mimic4_sample'):
            st.markdown("---")
            st.subheader(f"Current Task Data: {self.state['task_name']}")
            self._display_task_info()
            st.success("Task data is ready! Proceed to 'Model Training & Evaluation'.")
        elif 'task_name' in self.state and self.state['task_name'] is not None and self.state['mimic4_sample'] is None:
             # Handle case where processing was attempted but failed
             st.info(f"Processing for task '{self.state['task_name']}' did not yield samples. Please check data requirements or try another task.")
        else:
             st.info("No task data has been processed yet.")


    def _display_task_info(self):
        """Displays information about the processed task data."""
        sample_dataset = self.state.get('mimic4_sample')
        if not sample_dataset:
            return

        st.write(f"Number of samples: {len(sample_dataset)}")

        # Display label distribution safely
        try:
            # Check if the method exists and the dataset is not empty
            if hasattr(sample_dataset, "get_label_distribution") and len(sample_dataset) > 0:
                label_dist = sample_dataset.get_label_distribution()
                st.subheader("Label Distribution")
                if isinstance(label_dist, dict):
                    # Handle dictionary distributions (e.g., multiclass, binary)
                    if not label_dist:
                         st.caption("Label distribution dictionary is empty.")
                    else:
                        label_df = pd.DataFrame(list(label_dist.items()), columns=['Label', 'Count'])
                        label_df = label_df.sort_values('Count', ascending=False)
                        fig = px.bar(label_df, x='Label', y='Count', title='Label Distribution', text_auto=True)
                        st.plotly_chart(fig, use_container_width=True)
                        # st.dataframe(label_df) # Optional: show table too
                elif isinstance(label_dist, (int, float, np.number)):
                     # Handle single number distributions (e.g., regression tasks might return mean)
                     st.metric("Average Label Value", f"{label_dist:.4f}")
                elif isinstance(label_dist, list):
                     # Handle list distributions (potentially multilabel summaries)
                     st.write("Label Summary (list format):")
                     st.json(label_dist[:10]) # Show first 10 elements if it's a long list
                else:
                    # Fallback for other types
                    st.write("Label Distribution/Summary:")
                    st.json(label_dist) # Display whatever format it is
            else:
                 st.caption("Label distribution information not available for this task or dataset is empty.")

        except NotImplementedError:
             st.caption("Label distribution calculation is not implemented for this dataset type.")
        except Exception as e:
            st.error(f"Could not display label distribution: {e}")


        # Show sample data structure
        st.subheader("Sample Data Structure")
        if len(sample_dataset) > 0:
            sample = sample_dataset[0] # Get the first sample
            with st.expander("View First Sample Details"):
                 # Display sample keys and types, and potentially label
                 sample_info = {}
                 for key, value in sample.items():
                      sample_info[key] = type(value).__name__
                 st.json(sample_info)
                 if 'label' in sample:
                      st.write("**Label Value (first sample):**")
                      st.write(sample['label'])
                 # Optionally show feature content for the first sample
                 # st.write("**Feature Content (first sample):**")
                 # st.json({k: v for k, v in sample.items() if k != 'label'})
        else:
             st.info("No samples available to show structure.")


# --- Training Handling ---
class TrainingHandler:
    """Handles model training and evaluation."""

    MODEL_MAP = {
        'Transformer': Transformer,
        'RETAIN': RETAIN,
        'RNN': RNN,
        'MLP': MLP,
        'CNN': CNN
    }

    MODEL_DESCRIPTIONS = {
        "Transformer": "Self-attention mechanism for time-step dependencies.",
        "RETAIN": "Reverse Time Attention for interpretable predictions.",
        "RNN": "Recurrent Neural Network (GRU/LSTM) for sequential data.",
        "MLP": "Multi-Layer Perceptron baseline.",
        "CNN": "Convolutional Neural Network for pattern detection."
    }

    def __init__(self, session_state):
        """Initializes the TrainingHandler."""
        self.state = session_state
        # Initialize state
        if 'model' not in self.state: self.state['model'] = None
        if 'metrics' not in self.state: self.state['metrics'] = None
        if 'test_loader' not in self.state: self.state['test_loader'] = None
        if 'trained_model_type' not in self.state: self.state['trained_model_type'] = None


    def render(self):
        """Renders the Model Training & Evaluation page."""
        st.header("🏋️ Model Training & Evaluation")

        sample_dataset = self.state.get('mimic4_sample')
        task_name = self.state.get('task_name')

        if not sample_dataset or not task_name:
            st.warning("⚠️ Please process a task first in the 'Predictive Tasks' section.")
            st.stop()

        st.subheader(f"Training Models for: {task_name}")

        # --- Model Configuration ---
        st.markdown("### 1. Configure Model")
        col1, col2 = st.columns([1, 2])
        with col1:
            model_type = st.selectbox("Choose model type", list(self.MODEL_MAP.keys()))
        with col2:
             st.caption(self.MODEL_DESCRIPTIONS.get(model_type, ""))

        # --- Feature and Label Selection ---
        st.markdown("### 2. Select Features & Label")
        if len(sample_dataset) > 0:
            sample = sample_dataset[0]
            available_features = [k for k in sample.keys() if k not in ['label', 'patient_id', 'visit_id']] # Exclude common non-feature keys
            default_features = available_features[:min(len(available_features), 3)] # Default to first 3 features

            feature_keys = st.multiselect(
                "Select features to use",
                available_features,
                default=default_features
            )

            # Label key selection (usually 'label', but handle nested dicts if necessary)
            label_key = 'label' # Default assumption
            if 'label' not in sample:
                 st.error("❌ Critical Error: 'label' key not found in the sample data generated by the task function. Cannot proceed.")
                 st.stop()
            # Optional: Handle nested labels if your task function creates them
            # if isinstance(sample['label'], dict):
            #     label_keys = list(sample['label'].keys())
            #     label_key = st.selectbox("Select specific label key", label_keys)

            st.caption(f"Using label key: `{label_key}`")

            if not feature_keys:
                 st.warning("⚠️ Please select at least one feature.")
                 can_train = False
            else:
                 can_train = True

        else:
            st.error("❌ No samples found in the processed task data. Cannot configure features.")
            can_train = False


        # --- Training Parameters ---
        st.markdown("### 3. Set Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 1, 100, 10) # Increased max epochs
            batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128, 256], value=64)
        with col2:
             # Default mode based on task, allow override
             default_mode = self.state.get('task_mode', 'binary') # Get default mode stored during task processing
             mode_options = ["binary", "multiclass", "multilabel"] # Add regression if needed
             try:
                  default_index = mode_options.index(default_mode)
             except ValueError:
                  default_index = 0 # Fallback to binary if mode not found
             mode = st.selectbox("Prediction Mode", mode_options, index=default_index)

             learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2], # More options
                value=1e-4) # Common default for transformers


        # --- Training Execution ---
        st.markdown("### 4. Train & Evaluate")
        if st.button("Start Training", type="primary", key="train_button", disabled=not can_train):
            ModelClass = self.MODEL_MAP.get(model_type)
            if not ModelClass:
                st.error(f"Model class for '{model_type}' not found.")
                st.stop()

            # Clear previous results before starting new training
            self.state['model'] = None
            self.state['metrics'] = None
            self.state['test_loader'] = None
            self.state['trained_model_type'] = None


            try:
                # --- Data Splitting and Loading ---
                with st.spinner("🔄 Splitting data and creating dataloaders..."):
                    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.7, 0.1, 0.2])
                    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
                    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
                    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)
                    st.caption(f"Data split: Train ({len(train_ds)}), Val ({len(val_ds)}), Test ({len(test_ds)}) samples.")

                # --- Model Initialization ---
                with st.spinner("🛠️ Initializing model..."):
                    model = ModelClass(
                        dataset=sample_dataset, # Provide the full sample dataset for vocab building
                        feature_keys=feature_keys,
                        label_key=label_key,
                        mode=mode,
                        # Add other model-specific hyperparameters here if needed
                        # e.g., embedding_dim=128, hidden_dim=256, n_layers=2
                    )

                # --- Training ---
                trainer = Trainer(model=model, device=None) # Use default device (GPU if available)
                st.write(f"🚀 Training **{model_type}** for **{epochs}** epochs...")
                progress_bar = st.progress(0)
                progress_text = st.empty()
                metrics_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_metric': []} # To store metrics per epoch

                # Training callback
                def progress_callback(epoch, loss, val_metrics=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    val_loss = val_metrics.get('loss', float('nan')) if val_metrics else float('nan')
                    # Display a primary validation metric if available (e.g., roc_auc for binary)
                    primary_metric_key = 'roc_auc' if mode == 'binary' else 'macro_f1' if mode == 'multiclass' else 'macro_auc' if mode == 'multilabel' else None
                    primary_metric_val = val_metrics.get(primary_metric_key, float('nan')) if val_metrics and primary_metric_key else float('nan')

                    progress_text.text(f"Epoch {epoch+1}/{epochs}: Train Loss={loss:.4f} | Val Loss={val_loss:.4f}" +
                                       (f" | Val {primary_metric_key}={primary_metric_val:.4f}" if not np.isnan(primary_metric_val) else ""))

                    # Store metrics
                    metrics_history['epoch'].append(epoch + 1)
                    metrics_history['train_loss'].append(loss)
                    metrics_history['val_loss'].append(val_loss)
                    metrics_history['val_metric'].append(primary_metric_val)


                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    # Pass the callback correctly
                    # The trainer expects a callback function, not the result of calling it
                    monitor='loss', # Monitor validation loss by default
                    monitor_mode='min',
                    callbacks=[progress_callback] # Pass callback in a list if using PyHealth's callback system, or adjust if it expects a single function
                    # Note: PyHealth trainer's callback signature might differ. Adjust as needed.
                    # If the above callback doesn't work, you might need to manually loop epochs
                    # or use a simpler progress update within the training loop if modifying Trainer isn't desired.
                )
                progress_text.text("✅ Training complete!")

                # --- Evaluation ---
                with st.spinner("🧪 Evaluating model on test set..."):
                    # Ensure model is in evaluation mode
                    # The trainer.inference method should handle this, but double-check if issues arise
                    # model.eval()
                    y_true, y_prob, _, loss = trainer.inference(test_loader, additional_outputs=['loss']) # Get loss as well

                    # Choose metrics function based on mode
                    metrics_fn_map = {
                        'binary': binary_metrics_fn,
                        'multiclass': multiclass_metrics_fn,
                        'multilabel': multilabel_metrics_fn
                    }
                    metrics_fn = metrics_fn_map.get(mode)

                    if metrics_fn:
                         # Adjust metrics calculation based on function signature (e.g., thresholds)
                         metrics = metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc", "accuracy", "f1", "precision", "recall"]) # Request common metrics
                         metrics['test_loss'] = loss # Add test loss
                         st.success("✅ Evaluation complete!")
                         # Store results
                         self.state['model'] = model
                         self.state['metrics'] = metrics
                         self.state['test_loader'] = test_loader # Keep loader if needed for patient analytics
                         self.state['trained_model_type'] = model_type
                         st.session_state['metrics_history'] = metrics_history # Store history for plotting
                    else:
                         st.error(f"Metrics function for mode '{mode}' not found.")
                         self.state['metrics'] = {'error': f"Unsupported mode '{mode}' for metrics."}


            except Exception as e:
                st.error(f"❌ An error occurred during training/evaluation: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}") # Show traceback for debugging


        # --- Display Results ---
        if self.state.get('metrics'):
            st.markdown("---")
            st.subheader(f"📊 Evaluation Results ({self.state.get('trained_model_type', 'N/A')} - {task_name})")
            metrics = self.state['metrics']
            mode = self.state.get('task_mode', 'N/A') # Get mode used for training

            # Display training history plot
            if 'metrics_history' in st.session_state:
                 history = st.session_state['metrics_history']
                 history_df = pd.DataFrame(history)
                 fig_loss = px.line(history_df, x='epoch', y=['train_loss', 'val_loss'], title='Training & Validation Loss')
                 fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
                 st.plotly_chart(fig_loss, use_container_width=True)

                 # Plot validation metric if available
                 if 'val_metric' in history_df.columns and history_df['val_metric'].notna().any():
                      primary_metric_key = 'roc_auc' if mode == 'binary' else 'macro_f1' if mode == 'multiclass' else 'macro_auc' if mode == 'multilabel' else 'Metric'
                      fig_metric = px.line(history_df.dropna(subset=['val_metric']), x='epoch', y='val_metric', title=f'Validation {primary_metric_key}')
                      fig_metric.update_layout(xaxis_title='Epoch', yaxis_title=primary_metric_key)
                      st.plotly_chart(fig_metric, use_container_width=True)


            # Display key metrics in cards
            st.markdown("##### Key Metrics")
            cols = st.columns(4)
            key_metrics_displayed = 0
            metric_map = { # Map common metrics to display names
                 'roc_auc': 'AUC ROC', 'pr_auc': 'AUC PR', 'accuracy': 'Accuracy',
                 'f1': 'F1 Score', 'macro_f1': 'Macro F1', 'micro_f1': 'Micro F1',
                 'macro_auc': 'Macro AUC', 'micro_auc': 'Micro AUC',
                 'precision': 'Precision', 'recall': 'Recall',
                 'jaccard': 'Jaccard', 'p_at_k': 'Precision@k', # Add others as needed
                 'test_loss': 'Test Loss'
            }

            for key, name in metric_map.items():
                 if key in metrics and key_metrics_displayed < 4: # Limit to 4 cards
                     with cols[key_metrics_displayed % 4]:
                         st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>{name}</h3>
                                <h2>{metrics[key]:.4f}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                         )
                         key_metrics_displayed += 1

            # Display all metrics in a table
            st.markdown("##### All Metrics")
            try:
                # Filter out non-scalar metrics if any cause issues with DataFrame creation
                scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
                metrics_df = pd.DataFrame([scalar_metrics])
                st.dataframe(metrics_df)
            except Exception as e:
                st.error(f"Could not display all metrics in table format: {e}")
                st.json(metrics) # Fallback to JSON

            # --- Save Model Option ---
            st.markdown("### 5. Save Model (Optional)")
            save_dir = st.text_input("Save Directory", "./saved_models/")
            model_filename = st.text_input("Model Filename", f"{task_name.replace(' ', '_')}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.pt")

            if st.button("Save Trained Model"):
                model_to_save = self.state.get('model')
                if model_to_save:
                    try:
                        save_path = os.path.join(save_dir, model_filename)
                        os.makedirs(save_dir, exist_ok=True)
                        # Use PyHealth's save_model utility
                        save_model(model_to_save, save_path)
                        # Save metrics alongside the model
                        metrics_path = save_path.replace(".pt", "_metrics.json")
                        with open(metrics_path, 'w') as f:
                            # Convert numpy types to native Python types for JSON serialization
                            serializable_metrics = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in metrics.items()}
                            json.dump(serializable_metrics, f, indent=4)
                        st.success(f"✅ Model saved to `{save_path}` and metrics to `{metrics_path}`")
                    except Exception as e:
                        st.error(f"❌ Error saving model: {e}")
                else:
                    st.warning("⚠️ No trained model available to save.")


# --- Patient Analytics Handling ---
class AnalyticsHandler:
    """Handles individual patient analytics and report generation."""

    def __init__(self, session_state):
        """Initializes the AnalyticsHandler."""
        self.state = session_state

    def render(self):
        """Renders the Patient Analytics page."""
        st.header("🧑‍⚕️ Patient Analytics")

        dataset = self.state.get('mimic_dataset')
        if not dataset or not dataset.patients:
            st.warning("⚠️ Please load a dataset with patient data first ('Data Loading' section).")
            st.stop()

        # --- Patient Selection ---
        st.subheader("Select Patient")
        patient_ids = list(dataset.patients.keys())
        # Use index=None for default selection prompt if list is long
        default_index = 0 if patient_ids else None # Select first patient by default
        selected_patient_id = st.selectbox("Select Patient ID", patient_ids, index=default_index, help="Choose a patient to analyze.")

        if not selected_patient_id:
            st.info("Please select a patient ID.")
            st.stop()

        patient = dataset.patients[selected_patient_id]

        # --- Patient Information ---
        st.subheader(f"Patient Overview: {selected_patient_id}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gender", getattr(patient, 'gender', 'N/A'))
        with col2:
             # Display age if available and numeric
             age = getattr(patient, 'age', None)
             st.metric("Age", f"{int(age)}" if age is not None and isinstance(age, (int, float)) else 'N/A')
        with col3:
            st.metric("Total Visits", len(patient.visits))
        # Add more demographics if available (e.g., race)
        st.caption(f"Race: {getattr(patient, 'race', 'N/A')}, Ethnicity: {getattr(patient, 'ethnicity', 'N/A')}") # Assuming ethnicity might be on patient


        # --- Visit Selection & Details ---
        st.subheader("Visit Details")
        visit_ids = list(patient.visits.keys())
        if not visit_ids:
            st.info("This patient has no visit records in the loaded data.")
            st.stop()

        selected_visit_id = st.selectbox("Select Visit ID", visit_ids, help="Choose a specific hospital visit.")
        visit = patient.visits[selected_visit_id]

        # Visit Timestamps and LOS
        adm_time = getattr(visit, 'encounter_time', None)
        dis_time = getattr(visit, 'discharge_time', None)
        los_days = None
        if adm_time and dis_time:
             # Ensure they are datetime objects
             if isinstance(adm_time, str): adm_time = pd.to_datetime(adm_time, errors='coerce')
             if isinstance(dis_time, str): dis_time = pd.to_datetime(dis_time, errors='coerce')
             if pd.notna(adm_time) and pd.notna(dis_time):
                  los_delta = dis_time - adm_time
                  los_days = los_delta.total_seconds() / (24 * 3600)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Admission Time", str(adm_time.date()) if adm_time else "N/A")
        with col2: st.metric("Discharge Time", str(dis_time.date()) if dis_time else "N/A")
        with col3: st.metric("Length of Stay (Days)", f"{los_days:.2f}" if los_days is not None else "N/A")


        # --- Event Exploration for Selected Visit ---
        st.subheader("Events During Visit")
        event_types = list(visit.event_list_dict.keys())

        if not event_types:
             st.info("No specific events recorded for this visit in the loaded tables.")
        else:
             # Allow user to select event type to view
             selected_event_type = st.selectbox("Select event type to view details:", sorted(event_types))

             if selected_event_type:
                 events = visit.get_event_list(selected_event_type)
                 if events:
                     try:
                         event_dicts = [e.__dict__ for e in events]
                         event_df = pd.DataFrame(event_dicts)

                         # Convert time columns to datetime if they exist
                         time_cols = ['time', 'charttime', 'starttime', 'endtime', 'intime', 'outtime']
                         for tc in time_cols:
                              if tc in event_df.columns:
                                   event_df[tc] = pd.to_datetime(event_df[tc], errors='coerce')

                         # Try to sort by time if a time column exists
                         time_col_found = None
                         for tc in time_cols:
                              if tc in event_df.columns and pd.api.types.is_datetime64_any_dtype(event_df[tc]):
                                   time_col_found = tc
                                   break
                         if time_col_found:
                              event_df = event_df.sort_values(by=time_col_found)

                         st.dataframe(event_df)

                         # --- Event-Specific Visualizations ---
                         self._visualize_events(event_df, selected_event_type)

                     except Exception as e:
                         st.error(f"Error processing or displaying {selected_event_type} events: {e}")
                         st.write(events[:10]) # Show raw data on error
                 else:
                     st.info(f"No '{selected_event_type}' events found for this visit.")


        # --- Model Predictions (if model is loaded/trained) ---
        st.subheader("Clinical Predictions")
        model = self.state.get('model')
        trained_task_name = self.state.get('task_name') # Task the model was trained on

        if model and trained_task_name:
            st.write(f"Using model trained for: **{trained_task_name}**")

            if st.button(f"Run {trained_task_name} Prediction for this Visit", key="predict_patient"):
                with st.spinner("Generating sample and running prediction..."):
                    try:
                        # Get the correct task function used for training
                        task_fn = PredictionHandler.TASK_FUNCTIONS.get(trained_task_name)
                        if not task_fn:
                             st.error(f"Task function for '{trained_task_name}' not found.")
                             st.stop()

                        # Generate the sample for this specific patient/visit using the task function
                        # This requires the patient and visit objects
                        # Note: Some task functions might require the full dataset object too
                        sample_dict = task_fn(patient=patient, visit=visit, patient_id=selected_patient_id) # Adjust args as needed by task_fn

                        if not sample_dict:
                            st.error("Could not generate a valid sample for this patient/visit using the task function. Prediction aborted.")
                        else:
                            # Convert the sample dict to a PyHealth Sample object if needed by the model
                            # (Check if model.predict expects list of dicts or list of Sample objects)
                            from pyhealth.datasets import Sample
                            sample_obj = Sample.from_dict(sample_dict) # Use from_dict class method

                            # Make prediction (model.predict usually expects a list)
                            # Ensure model is in eval mode
                            model.eval()
                            import torch # Assuming PyTorch backend
                            with torch.no_grad():
                                 # The predict method might return raw logits or probabilities
                                 # It might also return dict with 'prediction', 'logit', 'prob' keys
                                 prediction_output = model.predict([sample_obj]) # Pass as a list

                            # Process prediction output (this depends heavily on the model's predict implementation)
                            # Assuming prediction_output is a list containing results for the single sample
                            if isinstance(prediction_output, list) and len(prediction_output) > 0:
                                 result = prediction_output[0] # Get result for the first (only) sample

                                 # Handle different possible output formats from model.predict
                                 y_prob = None
                                 if isinstance(result, dict):
                                      y_prob = result.get('prob', result.get('logit')) # Prefer probabilities
                                 elif isinstance(result, (np.ndarray, torch.Tensor)):
                                      y_prob = result
                                 else:
                                      st.error(f"Unexpected prediction output format: {type(result)}")
                                      st.stop()

                                 if y_prob is not None:
                                      # Convert to numpy array if it's a tensor
                                      if hasattr(y_prob, 'cpu'): # Check if it's a tensor
                                          y_prob = y_prob.cpu().numpy()

                                      st.subheader("Prediction Result")
                                      self._display_prediction_result(y_prob, trained_task_name, model)
                                 else:
                                      st.error("Could not extract probabilities or logits from prediction output.")

                            else:
                                 st.error("Prediction output was empty or not in the expected list format.")

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")

        else:
            st.info("No trained model available in the current session. Train a model in the 'Model Training & Evaluation' section first.")


        # --- Generate Patient Report ---
        # st.subheader("Generate Report")
        # if st.button("Generate Comprehensive Patient Report", key="generate_report"):
        #     self._generate_report(patient, visit, selected_patient_id, selected_visit_id)


    def _visualize_events(self, event_df, event_type):
        """Generates visualizations specific to event types."""
        st.markdown("---")
        st.write(f"**Visualizations for {event_type}:**")

        # --- Lab Events Visualization ---
        if event_type == "labevents" and not event_df.empty:
            numeric_col = 'value_num' # PyHealth often adds this
            value_col = 'value'
            test_col = 'test_name'
            time_col = 'time' # Or 'charttime'

            # Ensure required columns exist
            if not all(c in event_df.columns for c in [test_col, value_col, time_col]):
                 st.caption(f"Cannot visualize: Missing required columns ({test_col}, {value_col}, {time_col}).")
                 return

            # Attempt to convert value column to numeric
            event_df[numeric_col] = pd.to_numeric(event_df[value_col], errors='coerce')
            plot_df = event_df.dropna(subset=[time_col, numeric_col])

            if not plot_df.empty:
                st.write(f"Visualizing numeric lab results over time.")
                unique_tests = plot_df[test_col].unique()
                # Select top N tests by frequency for default view
                top_tests = plot_df[test_col].value_counts().head(5).index.tolist()
                selected_tests = st.multiselect(
                    f"Select Lab Tests to Plot",
                    unique_tests,
                    default=top_tests
                )

                if selected_tests:
                    fig_df = plot_df[plot_df[test_col].isin(selected_tests)]
                    if not fig_df.empty:
                         fig = px.line(fig_df, x=time_col, y=numeric_col, color=test_col,
                                       title='Lab Test Results Over Time', markers=True,
                                       labels={numeric_col: 'Value', time_col: 'Time'})
                         fig.update_layout(legend_title_text='Lab Test')
                         st.plotly_chart(fig, use_container_width=True)
                    else:
                         st.info("No numeric data available for the selected lab tests in this visit.")
                else:
                     st.info("Select at least one lab test to plot.")
            else:
                st.caption("No numeric lab values with valid timestamps found for plotting.")


        # --- Chart Events Visualization ---
        elif event_type == "chartevents" and not event_df.empty:
            numeric_col = 'valuenum' # Common in MIMIC chartevents
            value_col = 'value'
            item_col = 'itemid' # Or potentially a mapped label if d_items was loaded
            time_col = 'charttime'

             # Try to map itemid to label if d_items is available
            d_items_df = None
            if 'mimic_dataset' in self.state and hasattr(self.state['mimic_dataset'], 'extra_tables') and 'd_items' in self.state['mimic_dataset'].extra_tables:
                 d_items_df = self.state['mimic_dataset'].extra_tables['d_items'][['itemid', 'label']]
                 event_df = pd.merge(event_df, d_items_df, on='itemid', how='left')
                 item_col = 'label' # Use label for selection/display if merge successful
            elif 'label' not in event_df.columns:
                 event_df['label'] = event_df[item_col].astype(str) # Use itemid as label if no mapping

            # Ensure required columns exist
            if not all(c in event_df.columns for c in [item_col, value_col, time_col]):
                 st.caption(f"Cannot visualize: Missing required columns ({item_col}, {value_col}, {time_col}).")
                 return

             # Use 'valuenum' if available, otherwise try converting 'value'
            if numeric_col not in event_df.columns:
                 event_df[numeric_col] = pd.to_numeric(event_df[value_col], errors='coerce')

            plot_df = event_df.dropna(subset=[time_col, numeric_col])


            if not plot_df.empty:
                st.write(f"Visualizing numeric chart events (vitals, etc.) over time.")
                unique_items = plot_df[item_col].unique()
                top_items = plot_df[item_col].value_counts().head(5).index.tolist()

                selected_items = st.multiselect(
                    f"Select Chart Events to Plot",
                    unique_items,
                    default=top_items
                )

                if selected_items:
                    fig_df = plot_df[plot_df[item_col].isin(selected_items)]
                    if not fig_df.empty:
                         fig = px.line(fig_df, x=time_col, y=numeric_col, color=item_col,
                                       title='Chart Events Over Time', markers=True,
                                       labels={numeric_col: 'Value', time_col: 'Time'})
                         fig.update_layout(legend_title_text='Chart Event')
                         st.plotly_chart(fig, use_container_width=True)
                    else:
                         st.info("No numeric data available for the selected chart events in this visit.")
                else:
                     st.info("Select at least one chart event to plot.")
            else:
                st.caption("No numeric chart events with valid timestamps found for plotting.")

        # --- Add visualizations for other event types as needed ---
        # elif event_type == "..." : ...

        else:
            st.caption(f"No specific visualization implemented for '{event_type}'.")


    def _display_prediction_result(self, y_prob, task_name, model):
        """Formats and displays the prediction result based on the task."""

        # Ensure y_prob is a numpy array
        if not isinstance(y_prob, np.ndarray):
             y_prob = np.array(y_prob)

        # Remove batch dimension if present (prediction was for one sample)
        if y_prob.ndim > 1 and y_prob.shape[0] == 1:
             y_prob = y_prob.squeeze(0)

        # --- Drug Recommendation ---
        if task_name == 'Drug Recommendation':
            st.write("Top 5 Recommended Drugs (Codes):")
            if hasattr(model, 'label_tokenizer'):
                # Get top k indices and their probabilities
                top_k = 5
                # Ensure y_prob is 1D for argsort
                if y_prob.ndim > 1:
                     st.warning(f"Drug prediction has unexpected shape {y_prob.shape}, attempting to flatten.")
                     y_prob = y_prob.flatten()

                if y_prob.ndim == 1:
                    top_indices = np.argsort(y_prob)[::-1][:top_k]
                    top_probs = y_prob[top_indices]
                    # Decode indices to drug codes using the tokenizer
                    try:
                        top_codes = [model.label_tokenizer.idx2token[idx] for idx in top_indices] # Access idx2token directly
                        results_df = pd.DataFrame({'Drug Code': top_codes, 'Probability': top_probs})
                        st.dataframe(results_df)
                        # Bar chart
                        fig = px.bar(results_df, x='Drug Code', y='Probability', title='Top 5 Drug Recommendations', text_auto='.2f')
                        fig.update_layout(xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                    except AttributeError:
                         st.error("Model's label_tokenizer does not have 'idx2token'. Cannot decode drug codes.")
                         st.write("Raw top indices:", top_indices)
                         st.write("Probabilities:", top_probs)
                    except IndexError:
                         st.error("Index out of bounds when accessing label_tokenizer. Vocabulary might mismatch.")
                         st.write("Raw top indices:", top_indices)
                         st.write("Probabilities:", top_probs)
                else:
                     st.error(f"Cannot determine top drugs from prediction shape: {y_prob.shape}")


            else:
                st.warning("Model does not have a 'label_tokenizer'. Cannot decode drug codes.")
                st.write("Raw Prediction Output (Probabilities/Logits):")
                st.write(y_prob)

        # --- Mortality Prediction ---
        elif task_name == 'Mortality Prediction':
            # Assuming binary classification: prob of class 1 (mortality)
            mortality_prob = y_prob[1] if len(y_prob) > 1 else y_prob[0] # Handle single output or [prob_0, prob_1]
            st.metric("Predicted Mortality Risk", f"{mortality_prob:.2%}")
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=mortality_prob,
                title={'text': "Mortality Risk"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#d9534f"}, # Red
                       'steps': [{'range': [0, 0.2], 'color': "#5cb85c"}, # Green
                                 {'range': [0.2, 0.5], 'color': "#f0ad4e"}, # Orange
                                 {'range': [0.5, 1], 'color': "#d9534f"}]})) # Red
            st.plotly_chart(fig, use_container_width=True)

        # --- Readmission Prediction ---
        elif task_name == 'Readmission Prediction':
            readmission_prob = y_prob[1] if len(y_prob) > 1 else y_prob[0]
            st.metric("Predicted Readmission Risk", f"{readmission_prob:.2%}")
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=readmission_prob,
                title={'text': "Readmission Risk"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#f0ad4e"}, # Orange
                       'steps': [{'range': [0, 0.3], 'color': "#5cb85c"}, # Green
                                 {'range': [0.3, 0.7], 'color': "#f0ad4e"}, # Orange
                                 {'range': [0.7, 1], 'color': "#d9534f"}]})) # Red
            st.plotly_chart(fig, use_container_width=True)

        # --- Length of Stay Prediction ---
        elif task_name == 'Length of Stay Prediction':
            # Assuming multiclass classification where class index maps to LOS category
            # Or could be regression outputting a single value
            if y_prob.ndim == 0 or len(y_prob) == 1: # Regression case
                 predicted_los = y_prob.item() # Get scalar value
                 st.metric("Predicted Length of Stay (Days)", f"{predicted_los:.2f}")
            else: # Classification case
                 predicted_class_index = np.argmax(y_prob)
                 predicted_prob = y_prob[predicted_class_index]
                 # Need mapping from index to LOS category (from task function or tokenizer)
                 if hasattr(model, 'label_tokenizer'):
                      try:
                           predicted_category = model.label_tokenizer.idx2token[predicted_class_index]
                           st.metric(f"Predicted LOS Category", f"{predicted_category}")
                           st.caption(f"(Probability: {predicted_prob:.2%})")
                           # Optional: Show probabilities for all categories
                           # los_cats = [model.label_tokenizer.idx2token[i] for i in range(len(y_prob))]
                           # los_probs_df = pd.DataFrame({'Category': los_cats, 'Probability': y_prob})
                           # st.dataframe(los_probs_df.sort_values('Probability', ascending=False))
                      except Exception as e:
                           st.error(f"Could not decode LOS category using label_tokenizer: {e}")
                           st.metric("Predicted LOS Class Index", predicted_class_index)
                           st.caption(f"(Probability: {predicted_prob:.2%})")
                 else:
                      st.metric("Predicted LOS Class Index", predicted_class_index)
                      st.caption(f"(Probability: {predicted_prob:.2%})")
                 st.write("Raw Class Probabilities:", y_prob)

        # --- Fallback for other tasks ---
        else:
            st.write("Prediction Output:")
            st.write(y_prob)


    def _generate_report(self, patient, visit, patient_id, visit_id):
        """Generates a basic HTML report for the patient visit."""
        # This function needs significant expansion to be comprehensive
        st.markdown("---")
        st.write("Generating report...")

        # Basic Info
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Patient Report: {patient_id} / Visit: {visit_id}</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
                h2 {{ color: #1976D2; border-bottom: 1px solid #eee; padding-bottom: 5px;}}
                h3 {{ color: #2196F3; }}
                p {{ margin: 5px 0; }}
                strong {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Patient Clinical Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="section">
                <h2>Patient Information</h2>
                <p><strong>Patient ID:</strong> {patient_id}</p>
                <p><strong>Gender:</strong> {getattr(patient, 'gender', 'N/A')}</p>
                <p><strong>Age:</strong> {getattr(patient, 'age', 'N/A')}</p>
                <p><strong>Race:</strong> {getattr(patient, 'race', 'N/A')}</p>
            </div>

            <div class="section">
                <h2>Visit Summary</h2>
                <p><strong>Visit ID:</strong> {visit_id}</p>
                <p><strong>Admission:</strong> {getattr(visit, 'encounter_time', 'N/A')}</p>
                <p><strong>Discharge:</strong> {getattr(visit, 'discharge_time', 'N/A')}</p>
            </div>
        """

        # Add sections for events (Diagnoses, Procedures, Meds, Labs)
        event_types_to_report = {
            "diagnoses_icd": "Diagnoses",
            "procedures_icd": "Procedures",
            "prescriptions": "Medications",
            "labevents": "Key Lab Results"
        }

        for event_type, title in event_types_to_report.items():
             if event_type in visit.event_list_dict:
                 events = visit.get_event_list(event_type)
                 if events:
                      report_html += f'<div class="section"><h2>{title}</h2><ul>'
                      # Limit number of items shown in report
                      for event in events[:15]:
                           # Extract relevant info based on event type
                           info = f"Code: {getattr(event, 'code', 'N/A')}" # Default
                           if event_type == "diagnoses_icd" or event_type == "procedures_icd":
                                info = f"{getattr(event, 'code', 'N/A')} - {getattr(event, 'description', 'N/A')}"
                           elif event_type == "prescriptions":
                                info = f"{getattr(event, 'drug', 'N/A')} ({getattr(event, 'dosage', 'N/A')}) Route: {getattr(event, 'route', 'N/A')}"
                           elif event_type == "labevents":
                                info = f"{getattr(event, 'test_name', 'N/A')}: {getattr(event, 'value', 'N/A')} {getattr(event, 'unit', '')} (Time: {getattr(event, 'time', 'N/A')})"

                           report_html += f"<li>{info}</li>"

                      if len(events) > 15: report_html += "<li>... and more</li>"
                      report_html += "</ul></div>"


        # Add Prediction Section (if available)
        # TODO: Add prediction results to the report if generated

        report_html += """
            <div class="section">
                <h2>Disclaimer</h2>
                <p>This report is auto-generated for informational purposes only and may not be complete. It should not substitute professional medical advice or diagnosis.</p>
            </div>
        </body>
        </html>
        """

        st.download_button(
            label="Download Patient Report (HTML)",
            data=report_html,
            file_name=f"patient_{patient_id}_visit_{visit_id}_report.html",
            mime="text/html",
        )


# --- Main Application ---
class App:
    """Main application class orchestrating the Streamlit app."""

    def __init__(self):
        """Initializes the UI and handlers."""
        self.ui = UI()
        # Pass session state to handlers
        self.data_handler = DataHandler(st.session_state)
        self.exploration_handler = ExplorationHandler(st.session_state)
        self.prediction_handler = PredictionHandler(st.session_state)
        self.training_handler = TrainingHandler(st.session_state)
        self.analytics_handler = AnalyticsHandler(st.session_state)

    def run(self):
        """Runs the Streamlit application."""
        app_mode = self.ui.render_sidebar()

        if app_mode == "Home":
            self.ui.render_home()
        elif app_mode == "Data Loading":
            self.ui.render_data_loading(self.data_handler)
        elif app_mode == "Data Exploration":
            self.exploration_handler.render()
        elif app_mode == "Predictive Tasks":
            self.prediction_handler.render()
        elif app_mode == "Model Training & Evaluation":
            self.training_handler.render()
        elif app_mode == "Patient Analytics":
            self.analytics_handler.render()

# --- Entry Point ---
if __name__ == "__main__":
    app = App()
    app.run()
