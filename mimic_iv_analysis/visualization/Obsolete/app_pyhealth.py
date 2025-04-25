# Standard library imports
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

# TODO: refactor the @app_pyhealth.py to be more modular and clean (using python classes, etc.)

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

# Extended dataset to support generic loading of additional tables
class ExtendedMIMIC4Dataset(MIMIC4Dataset):
    def __init__(self, root, tables, code_mapping=None, dev=False, refresh_cache=False):
        # root should point to hosp/ directory
        super().__init__(root=root, tables=tables, code_mapping=code_mapping, dev=dev, refresh_cache=refresh_cache)
        # store full dataset root (parent of hosp/)
        self.full_root = os.path.dirname(root)
        # container for extra table DataFrames
        self.extra_tables = {}

    def parse_tables(self):
        # First parse basic info (patients & admissions)
        patients = self.parse_basic_info({})
        # define which tables go to hosp and icu
        hosp_tables = {
            'admissions', 'patients', 'transfers', 'diagnoses_icd',
            'procedures_icd', 'prescriptions', 'labevents', 'd_labitems',
            'microbiologyevents', 'pharmacy', 'poe', 'poe_detail', 'services'
        }
        icu_tables = {
            'icustays', 'chartevents', 'd_items', 'inputevents',
            'outputevents', 'procedureevents', 'datetimeevents', 'ingredientevents', 'caregiver'
        }
        # loop through requested tables
        for t in self.tables:
            base = t.lower().rstrip('.csv').rstrip('.gz')
            # if a PyHealth parser exists, use it
            parse_fn = getattr(self, f"parse_{base}", None)
            if callable(parse_fn):
                patients = parse_fn(patients)
            else:
                # generic load into extra_tables
                if base in hosp_tables:
                    path = os.path.join(self.root, f"{base}.csv.gz")
                    if not os.path.exists(path):
                        path = os.path.join(self.root, f"{base}.csv")
                elif base in icu_tables:
                    path = os.path.join(self.full_root, 'icu', f"{base}.csv.gz")
                    if not os.path.exists(path):
                        path = os.path.join(self.full_root, 'icu', f"{base}.csv")
                else:
                    # unknown table, skip
                    continue
                try:
                    df = pd.read_csv(path)
                    self.extra_tables[base] = df
                except Exception as e:
                    print(f"Error loading extra table {base}: {e}")
        return patients

# Set page configuration
st.set_page_config(
    page_title="MIMIC-IV Analysis with PyHealth",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    h1 {
        color: #0e4a80;
    }
    h2 {
        color: #1976D2;
    }
    h3 {
        color: #2196F3;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .report-section {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("MIMIC-IV v3.1 Analysis Tool with PyHealth")
st.markdown("This app allows you to load, explore, and analyze MIMIC-IV v3.1 healthcare data using the PyHealth library. You can perform various analysis tasks including mortality prediction, drug recommendation, readmission prediction, and length of stay prediction.")

# Sidebar for navigation and settings
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Home", "Data Loading", "Data Exploration", "Predictive Tasks", "Model Training & Evaluation", "Patient Analytics"]
)

# Global variables
dataset = None
mimic4_sample = None
task_fn = None
model = None

# Helper functions
def load_mimic_data(root_path, tables, code_mapping=None, dev=False):
    """
    Load MIMIC-IV v3.1 dataset using PyHealth's native MIMIC4Dataset loader.
    Handles the directory structure with hosp/ and icu/ subdirectories.

    Args:
        root_path (str): Path to MIMIC-IV dataset root directory (containing hosp/ and icu/ subdirs)
        tables (list): List of tables to load (e.g., ["admissions", "patients"])
        code_mapping (dict, optional): Dictionary for code system mapping
        dev (bool): If True, loads a smaller subset for development

    Returns:
        MIMIC4Dataset object or None if loading fails

    Example table structure:
        hosp/: admissions, patients, labevents, diagnoses_icd, procedures_icd, etc.
        icu/: chartevents, icustays, inputevents, outputevents, etc.
    """
    try:
        with st.spinner('Loading MIMIC-IV dataset... This might take some time.'):
            # Verify directory structure
            hosp_dir = os.path.join(root_path, 'hosp')
            icu_dir = os.path.join(root_path, 'icu')

            if not os.path.exists(hosp_dir) or not os.path.exists(icu_dir):
                st.error(f"Expected MIMIC-IV v3.1 directory structure with hosp/ and icu/ subdirectories in {root_path}")
                return None

            # Map table names to their correct subdirectory
            hosp_tables = {
                'admissions', 'patients', 'transfers', 'diagnoses_icd',
                'procedures_icd', 'prescriptions', 'labevents', 'd_labitems',
                'microbiologyevents', 'pharmacy', 'poe', 'poe_detail', 'services'
            }

            icu_tables = {
                'icustays', 'chartevents', 'd_items', 'inputevents',
                'outputevents', 'procedureevents', 'datetimeevents'
            }

            # Process table names to match PyHealth's expectations
            resolved_tables = []
            for t in tables:
                # Remove any file extensions and convert to lowercase
                base = t.lower()
                if base.endswith('.csv.gz'):
                    base = base[:-7]
                elif base.endswith('.csv'):
                    base = base[:-4]

                # Verify table exists in either hosp/ or icu/
                if base in hosp_tables:
                    file_path = os.path.join(hosp_dir, f"{base}.csv.gz")
                    if not os.path.exists(file_path):
                        file_path = os.path.join(hosp_dir, f"{base}.csv")
                        if not os.path.exists(file_path):
                            st.warning(f"Table {base} not found in hosp/ directory")
                            continue
                elif base in icu_tables:
                    file_path = os.path.join(icu_dir, f"{base}.csv.gz")
                    if not os.path.exists(file_path):
                        file_path = os.path.join(icu_dir, f"{base}.csv")
                        if not os.path.exists(file_path):
                            st.warning(f"Table {base} not found in icu/ directory")
                            continue
                else:
                    st.warning(f"Unknown table {base} - not in hosp/ or icu/ schema")
                    continue

                resolved_tables.append(base)

            if not resolved_tables:
                st.error("No valid tables found to load")
                return None

            # resolved_tables may include any v3.1 table; Extended subclass will store extras

            # Load dataset using PyHealth's MIMIC4Dataset (hospital CSVs live under hosp/)
            dataset = ExtendedMIMIC4Dataset(
                root=hosp_dir,  # hosp/ directory for core CSVs; ExtendedMIMC4Dataset handles extras
                tables=resolved_tables,
                code_mapping=code_mapping,
                dev=dev
            )

            return dataset

    except Exception as e:
        st.error(f"Error loading MIMIC-IV dataset: {str(e)}")
        return None

def display_dataset_info(dataset):
    """Display MIMIC-IV dataset information"""
    if dataset:
        st.success("Dataset loaded successfully!")

        # Basic dataset statistics
        st.subheader("Dataset Statistics")
        # stats = dataset.stat() # This returns a string, not a dict

        # Calculate stats directly
        num_patients = len(dataset.patients)
        num_visits = sum(len(p) for p in dataset.patients.values())
        # Note: Calculating total events requires iterating all visits/events and might be slow
        # num_events = sum(len(v.get_event_list(table))
        #                  for p in dataset.patients.values()
        #                  for v in p
        #                  for table in v.available_tables)

        col1, col2 = st.columns(2) # Adjusted columns as event count is intensive
        with col1:
            # st.metric("Number of Patients", stats["patient_num"])
            st.metric("Number of Patients", num_patients)
        with col2:
            # st.metric("Number of Visits", stats["visit_num"])
            st.metric("Number of Visits", num_visits)
        # with col3:
            # st.metric("Total Events", stats["event_num"])
            # st.metric("Total Events", num_events) # Omitted for performance

        # Display sample patients
        st.subheader("Sample Patient Data")
        sample_patients = list(dataset.patients.keys())[:5]
        for i, patient_id in enumerate(sample_patients):
            with st.expander(f"Patient {patient_id}"):
                patient_data = dataset.patients[patient_id]
                visits = patient_data.visits

                for visit_id, visit in visits.items():
                    st.write(f"Visit ID: {visit_id}")

                    # Basic visit information
                    st.write(f"- Admission Time: {visit.encounter_time}")
                    st.write(f"- Discharge Time: {visit.discharge_time}")

                    # Events summary
                    event_types = list(visit.event_list_dict.keys())
                    st.write(f"- Event Types: {', '.join(event_types)}")

                    # Show a sample of events for each type
                    for event_type in event_types:
                        events = visit.get_event_list(event_type)
                        if events and len(events) > 0:
                            with st.expander(f"Sample {event_type} events"):
                                event_df = pd.DataFrame([e.__dict__ for e in events[:5]])
                                st.dataframe(event_df)

def set_task(dataset, task_name):
    """Set a specific task for the dataset"""
    task_functions = {
        'drug_recommendation': drug_recommendation_mimic4_fn,
        'mortality_prediction': mortality_prediction_mimic4_fn,
        'readmission_prediction': readmission_prediction_mimic4_fn,
        'length_of_stay_prediction': length_of_stay_prediction_mimic4_fn
    }

    if task_name in task_functions:
        task_fn = task_functions[task_name]
        with st.spinner(f'Processing {task_name} task...'):
            try:
                sample_dataset = dataset.set_task(task_fn=task_fn)
                return sample_dataset, task_fn
            except Exception as e:
                st.error(f"Error setting task: {e}")
                return None, None
    else:
        st.error(f"Unknown task: {task_name}")
        return None, None

def train_model(model_type, sample_dataset, feature_keys, label_key, mode, epochs=10, batch_size=32):
    """Train a model on the prepared dataset"""
    # Split the dataset
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.7, 0.1, 0.2])

    # Create dataloaders
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    # Select the model
    model_map = {
        'Transformer': Transformer,
        'RETAIN': RETAIN,
        'RNN': RNN,
        'MLP': MLP,
        'CNN': CNN
    }

    ModelClass = model_map.get(model_type)
    if not ModelClass:
        st.error(f"Unknown model type: {model_type}")
        return None, None, None

    # Initialize model
    model = ModelClass(
        dataset=sample_dataset,
        feature_keys=feature_keys,
        label_key=label_key,
        mode=mode
    )

    # Train the model
    from pyhealth.trainer import Trainer

    trainer = Trainer(model=model)

    # Create a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Training callback to update progress
    def progress_callback(epoch, loss, val_loss=None):
        progress = epoch / epochs
        progress_bar.progress(progress)
        progress_text.text(f"Epoch {epoch}/{epochs}: loss={loss:.4f}" +
                           (f", val_loss={val_loss:.4f}" if val_loss is not None else ""))

    # Train the model
    with st.spinner('Training model...'):
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            callback=progress_callback
        )

    # Evaluate the model
    with st.spinner('Evaluating model...'):
        y_true, y_prob = trainer.inference(test_loader)

        # Choose the right metrics function based on the mode
        if mode == 'binary':
            metrics = binary_metrics_fn(y_true, y_prob)
        elif mode == 'multiclass':
            metrics = multiclass_metrics_fn(y_true, y_prob)
        elif mode == 'multilabel':
            metrics = multilabel_metrics_fn(y_true, y_prob)
        # elif mode == 'multitask':
        #     metrics = multitask_metrics_fn(y_true, y_prob)
        else:
            metrics = {'error': 'Unknown mode'}

    return model, metrics, test_loader

# Home page
if app_mode == "Home":
    st.header("Welcome to the MIMIC-IV Analysis Tool")

    st.markdown("""
    This application provides a comprehensive interface for analyzing MIMIC-IV v3.1 healthcare data using the PyHealth library.

    ### Key Features:
    - **Data Loading**: Connect to your MIMIC-IV v3.1 dataset
    - **Data Exploration**: Explore patient demographics, diagnoses, procedures, and more
    - **Predictive Tasks**: Perform various healthcare predictive tasks including:
        - Mortality prediction
        - Drug recommendation
        - Readmission prediction
        - Length of stay prediction
    - **Model Training & Evaluation**: Train and evaluate machine learning models
    - **Patient Analytics**: Analyze individual patient journeys and trends

    ### Getting Started:
    1. Navigate to the **Data Loading** section to connect to your MIMIC-IV dataset
    2. Explore the data in the **Data Exploration** section
    3. Choose a predictive task to analyze in the **Predictive Tasks** section
    4. Train and evaluate models in the **Model Training & Evaluation** section

    ### About MIMIC-IV:
    MIMIC-IV is a large, freely accessible database comprising de-identified health-related data associated with patients admitted to the intensive care units at Beth Israel Deaconess Medical Center. It includes clinical data such as demographics, vital signs, laboratory tests, medications, and more.

    ### About PyHealth:
    PyHealth is a Python library designed for healthcare AI research and applications. It provides easy-to-use interfaces for healthcare data processing, feature extraction, and model training on various healthcare predictive tasks.
    """)

    st.info("To begin, select 'Data Loading' from the navigation menu on the left.")

# Data Loading page
elif app_mode == "Data Loading":
    st.header("Load MIMIC-IV Dataset")

    st.markdown("""
    Use this section to connect to your MIMIC-IV v3.1 dataset. You'll need to provide the path to the dataset and select the tables to load.
    """)

    # Dataset path input
    root_path = st.text_input("MIMIC-IV Dataset Path", "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1")

    # Available tables selection
    st.subheader("Select Tables to Load")

    # Group tables by module
    st.markdown("#### Hospital (hosp) Module")
    hosp_tables = st.multiselect(
        "Select hospital tables",
        ["admissions", "patients", "transfers", "diagnoses_icd", "procedures_icd",
         "prescriptions", "labevents", "d_labitems", "microbiologyevents", "pharmacy",
         "poe", "poe_detail", "services"]
    )

    st.markdown("#### ICU Module")
    icu_tables = st.multiselect(
        "Select ICU tables",
        ["icustays", "chartevents", "d_items", "inputevents", "outputevents",
         "procedureevents", "datetimeevents"]
    )

    # Combine selected tables
    selected_tables = hosp_tables + icu_tables

    # Advanced options
    with st.expander("Advanced Options"):
        st.markdown("#### Code Mapping")
        st.markdown("Map medical codes from one coding system to another (e.g., NDC to ATC)")

        code_mapping_enabled = st.checkbox("Enable Code Mapping")
        code_mapping = None

        if code_mapping_enabled:
            mapping_source = st.selectbox("Source Coding System", ["NDC", "ICD9CM", "ICD10CM", "ICD10PCS"])
            mapping_target = st.selectbox("Target Coding System", ["ATC", "CCSCM", "ICD10CM", "CCSPCS"])
            mapping_level = st.slider("Mapping Level (for hierarchical coding systems)", 1, 5, 3)

            code_mapping = {mapping_source: (mapping_target, {"target_kwargs": {"level": mapping_level}})}

            st.code(f"Code Mapping: {code_mapping}")

        # Development mode option (for faster loading with subset of data)
        dev_mode = st.checkbox("Development Mode (loads smaller subset of data)", True)

    # Load button
    if st.button("Load Dataset", type="primary"):
        if not root_path:
            st.error("Please enter a valid path to the MIMIC-IV dataset")
        elif not selected_tables:
            st.error("Please select at least one table to load")
        else:
            # Verify directory structure
            hosp_dir = os.path.join(root_path, 'hosp')
            icu_dir = os.path.join(root_path, 'icu')

            if not os.path.exists(hosp_dir) or not os.path.exists(icu_dir):
                st.error(f"Expected MIMIC-IV v3.1 directory structure with hosp/ and icu/ subdirectories in {root_path}")
            else:
                # Load the dataset
                dataset = load_mimic_data(root_path, selected_tables, code_mapping, dev_mode)

                if dataset:
                    # Store the dataset in session state for other pages to access
                    st.session_state['mimic_dataset'] = dataset

                    # Display dataset info
                    display_dataset_info(dataset)

                    # Success message with next steps
                    st.success("✅ Dataset loaded successfully! You can now proceed to Data Exploration or Predictive Tasks.")
                    st.markdown("**Next Steps:**")
                    st.markdown("1. Go to **Data Exploration** to analyze the dataset content")
                    st.markdown("2. Go to **Predictive Tasks** to perform healthcare predictive tasks")

    # Display info about previously loaded dataset
    if 'mimic_dataset' in st.session_state:
        st.subheader("Previously Loaded Dataset")
        display_dataset_info(st.session_state['mimic_dataset'])

# Data Exploration page
elif app_mode == "Data Exploration":
    st.header("Explore MIMIC-IV Dataset")

    # Check if dataset is loaded
    if 'mimic_dataset' not in st.session_state:
        st.warning("Please load a dataset first in the Data Loading section.")
        st.stop()

    dataset = st.session_state['mimic_dataset']

    # Dataset overview
    st.subheader("Dataset Overview")

    # Show dataset statistics
    stats = dataset.stat()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Patients", stats["patient_num"])
    with col2:
        st.metric("Number of Visits", stats["visit_num"])
    with col3:
        st.metric("Total Events", stats["event_num"])

    # Exploration options
    st.subheader("Exploration Options")

    exploration_option = st.radio(
        "Select what to explore:",
        ["Patient Demographics", "Diagnoses", "Procedures", "Medications", "Lab Tests", "ICU Stays", "Extra Tables"]
    )

    # Extract and display data based on selection
    if exploration_option == "Patient Demographics":
        st.subheader("Patient Demographics")

        # Extract patient data
        patient_data = []
        for patient_id, patient in dataset.patients.items():
            # Get the first visit for basic info
            if patient.visits:
                first_visit = list(patient.visits.values())[0]
                patient_data.append({
                    "patient_id": patient_id,
                    "gender": patient.gender if hasattr(patient, 'gender') else "Unknown",
                    "age": patient.age if hasattr(patient, 'age') else "Unknown",
                    "num_visits": len(patient.visits),
                })

        if patient_data:
            patient_df = pd.DataFrame(patient_data)

            # Display sample of patient data
            st.write("Sample Patient Data:")
            st.dataframe(patient_df.head(10))

            # Gender distribution
            if 'gender' in patient_df.columns:
                st.subheader("Gender Distribution")
                gender_counts = patient_df['gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']

                fig = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution')
                st.plotly_chart(fig)

            # Age distribution
            if 'age' in patient_df.columns and patient_df['age'].dtype != object:
                st.subheader("Age Distribution")
                fig = px.histogram(patient_df, x='age', nbins=20, title='Age Distribution')
                fig.update_layout(xaxis_title='Age', yaxis_title='Count')
                st.plotly_chart(fig)

            # Number of visits distribution
            st.subheader("Visits per Patient")
            fig = px.histogram(patient_df, x='num_visits', nbins=20, title='Number of Visits per Patient')
            fig.update_layout(xaxis_title='Number of Visits', yaxis_title='Count')
            st.plotly_chart(fig)
        else:
            st.info("No patient demographic data available.")

    elif exploration_option == "Diagnoses":
        st.subheader("Diagnoses Analysis")

        # Extract diagnosis data
        diagnosis_data = []

        for patient_id, patient in dataset.patients.items():
            for visit_id, visit in patient.visits.items():
                # Check if diagnoses events exist
                diagnoses = visit.get_event_from_type("diagnosis_icd")
                if diagnoses:
                    for diag in diagnoses:
                        diagnosis_data.append({
                            "patient_id": patient_id,
                            "visit_id": visit_id,
                            "icd_code": diag.code if hasattr(diag, 'code') else "Unknown",
                            "icd_version": diag.icd_version if hasattr(diag, 'icd_version') else "Unknown",
                            "description": diag.description if hasattr(diag, 'description') else "Unknown",
                        })

        if diagnosis_data:
            diagnosis_df = pd.DataFrame(diagnosis_data)

            # Display sample of diagnosis data
            st.write("Sample Diagnosis Data:")
            st.dataframe(diagnosis_df.head(10))

            # Top diagnoses
            st.subheader("Top 20 Diagnoses")
            top_diagnoses = diagnosis_df['icd_code'].value_counts().reset_index().head(20)
            top_diagnoses.columns = ['ICD Code', 'Count']

            fig = px.bar(top_diagnoses, x='ICD Code', y='Count', title='Top 20 Diagnoses')
            st.plotly_chart(fig)

            # ICD version distribution
            if 'icd_version' in diagnosis_df.columns:
                st.subheader("ICD Version Distribution")
                version_counts = diagnosis_df['icd_version'].value_counts().reset_index()
                version_counts.columns = ['ICD Version', 'Count']

                fig = px.pie(version_counts, values='Count', names='ICD Version', title='ICD Version Distribution')
                st.plotly_chart(fig)
        else:
            st.info("No diagnosis data available. Make sure you've loaded the diagnoses_icd table.")

    elif exploration_option == "Procedures":
        st.subheader("Procedures Analysis")

        # Extract procedure data
        procedure_data = []

        for patient_id, patient in dataset.patients.items():
            for visit_id, visit in patient.visits.items():
                # Check if procedures events exist
                procedures = visit.get_event_from_type("procedures_icd")
                if procedures:
                    for proc in procedures:
                        procedure_data.append({
                            "patient_id": patient_id,
                            "visit_id": visit_id,
                            "icd_code": proc.code if hasattr(proc, 'code') else "Unknown",
                            "icd_version": proc.icd_version if hasattr(proc, 'icd_version') else "Unknown",
                            "description": proc.description if hasattr(proc, 'description') else "Unknown",
                        })

        if procedure_data:
            procedure_df = pd.DataFrame(procedure_data)

            # Display sample of procedure data
            st.write("Sample Procedure Data:")
            st.dataframe(procedure_df.head(10))

            # Top procedures
            st.subheader("Top 20 Procedures")
            top_procedures = procedure_df['icd_code'].value_counts().reset_index().head(20)
            top_procedures.columns = ['ICD Code', 'Count']

            fig = px.bar(top_procedures, x='ICD Code', y='Count', title='Top 20 Procedures')
            st.plotly_chart(fig)
        else:
            st.info("No procedure data available. Make sure you've loaded the procedures_icd table.")

    elif exploration_option == "Medications":
        st.subheader("Medications Analysis")

        # Extract medication data
        medication_data = []

        for patient_id, patient in dataset.patients.items():
            for visit_id, visit in patient.visits.items():
                # Check if prescriptions events exist
                prescriptions = visit.get_event_from_type("prescriptions")
                if prescriptions:
                    for med in prescriptions:
                        medication_data.append({
                            "patient_id": patient_id,
                            "visit_id": visit_id,
                            "drug": med.drug if hasattr(med, 'drug') else "Unknown",
                            "drug_code": med.code if hasattr(med, 'code') else "Unknown",
                            "dosage": med.dosage if hasattr(med, 'dosage') else "Unknown",
                            "route": med.route if hasattr(med, 'route') else "Unknown",
                        })

        if medication_data:
            medication_df = pd.DataFrame(medication_data)

            # Display sample of medication data
            st.write("Sample Medication Data:")
            st.dataframe(medication_df.head(10))

            # Top medications
            st.subheader("Top 20 Medications")
            top_meds = medication_df['drug'].value_counts().reset_index().head(20)
            top_meds.columns = ['Medication', 'Count']

            fig = px.bar(top_meds, x='Medication', y='Count', title='Top 20 Medications')
            st.plotly_chart(fig)

            # Medication routes
            if 'route' in medication_df.columns:
                st.subheader("Medication Routes")
                route_counts = medication_df['route'].value_counts().reset_index()
                route_counts.columns = ['Route', 'Count']

                fig = px.pie(route_counts, values='Count', names='Route', title='Medication Routes')
                st.plotly_chart(fig)
        else:
            st.info("No medication data available. Make sure you've loaded the prescriptions table.")

    elif exploration_option == "Lab Tests":
        st.subheader("Lab Tests Analysis")

        # Extract lab data
        lab_data = []

        for patient_id, patient in dataset.patients.items():
            for visit_id, visit in patient.visits.items():
                # Check if labevents exist
                labs = visit.get_event_from_type("labevents")
                if labs:
                    for lab in labs:
                        lab_data.append({
                            "patient_id": patient_id,
                            "visit_id": visit_id,
                            "test_name": lab.test_name if hasattr(lab, 'test_name') else "Unknown",
                            "test_code": lab.code if hasattr(lab, 'code') else "Unknown",
                            "value": lab.value if hasattr(lab, 'value') else None,
                            "unit": lab.unit if hasattr(lab, 'unit') else "Unknown",
                        })

        if lab_data:
            lab_df = pd.DataFrame(lab_data)

            # Display sample of lab data
            st.write("Sample Lab Test Data:")
            st.dataframe(lab_df.head(10))

            # Top lab tests
            st.subheader("Top 20 Lab Tests")
            top_labs = lab_df['test_name'].value_counts().reset_index().head(20)
            top_labs.columns = ['Lab Test', 'Count']

            fig = px.bar(top_labs, x='Lab Test', y='Count', title='Top 20 Lab Tests')
            st.plotly_chart(fig)
        else:
            st.info("No lab test data available. Make sure you've loaded the labevents table.")

    elif exploration_option == "ICU Stays":
        st.subheader("ICU Stays Analysis")

        # Extract ICU stay data
        icu_data = []

        for patient_id, patient in dataset.patients.items():
            for visit_id, visit in patient.visits.items():
                # Check for ICU stays
                if hasattr(visit, 'icu_stays') and visit.icu_stays:
                    for icu_stay_id, icu_stay in visit.icu_stays.items():
                        icu_data.append({
                            "patient_id": patient_id,
                            "visit_id": visit_id,
                            "icu_stay_id": icu_stay_id,
                            "intime": icu_stay.intime if hasattr(icu_stay, 'intime') else None,
                            "outtime": icu_stay.outtime if hasattr(icu_stay, 'outtime') else None,
                            "icu_type": icu_stay.icu_type if hasattr(icu_stay, 'icu_type') else "Unknown",
                        })

        if icu_data:
            icu_df = pd.DataFrame(icu_data)

            # Calculate LOS for ICU stays with valid in/out times
            if 'intime' in icu_df.columns and 'outtime' in icu_df.columns:
                # Convert to datetime if they're strings
                if icu_df['intime'].dtype == object:
                    icu_df['intime'] = pd.to_datetime(icu_df['intime'], errors='coerce')
                if icu_df['outtime'].dtype == object:
                    icu_df['outtime'] = pd.to_datetime(icu_df['outtime'], errors='coerce')

                # Calculate length of stay in days
                icu_df['los_days'] = (icu_df['outtime'] - icu_df['intime']).dt.total_seconds() / (24 * 3600)
                icu_df = icu_df[icu_df['los_days'] >= 0]  # Filter out invalid LOS

            # Display sample of ICU data
            st.write("Sample ICU Stay Data:")
            st.dataframe(icu_df.head(10))

            # ICU length of stay distribution
            if 'los_days' in icu_df.columns:
                st.subheader("ICU Length of Stay Distribution")

                # Cap LOS to 30 days for better visualization
                los_capped = icu_df[icu_df['los_days'] <= 30]

                fig = px.histogram(los_capped, x='los_days', nbins=30, title='ICU Length of Stay Distribution (capped at 30 days)')
                fig.update_layout(xaxis_title='Length of Stay (days)', yaxis_title='Count')
                st.plotly_chart(fig)

                # LOS statistics
                st.write("ICU Length of Stay Statistics:")
                los_stats = icu_df['los_days'].describe()
                st.write(los_stats)

            # ICU type distribution
            if 'icu_type' in icu_df.columns:
                st.subheader("ICU Type Distribution")
                icu_type_counts = icu_df['icu_type'].value_counts().reset_index()
                icu_type_counts.columns = ['ICU Type', 'Count']

                fig = px.pie(icu_type_counts, values='Count', names='ICU Type', title='ICU Type Distribution')
                st.plotly_chart(fig)
        else:
            st.info("No ICU stay data available. Make sure you've loaded the icustays table.")

    elif exploration_option == "Extra Tables":
        st.subheader("Extra Tables")
        # Show any tables loaded in ExtendedMIMIC4Dataset.extra_tables
        extra = getattr(dataset, 'extra_tables', None)
        if extra and len(extra) > 0:
            table_name = st.selectbox("Select table to view", list(extra.keys()))
            df = extra.get(table_name)
            if df is not None:
                st.write(f"Dataset shape: {df.shape}")
                st.dataframe(df.head(20))
            else:
                st.info("No data available for this table.")
        else:
            st.info("No extra tables loaded. Please select additional tables in Data Loading.")

# Predictive Tasks page
elif app_mode == "Predictive Tasks":
    st.header("Healthcare Predictive Tasks")

    # Check if dataset is loaded
    if 'mimic_dataset' not in st.session_state:
        st.warning("Please load a dataset first in the Data Loading section.")
        st.stop()

    dataset = st.session_state['mimic_dataset']

    # Task selection
    st.subheader("Select a Predictive Task")

    task_name = st.selectbox(
        "Choose a task to perform",
        ["drug_recommendation", "mortality_prediction", "readmission_prediction", "length_of_stay_prediction"]
    )

    # Task descriptions
    task_descriptions = {
        "drug_recommendation": "Predict medications that should be prescribed based on patient history and current diagnoses.",
        "mortality_prediction": "Predict the probability of in-hospital mortality based on patient data.",
        "readmission_prediction": "Predict the likelihood of patient readmission within a specified timeframe.",
        "length_of_stay_prediction": "Predict the expected length of stay for a patient in the hospital."
    }

    st.markdown(f"**Task Description:** {task_descriptions[task_name]}")

    # Set up the task
    if st.button("Process Task Data", type="primary"):
        with st.spinner(f"Processing {task_name} data..."):
            mimic4_sample, task_fn = set_task(dataset, task_name)

            if mimic4_sample is not None:
                # Store in session state
                st.session_state['mimic4_sample'] = mimic4_sample
                st.session_state['task_name'] = task_name
                st.session_state['task_fn'] = task_fn

                # Display task data info
                st.success(f"Task data processed successfully! {len(mimic4_sample)} samples created.")

                # Display sample distribution
                if hasattr(mimic4_sample, "get_label_distribution"):
                    label_dist = mimic4_sample.get_label_distribution()

                    st.subheader("Label Distribution")

                    if isinstance(label_dist, dict):
                        # Convert dict to DataFrame for display
                        label_df = pd.DataFrame(list(label_dist.items()), columns=['Label', 'Count'])

                        # Sort by count descending
                        label_df = label_df.sort_values('Count', ascending=False)

                        # Display as bar chart
                        fig = px.bar(label_df, x='Label', y='Count', title='Label Distribution')
                        st.plotly_chart(fig)

                        # Display as dataframe
                        st.dataframe(label_df)
                    else:
                        st.write("Label Distribution:", label_dist)

                # Show sample data
                st.subheader("Sample Data")

                # Get a few samples to display
                sample_indices = list(range(min(5, len(mimic4_sample))))
                for idx in sample_indices:
                    sample = mimic4_sample[idx]
                    with st.expander(f"Sample {idx+1}"):
                        # Convert sample to DataFrame for better display
                        sample_df = pd.DataFrame([sample])
                        st.dataframe(sample_df)

                # Next steps
                st.info("You can now proceed to the 'Model Training & Evaluation' section to train models on this task.")
            else:
                st.error("Failed to process task data. Please check if you have the necessary tables loaded.")

    # Display info about previously processed task
    if 'mimic4_sample' in st.session_state:
        st.subheader(f"Previously Processed Task: {st.session_state['task_name']}")
        st.write(f"Number of samples: {len(st.session_state['mimic4_sample'])}")

        # Next steps
        st.success("Task data is ready! Go to 'Model Training & Evaluation' to train models.")

# Model Training & Evaluation page
elif app_mode == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")

    # Check if task data is processed
    if 'mimic4_sample' not in st.session_state:
        st.warning("Please process a task first in the Predictive Tasks section.")
        st.stop()

    mimic4_sample = st.session_state['mimic4_sample']
    task_name = st.session_state['task_name']

    st.subheader(f"Training Models for: {task_name}")

    # Model selection
    st.markdown("### Select Model")

    model_type = st.selectbox(
        "Choose a model",
        ["Transformer", "RETAIN", "RNN", "MLP", "CNN"]
    )

    # Model descriptions
    model_descriptions = {
        "Transformer": "Uses self-attention mechanism to capture dependencies between different time steps.",
        "RETAIN": "Reverse Time Attention model for interpretable predictions in healthcare.",
        "RNN": "Recurrent Neural Network (GRU/LSTM) for sequential healthcare data.",
        "MLP": "Multi-Layer Perceptron for baseline predictions.",
        "CNN": "Convolutional Neural Network for pattern detection in healthcare data."
    }

    st.markdown(f"**Model Description:** {model_descriptions[model_type]}")

    # Feature selection
    st.markdown("### Select Features")

    # Get available features from sample
    if len(mimic4_sample) > 0:
        sample = mimic4_sample[0]
        available_features = [k for k in sample.keys() if k != 'label']

        feature_keys = st.multiselect(
            "Select features to use",
            available_features,
            default=available_features[:2]  # Default: select first two features
        )

        # Get label info
        if 'label' in sample:
            label_key = 'label'
            if isinstance(sample['label'], dict):
                label_keys = list(sample['label'].keys())
                label_key = st.selectbox("Select label key", label_keys)

            st.write(f"Label key: {label_key}")
        else:
            st.error("No label found in the sample data.")
            st.stop()
    else:
        st.error("No samples found in the processed data.")
        st.stop()

    # Training parameters
    st.markdown("### Training Parameters")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Number of Epochs", 1, 50, 10)
        batch_size = st.slider("Batch Size", 8, 128, 32)

    with col2:
        mode = st.selectbox(
            "Prediction Mode",
            ["binary", "multiclass", "multilabel"],
            index=0 if task_name in ["mortality_prediction", "readmission_prediction"] else 2
        )

        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )

    # Train button
    if st.button("Train Model", type="primary"):
        # Train the model
        model, metrics, test_loader = train_model(
            model_type=model_type,
            sample_dataset=mimic4_sample,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            epochs=epochs,
            batch_size=batch_size
        )

        if model is not None and metrics is not None:
            # Store in session state
            st.session_state['model'] = model
            st.session_state['metrics'] = metrics
            st.session_state['test_loader'] = test_loader

            # Display evaluation metrics
            st.subheader("Evaluation Metrics")

            # Create metrics cards based on the mode
            if mode == 'binary':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>AUC-ROC</h3>
                            <h2>{metrics['roc_auc']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Accuracy</h3>
                            <h2>{metrics['accuracy']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>F1 Score</h3>
                            <h2>{metrics['f1']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Precision and recall
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Precision</h3>
                            <h2>{metrics['precision']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Recall</h3>
                            <h2>{metrics['recall']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            elif mode == 'multiclass':
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Accuracy</h3>
                            <h2>{metrics['accuracy']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Macro F1</h3>
                            <h2>{metrics['macro_f1']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Weighted metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Weighted Precision</h3>
                            <h2>{metrics['weighted_precision']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Weighted Recall</h3>
                            <h2>{metrics['weighted_recall']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Per-class metrics
                if 'class_metrics' in metrics:
                    st.subheader("Per-Class Metrics")
                    class_metrics_df = pd.DataFrame(metrics['class_metrics'])
                    st.dataframe(class_metrics_df)

            elif mode == 'multilabel':
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Macro AUC</h3>
                            <h2>{metrics['macro_auc']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Micro AUC</h3>
                            <h2>{metrics['micro_auc']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Precision and recall
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Precision@5</h3>
                            <h2>{metrics['p_5']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Recall@5</h3>
                            <h2>{metrics['r_5']:.4f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Display all metrics in table format
            st.subheader("All Metrics")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df)

            # Save model option
            st.subheader("Save Model")
            model_name = st.text_input("Model Name", f"{task_name}_{model_type}_model")
            save_dir = st.text_input("Save Directory", "./saved_models/")

            if st.button("Save Model"):
                try:
                    # Make sure the directory exists
                    os.makedirs(save_dir, exist_ok=True)

                    # Save the model
                    from pyhealth.trainer import save_model
                    save_model(model, os.path.join(save_dir, model_name))

                    # Save metrics
                    with open(os.path.join(save_dir, f"{model_name}_metrics.json"), 'w') as f:
                        json.dump(metrics, f)

                    st.success(f"Model saved successfully to {os.path.join(save_dir, model_name)}!")
                except Exception as e:
                    st.error(f"Error saving model: {e}")
        else:
            st.error("Model training failed. Please check the logs for errors.")

    # Display previous model results if available
    if 'metrics' in st.session_state:
        st.subheader("Previous Model Results")
        st.write(f"Model Type: {model_type}")
        st.write(f"Task: {task_name}")

        # Display key metrics based on the mode
        metrics = st.session_state['metrics']

        if 'roc_auc' in metrics:
            st.metric("AUC-ROC", f"{metrics['roc_auc']:.4f}")
        if 'accuracy' in metrics:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        if 'f1' in metrics:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")
        if 'macro_auc' in metrics:
            st.metric("Macro AUC", f"{metrics['macro_auc']:.4f}")

# Patient Analytics page
elif app_mode == "Patient Analytics":
    st.header("Patient Analytics")

    # Check if dataset is loaded
    if 'mimic_dataset' not in st.session_state:
        st.warning("Please load a dataset first in the Data Loading section.")
        st.stop()

    dataset = st.session_state['mimic_dataset']

    # Patient selection
    st.subheader("Select Patient")

    # Get all patient IDs
    patient_ids = list(dataset.patients.keys())

    selected_patient_id = st.selectbox("Select Patient ID", patient_ids)

    if selected_patient_id:
        # Get the patient data
        patient = dataset.patients[selected_patient_id]

        # Display patient info
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Patient ID:** {selected_patient_id}")
            if hasattr(patient, 'gender'):
                st.markdown(f"**Gender:** {patient.gender}")
            if hasattr(patient, 'age'):
                st.markdown(f"**Age:** {patient.age}")

        with col2:
            st.markdown(f"**Number of Visits:** {len(patient.visits)}")
            if hasattr(patient, 'race'):
                st.markdown(f"**Race:** {patient.race}")

        # Display patient visits
        st.subheader("Hospital Visits")

        visits = patient.visits

        # Create a timeline of visits
        visit_data = []
        for visit_id, visit in visits.items():
            visit_data.append({
                "visit_id": visit_id,
                "start_time": visit.encounter_time if hasattr(visit, 'encounter_time') else None,
                "end_time": visit.discharge_time if hasattr(visit, 'discharge_time') else None,
                "length_of_stay": (visit.discharge_time - visit.encounter_time).total_seconds() / (24 * 3600)
                if hasattr(visit, 'discharge_time') and hasattr(visit, 'encounter_time') else None
            })

        if visit_data:
            visit_df = pd.DataFrame(visit_data)

            # Sort by start time
            if 'start_time' in visit_df.columns and visit_df['start_time'].dtype != object:
                visit_df = visit_df.sort_values('start_time')

            # Display visit info
            st.write("Visits Timeline:")
            st.dataframe(visit_df)

            # Select visit for detailed view
            st.subheader("Visit Details")

            selected_visit_id = st.selectbox("Select Visit ID", list(visits.keys()))

            if selected_visit_id:
                # Get the visit data
                visit = visits[selected_visit_id]

                # Visit summary
                col1, col2, col3 = st.columns(3)

                with col1:
                    if hasattr(visit, 'encounter_time'):
                        st.markdown(f"**Admission Time:** {visit.encounter_time}")
                with col2:
                    if hasattr(visit, 'discharge_time'):
                        st.markdown(f"**Discharge Time:** {visit.discharge_time}")
                with col3:
                    if hasattr(visit, 'encounter_time') and hasattr(visit, 'discharge_time'):
                        los = (visit.discharge_time - visit.encounter_time).total_seconds() / (24 * 3600)
                        st.markdown(f"**Length of Stay:** {los:.2f} days")

                # Get event types for this visit
                event_types = list(visit.event_list_dict.keys())

                # Display events by type
                for event_type in event_types:
                    with st.expander(f"{event_type.capitalize()} Events"):
                        events = visit.get_event_list(event_type)

                        if events:
                            # Convert events to dataframe
                            event_data = []
                            for event in events:
                                event_dict = event.__dict__
                                event_data.append(event_dict)

                            event_df = pd.DataFrame(event_data)
                            st.dataframe(event_df)

                            # Event-specific visualizations
                            if event_type == "labevents" and len(event_df) > 0:
                                st.subheader("Lab Tests Timeline")

                                if 'time' in event_df.columns and 'value' in event_df.columns and 'test_name' in event_df.columns:
                                    # Convert time to datetime if it's a string
                                    if event_df['time'].dtype == object:
                                        event_df['time'] = pd.to_datetime(event_df['time'], errors='coerce')

                                    # Select tests to visualize
                                    unique_tests = event_df['test_name'].unique()
                                    selected_tests = st.multiselect(
                                        "Select Lab Tests to Visualize",
                                        unique_tests,
                                        default=unique_tests[:3] if len(unique_tests) > 0 else []
                                    )

                                    if selected_tests:
                                        # Filter data for selected tests
                                        filtered_df = event_df[event_df['test_name'].isin(selected_tests)]

                                        # Check if there are numeric values
                                        filtered_df['value_num'] = pd.to_numeric(filtered_df['value'], errors='coerce')
                                        filtered_df = filtered_df.dropna(subset=['value_num', 'time'])

                                        if len(filtered_df) > 0:
                                            # Create timeline plot
                                            fig = px.line(
                                                filtered_df,
                                                x='time',
                                                y='value_num',
                                                color='test_name',
                                                title='Lab Test Results Over Time',
                                                labels={'time': 'Date', 'value_num': 'Value', 'test_name': 'Test'}
                                            )
                                            st.plotly_chart(fig)
                                        else:
                                            st.info("No numeric values available for the selected tests.")

                            elif event_type == "chartevents" and len(event_df) > 0:
                                st.subheader("Vital Signs Timeline")

                                if 'time' in event_df.columns and 'value' in event_df.columns and 'code' in event_df.columns:
                                    # Convert time to datetime if it's a string
                                    if event_df['time'].dtype == object:
                                        event_df['time'] = pd.to_datetime(event_df['time'], errors='coerce')

                                    # Select vitals to visualize
                                    unique_vitals = event_df['code'].unique()
                                    selected_vitals = st.multiselect(
                                        "Select Vital Signs to Visualize",
                                        unique_vitals,
                                        default=unique_vitals[:3] if len(unique_vitals) > 0 else []
                                    )

                                    if selected_vitals:
                                        # Filter data for selected vitals
                                        filtered_df = event_df[event_df['code'].isin(selected_vitals)]

                                        # Check if there are numeric values
                                        filtered_df['value_num'] = pd.to_numeric(filtered_df['value'], errors='coerce')
                                        filtered_df = filtered_df.dropna(subset=['value_num', 'time'])

                                        if len(filtered_df) > 0:
                                            # Create timeline plot
                                            fig = px.line(
                                                filtered_df,
                                                x='time',
                                                y='value_num',
                                                color='code',
                                                title='Vital Signs Over Time',
                                                labels={'time': 'Date', 'value_num': 'Value', 'code': 'Vital Sign'}
                                            )
                                            st.plotly_chart(fig)
                                        else:
                                            st.info("No numeric values available for the selected vital signs.")
                        else:
                            st.info(f"No {event_type} events found for this visit.")

                # ICU Stays for this visit
                if hasattr(visit, 'icu_stays') and visit.icu_stays:
                    st.subheader("ICU Stays")

                    icu_data = []
                    for icu_stay_id, icu_stay in visit.icu_stays.items():
                        icu_data.append({
                            "icu_stay_id": icu_stay_id,
                            "intime": icu_stay.intime if hasattr(icu_stay, 'intime') else None,
                            "outtime": icu_stay.outtime if hasattr(icu_stay, 'outtime') else None,
                            "icu_type": icu_stay.icu_type if hasattr(icu_stay, 'icu_type') else "Unknown",
                            "los_days": (icu_stay.outtime - icu_stay.intime).total_seconds() / (24 * 3600)
                            if hasattr(icu_stay, 'outtime') and hasattr(icu_stay, 'intime') else None
                        })

                    icu_df = pd.DataFrame(icu_data)
                    st.dataframe(icu_df)

                # If model is trained, make predictions for this patient
                if 'model' in st.session_state and st.session_state['task_name']:
                    st.subheader("Model Predictions")

                    st.write(f"Task: {st.session_state['task_name']}")

                    if st.button("Make Prediction for this Patient"):
                        try:
                            # Create a sample for this patient/visit
                            from pyhealth.tasks import drug_recommendation_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn, length_of_stay_prediction_mimic4_fn

                            task_functions = {
                                'drug_recommendation': drug_recommendation_mimic4_fn,
                                'mortality_prediction': mortality_prediction_mimic4_fn,
                                'readmission_prediction': readmission_prediction_mimic4_fn,
                                'length_of_stay_prediction': length_of_stay_prediction_mimic4_fn
                            }

                            task_fn = task_functions[st.session_state['task_name']]

                            # Generate sample for this patient/visit
                            sample = task_fn(patient_id=selected_patient_id, visit_id=selected_visit_id, patient=patient, visit=visit)

                            if sample:
                                # Make prediction
                                model = st.session_state['model']

                                # Convert sample to a format suitable for the model
                                from pyhealth.datasets import Sample
                                sample_obj = Sample(**sample)

                                # Make prediction
                                prediction = model.predict([sample_obj])

                                # Display prediction
                                st.subheader("Prediction Result")

                                if st.session_state['task_name'] == 'drug_recommendation':
                                    # Get top predicted drugs
                                    if hasattr(model, 'label_tokenizer'):
                                        # Get top 5 drug predictions
                                        top_indices = np.argsort(prediction[0])[::-1][:5]
                                        top_drug_codes = [model.label_tokenizer.decode(idx) for idx in top_indices]
                                        top_probs = [prediction[0][idx] for idx in top_indices]

                                        # Display predictions
                                        results_df = pd.DataFrame({
                                            'Drug Code': top_drug_codes,
                                            'Probability': top_probs
                                        })
                                        st.dataframe(results_df)

                                        # Visualize top drugs
                                        fig = px.bar(
                                            results_df,
                                            x='Drug Code',
                                            y='Probability',
                                            title='Top 5 Drug Recommendations'
                                        )
                                        st.plotly_chart(fig)

                                elif st.session_state['task_name'] == 'mortality_prediction':
                                    # Display mortality risk
                                    mortality_risk = prediction[0][0]

                                    # Gauge chart for mortality risk
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=mortality_risk,
                                        title={'text': "Mortality Risk"},
                                        gauge={
                                            'axis': {'range': [0, 1]},
                                            'bar': {'color': "darkred"},
                                            'steps': [
                                                {'range': [0, 0.3], 'color': "green"},
                                                {'range': [0.3, 0.7], 'color': "orange"},
                                                {'range': [0.7, 1], 'color': "red"}
                                            ]
                                        }
                                    ))
                                    st.plotly_chart(fig)

                                elif st.session_state['task_name'] == 'readmission_prediction':
                                    # Display readmission risk
                                    readmission_risk = prediction[0][0]

                                    # Gauge chart for readmission risk
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=readmission_risk,
                                        title={'text': "Readmission Risk"},
                                        gauge={
                                            'axis': {'range': [0, 1]},
                                            'bar': {'color': "darkblue"},
                                            'steps': [
                                                {'range': [0, 0.3], 'color': "green"},
                                                {'range': [0.3, 0.7], 'color': "yellow"},
                                                {'range': [0.7, 1], 'color': "orange"}
                                            ]
                                        }
                                    ))
                                    st.plotly_chart(fig)

                                elif st.session_state['task_name'] == 'length_of_stay_prediction':
                                    # Display predicted LOS
                                    predicted_los = prediction[0][0]

                                    # Convert to days if needed
                                    if predicted_los > 100:  # Likely in hours
                                        predicted_los_days = predicted_los / 24
                                    else:
                                        predicted_los_days = predicted_los

                                    st.metric("Predicted Length of Stay", f"{predicted_los_days:.2f} days")

                                    # Compare to actual LOS if available
                                    if hasattr(visit, 'encounter_time') and hasattr(visit, 'discharge_time'):
                                        actual_los = (visit.discharge_time - visit.encounter_time).total_seconds() / (24 * 3600)
                                        st.metric("Actual Length of Stay", f"{actual_los:.2f} days",
                                                delta=f"{predicted_los_days - actual_los:.2f} days")
                            else:
                                st.error("Could not generate a sample for this patient/visit for the selected task.")
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")

                # Generate patient report
                st.subheader("Generate Patient Report")

                if st.button("Generate Comprehensive Patient Report"):
                    # Create a report
                    st.markdown("""
                    <div class="report-section">
                        <h2>Patient Clinical Report</h2>
                        <p>This report summarizes the clinical information for the selected patient.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Basic info
                    st.markdown(f"""
                    <div class="report-section">
                        <h3>Patient Information</h3>
                        <p><strong>Patient ID:</strong> {selected_patient_id}</p>
                        <p><strong>Gender:</strong> {patient.gender if hasattr(patient, 'gender') else "Unknown"}</p>
                        <p><strong>Age:</strong> {patient.age if hasattr(patient, 'age') else "Unknown"}</p>
                        <p><strong>Race:</strong> {patient.race if hasattr(patient, 'race') else "Unknown"}</p>
                        <p><strong>Number of Hospital Visits:</strong> {len(patient.visits)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Current visit summary
                    # Calculate LOS if dates are available
                    los_text = "Unknown"
                    if hasattr(visit, 'encounter_time') and hasattr(visit, 'discharge_time'):
                        los_days = (visit.discharge_time - visit.encounter_time).total_seconds() / (24 * 3600)
                        los_text = f"{los_days:.2f} days"

                    st.markdown(f"""
                    <div class="report-section">
                        <h3>Current Visit Summary</h3>
                        <p><strong>Visit ID:</strong> {selected_visit_id}</p>
                        <p><strong>Admission Time:</strong> {visit.encounter_time if hasattr(visit, 'encounter_time') else "Unknown"}</p>
                        <p><strong>Discharge Time:</strong> {visit.discharge_time if hasattr(visit, 'discharge_time') else "Unknown"}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Diagnoses
                    diagnoses = visit.get_event_from_type("diagnoses_icd") if "diagnoses_icd" in event_types else []
                    if diagnoses:
                        st.markdown("""
                        <div class="report-section">
                            <h3>Diagnoses</h3>
                        """, unsafe_allow_html=True)

                        for i, diag in enumerate(diagnoses[:10]):
                            st.markdown(f"""
                            <p>{i+1}. {diag.description if hasattr(diag, 'description') else diag.code}</p>
                            """, unsafe_allow_html=True)

                        if len(diagnoses) > 10:
                            st.markdown(f"""
                            <p>... and {len(diagnoses) - 10} more diagnoses</p>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Procedures
                    procedures = visit.get_event_from_type("procedures_icd") if "procedures_icd" in event_types else []
                    if procedures:
                        st.markdown("""
                        <div class="report-section">
                            <h3>Procedures</h3>
                        """, unsafe_allow_html=True)

                        for i, proc in enumerate(procedures[:10]):
                            st.markdown(f"""
                            <p>{i+1}. {proc.description if hasattr(proc, 'description') else proc.code}</p>
                            """, unsafe_allow_html=True)

                        if len(procedures) > 10:
                            st.markdown(f"""
                            <p>... and {len(procedures) - 10} more procedures</p>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Medications
                    medications = visit.get_event_from_type("prescriptions") if "prescriptions" in event_types else []
                    if medications:
                        st.markdown("""
                        <div class="report-section">
                            <h3>Medications</h3>
                        """, unsafe_allow_html=True)

                        for i, med in enumerate(medications[:10]):
                            st.markdown(f"""
                            <p>{i+1}. {med.drug if hasattr(med, 'drug') else med.code} {f"({med.dosage})" if hasattr(med, 'dosage') and med.dosage else ""}</p>
                            """, unsafe_allow_html=True)

                        if len(medications) > 10:
                            st.markdown(f"""
                            <p>... and {len(medications) - 10} more medications</p>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Lab results
                    labs = visit.get_event_from_type("labevents") if "labevents" in event_types else []
                    if labs:
                        # Get the most recent labs
                        labs_sorted = sorted(labs, key=lambda x: x.time if hasattr(x, 'time') else datetime.min, reverse=True)

                        st.markdown("""
                        <div class="report-section">
                            <h3>Recent Laboratory Results</h3>
                        """, unsafe_allow_html=True)

                        for i, lab in enumerate(labs_sorted[:10]):
                            st.markdown(f"""
                            <p>{i+1}. {lab.test_name if hasattr(lab, 'test_name') else lab.code}: {lab.value if hasattr(lab, 'value') else "Unknown"} {lab.unit if hasattr(lab, 'unit') else ""}</p>
                            """, unsafe_allow_html=True)

                        if len(labs) > 10:
                            st.markdown(f"""
                            <p>... and {len(labs) - 10} more lab results</p>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Model predictions
                    if 'model' in st.session_state and st.session_state['task_name']:
                        st.markdown(f"""
                        <div class="report-section">
                            <h3>Clinical Predictions</h3>
                            <p>The following predictions are based on our trained machine learning models:</p>
                            <p><strong>{st.session_state['task_name'].replace('_', ' ').title()}:</strong> [Run prediction to see results]</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Report summary
                    st.markdown("""
                    <div class="report-section">
                        <h3>Summary</h3>
                        <p>This patient report provides a comprehensive overview of the patient's clinical information during their hospital stay. For more detailed information, please refer to the specific sections in the application.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Download report option
                    st.download_button(
                        "Download Patient Report (HTML)",
                        """
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Patient Clinical Report</title>
                            <style>
                                body { font-family: Arial, sans-serif; margin: 40px; }
                                .report-section { margin-bottom: 30px; }
                                h1 { color: #2C3E50; }
                                h2 { color: #3498DB; }
                                h3 { color: #2980B9; }
                            </style>
                        </head>
                        <body>
                            <h1>Patient Clinical Report</h1>
                            <p>Report generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

                            <div class="report-section">
                                <h2>Patient Information</h2>
                                <p><strong>Patient ID:</strong> """ + selected_patient_id + """</p>
                                <p><strong>Gender:</strong> """ + (patient.gender if hasattr(patient, 'gender') else "Unknown") + """</p>
                                <p><strong>Age:</strong> """ + str(patient.age if hasattr(patient, 'age') else "Unknown") + """</p>
                                <p><strong>Race:</strong> """ + (patient.race if hasattr(patient, 'race') else "Unknown") + """</p>
                                <p><strong>Number of Hospital Visits:</strong> """ + str(len(patient.visits)) + """</p>
                            </div>

                            <div class="report-section">
                                <h2>Visit Summary</h2>
                                <p><strong>Visit ID:</strong> """ + selected_visit_id + """</p>
                                <p><strong>Admission Time:</strong> """ + str(visit.encounter_time if hasattr(visit, 'encounter_time') else "Unknown") + """</p>
                                <p><strong>Discharge Time:</strong> """ + str(visit.discharge_time if hasattr(visit, 'discharge_time') else "Unknown") + """</p>
                            </div>

                            <!-- Additional sections would be added here -->

                            <div class="report-section">
                                <h2>Disclaimer</h2>
                                <p>This report is generated for informational purposes only and should not be used for clinical decision making without consultation with a healthcare professional.</p>
                            </div>
                        </body>
                        </html>
                        """,
                        file_name=f"patient_{selected_patient_id}_report.html",
                        mime="text/html"
                    )
        else:
            st.info("No visits found for this patient.")
    else:
        st.info("Please select a patient ID.")
