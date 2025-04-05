import streamlit as st
if not hasattr(st, 'cache_data'):
    st.cache_data = st.cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from mimic_iv_analysis.data.data_loader import MIMICDataLoader
from mimic_iv_analysis.core.eda_module import ExploratoryDataAnalysis
from mimic_iv_analysis.core.order_pattern_module import OrderPatternAnalysis
from mimic_iv_analysis.core.patient_trajectory_module import PatientTrajectoryAnalysis
from mimic_iv_analysis.core.predictive_modeling_module import PredictiveModeling
from mimic_iv_analysis.core.clinical_interpretation_module import ClinicalInterpretation

# Set page configuration
st.set_page_config(
    page_title="MIMIC-IV Discharge Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare-themed styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stApp {background-color: #f8f9fa;}
    .css-18e3th9 {padding-top: 1rem;}
    h1, h2, h3 {color: #2c3e50;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e6f3ff;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4682b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Application title and description
st.title("MIMIC-IV Discharge Analysis Dashboard")
st.markdown("""
    <div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px;'>
    This application provides comprehensive analysis tools for the MIMIC-IV healthcare dataset with a focus on discharge prediction.
    Navigate through different sections using the sidebar menu to explore patient data, analyze order patterns,
    visualize patient trajectories, build predictive models, and interpret clinical patterns.
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation and data loading
st.sidebar.title("Navigation")

# Data path input
default_mimic_path = os.environ.get("DEMO_MIMIC_PATH", "mimic-iv-3.1")
mimic_path = st.sidebar.text_input(
    "MIMIC-IV Dataset Path",
    value=default_mimic_path,
    help="Enter the path to the MIMIC-IV dataset directory"
)

# Function to check if the path exists
def validate_mimic_path(path):
    if not os.path.exists(path):
        st.sidebar.error(f"Path not found: {path}")
        return False

    required_dirs = ['hosp', 'icu']
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(path, dir_name)):
            st.sidebar.warning(f"Missing directory: {dir_name}")
            return False
    return True

# Validate the path
path_valid = validate_mimic_path(mimic_path)

if not path_valid:
    st.warning("⚠️ MIMIC-IV dataset not found at the specified path.")
    st.info("""
    ### Demo Mode
    This application is running in demo mode. Some features may be limited.

    To use with the actual MIMIC-IV dataset:
    1. Download the MIMIC-IV dataset from [PhysioNet](https://physionet.org/content/mimiciv/)
    2. Extract the files to a directory on your computer
    3. Enter the path to that directory in the sidebar
    """)

# Navigation options
app_mode = st.sidebar.radio(
    "Select Section",
    ["Home", "Data Explorer", "Order Pattern Analysis", "Patient Trajectory",
     "Predictive Modeling", "Clinical Interpretation"]
)

# Cache for data loading
@st.cache_data(ttl=3600, show_spinner=False)
def load_patient_data(mimic_path):
    try:
        file_path = os.path.join(mimic_path, 'hosp', 'patients.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path, low_memory=False)
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading patient data: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_admissions_data(mimic_path):
    try:
        file_path = os.path.join(mimic_path, 'hosp', 'admissions.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path, low_memory=False)
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading admissions data: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_transfers_data(mimic_path):
    try:
        file_path = os.path.join(mimic_path, 'hosp', 'transfers.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path, low_memory=False)
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading transfers data: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_poe_data(mimic_path):
    try:
        file_path = os.path.join(mimic_path, 'hosp', 'poe.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path, low_memory=False)
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading POE data: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_poe_detail_data(mimic_path):
    try:
        file_path = os.path.join(mimic_path, 'hosp', 'poe_detail.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path, low_memory=False)
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading POE detail data: {e}")
        return None

# Load data if path is valid
if path_valid:
    with st.sidebar.spinner("Loading data..."):
        patients_df = load_patient_data(mimic_path)
        admissions_df = load_admissions_data(mimic_path)
        transfers_df = load_transfers_data(mimic_path)
        poe_df = load_poe_data(mimic_path)
        poe_detail_df = load_poe_detail_data(mimic_path)

        if all([patients_df is not None, admissions_df is not None, transfers_df is not None,
                poe_df is not None, poe_detail_df is not None]):
            st.sidebar.success("Data loaded successfully!")
        else:
            st.sidebar.error("Some data files could not be loaded. Check the path and file structure.")
else:
    st.sidebar.warning("Please provide a valid path to the MIMIC-IV dataset.")
    st.sidebar.info("You can continue exploring the application interface, but data analysis features will be limited.")

# Home page
if app_mode == "Home":
    st.header("Welcome to the MIMIC-IV Discharge Analysis Dashboard")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("About this Application")
        st.markdown("""
        This comprehensive dashboard is designed to analyze the MIMIC-IV healthcare dataset with a focus on discharge prediction.
        The application provides tools for data exploration, order pattern analysis, patient trajectory visualization,
        predictive modeling, and clinical interpretation.

        ### Key Features:

        - **Data Explorer**: Analyze patient demographics, hospital utilization patterns, length of stay distributions, and admission/discharge trends
        - **Order Pattern Analysis**: Visualize order types throughout hospital stays, order density by time, IV to oral medication transitions, and sequential patterns
        - **Patient Trajectory**: Interactive patient journey visualization, pathway clustering, dimensionality reduction, and survival analysis
        - **Predictive Modeling**: Build and evaluate discharge prediction models with various algorithms
        - **Clinical Interpretation**: Compare provider order patterns, analyze unit-specific discharge patterns, and identify key order sequences

        ### Getting Started:

        1. Enter the path to your MIMIC-IV dataset in the sidebar
        2. Navigate to different sections using the sidebar menu
        3. Use the filters in each section to focus on specific patient cohorts or time periods
        4. Explore the visualizations and analysis tools to gain insights into discharge patterns
        """)

    with col2:
        st.subheader("Dataset Information")
        if path_valid and all([patients_df is not None, admissions_df is not None, transfers_df is not None,
                              poe_df is not None, poe_detail_df is not None]):
            st.metric("Patients", f"{len(patients_df):,}")
            st.metric("Admissions", f"{len(admissions_df):,}")
            st.metric("Transfers", f"{len(transfers_df):,}")
            st.metric("Provider Orders", f"{len(poe_df):,}")
            st.metric("Order Details", f"{len(poe_detail_df):,}")
        else:
            st.info("Dataset statistics will be displayed once a valid MIMIC-IV dataset path is provided.")

# Data Explorer page
elif app_mode == "Data Explorer":
    st.header("Data Explorer")

    if not path_valid or not all([patients_df is not None, admissions_df is not None, transfers_df is not None]):
        st.warning("Please provide a valid path to the MIMIC-IV dataset to enable data exploration.")
        st.stop()

    # Create tabs for different analysis sections
    tabs = st.tabs(["Patient Demographics", "Hospital Utilization", "Length of Stay", "Time Series Analysis"])

    # Patient Demographics tab
    with tabs[0]:
        st.subheader("Patient Demographics and Cohort Statistics")

        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            st.write("#### Age Distribution")
            # Code for age distribution visualization
            st.info("Age distribution visualization will be displayed here.")

            # Gender distribution
            st.write("#### Gender Distribution")
            # Code for gender distribution visualization
            st.info("Gender distribution visualization will be displayed here.")

        with col2:
            # Ethnicity distribution
            st.write("#### Ethnicity Distribution")
            # Code for ethnicity distribution visualization
            st.info("Ethnicity distribution visualization will be displayed here.")

            # Insurance distribution
            st.write("#### Insurance Distribution")
            # Code for insurance distribution visualization
            st.info("Insurance distribution visualization will be displayed here.")

    # Hospital Utilization tab
    with tabs[1]:
        st.subheader("Hospital Utilization Patterns")

        # Admission type distribution
        st.write("#### Admission Type Distribution")
        # Code for admission type visualization
        st.info("Admission type distribution visualization will be displayed here.")

        # Department utilization
        st.write("#### Department Utilization")
        # Code for department utilization visualization
        st.info("Department utilization visualization will be displayed here.")

        # Bed occupancy over time
        st.write("#### Bed Occupancy Over Time")
        # Code for bed occupancy visualization
        st.info("Bed occupancy visualization will be displayed here.")

    # Length of Stay tab
    with tabs[2]:
        st.subheader("Length of Stay Distributions")

        # Overall LOS distribution
        st.write("#### Overall Length of Stay Distribution")
        # Code for overall LOS visualization
        st.info("Overall length of stay distribution visualization will be displayed here.")

        # LOS by department
        st.write("#### Length of Stay by Department")
        # Code for LOS by department visualization
        st.info("Length of stay by department visualization will be displayed here.")

        # LOS by admission type
        st.write("#### Length of Stay by Admission Type")
        # Code for LOS by admission type visualization
        st.info("Length of stay by admission type visualization will be displayed here.")

    # Time Series Analysis tab
    with tabs[3]:
        st.subheader("Interactive Time Series Analysis")

        # Time range selector
        st.write("#### Select Time Range")
        # Code for time range selector
        st.info("Time range selector will be displayed here.")

        # Admission/discharge patterns
        st.write("#### Admission and Discharge Patterns")
        # Code for admission/discharge patterns visualization
        st.info("Admission and discharge patterns visualization will be displayed here.")

        # Seasonal patterns
        st.write("#### Seasonal Patterns")
        # Code for seasonal patterns visualization
        st.info("Seasonal patterns visualization will be displayed here.")

# Order Pattern Analysis page
elif app_mode == "Order Pattern Analysis":
    st.header("Order Pattern Analysis")

    if not path_valid or not all([poe_df is not None, poe_detail_df is not None]):
        st.warning("Please provide a valid path to the MIMIC-IV dataset to enable order pattern analysis.")
        st.stop()

    # Create tabs for different analysis sections
    tabs = st.tabs(["Temporal Heatmaps", "Order Density", "IV to Oral Transition", "Sequential Patterns"])

    # Temporal Heatmaps tab
    with tabs[0]:
        st.subheader("Temporal Heatmaps of Order Types")

        # Order type selector
        st.write("#### Select Order Types")
        # Code for order type selector
        st.info("Order type selector will be displayed here.")

        # Temporal heatmap
        st.write("#### Temporal Heatmap")
        # Code for temporal heatmap visualization
        st.info("Temporal heatmap visualization will be displayed here.")

    # Order Density tab
    with tabs[1]:
        st.subheader("Order Density Visualization")

        # Time of day analysis
        st.write("#### Order Density by Time of Day")
        # Code for time of day visualization
        st.info("Order density by time of day visualization will be displayed here.")

        # Day of week analysis
        st.write("#### Order Density by Day of Week")
        # Code for day of week visualization
        st.info("Order density by day of week visualization will be displayed here.")

    # IV to Oral Transition tab
    with tabs[2]:
        st.subheader("IV to Oral Medication Transition Analysis")

        # Medication selector
        st.write("#### Select Medications")
        # Code for medication selector
        st.info("Medication selector will be displayed here.")

        # Transition analysis
        st.write("#### Transition Analysis")
        # Code for transition analysis visualization
        st.info("IV to oral transition analysis visualization will be displayed here.")

    # Sequential Patterns tab
    with tabs[3]:
        st.subheader("Sequential Pattern Mining of Orders")

        # Pattern mining parameters
        st.write("#### Pattern Mining Parameters")
        # Code for pattern mining parameters
        st.info("Pattern mining parameters will be displayed here.")

        # Sequential patterns
        st.write("#### Sequential Patterns")
        # Code for sequential patterns visualization
        st.info("Sequential patterns visualization will be displayed here.")

# Patient Trajectory page
elif app_mode == "Patient Trajectory":
    st.header("Patient Trajectory Analysis")

    if not path_valid or not all([patients_df is not None, admissions_df is not None, transfers_df is not None]):
        st.warning("Please provide a valid path to the MIMIC-IV dataset to enable patient trajectory analysis.")
        st.stop()

    # Initialize data loader and patient trajectory module
    data_loader = MIMICDataLoader(mimic_path)
    data_loader.data = {
        'patients': patients_df,
        'admissions': admissions_df,
        'transfers': transfers_df,
        'poe': poe_df,
        'poe_detail': poe_detail_df
    }
    data_loader.preprocess_all()

    patient_trajectory = PatientTrajectoryAnalysis(data_loader)

    # Create tabs for different analysis sections
    tabs = st.tabs(["Patient Journey", "Pathway Clustering", "Dimensionality Reduction", "Survival Analysis"])

    # Patient Journey tab
    with tabs[0]:
        patient_trajectory.patient_journey()

    # Pathway Clustering tab
    with tabs[1]:
        patient_trajectory.pathway_clustering()

    # Dimensionality Reduction tab
    with tabs[2]:
        patient_trajectory.dimensionality_reduction()

    # Survival Analysis tab
    with tabs[3]:
        patient_trajectory.survival_analysis()

# Predictive Modeling page
elif app_mode == "Predictive Modeling":
    st.header("Predictive Modeling Interface")

    if not path_valid or not all([patients_df is not None, admissions_df is not None,
                                 transfers_df is not None, poe_df is not None, poe_detail_df is not None]):
        st.warning("Please provide a valid path to the MIMIC-IV dataset to enable predictive modeling.")
        st.stop()

    # Initialize data loader and predictive modeling module
    data_loader = MIMICDataLoader(mimic_path)
    data_loader.data = {
        'patients': patients_df,
        'admissions': admissions_df,
        'transfers': transfers_df,
        'poe': poe_df,
        'poe_detail': poe_detail_df
    }
    data_loader.preprocess_all()

    predictive_modeling = PredictiveModeling(data_loader)

    # Create tabs for different analysis sections
    tabs = st.tabs(["Feature Selection", "Model Training", "Performance Visualization", "Feature Importance"])

    # Feature Selection tab
    with tabs[0]:
        predictive_modeling.feature_selection()

    # Model Training tab
    with tabs[1]:
        predictive_modeling.model_training()

    # Performance Visualization tab
    with tabs[2]:
        predictive_modeling.performance_visualization()

    # Feature Importance tab
    with tabs[3]:
        predictive_modeling.feature_importance()
        st.write("#### ROC Curves")
        # Code for ROC curves
        st.info("ROC curves will be displayed here.")

        # Calibration plots
        st.write("#### Calibration Plots")
        # Code for calibration plots
        st.info("Calibration plots will be displayed here.")

        # Confusion matrix
        st.write("#### Confusion Matrix")
        # Code for confusion matrix
        st.info("Confusion matrix will be displayed here.")

    # Feature Importance tab
    with tabs[3]:
        st.subheader("Feature Importance Visualization")

        # Feature importance plot
        st.write("#### Feature Importance")
        # Code for feature importance plot
        st.info("Feature importance visualization will be displayed here.")

        # Partial dependence plots
        st.write("#### Partial Dependence Plots")
        # Code for partial dependence plots
        st.info("Partial dependence plots will be displayed here.")

# Clinical Interpretation page
elif app_mode == "Clinical Interpretation":
    st.header("Clinical Interpretation")

    if not path_valid or not all([poe_df is not None, poe_detail_df is not None]):
        st.warning("Please provide a valid path to the MIMIC-IV dataset to enable clinical interpretation.")
        st.stop()

    # Initialize data loader and clinical interpretation module
    data_loader = MIMICDataLoader(mimic_path)
    data_loader.data = {
        'patients': patients_df,
        'admissions': admissions_df,
        'transfers': transfers_df,
        'poe': poe_df,
        'poe_detail': poe_detail_df
    }
    data_loader.preprocess_all()

    clinical_interpretation = ClinicalInterpretation(data_loader)

    # Create tabs for different analysis sections
    tabs = st.tabs(["Provider Order Patterns", "Unit-Specific Analysis", "Key Order Sequences"])

    # Provider Order Patterns tab
    with tabs[0]:
        clinical_interpretation.provider_order_patterns()

    # Unit-Specific Analysis tab
    with tabs[1]:
        clinical_interpretation.unit_specific_analysis()

    # Key Order Sequences tab
    with tabs[2]:
        clinical_interpretation.key_order_sequences()

# Add main function at the end of the file
def main():
    """Entry point function for the Streamlit application."""
    # The app is already defined above, so we don't need to do anything extra here
    pass

if __name__ == "__main__":
    main()
