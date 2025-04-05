#!/usr/bin/env python
"""
MIMIC-IV Discharge Analysis Dashboard

A Streamlit application for comprehensive analysis of the MIMIC-IV healthcare dataset
with a focus on discharge prediction. The application provides tools for data exploration,
order pattern analysis, patient trajectory visualization, predictive modeling, and
clinical interpretation.

Compatible with MIMIC-IV dataset version 3.1.
"""

import os
import sys
from pathlib import Path
import datetime
import time
import warnings

# Third-party imports
import streamlit as st
if not hasattr(st, 'cache_data'):
    st.cache_data = st.cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings
warnings.filterwarnings('ignore')

# Import custom modules
from mimic_iv_analysis.data.data_loader import MIMICDataLoader
from mimic_iv_analysis.core.eda_module import ExploratoryDataAnalysis
from mimic_iv_analysis.core.order_pattern_module import OrderPatternAnalysis
from mimic_iv_analysis.core.patient_trajectory_module import PatientTrajectoryAnalysis
from mimic_iv_analysis.core.predictive_modeling_module import PredictiveModeling
from mimic_iv_analysis.core.clinical_interpretation_module import ClinicalInterpretation


class MIMICDashboard:
    """Main class for the MIMIC-IV Discharge Analysis Dashboard.

    This class encapsulates all functionality for the Streamlit application,
    including page configuration, data loading, and rendering different sections
    of the dashboard.
    """

    def __init__(self):
        """Initialize the dashboard application."""
        self.configure_page()
        self.display_app_header()
        self.path_valid, self.app_mode, self.data, self.mimic_path = self.setup_sidebar_navigation()

        # Display demo mode warning if path is not valid
        if not self.path_valid:
            self.display_demo_mode_warning()

        # Initialize data loader if path is valid
        self.data_loader = None
        if self.path_valid and all(df is not None for df in self.data.values()):
            self.data_loader = MIMICDataLoader(self.mimic_path)
            self.data_loader.data = self.data
            self.data_loader.preprocess_all()

    def configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="MIMIC-IV Discharge Analysis",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for healthcare-themed styling with dark mode support
        st.markdown("""
        <style>
            /* Theme-aware styling using CSS variables */
            :root {
                --background-color: #f8f9fa;
                --text-color: #2c3e50;
                --tab-bg-color: #e6f3ff;
                --tab-selected-bg: #4682b4;
                --tab-selected-color: white;
            }

            /* Dark theme overrides */
            @media (prefers-color-scheme: dark) {
                :root {
                    --background-color: transparent;
                    --text-color: #f8f9fa;
                    --tab-bg-color: #2c3e50;
                    --tab-selected-bg: #4682b4;
                    --tab-selected-color: white;
                }
            }

            /* Apply theme variables */
            .main {background-color: var(--background-color);}
            .stApp {background-color: var(--background-color);}
            .css-18e3th9 {padding-top: 1rem;}
            h1, h2, h3 {color: var(--text-color);}
            .stTabs [data-baseweb="tab-list"] {gap: 2px;}
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: var(--tab-bg-color);
                border-radius: 4px 4px 0 0;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: var(--tab-selected-bg);
                color: var(--tab-selected-color);
            }

            /* Additional dark mode specific styles for Streamlit elements */
            [data-testid="stMarkdown"] {color: var(--text-color);}
            [data-testid="stText"] {color: var(--text-color);}
        </style>
        """, unsafe_allow_html=True)

    def display_app_header(self):
        """Display application title and description."""
        st.title("MIMIC-IV Discharge Analysis Dashboard")
        st.markdown("""
            <div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px;'>
            This application provides comprehensive analysis tools for the MIMIC-IV healthcare dataset with a focus on discharge prediction.
            Navigate through different sections using the sidebar menu to explore patient data, analyze order patterns,
            visualize patient trajectories, build predictive models, and interpret clinical patterns.
            </div>
        """, unsafe_allow_html=True)

    def setup_sidebar_navigation(self):
        """Set up sidebar navigation and data loading.

        Returns:
            tuple: Contains path_valid (bool), app_mode (str), data (dict), and mimic_path (str)
        """
        st.sidebar.title("Navigation")

        # Data path input
        default_mimic_path = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
        mimic_path = st.sidebar.text_input(
            "MIMIC-IV Dataset Path",
            value=default_mimic_path,
            help="Enter the path to the MIMIC-IV dataset directory"
        )

        # Validate the path
        path_valid = self.validate_mimic_path(mimic_path)

        # Navigation options
        app_mode = st.sidebar.radio(
            "Select Section",
            ["Home", "Data Explorer", "Order Pattern Analysis", "Patient Trajectory", "Predictive Modeling", "Clinical Interpretation"]
        )

        # Load data if path is valid
        data = {}
        if path_valid:
            # Use st.spinner instead of st.sidebar.spinner
            spinner_placeholder = st.sidebar.empty()
            spinner_placeholder.info("Loading data...")
            data = self.load_all_data(mimic_path)

            if all(df is not None for df in data.values()):
                spinner_placeholder.success("Data loaded successfully!")
            else:
                spinner_placeholder.error("Some data files could not be loaded. Check the path and file structure.")
        else:
            st.sidebar.warning("Please provide a valid path to the MIMIC-IV dataset.")
            st.sidebar.info("You can continue exploring the application interface, but data analysis features will be limited.")

        return path_valid, app_mode, data, mimic_path

    def validate_mimic_path(self, path):
        """Check if the path exists and contains required files.

        Args:
            path (str): Path to the MIMIC-IV dataset directory

        Returns:
            bool: True if path is valid, False otherwise
        """
        if not os.path.exists(path):
            st.sidebar.error(f"Path not found: {path}")
            return False

        # Check for common structures
        if os.path.exists(os.path.join(path, 'hosp')):
            # Standard MIMIC-IV structure
            return True

        # Check if CSV files exist directly in the path or in a parent directory
        required_files = ['patients.csv', 'admissions.csv', 'transfers.csv', 'poe.csv', 'poe_detail.csv']
        parent_dirs = [path, os.path.join(path, 'hosp'), os.path.dirname(path)]

        for parent_dir in parent_dirs:
            if all(os.path.exists(os.path.join(parent_dir, file)) for file in required_files):
                return True

        st.sidebar.warning("Could not find required MIMIC-IV files in the specified path.")
        st.sidebar.info("Please ensure the path contains patients.csv, admissions.csv, transfers.csv, poe.csv, and poe_detail.csv files.")
        return False

    def load_all_data(self, mimic_path):
        """Load all required data files from MIMIC-IV dataset.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory

        Returns:
            dict: Dictionary containing loaded dataframes
        """
        data = {
            'patients'  : self.load_patient_data(mimic_path),
            'admissions': self.load_admissions_data(mimic_path),
            'transfers' : self.load_transfers_data(mimic_path),
            'poe'       : self.load_poe_data(mimic_path),
            'poe_detail': self.load_poe_detail_data(mimic_path)
        }
        return data

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_patient_data(mimic_path):
        """Load patients data from MIMIC-IV.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory

        Returns:
            pandas.DataFrame: Patients data or None if file not found
        """
        # Try multiple possible locations for the file
        possible_paths = [
            os.path.join(mimic_path, 'hosp', 'patients.csv'),  # Standard MIMIC-IV structure
            os.path.join(mimic_path, 'patients.csv'),          # Files directly in the path
            os.path.join(os.path.dirname(mimic_path), 'patients.csv')  # Files in parent directory
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return MIMICDashboard._load_csv_file(path, "patient data")

        return MIMICDashboard._load_csv_file(possible_paths[0], "patient data")  # Will show error if not found

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_admissions_data(mimic_path):
        """Load admissions data from MIMIC-IV.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory

        Returns:
            pandas.DataFrame: Admissions data or None if file not found
        """
        # Try multiple possible locations for the file
        possible_paths = [
            os.path.join(mimic_path, 'hosp', 'admissions.csv'),  # Standard MIMIC-IV structure
            os.path.join(mimic_path, 'admissions.csv'),          # Files directly in the path
            os.path.join(os.path.dirname(mimic_path), 'admissions.csv')  # Files in parent directory
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return MIMICDashboard._load_csv_file(path, "admissions data")

        return MIMICDashboard._load_csv_file(possible_paths[0], "admissions data")  # Will show error if not found

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_transfers_data(mimic_path):
        """Load transfers data from MIMIC-IV.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory

        Returns:
            pandas.DataFrame: Transfers data or None if file not found
        """
        # Try multiple possible locations for the file
        possible_paths = [
            os.path.join(mimic_path, 'hosp', 'transfers.csv'),  # Standard MIMIC-IV structure
            os.path.join(mimic_path, 'transfers.csv'),          # Files directly in the path
            os.path.join(os.path.dirname(mimic_path), 'transfers.csv')  # Files in parent directory
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return MIMICDashboard._load_csv_file(path, "transfers data")

        return MIMICDashboard._load_csv_file(possible_paths[0], "transfers data")  # Will show error if not found

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_poe_data(mimic_path):
        """Load provider order entry data from MIMIC-IV.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory

        Returns:
            pandas.DataFrame: POE data or None if file not found
        """
        # Try multiple possible locations for the file
        possible_paths = [
            os.path.join(mimic_path, 'hosp', 'poe.csv'),  # Standard MIMIC-IV structure
            os.path.join(mimic_path, 'poe.csv'),          # Files directly in the path
            os.path.join(os.path.dirname(mimic_path), 'poe.csv')  # Files in parent directory
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return MIMICDashboard._load_csv_file(path, "POE data")

        return MIMICDashboard._load_csv_file(possible_paths[0], "POE data")  # Will show error if not found

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_poe_detail_data(mimic_path):
        """Load provider order entry detail data from MIMIC-IV.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset directory

        Returns:
            pandas.DataFrame: POE detail data or None if file not found
        """
        # Try multiple possible locations for the file
        possible_paths = [
            os.path.join(mimic_path, 'hosp', 'poe_detail.csv'),  # Standard MIMIC-IV structure
            os.path.join(mimic_path, 'poe_detail.csv'),          # Files directly in the path
            os.path.join(os.path.dirname(mimic_path), 'poe_detail.csv')  # Files in parent directory
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return MIMICDashboard._load_csv_file(path, "POE detail data")

        return MIMICDashboard._load_csv_file(possible_paths[0], "POE detail data")  # Will show error if not found

    @staticmethod
    def _load_csv_file(file_path, description):
        """Helper function to load CSV files with error handling.

        Args:
            file_path (str): Path to the CSV file
            description (str): Description of the data being loaded

        Returns:
            pandas.DataFrame: Loaded data or None if file not found or error occurs
        """
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path, low_memory=False)
            else:
                st.error(f"File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"Error loading {description}: {e}")
            return None

    def display_demo_mode_warning(self):
        """Display warning and information for demo mode."""
        st.warning("⚠️ MIMIC-IV dataset not found at the specified path.")
        st.info("""
        ### Demo Mode
        This application is running in demo mode. Some features may be limited.

        To use with the actual MIMIC-IV dataset:
        1. Download the MIMIC-IV dataset from [PhysioNet](https://physionet.org/content/mimiciv/)
        2. Extract the files to a directory on your computer
        3. Enter the path to that directory in the sidebar
        """)

    def render_home_page(self):
        """Render the home page of the application."""
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
            if self.path_valid and all(df is not None for df in self.data.values()):
                st.metric("Patients", f"{len(self.data['patients']):,}")
                st.metric("Admissions", f"{len(self.data['admissions']):,}")
                st.metric("Transfers", f"{len(self.data['transfers']):,}")
                st.metric("Provider Orders", f"{len(self.data['poe']):,}")
                st.metric("Order Details", f"{len(self.data['poe_detail']):,}")
            else:
                st.info("Dataset statistics will be displayed once a valid MIMIC-IV dataset path is provided.")

    def render_data_explorer(self):
        """Render the data explorer page."""
        st.header("Data Explorer")

        if not self.path_valid or not all(self.data.get(key) is not None for key in ['patients', 'admissions', 'transfers']):
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable data exploration.")
            return

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

    def render_order_pattern_analysis(self):
        """Render the order pattern analysis page."""
        st.header("Order Pattern Analysis")

        if not self.path_valid or not all(self.data.get(key) is not None for key in ['poe', 'poe_detail']):
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable order pattern analysis.")
            return

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

    def render_patient_trajectory(self):
        """Render the patient trajectory analysis page."""
        st.header("Patient Trajectory Analysis")

        if not self.path_valid or not all(self.data.get(key) is not None for key in ['patients', 'admissions', 'transfers']):
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable patient trajectory analysis.")
            return

        # Create tabs for different analysis sections
        tabs = st.tabs(["Patient Journey", "Pathway Clustering", "Dimensionality Reduction", "Survival Analysis"])

        # Patient Journey tab
        with tabs[0]:
            self.data_loader.patient_trajectory_module.patient_journey()

        # Pathway Clustering tab
        with tabs[1]:
            self.data_loader.patient_trajectory_module.pathway_clustering()

        # Dimensionality Reduction tab
        with tabs[2]:
            self.data_loader.patient_trajectory_module.dimensionality_reduction()

        # Survival Analysis tab
        with tabs[3]:
            self.data_loader.patient_trajectory_module.survival_analysis()

    def render_predictive_modeling(self):
        """Render the predictive modeling page."""
        st.header("Predictive Modeling Interface")

        if not self.path_valid or not all(self.data.get(key) is not None for key in ['patients', 'admissions', 'transfers', 'poe', 'poe_detail']):
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable predictive modeling.")
            return

        # Create tabs for different analysis sections
        tabs = st.tabs(["Feature Selection", "Model Training", "Performance Visualization", "Feature Importance"])

        # Feature Selection tab
        with tabs[0]:
            self.data_loader.predictive_modeling_module.feature_selection()

        # Model Training tab
        with tabs[1]:
            self.data_loader.predictive_modeling_module.model_training()

        # Performance Visualization tab
        with tabs[2]:
            self.data_loader.predictive_modeling_module.performance_visualization()

        # Feature Importance tab
        with tabs[3]:
            self.data_loader.predictive_modeling_module.feature_importance()

    def render_clinical_interpretation(self):
        """Render the clinical interpretation page."""
        st.header("Clinical Interpretation")

        if not self.path_valid or not all(self.data.get(key) is not None for key in ['poe', 'poe_detail']):
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable clinical interpretation.")
            return

        # Create tabs for different analysis sections
        tabs = st.tabs(["Provider Order Patterns", "Unit-Specific Analysis", "Key Order Sequences"])

        # Provider Order Patterns tab
        with tabs[0]:
            self.data_loader.clinical_interpretation_module.provider_order_patterns()

        # Unit-Specific Analysis tab
        with tabs[1]:
            self.data_loader.clinical_interpretation_module.unit_specific_analysis()

        # Key Order Sequences tab
        with tabs[2]:
            self.data_loader.clinical_interpretation_module.key_order_sequences()

    def run(self):
        """Run the dashboard application based on the selected mode."""
        # Render the selected page based on app_mode
        if self.app_mode == "Home":
            self.render_home_page()
        elif self.app_mode == "Data Explorer":
            self.render_data_explorer()
        elif self.app_mode == "Order Pattern Analysis":
            self.render_order_pattern_analysis()
        elif self.app_mode == "Patient Trajectory":
            self.render_patient_trajectory()
        elif self.app_mode == "Predictive Modeling":
            self.render_predictive_modeling()
        elif self.app_mode == "Clinical Interpretation":
            self.render_clinical_interpretation()


def main():
    """Main entry point for the Streamlit application."""
    dashboard = MIMICDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
