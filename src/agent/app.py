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


class StreamlitApp:
    """Main class for the MIMIC-IV Discharge Analysis Dashboard.

    This class encapsulates all functionality for the Streamlit application,
    including page configuration, data loading, and rendering different sections
    of the dashboard.
    """

    def __init__(self):
        """Initialize the dashboard application."""
        self.configure_page()
        self.display_app_header()

        # Initialize empty data structures
        self.data = {}
        self.data_loader = None
        self.preprocessed_data = {}

        # Setup navigation and path validation
        self.path_valid, self.app_mode, self.mimic_path = self.setup_sidebar_navigation()

        # Display demo mode warning if path is not valid
        if not self.path_valid:
            self.display_demo_mode_warning()

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
        """Set up sidebar navigation and validate dataset path.

        Returns:
            tuple: Contains path_valid (bool), app_mode (str), and mimic_path (str)
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

        # Add Load Dataset button
        if path_valid:
            if st.sidebar.button("Load Dataset", key="load_dataset_button", help="Click to load the MIMIC-IV dataset"):
                # Create a status text element in the sidebar
                status_text = st.sidebar.empty()
                status_text.text("⏳ Initializing data loader...")

                # Use the main spinner for the loading animation
                with st.spinner("Loading MIMIC-IV dataset..."):
                    try:
                        # Initialize data loader
                        self.data_loader = MIMICDataLoader(mimic_path)
                        status_text.text("⏳ Loading data files...")

                        # Load and preprocess data
                        self.data = self.data_loader.load_all_data()
                        if self.data and all(df is not None for df in self.data.values()):
                            status_text.text("⏳ Preprocessing data...")
                            self.preprocessed_data = self.data_loader.preprocess_all_data()
                            status_text.text("✅ Dataset loaded and preprocessed successfully!")
                            st.session_state['data_loaded'] = True
                        else:
                            status_text.text("❌ Error: Some data files could not be loaded")
                            st.session_state['data_loaded'] = False
                    except Exception as e:
                        status_text.text(f"❌ Error: {str(e)}")
                        st.session_state['data_loaded'] = False
        else:
            st.sidebar.warning("Please provide a valid path to the MIMIC-IV dataset.")
            st.sidebar.info("You can continue exploring the application interface, but data analysis features will be limited.")
            st.session_state['data_loaded'] = False

        # Navigation options
        app_mode = st.sidebar.radio(
            "Select Section",
            ["Home", "Data Explorer", "Order Pattern Analysis", "Patient Trajectory", "Predictive Modeling", "Clinical Interpretation"]
        )

        return path_valid, app_mode, mimic_path

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

    def check_data_loaded(self):
        """Check if data is loaded and preprocessed.

        Returns:
            bool: True if data is loaded and preprocessed, False otherwise
        """
        if not st.session_state.get('data_loaded', False):
            st.warning("Please click the 'Load Dataset' button in the sidebar to load the data before proceeding.")
            return False
        return True

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
            if self.path_valid and self.data and all(df is not None for df in self.data.values()):
                st.metric("Patients", f"{len(self.data['patients']):,}")
                st.metric("Admissions", f"{len(self.data['admissions']):,}")
                st.metric("Transfers", f"{len(self.data['transfers']):,}")
                st.metric("Provider Orders", f"{len(self.data['poe']):,}")
                st.metric("Order Details", f"{len(self.data['poe_detail']):,}")
            else:
                if self.path_valid:
                    st.info("Click the 'Load Dataset' button in the sidebar to load and display dataset statistics.")
                else:
                    st.info("Dataset statistics will be displayed once a valid MIMIC-IV dataset path is provided and data is loaded.")

    def render_data_explorer(self):
        """Render the data explorer page."""
        st.header("Data Explorer")

        if not self.path_valid:
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable data exploration.")
            return

        if not self.check_data_loaded():
            return

        # Create tabs for different analysis sections
        tabs = st.tabs([
            "Patient Demographics",
            "Hospital Utilization",
            "Length of Stay",
            "Diagnoses",
            "Medications",
            "Lab Results",
            "Time Series Analysis"
        ])

        # Patient Demographics tab
        with tabs[0]:
            st.subheader("Patient Demographics and Cohort Statistics")

            col1, col2 = st.columns(2)

            with col1:
                # Age distribution
                if 'patients' in self.data and 'anchor_age' in self.data['patients'].columns:
                    st.write("#### Age Distribution")
                    fig = px.histogram(self.data['patients'], x='anchor_age', nbins=50,
                                     title='Patient Age Distribution')
                    st.plotly_chart(fig)

                # Gender distribution
                if 'patients' in self.data and 'gender' in self.data['patients'].columns:
                    st.write("#### Gender Distribution")
                    gender_counts = self.data['patients']['gender'].value_counts()
                    fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                               title='Patient Gender Distribution')
                    st.plotly_chart(fig)

            with col2:
                # Insurance distribution
                if 'admissions' in self.data and 'insurance' in self.data['admissions'].columns:
                    st.write("#### Insurance Distribution")
                    insurance_counts = self.data['admissions']['insurance'].value_counts()
                    fig = px.pie(values=insurance_counts.values, names=insurance_counts.index,
                               title='Insurance Distribution')
                    st.plotly_chart(fig)

        # Hospital Utilization tab
        with tabs[1]:
            st.subheader("Hospital Utilization Patterns")

            # Admission type distribution
            if 'admissions' in self.data and 'admission_type' in self.data['admissions'].columns:
                st.write("#### Admission Type Distribution")
                admission_type_counts = self.data['admissions']['admission_type'].value_counts()
                fig = px.bar(x=admission_type_counts.index, y=admission_type_counts.values,
                           title='Admission Type Distribution')
                st.plotly_chart(fig)

            # Department/Service utilization
            if 'transfers' in self.data and 'careunit' in self.data['transfers'].columns:
                st.write("#### Care Unit Utilization")
                careunit_counts = self.data['transfers']['careunit'].value_counts()
                fig = px.bar(x=careunit_counts.index, y=careunit_counts.values,
                           title='Care Unit Utilization')
                st.plotly_chart(fig)

        # Length of Stay tab
        with tabs[2]:
            st.subheader("Length of Stay Distributions")

            if 'admissions' in self.data and all(col in self.data['admissions'].columns for col in ['admittime', 'dischtime']):
                # Calculate LOS
                self.data['admissions']['los_days'] = (pd.to_datetime(self.data['admissions']['dischtime']) -
                                                      pd.to_datetime(self.data['admissions']['admittime'])).dt.total_seconds() / (24*60*60)

                # Overall LOS distribution
                st.write("#### Overall Length of Stay Distribution")
                fig = px.histogram(self.data['admissions'], x='los_days', nbins=50,
                                 title='Length of Stay Distribution (Days)')
                st.plotly_chart(fig)

        # Diagnoses tab
        with tabs[3]:
            st.subheader("Diagnosis Patterns")

            if all(key in self.data for key in ['diagnoses_icd', 'd_icd_diagnoses']):
                # Merge diagnoses with descriptions
                diagnoses_df = pd.merge(self.data['diagnoses_icd'], self.data['d_icd_diagnoses'],
                                      on='icd_code', how='left')

                # Top 20 diagnoses
                st.write("#### Most Common Diagnoses")
                diagnosis_counts = diagnoses_df['long_title'].value_counts().head(20)
                fig = px.bar(x=diagnosis_counts.index, y=diagnosis_counts.values,
                           title='Top 20 Most Common Diagnoses')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)

        # Medications tab
        with tabs[4]:
            st.subheader("Medication Analysis")

            if 'pharmacy' in self.data:
                # Top 20 medications
                st.write("#### Most Commonly Prescribed Medications")
                if 'medication' in self.data['pharmacy'].columns:
                    med_counts = self.data['pharmacy']['medication'].value_counts().head(20)
                    fig = px.bar(x=med_counts.index, y=med_counts.values,
                               title='Top 20 Most Common Medications')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)

        # Lab Results tab
        with tabs[5]:
            st.subheader("Laboratory Test Analysis")

            if 'labevents' in self.data:
                # Top 20 lab tests
                st.write("#### Most Common Laboratory Tests")
                if 'itemid' in self.data['labevents'].columns:
                    lab_counts = self.data['labevents']['itemid'].value_counts().head(20)
                    fig = px.bar(x=lab_counts.index, y=lab_counts.values,
                               title='Top 20 Most Common Laboratory Tests')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)

        # Time Series Analysis tab
        with tabs[6]:
            st.subheader("Time Series Analysis")

            if 'admissions' in self.data and 'admittime' in self.data['admissions'].columns:
                # Convert admittime to datetime
                self.data['admissions']['admittime'] = pd.to_datetime(self.data['admissions']['admittime'])

                # Admissions over time
                st.write("#### Admissions Over Time")
                admissions_by_date = self.data['admissions']['admittime'].dt.date.value_counts().sort_index()
                fig = px.line(x=admissions_by_date.index, y=admissions_by_date.values,
                            title='Daily Admission Counts')
                st.plotly_chart(fig)

                # Admissions by day of week
                st.write("#### Admissions by Day of Week")
                admissions_by_dow = self.data['admissions']['admittime'].dt.day_name().value_counts()
                fig = px.bar(x=admissions_by_dow.index, y=admissions_by_dow.values,
                           title='Admissions by Day of Week')
                st.plotly_chart(fig)

    def render_order_pattern_analysis(self):
        """Render the order pattern analysis page."""
        st.header("Order Pattern Analysis")

        if not self.path_valid:
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable order pattern analysis.")
            return

        if not self.check_data_loaded():
            return

        # Create tabs for different analysis sections
        tabs = st.tabs([
            "Order Overview",
            "Medication Orders",
            "Lab Orders",
            "Order Timing",
            "Sequential Patterns"
        ])

        # Order Overview tab
        with tabs[0]:
            st.subheader("Overview of Orders")

            if 'poe' in self.data and 'order_type' in self.data['poe'].columns:
                # Order type distribution
                st.write("#### Distribution of Order Types")
                order_type_counts = self.data['poe']['order_type'].value_counts()
                fig = px.pie(values=order_type_counts.values, names=order_type_counts.index,
                           title='Distribution of Order Types')
                st.plotly_chart(fig)

                # Orders per admission
                if 'hadm_id' in self.data['poe'].columns:
                    st.write("#### Orders per Admission")
                    orders_per_admission = self.data['poe'].groupby('hadm_id').size()
                    fig = px.histogram(x=orders_per_admission.values, nbins=50,
                                     title='Distribution of Orders per Admission')
                    fig.update_layout(xaxis_title='Number of Orders', yaxis_title='Count of Admissions')
                    st.plotly_chart(fig)

        # Medication Orders tab
        with tabs[1]:
            st.subheader("Medication Order Analysis")

            if all(df in self.data for df in ['pharmacy', 'poe']):
                # Filter medication orders
                med_orders = self.data['poe'][self.data['poe']['order_type'] == 'Medications'].copy()

                col1, col2 = st.columns(2)

                with col1:
                    # Top 20 medications by frequency
                    if 'pharmacy' in self.data and 'medication' in self.data['pharmacy'].columns:
                        st.write("#### Most Frequently Ordered Medications")
                        med_counts = self.data['pharmacy']['medication'].value_counts().head(20)
                        fig = px.bar(x=med_counts.index, y=med_counts.values,
                                   title='Top 20 Medications by Order Frequency')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                with col2:
                    # Medication routes
                    if 'pharmacy' in self.data and 'route' in self.data['pharmacy'].columns:
                        st.write("#### Medication Routes")
                        route_counts = self.data['pharmacy']['route'].value_counts()
                        fig = px.pie(values=route_counts.values, names=route_counts.index,
                                   title='Distribution of Medication Routes')
                        st.plotly_chart(fig)

                # Medication timing analysis
                if 'ordertime' in med_orders.columns:
                    st.write("#### Medication Order Timing")
                    med_orders['hour'] = pd.to_datetime(med_orders['ordertime']).dt.hour
                    hourly_orders = med_orders['hour'].value_counts().sort_index()
                    fig = px.line(x=hourly_orders.index, y=hourly_orders.values,
                                title='Medication Orders by Hour of Day')
                    fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Number of Orders')
                    st.plotly_chart(fig)

        # Lab Orders tab
        with tabs[2]:
            st.subheader("Laboratory Order Analysis")

            if 'labevents' in self.data:
                col1, col2 = st.columns(2)

                with col1:
                    # Top 20 lab tests
                    if 'itemid' in self.data['labevents'].columns:
                        st.write("#### Most Common Laboratory Tests")
                        lab_counts = self.data['labevents']['itemid'].value_counts().head(20)
                        fig = px.bar(x=lab_counts.index, y=lab_counts.values,
                                   title='Top 20 Laboratory Tests')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                with col2:
                    # Lab result status distribution
                    if 'status' in self.data['labevents'].columns:
                        st.write("#### Lab Result Status Distribution")
                        status_counts = self.data['labevents']['status'].value_counts()
                        fig = px.pie(values=status_counts.values, names=status_counts.index,
                                   title='Distribution of Lab Result Status')
                        st.plotly_chart(fig)

                # Lab order timing
                if 'charttime' in self.data['labevents'].columns:
                    st.write("#### Laboratory Order Timing")
                    self.data['labevents']['hour'] = pd.to_datetime(self.data['labevents']['charttime']).dt.hour
                    hourly_labs = self.data['labevents']['hour'].value_counts().sort_index()
                    fig = px.line(x=hourly_labs.index, y=hourly_labs.values,
                                title='Laboratory Orders by Hour of Day')
                    fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Number of Orders')
                    st.plotly_chart(fig)

        # Order Timing tab
        with tabs[3]:
            st.subheader("Order Timing Analysis")

            if 'poe' in self.data and 'ordertime' in self.data['poe'].columns:
                # Convert ordertime to datetime
                self.data['poe']['ordertime'] = pd.to_datetime(self.data['poe']['ordertime'])

                # Orders by hour of day
                st.write("#### Orders by Hour of Day")
                self.data['poe']['hour'] = self.data['poe']['ordertime'].dt.hour
                hourly_orders = self.data['poe']['hour'].value_counts().sort_index()
                fig = px.line(x=hourly_orders.index, y=hourly_orders.values,
                            title='All Orders by Hour of Day')
                fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Number of Orders')
                st.plotly_chart(fig)

                # Orders by day of week
                st.write("#### Orders by Day of Week")
                self.data['poe']['dow'] = self.data['poe']['ordertime'].dt.day_name()
                dow_orders = self.data['poe']['dow'].value_counts()
                fig = px.bar(x=dow_orders.index, y=dow_orders.values,
                           title='Orders by Day of Week')
                st.plotly_chart(fig)

                # Heatmap of orders by hour and day
                st.write("#### Order Density Heatmap")
                self.data['poe']['dow_num'] = self.data['poe']['ordertime'].dt.dayofweek
                heatmap_data = pd.crosstab(self.data['poe']['dow_num'], self.data['poe']['hour'])
                fig = px.imshow(heatmap_data,
                              labels=dict(x='Hour of Day', y='Day of Week'),
                              title='Order Density by Day and Hour')
                st.plotly_chart(fig)

        # Sequential Patterns tab
        with tabs[4]:
            st.subheader("Sequential Order Pattern Analysis")

            if 'poe' in self.data and all(col in self.data['poe'].columns for col in ['hadm_id', 'ordertime', 'order_type']):
                # Get a sample admission for demonstration
                st.write("#### Order Sequence Example")
                sample_hadm = self.data['poe']['hadm_id'].iloc[0]
                sample_orders = self.data['poe'][self.data['poe']['hadm_id'] == sample_hadm].sort_values('ordertime')

                # Create a timeline of orders
                fig = px.timeline(sample_orders,
                                x_start='ordertime',
                                x_end='ordertime',
                                y='order_type',
                                title=f'Order Sequence Timeline for Admission {sample_hadm}')
                st.plotly_chart(fig)

                # Common order sequences
                st.write("#### Common Order Sequences")
                st.info("This section will show common sequences of orders across admissions. "
                       "The analysis requires processing the full dataset which may take some time.")

    def render_patient_trajectory(self):
        """Render the patient trajectory analysis page."""
        st.header("Patient Trajectory Analysis")

        if not self.path_valid:
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable patient trajectory analysis.")
            return

        if not self.check_data_loaded():
            return

        # Create tabs for different analysis sections
        tabs = st.tabs([
            "Patient Journey",
            "Diagnosis Trajectories",
            "Lab Value Trends",
            "Care Unit Transitions",
            "Outcome Analysis"
        ])

        # Patient Journey tab
        with tabs[0]:
            st.subheader("Individual Patient Journey")

            # Patient selector
            if 'patients' in self.data and 'subject_id' in self.data['patients'].columns:
                patient_ids = sorted(self.data['patients']['subject_id'].unique())
                selected_patient = st.selectbox("Select Patient ID", patient_ids)

                if selected_patient:
                    # Get patient admissions
                    patient_admissions = self.data['admissions'][
                        self.data['admissions']['subject_id'] == selected_patient
                    ].sort_values('admittime')

                    if not patient_admissions.empty:
                        st.write("#### Admission History")

                        # Create timeline of admissions
                        fig = px.timeline(
                            patient_admissions,
                            x_start='admittime',
                            x_end='dischtime',
                            y='admission_type',
                            title=f'Admission Timeline for Patient {selected_patient}'
                        )
                        st.plotly_chart(fig)

                        # Show diagnoses for each admission
                        st.write("#### Diagnoses by Admission")
                        for _, admission in patient_admissions.iterrows():
                            with st.expander(f"Admission {admission['hadm_id']} ({admission['admittime']} to {admission['dischtime']})"):
                                # Get diagnoses for this admission
                                admission_diagnoses = self.data['diagnoses_icd'][
                                    self.data['diagnoses_icd']['hadm_id'] == admission['hadm_id']
                                ]
                                if not admission_diagnoses.empty:
                                    # Merge with ICD codes to get descriptions
                                    diagnoses_with_desc = pd.merge(
                                        admission_diagnoses,
                                        self.data['d_icd_diagnoses'],
                                        on='icd_code',
                                        how='left'
                                    )
                                    st.write(diagnoses_with_desc[['icd_code', 'long_title']])

        # Diagnosis Trajectories tab
        with tabs[1]:
            st.subheader("Diagnosis Pattern Analysis")

            if all(key in self.data for key in ['diagnoses_icd', 'd_icd_diagnoses']):
                # Merge diagnoses with descriptions
                diagnoses_df = pd.merge(
                    self.data['diagnoses_icd'],
                    self.data['d_icd_diagnoses'],
                    on='icd_code',
                    how='left'
                )

                # Most common diagnosis sequences
                st.write("#### Common Diagnosis Sequences")

                # Group diagnoses by admission and sort by sequence number
                admission_diagnoses = diagnoses_df.sort_values(['hadm_id', 'seq_num'])

                # Get top 10 most common first diagnoses
                first_diagnoses = admission_diagnoses[admission_diagnoses['seq_num'] == 1]['long_title'].value_counts().head(10)

                fig = px.bar(
                    x=first_diagnoses.index,
                    y=first_diagnoses.values,
                    title='Top 10 Most Common Initial Diagnoses'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)

        # Lab Value Trends tab
        with tabs[2]:
            st.subheader("Laboratory Value Trends")

            if 'labevents' in self.data:
                # Get list of lab tests
                if 'itemid' in self.data['labevents'].columns:
                    lab_items = self.data['labevents']['itemid'].unique()
                    selected_lab = st.selectbox("Select Lab Test", lab_items)

                    if selected_lab:
                        # Get values for selected lab test
                        lab_values = self.data['labevents'][
                            self.data['labevents']['itemid'] == selected_lab
                        ].copy()

                        if 'valuenum' in lab_values.columns and 'charttime' in lab_values.columns:
                            lab_values['charttime'] = pd.to_datetime(lab_values['charttime'])

                            # Plot value distribution
                            st.write("#### Value Distribution")
                            fig = px.histogram(
                                lab_values,
                                x='valuenum',
                                nbins=50,
                                title=f'Distribution of Values for Lab Test {selected_lab}'
                            )
                            st.plotly_chart(fig)

                            # Plot trend over time
                            st.write("#### Trend Over Time")
                            daily_avg = lab_values.groupby(
                                lab_values['charttime'].dt.date
                            )['valuenum'].mean().reset_index()

                            fig = px.line(
                                daily_avg,
                                x='charttime',
                                y='valuenum',
                                title=f'Daily Average Values for Lab Test {selected_lab}'
                            )
                            st.plotly_chart(fig)

        # Care Unit Transitions tab
        with tabs[3]:
            st.subheader("Care Unit Transition Analysis")

            if 'transfers' in self.data:
                # Create transition matrix
                if 'careunit' in self.data['transfers'].columns:
                    transitions = self.data['transfers'].groupby(
                        ['hadm_id', 'careunit']
                    ).size().unstack(fill_value=0)

                    # Plot transition heatmap
                    st.write("#### Care Unit Transition Patterns")
                    fig = px.imshow(
                        transitions,
                        title='Care Unit Transition Matrix',
                        labels=dict(x='To Unit', y='From Unit')
                    )
                    st.plotly_chart(fig)

                    # Calculate average length of stay by unit
                    if all(col in self.data['transfers'].columns for col in ['intime', 'outtime']):
                        self.data['transfers']['los_hours'] = (
                            pd.to_datetime(self.data['transfers']['outtime']) -
                            pd.to_datetime(self.data['transfers']['intime'])
                        ).dt.total_seconds() / 3600

                        unit_los = self.data['transfers'].groupby('careunit')['los_hours'].mean()

                        st.write("#### Average Length of Stay by Unit")
                        fig = px.bar(
                            x=unit_los.index,
                            y=unit_los.values,
                            title='Average Length of Stay (Hours) by Care Unit'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

        # Outcome Analysis tab
        with tabs[4]:
            st.subheader("Patient Outcome Analysis")

            if 'admissions' in self.data:
                # Analyze discharge dispositions
                if 'discharge_location' in self.data['admissions'].columns:
                    st.write("#### Discharge Dispositions")
                    disposition_counts = self.data['admissions']['discharge_location'].value_counts()
                    fig = px.pie(
                        values=disposition_counts.values,
                        names=disposition_counts.index,
                        title='Distribution of Discharge Dispositions'
                    )
                    st.plotly_chart(fig)

                # Analyze readmissions
                if all(col in self.data['admissions'].columns for col in ['subject_id', 'admittime']):
                    st.write("#### Readmission Analysis")

                    # Sort admissions by patient and time
                    sorted_admissions = self.data['admissions'].sort_values(['subject_id', 'admittime'])

                    # Calculate days to next admission
                    sorted_admissions['next_admittime'] = sorted_admissions.groupby('subject_id')['admittime'].shift(-1)
                    sorted_admissions['days_to_readmission'] = (
                        pd.to_datetime(sorted_admissions['next_admittime']) -
                        pd.to_datetime(sorted_admissions['admittime'])
                    ).dt.total_seconds() / (24*3600)

                    # Plot readmission distribution
                    readmission_data = sorted_admissions[
                        sorted_admissions['days_to_readmission'].between(0, 365)
                    ]

                    fig = px.histogram(
                        readmission_data,
                        x='days_to_readmission',
                        nbins=50,
                        title='Distribution of Days to Readmission (within 1 year)'
                    )
                    st.plotly_chart(fig)

    def render_predictive_modeling(self):
        """Render the predictive modeling page."""
        st.header("Predictive Modeling Interface")

        if not self.path_valid:
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable predictive modeling.")
            return

        if not self.check_data_loaded():
            return

        # Create tabs for different analysis sections
        tabs = st.tabs([
            "Feature Selection",
            "Model Configuration",
            "Model Training",
            "Model Evaluation",
            "Feature Importance"
        ])

        # Feature Selection tab
        with tabs[0]:
            st.subheader("Feature Selection")

            st.write("#### Available Feature Groups")

            # Demographics features
            demographics_features = st.checkbox("Demographics Features", True)
            if demographics_features:
                st.write("Selected Demographics Features:")
                demo_cols = ['gender', 'age', 'race', 'marital_status', 'language']
                selected_demo = [col for col in demo_cols if col in self.data['patients'].columns]
                st.write(", ".join(selected_demo))

            # Admission features
            admission_features = st.checkbox("Admission Features", True)
            if admission_features:
                st.write("Selected Admission Features:")
                adm_cols = ['admission_type', 'admission_location', 'insurance', 'marital_status']
                selected_adm = [col for col in adm_cols if col in self.data['admissions'].columns]
                st.write(", ".join(selected_adm))

            # Diagnosis features
            diagnosis_features = st.checkbox("Diagnosis Features", True)
            if diagnosis_features and 'diagnoses_icd' in self.data:
                st.write("Selected Diagnosis Features:")
                st.write("- ICD Codes")
                st.write("- Diagnosis Sequence Numbers")
                st.write("- Primary Diagnoses")

            # Lab test features
            lab_features = st.checkbox("Laboratory Test Features", True)
            if lab_features and 'labevents' in self.data:
                st.write("Selected Lab Features:")
                if 'itemid' in self.data['labevents'].columns:
                    top_labs = self.data['labevents']['itemid'].value_counts().head(5)
                    st.write(f"Top {len(top_labs)} most common lab tests")

            # Medication features
            medication_features = st.checkbox("Medication Features", True)
            if medication_features and 'pharmacy' in self.data:
                st.write("Selected Medication Features:")
                if 'medication' in self.data['pharmacy'].columns:
                    top_meds = self.data['pharmacy']['medication'].value_counts().head(5)
                    st.write(f"Top {len(top_meds)} most common medications")

        # Model Configuration tab
        with tabs[1]:
            st.subheader("Model Configuration")

            # Target variable selection
            st.write("#### Select Prediction Target")
            target_variable = st.selectbox(
                "Target Variable",
                ["Length of Stay", "Readmission Risk", "Mortality Risk", "Discharge Disposition"]
            )

            # Model selection
            st.write("#### Select Model Type")
            model_type = st.selectbox(
                "Model Type",
                ["Logistic Regression", "Random Forest", "XGBoost", "Deep Learning"]
            )

            # Hyperparameter settings
            st.write("#### Model Hyperparameters")
            if model_type == "Logistic Regression":
                st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            elif model_type == "Random Forest":
                st.slider("Number of Trees", 10, 500, 100)
                st.slider("Max Depth", 2, 20, 10)
            elif model_type == "XGBoost":
                st.slider("Learning Rate", 0.01, 0.3, 0.1)
                st.slider("Max Depth", 2, 15, 6)
            elif model_type == "Deep Learning":
                st.slider("Number of Layers", 1, 5, 3)
                st.slider("Neurons per Layer", 32, 512, 128)

        # Model Training tab
        with tabs[2]:
            st.subheader("Model Training")

            # Training settings
            st.write("#### Training Settings")
            train_size = st.slider("Training Data Percentage", 60, 90, 80)
            use_cross_val = st.checkbox("Use Cross-Validation", True)
            if use_cross_val:
                n_folds = st.slider("Number of Folds", 3, 10, 5)

            # Start training button
            if st.button("Start Training"):
                st.info("Model training would start here. This is a placeholder for the actual training process.")
                st.progress(100)
                st.success("Training complete! (Placeholder)")

        # Model Evaluation tab
        with tabs[3]:
            st.subheader("Model Evaluation")

            # Placeholder metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", "0.85")
            with col2:
                st.metric("Precision", "0.83")
            with col3:
                st.metric("Recall", "0.87")

            # ROC curve placeholder
            st.write("#### ROC Curve")
            st.info("ROC curve visualization would appear here")

            # Confusion matrix placeholder
            st.write("#### Confusion Matrix")
            st.info("Confusion matrix visualization would appear here")

            # Performance by subgroup
            st.write("#### Performance by Patient Subgroup")
            st.info("Subgroup analysis would appear here")

        # Feature Importance tab
        with tabs[4]:
            st.subheader("Feature Importance Analysis")

            # Global feature importance
            st.write("#### Global Feature Importance")
            st.info("Feature importance plot would appear here")

            # SHAP values
            st.write("#### SHAP Values")
            st.info("SHAP value visualization would appear here")

            # Feature interactions
            st.write("#### Feature Interactions")
            st.info("Feature interaction analysis would appear here")

    def render_clinical_interpretation(self):
        """Render the clinical interpretation page."""
        st.header("Clinical Interpretation")

        if not self.path_valid:
            st.warning("Please provide a valid path to the MIMIC-IV dataset to enable clinical interpretation.")
            return

        if not self.check_data_loaded():
            return

        # Create tabs for different analysis sections
        tabs = st.tabs([
            "Diagnosis Patterns",
            "Medication Patterns",
            "Lab Test Patterns",
            "Care Unit Analysis",
            "Temporal Patterns"
        ])

        # Diagnosis Patterns tab
        with tabs[0]:
            st.subheader("Diagnosis Pattern Analysis")

            if all(key in self.data for key in ['diagnoses_icd', 'd_icd_diagnoses']):
                # Merge diagnoses with descriptions
                diagnoses_df = pd.merge(
                    self.data['diagnoses_icd'],
                    self.data['d_icd_diagnoses'],
                    on='icd_code',
                    how='left'
                )

                # Most common diagnoses by admission type
                if 'admissions' in self.data:
                    st.write("#### Most Common Diagnoses by Admission Type")

                    # Merge with admissions to get admission type
                    diagnoses_with_adm = pd.merge(
                        diagnoses_df,
                        self.data['admissions'][['hadm_id', 'admission_type']],
                        on='hadm_id',
                        how='left'
                    )

                    # Get admission types for selection
                    adm_types = diagnoses_with_adm['admission_type'].unique()
                    selected_adm_type = st.selectbox("Select Admission Type", adm_types)

                    if selected_adm_type:
                        # Filter for selected admission type
                        filtered_diagnoses = diagnoses_with_adm[
                            diagnoses_with_adm['admission_type'] == selected_adm_type
                        ]

                        # Get top diagnoses
                        top_diagnoses = filtered_diagnoses['long_title'].value_counts().head(10)

                        fig = px.bar(
                            x=top_diagnoses.index,
                            y=top_diagnoses.values,
                            title=f'Top 10 Diagnoses for {selected_adm_type} Admissions'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                # Diagnosis co-occurrence
                st.write("#### Diagnosis Co-occurrence Analysis")

                # Get top diagnoses for co-occurrence analysis
                top_diagnoses = diagnoses_df['long_title'].value_counts().head(20)
                selected_diagnosis = st.selectbox("Select Primary Diagnosis", top_diagnoses.index)

                if selected_diagnosis:
                    # Find co-occurring diagnoses
                    primary_hadm_ids = diagnoses_df[
                        diagnoses_df['long_title'] == selected_diagnosis
                    ]['hadm_id'].unique()

                    co_occurring = diagnoses_df[
                        (diagnoses_df['hadm_id'].isin(primary_hadm_ids)) &
                        (diagnoses_df['long_title'] != selected_diagnosis)
                    ]['long_title'].value_counts().head(10)

                    fig = px.bar(
                        x=co_occurring.index,
                        y=co_occurring.values,
                        title=f'Top 10 Co-occurring Diagnoses with {selected_diagnosis}'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)

        # Medication Patterns tab
        with tabs[1]:
            st.subheader("Medication Pattern Analysis")

            if 'pharmacy' in self.data:
                # Medication usage by diagnosis
                st.write("#### Medication Usage by Diagnosis")

                if all(key in self.data for key in ['diagnoses_icd', 'd_icd_diagnoses']):
                    # Get top diagnoses for selection
                    diagnoses_df = pd.merge(
                        self.data['diagnoses_icd'],
                        self.data['d_icd_diagnoses'],
                        on='icd_code',
                        how='left'
                    )
                    top_diagnoses = diagnoses_df['long_title'].value_counts().head(20)
                    selected_diagnosis = st.selectbox("Select Diagnosis", top_diagnoses.index)

                    if selected_diagnosis:
                        # Find medications commonly prescribed for this diagnosis
                        diagnosis_hadm_ids = diagnoses_df[
                            diagnoses_df['long_title'] == selected_diagnosis
                        ]['hadm_id'].unique()

                        related_meds = self.data['pharmacy'][
                            self.data['pharmacy']['hadm_id'].isin(diagnosis_hadm_ids)
                        ]['medication'].value_counts().head(10)

                        fig = px.bar(
                            x=related_meds.index,
                            y=related_meds.values,
                            title=f'Top 10 Medications for {selected_diagnosis}'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                # Medication timing analysis
                st.write("#### Medication Timing Analysis")
                if 'starttime' in self.data['pharmacy'].columns:
                    self.data['pharmacy']['hour'] = pd.to_datetime(
                        self.data['pharmacy']['starttime']
                    ).dt.hour

                    hourly_med_orders = self.data['pharmacy']['hour'].value_counts().sort_index()

                    fig = px.line(
                        x=hourly_med_orders.index,
                        y=hourly_med_orders.values,
                        title='Medication Orders by Hour of Day'
                    )
                    st.plotly_chart(fig)

        # Lab Test Patterns tab
        with tabs[2]:
            st.subheader("Laboratory Test Pattern Analysis")

            if 'labevents' in self.data:
                # Lab test ordering patterns
                st.write("#### Lab Test Ordering Patterns")

                if 'itemid' in self.data['labevents'].columns:
                    # Get top lab tests
                    top_labs = self.data['labevents']['itemid'].value_counts().head(10)

                    fig = px.bar(
                        x=top_labs.index,
                        y=top_labs.values,
                        title='Top 10 Most Frequently Ordered Lab Tests'
                    )
                    st.plotly_chart(fig)

                # Lab test timing
                st.write("#### Lab Test Timing Analysis")
                if 'charttime' in self.data['labevents'].columns:
                    self.data['labevents']['hour'] = pd.to_datetime(
                        self.data['labevents']['charttime']
                    ).dt.hour

                    hourly_lab_orders = self.data['labevents']['hour'].value_counts().sort_index()

                    fig = px.line(
                        x=hourly_lab_orders.index,
                        y=hourly_lab_orders.values,
                        title='Lab Orders by Hour of Day'
                    )
                    st.plotly_chart(fig)

        # Care Unit Analysis tab
        with tabs[3]:
            st.subheader("Care Unit Pattern Analysis")

            if 'transfers' in self.data:
                # Care unit utilization
                st.write("#### Care Unit Utilization")

                if 'careunit' in self.data['transfers'].columns:
                    unit_counts = self.data['transfers']['careunit'].value_counts()

                    fig = px.pie(
                        values=unit_counts.values,
                        names=unit_counts.index,
                        title='Care Unit Utilization Distribution'
                    )
                    st.plotly_chart(fig)

                # Length of stay by unit
                st.write("#### Length of Stay by Care Unit")
                if all(col in self.data['transfers'].columns for col in ['intime', 'outtime', 'careunit']):
                    self.data['transfers']['los_hours'] = (
                        pd.to_datetime(self.data['transfers']['outtime']) -
                        pd.to_datetime(self.data['transfers']['intime'])
                    ).dt.total_seconds() / 3600

                    unit_los = self.data['transfers'].groupby('careunit')['los_hours'].mean()

                    fig = px.bar(
                        x=unit_los.index,
                        y=unit_los.values,
                        title='Average Length of Stay by Care Unit (Hours)'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)

        # Temporal Patterns tab
        with tabs[4]:
            st.subheader("Temporal Pattern Analysis")

            # Time range selector
            st.write("#### Select Time Range")
            if 'admissions' in self.data and 'admittime' in self.data['admissions'].columns:
                min_date = pd.to_datetime(self.data['admissions']['admittime']).min()
                max_date = pd.to_datetime(self.data['admissions']['admittime']).max()

                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )

                if len(date_range) == 2:
                    start_date, end_date = date_range

                    # Filter data for selected date range
                    filtered_admissions = self.data['admissions'][
                        (pd.to_datetime(self.data['admissions']['admittime']).dt.date >= start_date) &
                        (pd.to_datetime(self.data['admissions']['admittime']).dt.date <= end_date)
                    ]

                    # Daily admission counts
                    st.write("#### Daily Admission Counts")
                    daily_admissions = filtered_admissions['admittime'].dt.date.value_counts().sort_index()

                    fig = px.line(
                        x=daily_admissions.index,
                        y=daily_admissions.values,
                        title='Daily Admission Counts'
                    )
                    st.plotly_chart(fig)

                    # Admission type distribution over time
                    st.write("#### Admission Type Distribution Over Time")
                    admission_types = filtered_admissions.groupby(
                        [pd.to_datetime(filtered_admissions['admittime']).dt.date, 'admission_type']
                    ).size().unstack(fill_value=0)

                    fig = px.area(
                        admission_types,
                        title='Admission Types Over Time'
                    )
                    st.plotly_chart(fig)

    def run(self):
        """Run the dashboard application based on the selected mode."""
        # Display data loading status
        if self.path_valid and not self.data:
            st.sidebar.info("⚠️ Dataset not loaded. Click the 'Load Dataset' button to load the data.")

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
    dashboard = StreamlitApp()
    dashboard.run()


if __name__ == "__main__":
    main()
