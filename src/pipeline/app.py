import streamlit as st
import os
import sys
import importlib.util
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import traceback
import logging
from typing import Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the MimicPreprocessor class
from src.preprocessing.mimic_preprocessor import MimicPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="MIMIC-IV Analysis Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Prepare session state
if 'pipeline_steps' not in st.session_state:
    st.session_state.pipeline_steps = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'step_completed' not in st.session_state:
    st.session_state.step_completed = {}

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    # Import preprocessing utilities
    import mimic4_preprocess_util as preproc_util
    from preprocess_outcomes import load_data, pickle_data, reparsing, split_data

    # Check if preprocessing directory modules are available
    preprocessing_dirs_available = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessing'))
    if preprocessing_dirs_available:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessing'))

        # Try importing from preprocessing subdirectories
        try:
            from preprocessing.day_intervals_preproc.day_intervals_cohort import extract_data
            from preprocessing.hosp_module_preproc.feature_selection_hosp import feature_nonicu
            from preprocessing.hosp_module_preproc.feature_selection_icu import feature_icu
            PREPROCESSING_MODULES_AVAILABLE = True
        except ImportError as e:
            print(f"Warning: Could not import preprocessing submodules: {str(e)}")
            PREPROCESSING_MODULES_AVAILABLE = False
    else:
        PREPROCESSING_MODULES_AVAILABLE = False

    # Try importing model module
    try:
        from model_module import ModelModule
        MODEL_MODULE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import model module: {str(e)}")
        MODEL_MODULE_AVAILABLE = False

    # Set overall availability flag
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    error_details = traceback.format_exc()

    # This will be shown in the terminal but also stored for UI display
    import_error_message = f"Warning: Could not import project modules. Error: {str(e)}\n\nDetails: {error_details}"
    print(import_error_message)

    # Store the error message in a variable that can be accessed later
    # This will be used in the UI to show the error to the user
    import_error_for_ui = f"Failed to import required modules: {str(e)}"


# Define a MimicPreprocessor class to encapsulate the preprocessing functionality
class MimicPreprocessor:
    """
    A wrapper class for MIMIC-IV preprocessing functionality using existing code.
    This class serves as a bridge between the existing preprocessing code and the dashboard.
    """

    def __init__(self, mimic_path, output_dir="./data"):
        """Initialize the preprocessing module"""
        self.mimic_path = mimic_path
        self.output_dir = output_dir

        # Create output directories if they don't exist
        os.makedirs(f"{output_dir}/cohort", exist_ok=True)
        os.makedirs(f"{output_dir}/features", exist_ok=True)
        os.makedirs(f"{output_dir}/summary", exist_ok=True)

        # Default parameters
        self.use_icu = False
        self.remove_outliers = True
        self.impute_missing = True
        self.outlier_threshold = 95
        self.left_threshold = 5
        self.disease_label = ""

        # Feature paths and statistics
        self.feature_paths = {}
        self.cohort_path = None
        self.statistics = {
            "cohort": {},
            "features": {},
            "warnings": []
        }

    def read_patient_data(self):
        """Read patient data from MIMIC-IV"""
        try:
            patients = preproc_util.read_patients_table(self.mimic_path)
            return patients
        except Exception as e:
            self.statistics["warnings"].append(f"Error reading patient data: {str(e)}")
            return None

    def read_admissions_data(self):
        """Read admissions data from MIMIC-IV"""
        try:
            admits = preproc_util.read_admissions_table(self.mimic_path)
            return admits
        except Exception as e:
            self.statistics["warnings"].append(f"Error reading admissions data: {str(e)}")
            return None

    def read_icu_stays(self):
        """Read ICU stays data from MIMIC-IV"""
        try:
            stays = preproc_util.read_icustays_table(self.mimic_path)
            stays = preproc_util.clean_stays(stays)
            return stays
        except Exception as e:
            self.statistics["warnings"].append(f"Error reading ICU stays data: {str(e)}")
            return None

    def create_mortality_cohort(self, output_name="mortality_cohort"):
        """Create a mortality prediction cohort"""
        try:
            # Read tables
            patients = self.read_patient_data()
            admits = self.read_admissions_data()

            if patients is None or admits is None:
                raise ValueError("Could not read required tables")

            # Merge tables
            cohort = preproc_util.merge_on_subject(patients, admits)

            # Add mortality information
            cohort = preproc_util.add_inhospital_mortality_to_icustays(cohort)

            # Save cohort
            cohort_path = f"{self.output_dir}/cohort/{output_name}.csv.gz"
            cohort.to_csv(cohort_path, compression="gzip")
            self.cohort_path = cohort_path

            # Record statistics
            self.statistics["cohort"] = {
                "total_patients": cohort["subject_id"].nunique(),
                "total_admissions": cohort.shape[0],
                "positive_label_count": cohort["mortality_inhospital"].sum(),
                "negative_label_count": cohort.shape[0] - cohort["mortality_inhospital"].sum()
            }

            return cohort_path
        except Exception as e:
            self.statistics["warnings"].append(f"Error creating mortality cohort: {str(e)}")
            return None

    def create_length_of_stay_cohort(self, threshold=7, output_name="los_cohort"):
        """Create a length of stay prediction cohort"""
        try:
            # Read tables
            patients = self.read_patient_data()
            admits = self.read_admissions_data()

            if patients is None or admits is None:
                raise ValueError("Could not read required tables")

            # Merge tables
            cohort = preproc_util.merge_on_subject(patients, admits)

            # Calculate length of stay
            cohort['los_days'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / (24 * 3600)

            # Create label
            cohort['extended_los'] = (cohort['los_days'] > threshold).astype(int)

            # Save cohort
            cohort_path = f"{self.output_dir}/cohort/{output_name}.csv.gz"
            cohort.to_csv(cohort_path, compression="gzip")
            self.cohort_path = cohort_path

            # Record statistics
            self.statistics["cohort"] = {
                "total_patients": cohort["subject_id"].nunique(),
                "total_admissions": cohort.shape[0],
                "positive_label_count": cohort["extended_los"].sum(),
                "negative_label_count": cohort.shape[0] - cohort["extended_los"].sum()
            }

            return cohort_path
        except Exception as e:
            self.statistics["warnings"].append(f"Error creating LOS cohort: {str(e)}")
            return None

    def extract_features(self, cohort_path, feature_categories=None):
        """Extract features based on the available code"""
        try:
            # Default feature categories
            if feature_categories is None:
                feature_categories = {
                    "diagnoses": True,
                    "procedures": True,
                    "medications": True,
                    "lab_tests": True
                }

            # Feature paths
            feature_paths = {}

            # Extract features based on which module is available
            if PREPROCESSING_MODULES_AVAILABLE:
                # Use the imported preprocessing modules
                if self.use_icu:
                    feature_icu(
                        cohort_output=os.path.basename(cohort_path).replace(".csv.gz", ""),
                        version_path=self.mimic_path,
                        diag_flag=feature_categories.get("diagnoses", True),
                        proc_flag=feature_categories.get("procedures", True),
                        med_flag=feature_categories.get("medications", True),
                        lab_flag=feature_categories.get("lab_tests", True)
                    )
                else:
                    feature_nonicu(
                        cohort_output=os.path.basename(cohort_path).replace(".csv.gz", ""),
                        version_path=self.mimic_path,
                        diag_flag=feature_categories.get("diagnoses", True),
                        proc_flag=feature_categories.get("procedures", True),
                        med_flag=feature_categories.get("medications", True),
                        lab_flag=feature_categories.get("lab_tests", True)
                    )

                # Set feature paths
                feature_paths = {
                    "diagnoses": f"{self.output_dir}/features/preproc_diag.csv.gz" if feature_categories.get("diagnoses", True) else None,
                    "procedures": f"{self.output_dir}/features/preproc_proc.csv.gz" if feature_categories.get("procedures", True) else None,
                    "medications": f"{self.output_dir}/features/preproc_med.csv.gz" if feature_categories.get("medications", True) else None,
                    "lab_tests": f"{self.output_dir}/features/preproc_labs.csv.gz" if feature_categories.get("lab_tests", True) else None
                }
            else:
                # Use basic feature extraction based on mimic4_preprocess_util
                # Read cohort
                cohort = pd.read_csv(cohort_path, compression="gzip")

                # Read tables directly using the utility functions
                if feature_categories.get("diagnoses", True):
                    try:
                        diag_path = os.path.join(self.mimic_path, 'hosp/diagnoses_icd.csv.gz')
                        diagnoses = pd.read_csv(diag_path, compression="gzip")
                        diagnoses = diagnoses[diagnoses['subject_id'].isin(cohort['subject_id'])]
                        diagnoses.to_csv(f"{self.output_dir}/features/preproc_diag.csv.gz", compression="gzip")
                        feature_paths["diagnoses"] = f"{self.output_dir}/features/preproc_diag.csv.gz"
                    except Exception as e:
                        self.statistics["warnings"].append(f"Error extracting diagnoses: {str(e)}")

                if feature_categories.get("procedures", True):
                    try:
                        proc_path = os.path.join(self.mimic_path, 'hosp/procedures_icd.csv.gz')
                        procedures = pd.read_csv(proc_path, compression="gzip")
                        procedures = procedures[procedures['subject_id'].isin(cohort['subject_id'])]
                        procedures.to_csv(f"{self.output_dir}/features/preproc_proc.csv.gz", compression="gzip")
                        feature_paths["procedures"] = f"{self.output_dir}/features/preproc_proc.csv.gz"
                    except Exception as e:
                        self.statistics["warnings"].append(f"Error extracting procedures: {str(e)}")

                if feature_categories.get("medications", True):
                    try:
                        med_path = os.path.join(self.mimic_path, 'hosp/prescriptions.csv.gz')
                        medications = pd.read_csv(med_path, compression="gzip")
                        medications = medications[medications['subject_id'].isin(cohort['subject_id'])]
                        medications.to_csv(f"{self.output_dir}/features/preproc_med.csv.gz", compression="gzip")
                        feature_paths["medications"] = f"{self.output_dir}/features/preproc_med.csv.gz"
                    except Exception as e:
                        self.statistics["warnings"].append(f"Error extracting medications: {str(e)}")

                if feature_categories.get("lab_tests", True):
                    try:
                        lab_path = os.path.join(self.mimic_path, 'hosp/labevents.csv.gz')
                        labs = pd.read_csv(lab_path, compression="gzip", nrows=100000)  # Limited rows due to size
                        labs = labs[labs['subject_id'].isin(cohort['subject_id'])]
                        labs.to_csv(f"{self.output_dir}/features/preproc_labs.csv.gz", compression="gzip")
                        feature_paths["lab_tests"] = f"{self.output_dir}/features/preproc_labs.csv.gz"
                    except Exception as e:
                        self.statistics["warnings"].append(f"Error extracting lab tests: {str(e)}")

            # Store feature paths
            self.feature_paths = feature_paths

            # Collect feature statistics
            self.collect_feature_statistics(feature_categories)

            return feature_paths
        except Exception as e:
            self.statistics["warnings"].append(f"Error in feature extraction: {str(e)}")
            return {}

    def collect_feature_statistics(self, feature_categories):
        """Collect statistics about extracted features"""
        try:
            feature_stats = {}

            for feature_type, path in self.feature_paths.items():
                if path and os.path.exists(path):
                    try:
                        df = pd.read_csv(path, compression="gzip")

                        # Get column names based on feature type
                        id_col = None
                        if feature_type == "diagnoses" and "new_icd_code" in df.columns:
                            id_col = "new_icd_code"
                        elif feature_type == "diagnoses" and "icd_code" in df.columns:
                            id_col = "icd_code"
                        elif feature_type == "procedures" and "icd_code" in df.columns:
                            id_col = "icd_code"
                        elif feature_type == "medications" and "drug_name" in df.columns:
                            id_col = "drug_name"
                        elif feature_type == "medications" and "drug" in df.columns:
                            id_col = "drug"
                        elif feature_type == "lab_tests" and "itemid" in df.columns:
                            id_col = "itemid"

                        if id_col:
                            feature_stats[feature_type] = {
                                "total_records": df.shape[0],
                                "unique_codes": df[id_col].nunique(),
                                "patients": df["subject_id"].nunique(),
                                "admissions": df["hadm_id"].nunique() if "hadm_id" in df.columns else 0
                            }
                    except Exception as e:
                        self.statistics["warnings"].append(f"Error reading {feature_type} file: {str(e)}")

            self.statistics["features"] = feature_stats
        except Exception as e:
            self.statistics["warnings"].append(f"Error collecting feature statistics: {str(e)}")

    def generate_summary(self, feature_categories=None):
        """Generate summary statistics for features"""
        try:
            # Default feature categories
            if feature_categories is None:
                feature_categories = {
                    "diagnoses": True,
                    "procedures": True,
                    "medications": True,
                    "lab_tests": True
                }

            summary_paths = {}

            # For each feature type, generate basic summary
            for feature_type, path in self.feature_paths.items():
                if path and os.path.exists(path):
                    try:
                        df = pd.read_csv(path, compression="gzip")

                        # Get key column based on feature type
                        id_col = None
                        if feature_type == "diagnoses" and "new_icd_code" in df.columns:
                            id_col = "new_icd_code"
                        elif feature_type == "diagnoses" and "icd_code" in df.columns:
                            id_col = "icd_code"
                        elif feature_type == "procedures" and "icd_code" in df.columns:
                            id_col = "icd_code"
                        elif feature_type == "medications" and "drug_name" in df.columns:
                            id_col = "drug_name"
                        elif feature_type == "medications" and "drug" in df.columns:
                            id_col = "drug"
                        elif feature_type == "lab_tests" and "itemid" in df.columns:
                            id_col = "itemid"

                        if id_col:
                            # Generate frequency table
                            summary = df[id_col].value_counts().reset_index()
                            summary.columns = [id_col, 'count']

                            # Add percentage
                            summary['percentage'] = 100 * summary['count'] / summary['count'].sum()

                            # Save summary
                            summary_path = f"{self.output_dir}/summary/{feature_type}_summary.csv"
                            summary.to_csv(summary_path, index=False)
                            summary_paths[feature_type] = summary_path
                    except Exception as e:
                        self.statistics["warnings"].append(f"Error generating summary for {feature_type}: {str(e)}")

            return summary_paths
        except Exception as e:
            self.statistics["warnings"].append(f"Error generating summary: {str(e)}")
            return {}

    def run_preprocessing_pipeline(self, cohort_name, feature_categories=None,
                                 use_mortality=True, use_los=False, los_threshold=7,
                                 use_readmission=False, readmission_window=30,
                                 disease_label="", **kwargs):
        """Run the complete preprocessing pipeline"""
        try:
            # Reset statistics
            self.statistics = {
                "cohort": {},
                "features": {},
                "warnings": []
            }

            # Create cohort based on selection
            if use_mortality:
                cohort_path = self.create_mortality_cohort(output_name=cohort_name)
            elif use_los:
                cohort_path = self.create_length_of_stay_cohort(threshold=los_threshold, output_name=cohort_name)
            else:
                # Default to mortality
                cohort_path = self.create_mortality_cohort(output_name=cohort_name)

            if not cohort_path:
                raise ValueError("Failed to create cohort")

            # Extract features
            feature_paths = self.extract_features(cohort_path, feature_categories)

            # Generate summary
            summary_paths = self.generate_summary(feature_categories)

            # Return results
            return {
                "cohort": cohort_path,
                "diagnoses": feature_paths.get("diagnoses"),
                "procedures": feature_paths.get("procedures"),
                "medications": feature_paths.get("medications"),
                "lab_tests": feature_paths.get("lab_tests"),
                "summary": summary_paths,
                "statistics": self.statistics
            }
        except Exception as e:
            self.statistics["warnings"].append(f"Error in preprocessing pipeline: {str(e)}")
            return {
                "cohort": None,
                "diagnoses": None,
                "procedures": None,
                "medications": None,
                "lab_tests": None,
                "summary": {},
                "statistics": self.statistics
            }


# PipelineStep Base Class
class PipelineStep:
    def __init__(self, name: str, order: int, description: str = ""):
        self.name = name
        self.order = order
        self.description = description
        self.status = "Not Started"
        self.result = None

    def validate(self) -> bool:
        """Validate that the step can be executed"""
        raise NotImplementedError("Subclasses must implement validate()")

    def execute(self) -> Any:
        """Execute the pipeline step"""
        raise NotImplementedError("Subclasses must implement execute()")

    def render(self):
        """Render the pipeline step in the Streamlit UI"""
        st.subheader(f"Step {self.order}: {self.name}")

        if self.description:
            st.write(self.description)

        # If this step has not been completed yet
        if self.status != "Completed":
            execute_button = st.button(f"Execute {self.name}", key=f"execute_{self.name}")
            if execute_button:
                try:
                    if self.validate():
                        with st.spinner(f"Executing {self.name}..."):
                            self.result = self.execute()
                            self.status = "Completed"
                            st.session_state.step_completed[self.order] = True
                            st.success(f"{self.name} completed successfully!")
                    else:
                        st.error("Validation failed. Cannot execute step.")
                except Exception as e:
                    st.error(f"Error executing {self.name}: {str(e)}")
                    st.error(traceback.format_exc())
        else:
            st.success(f"{self.name} completed ✓")
            reset_button = st.button(f"Reset {self.name}", key=f"reset_{self.name}")
            if reset_button:
                self.status = "Not Started"
                self.result = None
                if self.order in st.session_state.step_completed:
                    del st.session_state.step_completed[self.order]
                st.experimental_rerun()

        # Render the result if available
        if self.result is not None:
            self.render_result()

    def render_result(self):
        """Render the result of the pipeline step"""
        st.write("Step completed successfully.")

        # Default rendering for result data
        if isinstance(self.result, pd.DataFrame):
            st.write("Result DataFrame:")
            st.dataframe(self.result.head())
            st.write(f"Shape: {self.result.shape}")


# Data Preprocessing Step
class DataPreprocessingStep(PipelineStep):
    def __init__(self, order: int):
        super().__init__(name="Data Preprocessing", order=order,
                        description="Preprocess the MIMIC-IV data for analysis")
        self.preprocessor = MimicPreprocessor()

    def validate(self) -> bool:
        """Validate that the data can be preprocessed"""
        # Check if MIMIC-IV data paths are configured
        if not self.preprocessor.config.get('data_path'):
            st.error("MIMIC-IV data path not configured. Please set it in the configuration.")
            return False
        return True

    def execute(self) -> Any:
        """Execute the data preprocessing step"""
        try:
            # Get parameters from the UI
            data_option = st.session_state.get('data_option', 'sample')
            features = st.session_state.get('selected_features', [])
            outcome = st.session_state.get('selected_outcome', None)

            # Log the preprocessing parameters
            logger.info(f"Preprocessing data with option: {data_option}")
            logger.info(f"Selected features: {features}")
            logger.info(f"Selected outcome: {outcome}")

            # Run the preprocessing
            processed_data = self.preprocessor.preprocess(
                data_option=data_option,
                features=features,
                outcome=outcome
            )

            return processed_data
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def render(self):
        """Render the data preprocessing step UI"""
        st.subheader(f"Step {self.order}: {self.name}")

        if self.description:
            st.write(self.description)

        # Data options
        st.write("#### Data Options")
        data_option = st.radio(
            "Select data to preprocess:",
            options=["sample", "full"],
            index=0,
            key="data_option"
        )

        # Feature selection
        st.write("#### Feature Selection")
        available_features = self.preprocessor.get_available_features()
        selected_features = st.multiselect(
            "Select features to include:",
            options=available_features,
            default=available_features[:5] if available_features else [],
            key="selected_features"
        )

        # Outcome selection
        st.write("#### Outcome Selection")
        available_outcomes = self.preprocessor.get_available_outcomes()
        selected_outcome = st.selectbox(
            "Select outcome variable:",
            options=available_outcomes,
            index=0 if available_outcomes else None,
            key="selected_outcome"
        )

        # Execute button
        if self.status != "Completed":
            execute_button = st.button(f"Execute {self.name}", key=f"execute_{self.name}")
            if execute_button:
                try:
                    if self.validate():
                        with st.spinner(f"Executing {self.name}..."):
                            self.result = self.execute()
                            self.status = "Completed"
                            st.session_state.step_completed[self.order] = True
                            st.success(f"{self.name} completed successfully!")
                    else:
                        st.error("Validation failed. Cannot execute step.")
                except Exception as e:
                    st.error(f"Error executing {self.name}: {str(e)}")
                    st.error(traceback.format_exc())
        else:
            st.success(f"{self.name} completed ✓")
            reset_button = st.button(f"Reset {self.name}", key=f"reset_{self.name}")
            if reset_button:
                self.status = "Not Started"
                self.result = None
                if self.order in st.session_state.step_completed:
                    del st.session_state.step_completed[self.order]
                st.experimental_rerun()

        # Render the result if available
        if self.result is not None:
            self.render_result()

    def render_result(self):
        """Render the preprocessing results"""
        st.write("### Preprocessing Results")

        if isinstance(self.result, dict):
            # Display main dataset
            if 'data' in self.result and isinstance(self.result['data'], pd.DataFrame):
                st.write("#### Preprocessed Data")
                st.dataframe(self.result['data'].head())
                st.write(f"Shape: {self.result['data'].shape}")

                # Display feature statistics
                st.write("#### Feature Statistics")
                st.write(self.result['data'].describe())

                # Display additional metadata if available
                if 'metadata' in self.result:
                    st.write("#### Metadata")
                    st.json(self.result['metadata'])
        elif isinstance(self.result, pd.DataFrame):
            st.write("#### Preprocessed Data")
            st.dataframe(self.result.head())
            st.write(f"Shape: {self.result.shape}")

            # Display feature statistics
            st.write("#### Feature Statistics")
            st.write(self.result.describe())
        else:
            st.write("Preprocessing completed, but no compatible data format was returned.")


class FeatureSelectionStep(PipelineStep):
    """Handles the feature selection step of the pipeline."""

    def render(self):
        st.header("Feature Selection")

        if not self.validate():
            return

        st.info("Feature selection will be implemented in the next version.")
        st.write("This feature will allow you to:")
        st.write("- Select specific features from each category")
        st.write("- Apply feature importance methods")
        st.write("- Generate feature importance plots")
        st.write("- Save selected feature sets")

    def validate(self):
        if not MODULES_AVAILABLE:
            if 'import_error_for_ui' in globals():
                st.error(f"Required modules not available: {import_error_for_ui}")
            else:
                st.error("Required modules not available. Please check your installation.")

            st.info("Possible solutions:\n"
                    "1. Make sure all dependencies are installed by running `pip install -r requirements.txt`\n"
                    "2. Check that all project files are in the correct directories\n"
                    "3. Restart the application")
            return False
        elif st.session_state.preprocessing_module is None:
            st.warning("Please run the preprocessing step first.")
            return False
        return True


class ModelTrainingStep(PipelineStep):
    """Handles the model training step of the pipeline."""

    def render(self):
        st.header("Model Training")

        if not self.validate():
            return

        # Model settings
        st.subheader("Model Settings")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type", ["lstm", "gru", "transformer"])
            hidden_size = st.number_input("Hidden Size", value=128, min_value=1)
            num_layers = st.number_input("Number of Layers", value=2, min_value=1)
        with col2:
            dropout = st.slider("Dropout", value=0.2, min_value=0.0, max_value=1.0)
            batch_size = st.number_input("Batch Size", value=32, min_value=1)
            learning_rate = st.number_input("Learning Rate", value=0.001, format="%.3f")

        # Training settings
        st.subheader("Training Settings")
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.number_input("Number of Epochs", value=10, min_value=1)
            target_column = st.text_input("Target Column", value="mortality")
        with col2:
            sensitive_columns = st.multiselect(
                "Sensitive Columns",
                options=["ethnicity", "gender", "age"],
                default=["ethnicity", "gender"]
            )

        # Run training
        if st.button("Train Model"):
            try:
                # Initialize model module
                st.session_state.model_module = ModelModule(
                    output_dir="./data/models"
                )

                # Set parameters
                st.session_state.model_module.model_type = model_type
                st.session_state.model_module.hidden_size = hidden_size
                st.session_state.model_module.num_layers = num_layers
                st.session_state.model_module.dropout = dropout
                st.session_state.model_module.batch_size = batch_size
                st.session_state.model_module.learning_rate = learning_rate
                st.session_state.model_module.num_epochs = num_epochs

                # Run pipeline
                with st.spinner("Training model..."):
                    results = st.session_state.model_module.run_model_pipeline(
                        feature_paths=st.session_state.preprocessing_module.feature_paths,
                        cohort_path=st.session_state.preprocessing_module.cohort_path,
                        target_column=target_column,
                        sensitive_columns=sensitive_columns,
                        model_name=f"{model_type}_{target_column}"
                    )

                    st.success("Model training completed successfully!")

                    # Display results
                    st.subheader("Model Metrics")
                    for metric, value in results["metrics"].items():
                        if metric != "fairness":
                            st.write(f"**{metric}:** {value:.4f}")

                    if "fairness" in results["metrics"]:
                        st.subheader("Fairness Metrics")
                        for attribute, metrics in results["metrics"]["fairness"].items():
                            st.write(f"**{attribute}:**")
                            for metric, value in metrics.items():
                                st.write(f"- {metric}: {value:.4f}")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.error("Check logs for more details")

    def validate(self):
        if not MODULES_AVAILABLE:
            if 'import_error_for_ui' in globals():
                st.error(f"Required modules not available: {import_error_for_ui}")
            else:
                st.error("Required modules not available. Please check your installation.")

            st.info("Possible solutions:\n"
                    "1. Make sure all dependencies are installed by running `pip install -r requirements.txt`\n"
                    "2. Check that all project files are in the correct directories\n"
                    "3. Restart the application")
            return False
        elif st.session_state.preprocessing_module is None:
            st.warning("Please run the preprocessing step first.")
            return False
        return True


class EvaluationStep(PipelineStep):
    """Handles the model evaluation step of the pipeline."""

    def render(self):
        st.header("Model Evaluation")

        if not self.validate():
            return

        st.info("Evaluation metrics will be displayed here.")
        st.write("This section will show:")
        st.write("- Model performance metrics")
        st.write("- ROC curves")
        st.write("- Precision-recall curves")
        st.write("- Confusion matrices")

    def validate(self):
        if not MODULES_AVAILABLE:
            if 'import_error_for_ui' in globals():
                st.error(f"Required modules not available: {import_error_for_ui}")
            else:
                st.error("Required modules not available. Please check your installation.")

            st.info("Possible solutions:\n"
                    "1. Make sure all dependencies are installed by running `pip install -r requirements.txt`\n"
                    "2. Check that all project files are in the correct directories\n"
                    "3. Restart the application")
            return False
        elif st.session_state.model_module is None:
            st.warning("Please train a model first.")
            return False
        return True


class FairnessAnalysisStep(PipelineStep):
    """Handles the fairness analysis step of the pipeline."""

    def render(self):
        st.header("Fairness Analysis")

        if not self.validate():
            return

        st.info("Fairness analysis will be displayed here.")
        st.write("This section will show:")
        st.write("- Demographic parity")
        st.write("- Equal opportunity")
        st.write("- Equalized odds")
        st.write("- Treatment equality")

    def validate(self):
        if not MODULES_AVAILABLE:
            if 'import_error_for_ui' in globals():
                st.error(f"Required modules not available: {import_error_for_ui}")
            else:
                st.error("Required modules not available. Please check your installation.")

            st.info("Possible solutions:\n"
                    "1. Make sure all dependencies are installed by running `pip install -r requirements.txt`\n"
                    "2. Check that all project files are in the correct directories\n"
                    "3. Restart the application")
            return False
        elif st.session_state.model_module is None:
            st.warning("Please train a model first.")
            return False
        return True


class StreamlitApp:
    """Main application class for the MIMIC-IV Data Pipeline."""

    def __init__(self):
        """Initialize the application."""
        self.steps = {
            "Dataset Loading": DatasetLoadingStep(),
            "Data Preprocessing": DataPreprocessingStep(),
            "Feature Selection": FeatureSelectionStep(),
            "Model Training": ModelTrainingStep(),
            "Evaluation": EvaluationStep(),
            "Fairness Analysis": FairnessAnalysisStep()
        }

        # Initialize session state
        if "preprocessing_module" not in st.session_state:
            st.session_state.preprocessing_module = None
        if "model_module" not in st.session_state:
            st.session_state.model_module = None
        if "current_step" not in st.session_state:
            st.session_state.current_step = "Dataset Loading"
        if "dataset_loaded" not in st.session_state:
            st.session_state.dataset_loaded = False
        if "mimic_path" not in st.session_state:
            st.session_state.mimic_path = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"

    def render_header(self):
        """Render the application header."""
        st.title("🏥 MIMIC-IV Data Pipeline")
        st.markdown("""
        This dashboard helps you process and analyze MIMIC-IV healthcare data through a step-by-step pipeline.
        Start by loading your dataset, then proceed through preprocessing, model training, and analysis.
        """)
        st.markdown("---")

    def render_main_content(self):
        """Render the main content with tabs for navigation."""
        # Create tabs for each step
        tabs = st.tabs(list(self.steps.keys()))

        # Render the content for each tab
        for i, (step_name, step) in enumerate(self.steps.items()):
            with tabs[i]:
                step.render()

    def render_footer(self):
        """Render the footer."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>MIMIC-IV Data Pipeline Dashboard | Built with Streamlit</p>
            <p>For documentation and support, please visit the <a href="https://github.com/your-repo/mimic-iv-pipeline">GitHub repository</a></p>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        """Run the application."""
        # Render the application
        self.render_header()
        self.render_main_content()
        self.render_footer()


def initialize_app():
    """Initialize the Streamlit app and return the pipeline steps"""
    # Create pipeline steps
    pipeline_steps = [
        DataPreprocessingStep(order=1)
    ]

    return pipeline_steps

def run_app():
    """Run the Streamlit application"""
    # Set up page configuration
    st.set_page_config(
        page_title="MIMIC-IV Analysis Pipeline",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Prepare session state
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = initialize_app()

    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

    if 'step_completed' not in st.session_state:
        st.session_state.step_completed = {}

    # Render app title and description
    st.title("MIMIC-IV Analysis Pipeline")
    st.write("""
    This application guides you through analyzing MIMIC-IV clinical data for healthcare research.
    Follow the steps sequentially to preprocess the data, build models, and analyze results.
    """)

    # Create sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        st.write("Select a step to navigate:")

        for i, step in enumerate(st.session_state.pipeline_steps):
            is_completed = st.session_state.step_completed.get(step.order, False)
            status_icon = "✅" if is_completed else "⏳"

            if st.button(f"{status_icon} {step.name}", key=f"nav_{step.order}"):
                st.session_state.current_step = i
                st.experimental_rerun()

    # Display current step
    current_step = st.session_state.pipeline_steps[st.session_state.current_step]
    current_step.render()

    # Navigation buttons
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.current_step > 0:
            if st.button("⬅️ Previous Step"):
                st.session_state.current_step -= 1
                st.experimental_rerun()

    with col2:
        if st.session_state.current_step < len(st.session_state.pipeline_steps) - 1:
            next_enabled = st.session_state.step_completed.get(
                st.session_state.pipeline_steps[st.session_state.current_step].order,
                False
            )

            if st.button("Next Step ➡️", disabled=not next_enabled):
                st.session_state.current_step += 1
                st.experimental_rerun()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        run_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())
        logger.error(f"Application error: {str(e)}", exc_info=True)
