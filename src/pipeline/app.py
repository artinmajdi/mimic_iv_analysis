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
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'step_completed' not in st.session_state:
        st.session_state.step_completed = {}
    if 'preprocessing_module' not in st.session_state:
        st.session_state.preprocessing_module = None
    if 'model_module' not in st.session_state:
        st.session_state.model_module = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'mimic_path' not in st.session_state:
        st.session_state.mimic_path = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"

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


class Config:
    """Configuration class for the MIMIC-IV preprocessor"""
    def __init__(self, data_path: str, output_dir: str = "./data"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.use_icu = False
        self.remove_outliers = True
        self.impute_missing = True
        self.outlier_threshold = 95
        self.left_threshold = 5


# Define a MimicPreprocessor class to encapsulate the preprocessing functionality
class MimicPreprocessor:
    """
    A wrapper class for MIMIC-IV preprocessing functionality using existing code.
    This class serves as a bridge between the existing preprocessing code and the dashboard.
    """

    def __init__(self, mimic_path: str = None, output_dir: str = "./data"):
        """Initialize the preprocessing module"""
        self.mimic_path = mimic_path or st.session_state.get('mimic_path')
        self.output_dir = output_dir
        self.config = Config(self.mimic_path, output_dir)

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

    def get_available_features(self) -> List[str]:
        """Return list of available features"""
        return [
            "diagnoses",
            "procedures",
            "medications",
            "lab_tests",
            "vital_signs",
            "demographics"
        ]

    def get_available_outcomes(self) -> List[str]:
        """Return list of available outcome variables"""
        return [
            "mortality",
            "length_of_stay",
            "readmission"
        ]

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


class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    @abstractmethod
    def run(self):
        """Run the pipeline step"""
        pass

    @abstractmethod
    def render(self):
        """Render the UI for this step"""
        pass

class DatasetLoadingStep(PipelineStep):
    """Step for loading and validating the MIMIC-IV dataset"""
    def __init__(self):
        self.preprocessor = None
        self.status = "not_started"
        self.error = None

    def run(self):
        """Run the dataset loading step"""
        try:
            # Initialize preprocessor
            self.preprocessor = MimicPreprocessor()

            # Validate MIMIC-IV path
            if not os.path.exists(self.preprocessor.mimic_path):
                raise FileNotFoundError(f"MIMIC-IV dataset not found at {self.preprocessor.mimic_path}")

            # Check for required subdirectories
            required_dirs = ['hosp', 'icu']  # Updated required directories
            for dir_name in required_dirs:
                dir_path = os.path.join(self.preprocessor.mimic_path, dir_name)
                if not os.path.exists(dir_path):
                    raise FileNotFoundError(f"Required directory '{dir_name}' not found in MIMIC-IV dataset")

            # Check for essential files
            required_files = {
                'hosp/admissions.csv.gz': 'admissions data',
                'icu/icustays.csv.gz': 'ICU stays data',
                'hosp/patients.csv.gz': 'patient data'
            }

            for file_path, description in required_files.items():
                full_path = os.path.join(self.preprocessor.mimic_path, file_path)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Required {description} file not found: {file_path}")

            st.session_state.preprocessing_module = self.preprocessor
            st.session_state.dataset_loaded = True
            self.status = "completed"
            return True
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Dataset loading failed: {str(e)}")
            return False

    def render(self):
        """Render the dataset loading UI"""
        st.header("Step 1: Dataset Loading")

        # Show MIMIC-IV path input
        mimic_path = st.text_input(
            "MIMIC-IV Dataset Path",
            value=st.session_state.get('mimic_path', ''),
            help="Path to the MIMIC-IV dataset directory"
        )

        if mimic_path != st.session_state.get('mimic_path'):
            st.session_state.mimic_path = mimic_path

        # Show status and errors
        if self.status == "completed":
            st.success("Dataset loaded successfully!")
        elif self.status == "error":
            st.error(f"Error loading dataset: {self.error}")

        # Show load button
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                success = self.run()
                if success:
                    st.session_state.step_completed['dataset_loading'] = True
                    st.rerun()


class DataPreprocessingStep(PipelineStep):
    """Step for preprocessing the MIMIC-IV dataset"""
    def __init__(self):
        self.status = "not_started"
        self.error = None
        self.preprocessed_data = None

    def run(self):
        """Run the preprocessing step"""
        try:
            if not st.session_state.preprocessing_module:
                raise ValueError("Preprocessor not initialized. Please complete dataset loading first.")

            preprocessor = st.session_state.preprocessing_module

            # Get preprocessing parameters from session state
            config = preprocessor.config

            # Run preprocessing
            self.preprocessed_data = preprocessor.preprocess_data()

            self.status = "completed"
            return True
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Preprocessing failed: {str(e)}")
            return False

    def render(self):
        """Render the preprocessing UI"""
        st.header("Step 2: Data Preprocessing")

        if not st.session_state.step_completed.get('dataset_loading'):
            st.warning("Please complete dataset loading first.")
            return

        # Preprocessing options
        st.subheader("Preprocessing Options")

        preprocessor = st.session_state.preprocessing_module
        config = preprocessor.config

        # Configure preprocessing options
        config.use_icu = st.checkbox("Include ICU data", value=config.use_icu)
        config.remove_outliers = st.checkbox("Remove outliers", value=config.remove_outliers)
        config.impute_missing = st.checkbox("Impute missing values", value=config.impute_missing)

        if config.remove_outliers:
            config.outlier_threshold = st.slider(
                "Outlier threshold (standard deviations)",
                min_value=1.0,
                max_value=5.0,
                value=float(config.outlier_threshold),
                step=0.5
            )

        # Show status and errors
        if self.status == "completed":
            st.success("Preprocessing completed successfully!")
            if self.preprocessed_data is not None:
                st.write("Preprocessed data shape:", self.preprocessed_data.shape)
        elif self.status == "error":
            st.error(f"Error during preprocessing: {self.error}")

        # Show preprocess button
        if st.button("Run Preprocessing"):
            with st.spinner("Preprocessing data..."):
                success = self.run()
                if success:
                    st.session_state.step_completed['preprocessing'] = True
                    st.rerun()


class FeatureSelectionStep(PipelineStep):
    """Step for selecting features and outcomes"""
    def __init__(self):
        self.status = "not_started"
        self.error = None
        self.selected_features = []
        self.selected_outcomes = []

    def run(self):
        """Run the feature selection step"""
        try:
            if not st.session_state.preprocessing_module:
                raise ValueError("Preprocessor not initialized. Please complete preprocessing first.")

            preprocessor = st.session_state.preprocessing_module

            # Validate feature selection
            if not self.selected_features:
                raise ValueError("No features selected")
            if not self.selected_outcomes:
                raise ValueError("No outcomes selected")

            # Store selections in session state
            st.session_state.selected_features = self.selected_features
            st.session_state.selected_outcomes = self.selected_outcomes

            self.status = "completed"
            return True
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Feature selection failed: {str(e)}")
            return False

    def render(self):
        """Render the feature selection UI"""
        st.header("Step 3: Feature Selection")

        if not st.session_state.step_completed.get('preprocessing'):
            st.warning("Please complete preprocessing first.")
            return

        preprocessor = st.session_state.preprocessing_module

        # Feature selection
        st.subheader("Select Features")
        available_features = preprocessor.get_available_features()
        self.selected_features = st.multiselect(
            "Select features for model training",
            options=available_features,
            default=self.selected_features or []
        )

        # Outcome selection
        st.subheader("Select Outcomes")
        available_outcomes = preprocessor.get_available_outcomes()
        self.selected_outcomes = st.multiselect(
            "Select outcomes to predict",
            options=available_outcomes,
            default=self.selected_outcomes or []
        )

        # Show status and errors
        if self.status == "completed":
            st.success("Feature selection completed!")
        elif self.status == "error":
            st.error(f"Error during feature selection: {self.error}")

        # Show confirm button
        if st.button("Confirm Selection"):
            with st.spinner("Confirming feature selection..."):
                success = self.run()
                if success:
                    st.session_state.step_completed['feature_selection'] = True
                    st.rerun()


class ModelTrainingStep(PipelineStep):
    """Step for training the machine learning model"""
    def __init__(self):
        self.status = "not_started"
        self.error = None
        self.model = None
        self.training_results = None

    def run(self):
        """Run the model training step"""
        try:
            if not st.session_state.model_module:
                raise ValueError("Model module not initialized")

            model_module = st.session_state.model_module

            # Get training parameters from session state
            params = st.session_state.get('training_params', {})

            # Train model
            self.model, self.training_results = model_module.train_model(
                features=st.session_state.selected_features,
                outcomes=st.session_state.selected_outcomes,
                **params
            )

            # Store model in session state
            st.session_state.trained_model = self.model
            st.session_state.training_results = self.training_results

            self.status = "completed"
            return True
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Model training failed: {str(e)}")
            return False

    def render(self):
        """Render the model training UI"""
        st.header("Step 4: Model Training")

        if not st.session_state.step_completed.get('feature_selection'):
            st.warning("Please complete feature selection first.")
            return

        # Model configuration
        st.subheader("Model Configuration")

        # Model type selection
        model_type = st.selectbox(
            "Select model type",
            options=["Random Forest", "XGBoost", "Logistic Regression"]
        )

        # Training parameters
        st.subheader("Training Parameters")

        params = {}
        if model_type == "Random Forest":
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100)
            params['max_depth'] = st.slider("Maximum depth", 2, 20, 10)
        elif model_type == "XGBoost":
            params['learning_rate'] = st.slider("Learning rate", 0.01, 0.3, 0.1)
            params['max_depth'] = st.slider("Maximum depth", 2, 20, 6)
        elif model_type == "Logistic Regression":
            params['C'] = st.slider("Regularization strength", 0.1, 10.0, 1.0)

        # Store parameters in session state
        st.session_state.training_params = params

        # Show status and errors
        if self.status == "completed":
            st.success("Model training completed!")
            if self.training_results:
                st.write("Training Results:")
                st.write(self.training_results)
        elif self.status == "error":
            st.error(f"Error during model training: {self.error}")

        # Show train button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                success = self.run()
                if success:
                    st.session_state.step_completed['model_training'] = True
                    st.rerun()


class EvaluationStep(PipelineStep):
    """Step for evaluating the trained model"""
    def __init__(self):
        self.status = "not_started"
        self.error = None
        self.evaluation_results = None

    def run(self):
        """Run the model evaluation step"""
        try:
            if not st.session_state.model_module:
                raise ValueError("Model module not initialized")

            model_module = st.session_state.model_module

            # Evaluate model
            self.evaluation_results = model_module.evaluate_model(
                model=st.session_state.trained_model,
                features=st.session_state.selected_features,
                outcomes=st.session_state.selected_outcomes
            )

            # Store results in session state
            st.session_state.evaluation_results = self.evaluation_results

            self.status = "completed"
            return True
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Model evaluation failed: {str(e)}")
            return False

    def render(self):
        """Render the model evaluation UI"""
        st.header("Step 5: Model Evaluation")

        if not st.session_state.step_completed.get('model_training'):
            st.warning("Please complete model training first.")
            return

        # Evaluation options
        st.subheader("Evaluation Options")

        metrics = st.multiselect(
            "Select evaluation metrics",
            options=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            default=["Accuracy", "ROC-AUC"]
        )

        # Show status and errors
        if self.status == "completed":
            st.success("Model evaluation completed!")
            if self.evaluation_results:
                st.write("Evaluation Results:")
                st.write(self.evaluation_results)
        elif self.status == "error":
            st.error(f"Error during evaluation: {self.error}")

        # Show evaluate button
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                success = self.run()
                if success:
                    st.session_state.step_completed['evaluation'] = True
                    st.rerun()


class FairnessAnalysisStep(PipelineStep):
    """Step for analyzing model fairness"""
    def __init__(self):
        self.status = "not_started"
        self.error = None
        self.fairness_results = None

    def run(self):
        """Run the fairness analysis step"""
        try:
            if not st.session_state.model_module:
                raise ValueError("Model module not initialized")

            model_module = st.session_state.model_module

            # Get fairness parameters from session state
            params = st.session_state.get('fairness_params', {})

            # Analyze fairness
            self.fairness_results = model_module.analyze_fairness(
                model=st.session_state.trained_model,
                sensitive_features=params.get('sensitive_features', []),
                metrics=params.get('metrics', [])
            )

            # Store results in session state
            st.session_state.fairness_results = self.fairness_results

            self.status = "completed"
            return True
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Fairness analysis failed: {str(e)}")
            return False

    def render(self):
        """Render the fairness analysis UI"""
        st.header("Step 6: Fairness Analysis")

        if not st.session_state.step_completed.get('evaluation'):
            st.warning("Please complete model evaluation first.")
            return

        # Fairness analysis options
        st.subheader("Fairness Analysis Options")

        # Select sensitive features
        sensitive_features = st.multiselect(
            "Select sensitive features",
            options=st.session_state.selected_features,
            default=[]
        )

        # Select fairness metrics
        fairness_metrics = st.multiselect(
            "Select fairness metrics",
            options=["Demographic Parity", "Equal Opportunity", "Equalized Odds"],
            default=["Demographic Parity"]
        )

        # Store parameters in session state
        st.session_state.fairness_params = {
            'sensitive_features': sensitive_features,
            'metrics': fairness_metrics
        }

        # Show status and errors
        if self.status == "completed":
            st.success("Fairness analysis completed!")
            if self.fairness_results:
                st.write("Fairness Analysis Results:")
                st.write(self.fairness_results)
        elif self.status == "error":
            st.error(f"Error during fairness analysis: {self.error}")

        # Show analyze button
        if st.button("Analyze Fairness"):
            with st.spinner("Analyzing model fairness..."):
                success = self.run()
                if success:
                    st.session_state.step_completed['fairness_analysis'] = True
                    st.rerun()


def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="MIMIC-IV Preprocessing Pipeline",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Title and description
    st.title("MIMIC-IV Preprocessing Pipeline")
    st.markdown("""
    This application helps you preprocess the MIMIC-IV dataset for machine learning tasks.
    Follow the steps below to prepare your data.
    """)

    # Check if modules are available
    if not MODULES_AVAILABLE:
        st.error(import_error_for_ui)
        st.stop()

    # Initialize pipeline steps if not already done
    if not st.session_state.pipeline_steps:
        st.session_state.pipeline_steps = [
            DatasetLoadingStep(),
            DataPreprocessingStep(),
            FeatureSelectionStep(),
            ModelTrainingStep(),
            EvaluationStep(),
            FairnessAnalysisStep()
        ]

    # Create sidebar for navigation
    with st.sidebar:
        st.header("Pipeline Steps")

        # Display progress
        total_steps = len(st.session_state.pipeline_steps)
        completed_steps = sum(1 for step in st.session_state.step_completed.values() if step)
        st.progress(completed_steps / total_steps)

        # Step selection
        for i, step in enumerate(st.session_state.pipeline_steps):
            step_name = step.__class__.__name__.replace("Step", "")
            is_completed = st.session_state.step_completed.get(step_name.lower(), False)
            is_current = i == st.session_state.current_step

            # Create a button for each step
            button_label = f"{'✅' if is_completed else '⏳'} {step_name}"
            if st.button(button_label, key=f"nav_{i}", disabled=not (is_completed or i <= completed_steps)):
                st.session_state.current_step = i
                st.rerun()

    # Add navigation buttons at the bottom
    st.write("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.session_state.current_step > 0:
            if st.button("⬅️ Previous Step"):
                st.session_state.current_step -= 1
                st.rerun()

    with col2:
        # Show current step number and name
        current_step = st.session_state.pipeline_steps[st.session_state.current_step]
        step_name = current_step.__class__.__name__.replace("Step", "")
        st.write(f"Current Step: {st.session_state.current_step + 1}/{total_steps} - {step_name}")

    with col3:
        if st.session_state.current_step < len(st.session_state.pipeline_steps) - 1:
            # Enable next button only if current step is completed
            current_step_name = st.session_state.pipeline_steps[st.session_state.current_step].__class__.__name__.replace("Step", "").lower()
            can_proceed = st.session_state.step_completed.get(current_step_name, False)

            if st.button("Next Step ➡️", disabled=not can_proceed):
                st.session_state.current_step += 1
                st.rerun()

    # Render current step
    current_step = st.session_state.pipeline_steps[st.session_state.current_step]
    current_step.render()


if __name__ == "__main__":
    main()
