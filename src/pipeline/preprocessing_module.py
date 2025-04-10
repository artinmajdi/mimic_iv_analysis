import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from preprocessing.day_intervals_preproc.day_intervals_cohort import *
    from preprocessing.hosp_module_preproc.feature_selection_hosp import *
    from preprocessing.hosp_module_preproc.feature_selection_icu import *
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("Warning: Could not import preprocessing modules")

class PreprocessingModule:
    """
    A wrapper class for the MIMIC-IV preprocessing functionality.
    This class provides a simplified interface to the preprocessing modules.
    """

    def __init__(self, mimic_path, output_dir="./data"):
        """
        Initialize the preprocessing module.

        Args:
            mimic_path (str): Path to the MIMIC-IV dataset
            output_dir (str): Directory to save output files
        """
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
        self.time_interval = 24
        self.disease_label = ""

    def extract_cohort(self, cohort_name="default_cohort"):
        """
        Extract a cohort from the MIMIC-IV dataset.

        Args:
            cohort_name (str): Name of the cohort to extract

        Returns:
            str: Path to the extracted cohort file
        """
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Preprocessing modules not available")

        try:
            # Extract cohort using the day_intervals_cohort module
            extract_data(
                use_ICU=self.use_icu,
                label=cohort_name,
                time=self.time_interval,
                icd_code="",
                root_dir=self.mimic_path,
                disease_label=self.disease_label,
                cohort_output=cohort_name,
                summary_output=None
            )

            return f"{self.output_dir}/cohort/{cohort_name}.csv.gz"
        except Exception as e:
            raise Exception(f"Error extracting cohort: {str(e)}")

    def extract_features(self, cohort_name, feature_categories=None):
        """
        Extract features from the cohort.

        Args:
            cohort_name (str): Name of the cohort to extract features from
            feature_categories (dict): Dictionary of feature categories to extract
                Format: {
                    "diagnoses": True,
                    "procedures": True,
                    "medications": True,
                    "lab_tests": True
                }

        Returns:
            dict: Dictionary of paths to the extracted feature files
        """
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Preprocessing modules not available")

        if feature_categories is None:
            feature_categories = {
                "diagnoses": True,
                "procedures": True,
                "medications": True,
                "lab_tests": True
            }

        try:
            # Extract features using the feature_selection_hosp or feature_selection_icu module
            if self.use_icu:
                # ICU features
                feature_icu(
                    cohort_output=cohort_name,
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True),
                    group_diag=True,
                    group_med=True,
                    group_proc=True,
                    clean_labs=self.remove_outliers,
                    impute_labs=self.impute_missing
                )
            else:
                # Hospital features
                feature_nonicu(
                    cohort_output=cohort_name,
                    version_path=self.mimic_path,
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True)
                )

            return {
                "diagnoses": f"{self.output_dir}/features/preproc_diag.csv.gz" if feature_categories.get("diagnoses", True) else None,
                "procedures": f"{self.output_dir}/features/preproc_proc.csv.gz" if feature_categories.get("procedures", True) else None,
                "medications": f"{self.output_dir}/features/preproc_med.csv.gz" if feature_categories.get("medications", True) else None,
                "lab_tests": f"{self.output_dir}/features/preproc_lab.csv.gz" if feature_categories.get("lab_tests", True) else None
            }
        except Exception as e:
            raise Exception(f"Error extracting features: {str(e)}")

    def generate_summary(self, feature_categories=None):
        """
        Generate summary statistics for the extracted features.

        Args:
            feature_categories (dict): Dictionary of feature categories to summarize
                Format: {
                    "diagnoses": True,
                    "procedures": True,
                    "medications": True,
                    "lab_tests": True
                }

        Returns:
            dict: Dictionary of paths to the summary files
        """
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Preprocessing modules not available")

        if feature_categories is None:
            feature_categories = {
                "diagnoses": True,
                "procedures": True,
                "medications": True,
                "lab_tests": True
            }

        try:
            # Generate summary using the feature_selection_hosp or feature_selection_icu module
            if self.use_icu:
                # ICU summary
                generate_summary_icu(
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True)
                )
            else:
                # Hospital summary
                generate_summary_hosp(
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True)
                )

            return {
                "diagnoses": f"{self.output_dir}/summary/diag_summary.csv.gz" if feature_categories.get("diagnoses", True) else None,
                "procedures": f"{self.output_dir}/summary/proc_summary.csv.gz" if feature_categories.get("procedures", True) else None,
                "medications": f"{self.output_dir}/summary/med_summary.csv.gz" if feature_categories.get("medications", True) else None,
                "lab_tests": f"{self.output_dir}/summary/lab_summary.csv.gz" if feature_categories.get("lab_tests", True) else None
            }
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

    def run_preprocessing_pipeline(self, cohort_name, feature_categories=None):
        """
        Run the complete preprocessing pipeline.

        Args:
            cohort_name (str): Name of the cohort to extract
            feature_categories (dict): Dictionary of feature categories to extract
                Format: {
                    "diagnoses": True,
                    "procedures": True,
                    "medications": True,
                    "lab_tests": True
                }

        Returns:
            dict: Dictionary of paths to the output files
        """
        # Extract cohort
        cohort_path = self.extract_cohort(cohort_name)

        # Extract features
        feature_paths = self.extract_features(cohort_name, feature_categories)

        # Generate summary
        summary_paths = self.generate_summary(feature_categories)

        return {
            "cohort": cohort_path,
            "features": feature_paths,
            "summary": summary_paths
        }
