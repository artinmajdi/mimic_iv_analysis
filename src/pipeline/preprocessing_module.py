import os
import sys
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from tqdm import tqdm

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from preprocessing.day_intervals_preproc.day_intervals_cohort import extract_data, get_visit_pts, partition_by_los, partition_by_readmit, partition_by_mort
    from preprocessing.day_intervals_preproc.disease_cohort import extract_diag_cohort
    from preprocessing.hosp_module_preproc.feature_selection_hosp import feature_nonicu, preprocess_features_hosp, generate_summary_hosp
    from preprocessing.hosp_module_preproc.feature_selection_icu import feature_icu, preprocess_features_icu, generate_summary_icu
    from utils.outlier_removal import outlier_imputation
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    print(f"Warning: Could not import preprocessing modules: {str(e)}")

class PreprocessingModule:
    """
    A comprehensive wrapper class for the MIMIC-IV preprocessing pipeline.
    This class provides a unified interface to all preprocessing modules.
    Compatible with MIMIC-IV v3.1 structure.
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

        # Check for trailing slash and add if missing
        if not self.mimic_path.endswith('/'):
            self.mimic_path += '/'

        # Create output directories if they don't exist
        os.makedirs(f"{output_dir}/cohort", exist_ok=True)
        os.makedirs(f"{output_dir}/features", exist_ok=True)
        os.makedirs(f"{output_dir}/summary", exist_ok=True)

        # Default preprocessing parameters
        self.use_icu = False
        self.remove_outliers = True
        self.impute_missing = True
        self.outlier_threshold = 95 # Percentile threshold for outlier removal
        self.left_threshold = 5 # Lower percentile for outlier removal
        self.imputation_method = "Mean"
        self.disease_label = ""

        # Feature parameters
        self.feature_paths = {}
        self.cohort_path = None
        self.summary_paths = {}

        # Statistics tracking
        self.statistics = {
            "cohort": {},
            "features": {},
            "warnings": []
        }

    def extract_cohort(self, cohort_name="default_cohort", use_mortality=True, use_los=False,
                      los_threshold=7, use_readmission=False, readmission_window=30, disease_label=""):
        """
        Extract a cohort from the MIMIC-IV dataset.

        Args:
            cohort_name (str): Name of the cohort to extract
            use_mortality (bool): Whether to include mortality labels
            use_los (bool): Whether to include length of stay labels
            los_threshold (int): Threshold for length of stay (days)
            use_readmission (bool): Whether to include readmission labels
            readmission_window (int): Window for readmission (days)
            disease_label (str): Optional ICD code to filter the cohort

        Returns:
            str: Path to the extracted cohort file
        """
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Preprocessing modules not available")

        try:
            # Save parameters
            self.disease_label = disease_label

            # Convert readmission window to hours for compatibility with the code
            time_interval_hours = readmission_window * 24 if use_readmission else 24

            # Extract cohort using the day_intervals_cohort module
            extract_data(
                use_ICU=self.use_icu,
                label=cohort_name,
                time=time_interval_hours,
                icd_code="",
                root_dir=self.mimic_path,
                disease_label=disease_label,
                cohort_output=cohort_name,
                summary_output=None
            )

            # Store cohort path
            cohort_path = f"{self.output_dir}/cohort/{cohort_name}.csv.gz"
            self.cohort_path = cohort_path

            # Generate statistics about the cohort
            try:
                cohort_df = pd.read_csv(cohort_path)
                self.statistics["cohort"] = {
                    "total_patients": cohort_df["subject_id"].nunique(),
                    "total_admissions": cohort_df.shape[0],
                    "icu_admissions": cohort_df[cohort_df["stay_id"].notna()].shape[0] if "stay_id" in cohort_df.columns else 0,
                    "positive_label_count": cohort_df["label"].sum() if "label" in cohort_df.columns else 0,
                    "negative_label_count": (cohort_df.shape[0] - cohort_df["label"].sum()) if "label" in cohort_df.columns else 0
                }
            except Exception as e:
                self.statistics["warnings"].append(f"Could not generate cohort statistics: {str(e)}")

            return cohort_path
        except Exception as e:
            error_msg = f"Error extracting cohort: {str(e)}"
            self.statistics["warnings"].append(error_msg)
            raise Exception(error_msg)

    def extract_features(self, cohort_name, feature_categories=None, diagnosis_grouping=None,
                        procedure_grouping=None, med_grouping=None):
        """
        Extract features from the cohort.

        Args:
            cohort_name (str): Name of the cohort to extract features from
            feature_categories (dict): Dictionary of feature categories to extract
            diagnosis_grouping (str): How to group ICD diagnosis codes
            procedure_grouping (str): How to select procedure code versions
            med_grouping (bool): Whether to group medications by generic names

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
            # Map UI selections to preprocessing parameters
            if diagnosis_grouping is None:
                diagnosis_grouping = "Keep both ICD-9 and ICD-10 codes"

            if procedure_grouping is None:
                procedure_grouping = "ICD-9 and ICD-10"

            group_diag_map = {
                "Keep both ICD-9 and ICD-10 codes": "Keep both ICD-9 and ICD-10 codes",
                "Convert ICD-9 to ICD-10 codes": "Convert ICD-9 to ICD-10 codes",
                "Convert ICD-9 to ICD-10 and group ICD-10 codes": "Convert ICD-9 to ICD-10 and group ICD-10 codes"
            }

            group_proc_map = {
                "ICD-9 and ICD-10": "ICD-9 and ICD-10",
                "ICD-10": "ICD-10"
            }

            # Extract features
            if self.use_icu:
                # ICU features extraction
                feature_icu(
                    cohort_output=cohort_name,
                    version_path=self.mimic_path,
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True)
                )

                # Process features
                preprocess_features_icu(
                    cohort_output=cohort_name,
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True),
                    group_diag=group_diag_map.get(diagnosis_grouping, group_diag_map["Keep both ICD-9 and ICD-10 codes"]),
                    group_med=med_grouping if med_grouping is not None else True,
                    group_proc=group_proc_map.get(procedure_grouping, group_proc_map["ICD-9 and ICD-10"]),
                    clean_labs=self.remove_outliers,
                    impute_labs=self.impute_missing,
                    thresh=self.outlier_threshold,
                    left_thresh=self.left_threshold
                )
            else:
                # Hospital features extraction
                feature_nonicu(
                    cohort_output=cohort_name,
                    version_path=self.mimic_path,
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True)
                )

                # Process features
                preprocess_features_hosp(
                    cohort_output=cohort_name,
                    diag_flag=feature_categories.get("diagnoses", True),
                    proc_flag=feature_categories.get("procedures", True),
                    med_flag=feature_categories.get("medications", True),
                    lab_flag=feature_categories.get("lab_tests", True),
                    group_diag=group_diag_map.get(diagnosis_grouping, group_diag_map["Keep both ICD-9 and ICD-10 codes"]),
                    group_med=med_grouping if med_grouping is not None else True,
                    group_proc=group_proc_map.get(procedure_grouping, group_proc_map["ICD-9 and ICD-10"]),
                    clean_labs=self.remove_outliers,
                    impute_labs=self.impute_missing,
                    thresh=self.outlier_threshold,
                    left_thresh=self.left_threshold
                )

            # Store paths to the extracted features
            feature_paths = {
                "diagnoses": f"{self.output_dir}/features/preproc_diag.csv.gz" if feature_categories.get("diagnoses", True) else None,
                "procedures": f"{self.output_dir}/features/preproc_proc.csv.gz" if feature_categories.get("procedures", True) else None,
                "medications": f"{self.output_dir}/features/preproc_med.csv.gz" if feature_categories.get("medications", True) else None,
                "lab_tests": f"{self.output_dir}/features/preproc_labs.csv.gz" if feature_categories.get("lab_tests", True) else None
            }

            self.feature_paths = feature_paths

            # Collect statistics about features
            self.collect_feature_statistics(feature_categories)

            return feature_paths
        except Exception as e:
            error_msg = f"Error extracting features: {str(e)}"
            self.statistics["warnings"].append(error_msg)
            raise Exception(error_msg)

    def collect_feature_statistics(self, feature_categories):
        """
        Collect statistics about the extracted features.

        Args:
            feature_categories (dict): Dictionary of feature categories
        """
        try:
            feature_stats = {}

            # Diagnoses
            if feature_categories.get("diagnoses", True) and os.path.exists(f"{self.output_dir}/features/preproc_diag.csv.gz"):
                try:
                    diag_df = pd.read_csv(f"{self.output_dir}/features/preproc_diag.csv.gz")
                    feature_stats["diagnoses"] = {
                        "unique_codes": diag_df["new_icd_code"].nunique() if "new_icd_code" in diag_df.columns else diag_df["icd_code"].nunique(),
                        "total_records": diag_df.shape[0],
                        "patients": diag_df["subject_id"].nunique(),
                        "admissions": diag_df["hadm_id"].nunique()
                    }
                except Exception as e:
                    self.statistics["warnings"].append(f"Could not read diagnoses feature file: {str(e)}")

            # Procedures
            if feature_categories.get("procedures", True) and os.path.exists(f"{self.output_dir}/features/preproc_proc.csv.gz"):
                try:
                    proc_df = pd.read_csv(f"{self.output_dir}/features/preproc_proc.csv.gz")
                    feature_stats["procedures"] = {
                        "unique_codes": proc_df["icd_code"].nunique(),
                        "total_records": proc_df.shape[0],
                        "patients": proc_df["subject_id"].nunique(),
                        "admissions": proc_df["hadm_id"].nunique()
                    }
                except Exception as e:
                    self.statistics["warnings"].append(f"Could not read procedures feature file: {str(e)}")

            # Medications
            if feature_categories.get("medications", True) and os.path.exists(f"{self.output_dir}/features/preproc_med.csv.gz"):
                try:
                    med_df = pd.read_csv(f"{self.output_dir}/features/preproc_med.csv.gz")
                    feature_stats["medications"] = {
                        "unique_drugs": med_df["drug_name"].nunique() if "drug_name" in med_df.columns else med_df["drug"].nunique(),
                        "total_records": med_df.shape[0],
                        "patients": med_df["subject_id"].nunique(),
                        "admissions": med_df["hadm_id"].nunique()
                    }
                except Exception as e:
                    self.statistics["warnings"].append(f"Could not read medications feature file: {str(e)}")

            # Lab tests
            if feature_categories.get("lab_tests", True) and os.path.exists(f"{self.output_dir}/features/preproc_labs.csv.gz"):
                try:
                    lab_df = pd.read_csv(f"{self.output_dir}/features/preproc_labs.csv.gz")
                    feature_stats["lab_tests"] = {
                        "unique_tests": lab_df["itemid"].nunique(),
                        "total_records": lab_df.shape[0],
                        "patients": lab_df["subject_id"].nunique(),
                        "admissions": lab_df["hadm_id"].nunique()
                    }
                except Exception as e:
                    self.statistics["warnings"].append(f"Could not read lab tests feature file: {str(e)}")

            self.statistics["features"] = feature_stats
        except Exception as e:
            self.statistics["warnings"].append(f"Error collecting feature statistics: {str(e)}")

    def generate_summary(self, feature_categories=None):
        """
        Generate summary statistics for the extracted features.

        Args:
            feature_categories (dict): Dictionary of feature categories to summarize

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
            # Generate summary using the appropriate module
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

            # Store paths to summary files
            summary_paths = {
                "diagnoses": f"{self.output_dir}/summary/diag_summary.csv" if feature_categories.get("diagnoses", True) else None,
                "procedures": f"{self.output_dir}/summary/proc_summary.csv" if feature_categories.get("procedures", True) else None,
                "medications": f"{self.output_dir}/summary/med_summary.csv" if feature_categories.get("medications", True) else None,
                "lab_tests": f"{self.output_dir}/summary/labs_summary.csv" if feature_categories.get("lab_tests", True) else None
            }

            self.summary_paths = summary_paths
            return summary_paths
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            self.statistics["warnings"].append(error_msg)
            raise Exception(error_msg)

    def run_preprocessing_pipeline(self, cohort_name, feature_categories=None,
                                  use_mortality=True, use_los=False, los_threshold=7,
                                  use_readmission=False, readmission_window=30,
                                  disease_label="", diagnosis_grouping=None,
                                  procedure_grouping=None, med_grouping=None):
        """
        Run the complete preprocessing pipeline.

        Args:
            cohort_name (str): Name of the cohort to extract
            feature_categories (dict): Dictionary of feature categories to extract
            use_mortality (bool): Whether to include mortality labels
            use_los (bool): Whether to include length of stay labels
            los_threshold (int): Threshold for length of stay (days)
            use_readmission (bool): Whether to include readmission labels
            readmission_window (int): Window for readmission (days)
            disease_label (str): Optional ICD code to filter the cohort
            diagnosis_grouping (str): How to group ICD diagnosis codes
            procedure_grouping (str): How to select procedure code versions
            med_grouping (bool): Whether to group medications by generic names

        Returns:
            dict: Dictionary of paths to the output files and statistics
        """
        # Reset statistics
        self.statistics = {
            "cohort": {},
            "features": {},
            "warnings": []
        }

        # Extract cohort
        cohort_path = self.extract_cohort(
            cohort_name=cohort_name,
            use_mortality=use_mortality,
            use_los=use_los,
            los_threshold=los_threshold,
            use_readmission=use_readmission,
            readmission_window=readmission_window,
            disease_label=disease_label
        )

        # Extract features
        feature_paths = self.extract_features(
            cohort_name=cohort_name,
            feature_categories=feature_categories,
            diagnosis_grouping=diagnosis_grouping,
            procedure_grouping=procedure_grouping,
            med_grouping=med_grouping
        )

        # Generate summary
        summary_paths = self.generate_summary(feature_categories)

        # Return all paths and statistics
        return {
            "cohort": cohort_path,
            "diagnoses": feature_paths.get("diagnoses"),
            "procedures": feature_paths.get("procedures"),
            "medications": feature_paths.get("medications"),
            "lab_tests": feature_paths.get("lab_tests"),
            "summary": summary_paths,
            "statistics": self.statistics
        }
