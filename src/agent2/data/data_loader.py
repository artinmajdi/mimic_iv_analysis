import dask.dataframe as dd
import pandas as pd
from pathlib import Path
import os

class DataLoader:
    def __init__(self, dataset_path=None):
        # Use environment variable if no path is provided
        if dataset_path is None:
            dataset_path = os.getenv('MIMIC_DATASET_PATH')
            if dataset_path is None:
                raise ValueError(
                    "Dataset path not provided and MIMIC_DATASET_PATH environment variable not set. "
                    "Please provide a dataset path or set the MIMIC_DATASET_PATH environment variable."
                )

        self.dataset_path = Path(dataset_path)
        self.patients = None
        self.admissions = None
        self.diagnoses = None
        self.poe = None
        self.poe_detail = None

        # Verify dataset directory exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found at: {self.dataset_path}")

    def load_patients(self):
        """Load patients table with memory-efficient processing"""
        if self.patients is None:
            self.patients = dd.read_csv(
                self.dataset_path / "hosp" / "patients.csv",
                dtype={
                    'subject_id': 'int64',
                    'gender': 'object',
                    'anchor_age': 'int64',
                    'anchor_year': 'int64',
                    'anchor_year_group': 'object',
                    'dod': 'object'
                }
            )
        return self.patients

    def load_admissions(self):
        """Load admissions table with memory-efficient processing"""
        if self.admissions is None:
            self.admissions = dd.read_csv(
                self.dataset_path / "hosp" / "admissions.csv",
                dtype={
                    'hadm_id': 'int64',
                    'subject_id': 'int64',
                    'admittime': 'object',
                    'dischtime': 'object',
                    'deathtime': 'object',
                    'admission_type': 'object',
                    'admission_location': 'object',
                    'discharge_location': 'object',
                    'insurance': 'object',
                    'language': 'object',
                    'marital_status': 'object',
                    'ethnicity': 'object',
                    'edregtime': 'object',
                    'edouttime': 'object',
                    'hospital_expire_flag': 'int64'
                }
            )
        return self.admissions

    def load_diagnoses(self):
        """Load diagnoses table with memory-efficient processing"""
        if self.diagnoses is None:
            self.diagnoses = dd.read_csv(
                self.dataset_path / "hosp" / "diagnoses_icd.csv",
                dtype={
                    'subject_id': 'int64',
                    'hadm_id': 'int64',
                    'seq_num': 'int64',
                    'icd_code': 'object',
                    'icd_version': 'int64'
                }
            )
        return self.diagnoses

    def load_poe(self):
        """Load provider order entry table with memory-efficient processing"""
        if self.poe is None:
            self.poe = dd.read_csv(
                self.dataset_path / "hosp" / "poe.csv",
                dtype={
                    'poe_id': 'object',
                    'poe_seq': 'int64',
                    'subject_id': 'int64',
                    'hadm_id': 'int64',
                    'ordertime': 'object',
                    'order_type': 'object',
                    'order_subtype': 'object',
                    'transaction_type': 'object',
                    'discontinue_of_poe_id': 'object'
                }
            )
            # Verify all required columns are present
            required_columns = ['poe_id', 'poe_seq', 'subject_id', 'hadm_id']
            missing_columns = [col for col in required_columns if col not in self.poe.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in poe table: {missing_columns}")
        return self.poe

    def load_poe_detail(self):
        """Load provider order entry detail table with memory-efficient processing"""
        if self.poe_detail is None:
            self.poe_detail = dd.read_csv(
                self.dataset_path / "hosp" / "poe_detail.csv",
                dtype={
                    'poe_id': 'object',
                    'field_name': 'object',
                    'field_value': 'object'
                }
            )
            # Verify all required columns are present
            required_columns = ['poe_id', 'field_name', 'field_value']
            missing_columns = [col for col in required_columns if col not in self.poe_detail.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in poe_detail table: {missing_columns}")
        return self.poe_detail

    def get_t2dm_patients(self):
        """Get Type 2 Diabetes patients cohort"""
        diagnoses = self.load_diagnoses()
        t2dm_patients = diagnoses[
            (diagnoses['icd_code'].str.startswith('E11')) &
            (diagnoses['icd_version'] == 10)
        ]['subject_id'].unique()
        return t2dm_patients.compute()  # Compute to get actual values

    def get_patient_orders(self, subject_ids):
        """
        Get provider orders for specific patients
        """
        try:
            # Load provider orders
            poe = self.load_poe()
            print(f"POE columns: {poe.columns.tolist()}")  # Debug print

            # Convert subject_ids to a list if it's a Dask Series
            if hasattr(subject_ids, 'compute'):
                subject_ids = subject_ids.compute()
            print(f"Subject IDs type: {type(subject_ids)}")  # Debug print

            # Filter orders for specified patients
            if 'subject_id' not in poe.columns:
                raise ValueError("'subject_id' column not found in poe table. Available columns: " +
                               ", ".join(poe.columns.tolist()))

            patient_orders = poe[poe['subject_id'].isin(subject_ids)]
            print(f"Filtered patient_orders shape: {patient_orders.shape}")  # Debug print

            # Load order details
            poe_detail = self.load_poe_detail()
            print(f"POE Detail columns: {poe_detail.columns.tolist()}")  # Debug print

            # Join with poe_detail for order specifics
            # Only use poe_id as the join key since that's the only common column
            if 'poe_id' not in patient_orders.columns:
                raise ValueError("'poe_id' column not found in patient_orders. Available columns: " +
                               ", ".join(patient_orders.columns.tolist()))
            if 'poe_id' not in poe_detail.columns:
                raise ValueError("'poe_id' column not found in poe_detail. Available columns: " +
                               ", ".join(poe_detail.columns.tolist()))

            patient_orders = patient_orders.merge(
                poe_detail,
                on='poe_id',
                how='left'
            )
            print(f"Merged patient_orders shape: {patient_orders.shape}")  # Debug print

            return patient_orders

        except Exception as e:
            print(f"Error in get_patient_orders: {str(e)}")
            print(f"POE columns: {poe.columns.tolist() if 'poe' in locals() else 'POE not loaded'}")
            print(f"POE Detail columns: {poe_detail.columns.tolist() if 'poe_detail' in locals() else 'POE Detail not loaded'}")
            print(f"Subject IDs type: {type(subject_ids) if 'subject_ids' in locals() else 'Subject IDs not available'}")
            raise
