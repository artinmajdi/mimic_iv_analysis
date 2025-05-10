#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating the usage of the DataLoader class.

This script shows how to:
1. Initialize the DataLoader
2. Scan the MIMIC-IV directory
3. Load a single MIMIC-IV table with various options
4. Apply filters to a loaded DataFrame
5. Get descriptive information for a table
6. Convert a loaded DataFrame to Parquet
7. Load a reference/dictionary table
8. Load a sampled patient cohort
9. Load a table and filter it by a set of values
10. Merge two DataFrames
11. Load and connect the six main MIMIC-IV tables

Usage:
    python data_loader_examples.py
"""

import os
import pandas as pd
import logging


# Import our modules
from mimic_iv_analysis.core.data_loader import DataLoader
from mimic_iv_analysis.core.filtering import Filtering # DataLoader uses this



# --- Configuration ---
# ! Replace this with the actual path to your MIMIC-IV dataset !
MIMIC_DATA_PATH = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"



class ExampleDataLoader:
    """Example script demonstrating the usage of the DataLoader class."""

    @staticmethod
    def configure_logging():
        """Set up logging configuration."""
        logging.basicConfig(
            level    = logging.INFO,
            format   = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers = [logging.StreamHandler()]
        )

    @staticmethod
    def example_scan_directory(loader: DataLoader):
        """Demonstrates scanning the MIMIC-IV directory."""
        logging.info("\\n--- Example: Scan MIMIC-IV Directory ---")
        if not check_mimic_path():
            return

        available_tables, file_paths, file_sizes, table_display_names = loader.scan_mimic_directory(MIMIC_DATA_PATH)

        if available_tables:
            logging.info(f"Found {len(available_tables)} modules: {list(available_tables.keys())}")
            for module, tables in available_tables.items():
                logging.info(f"  Module '{module}': {len(tables)} tables: {tables[:3]}...") # Print first 3

    # --- Helper Functions ---
    @staticmethod
    def setup_loader():
        """Initializes and returns a DataLoader instance."""
        return DataLoader()

    @staticmethod
    def check_mimic_path():
        """Checks if the MIMIC_DATA_PATH is set and valid."""
        if MIMIC_DATA_PATH == "/path/to/your/mimic-iv-3.1/dataset" or not os.path.exists(MIMIC_DATA_PATH):
            logging.warning(f"MIMIC_DATA_PATH is not set to a valid path: {MIMIC_DATA_PATH}")
            logging.warning("Many examples will not run correctly. Please update it in the script.")
            return False
        return True

    # --- Example Functions ---

    @staticmethod
    def example_scan_directory(loader: DataLoader):
        """Demonstrates scanning the MIMIC-IV directory."""
        logging.info("\\n--- Example: Scan MIMIC-IV Directory ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        available_tables, file_paths, file_sizes, table_display_names = loader.scan_mimic_directory(MIMIC_DATA_PATH)

        if available_tables:
            logging.info(f"Found {len(available_tables)} modules: {list(available_tables.keys())}")
            for module, tables in available_tables.items():
                logging.info(f"  Module \'{module}\' has {len(tables)} tables: {tables[:3]}...") # Print first 3
            logging.info(f"Example file path for ('hosp', 'patients'): {file_paths.get(('hosp', 'patients'))}")
            logging.info(f"Example file size for ('hosp', 'patients'): {file_sizes.get(('hosp', 'patients'))} MB")
            logging.info(f"Example display name for ('hosp', 'patients'): {table_display_names.get(('hosp', 'patients'))}")
        else:
            logging.info("No tables found. Check your MIMIC_DATA_PATH.")


    @staticmethod
    def example_load_mimic_table(loader: DataLoader):
        """Demonstrates loading a single MIMIC-IV table with various options."""
        logging.info("\\n--- Example: Load MIMIC Table (patients) ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        patients_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "patients.csv.gz")
        if not os.path.exists(patients_file_path):
            patients_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "patients.csv") # Try uncompressed
            if not os.path.exists(patients_file_path):
                logging.warning(f"patients.csv.gz or patients.csv not found in {os.path.join(MIMIC_DATA_PATH, 'hosp')}. Skipping.")
                return

        # 1. Load with sampling (Pandas)
        logging.info("Loading \'patients\' table with sampling (100 rows, Pandas)...")
        df_sampled_pd, total_rows_pd = loader.load_mimic_table(patients_file_path, sample_size=100, use_dask=False)
        if df_sampled_pd is not None:
            logging.info(f"Loaded {len(df_sampled_pd)} rows (sampled) out of {total_rows_pd} total (Pandas). Columns: {df_sampled_pd.columns.tolist()}")
        else:
            logging.warning("Failed to load sampled (Pandas).")

        # 2. Load with sampling (Dask)
        logging.info("Loading \'patients\' table with sampling (100 rows, Dask)...")
        df_sampled_dd, total_rows_dd = loader.load_mimic_table(patients_file_path, sample_size=100, use_dask=True)
        if df_sampled_dd is not None:
            logging.info(f"Loaded {len(df_sampled_dd)} rows (sampled) out of {total_rows_dd} total (Dask). Columns: {df_sampled_dd.columns.tolist()}")
        else:
            logging.warning("Failed to load sampled (Dask).")

        # 3. Load full table (Pandas) - careful with large tables
        # For demonstration, we'll try loading a smaller table if patients is too large
        # Or use a known smaller table like d_labitems
        d_labitems_path = os.path.join(MIMIC_DATA_PATH, "hosp", "d_labitems.csv.gz")
        if not os.path.exists(d_labitems_path):
            d_labitems_path = os.path.join(MIMIC_DATA_PATH, "hosp", "d_labitems.csv")

        if os.path.exists(d_labitems_path):
            logging.info("Loading \'d_labitems\' table fully (Pandas)...")
            df_full_pd, total_rows_full_pd = loader.load_mimic_table(d_labitems_path, sample_size=None, use_dask=False) # sample_size=None loads full
            if df_full_pd is not None:
                logging.info(f"Loaded {len(df_full_pd)} rows (full) out of {total_rows_full_pd} total (Pandas).")
            else:
                logging.warning("Failed to load d_labitems full (Pandas).")
        else:
            logging.info("d_labitems.csv.gz or .csv not found, skipping full load example for it.")


    @staticmethod
    def example_apply_filters(loader: DataLoader):
        """Demonstrates applying filters to a loaded DataFrame."""
        logging.info("\\n--- Example: Apply Filters ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # First, load a small sample of admissions data
        admissions_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "admissions.csv.gz")
        if not os.path.exists(admissions_file_path):
            admissions_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "admissions.csv")

        if not os.path.exists(admissions_file_path):
            logging.warning(f"admissions.csv.gz or admissions.csv not found in {os.path.join(MIMIC_DATA_PATH, 'hosp')}. Skipping apply_filters example.")
            return

        df_admissions, _ = loader.load_mimic_table(admissions_file_path, sample_size=5000, use_dask=False)
        if df_admissions is None or df_admissions.empty:
            logging.warning("Could not load admissions data for filter example.")
            return

        logging.info(f"Loaded {len(df_admissions)} admissions for filtering.")

        filter_params = {
            'apply_age_range': True, 'min_age': 30, 'max_age': 60,
            'apply_encounter_timeframe': False, # Needs patients table, complex for simple example
            'apply_t2dm_diagnosis': False,      # Needs diagnoses_icd table
            'apply_valid_admission_discharge': True,
            'apply_inpatient_stay': True, 'admission_types': ['EW EMER.', 'URGENT', 'DIRECT EMER.', 'EU OBSERVATION'], # Example types
            'exclude_in_hospital_death': True
        }

        # Note: The DataLoader.apply_filters method is designed to work within the Streamlit app\'s
        # session state context (st.session_state.mimic_path) to load related tables.
        # For this standalone example, we\'d typically pass pre-loaded patients_df, admissions_df etc.
        # or the Filtering class directly.
        # To simplify, we demonstrate what load_mimic_table does internally with filter_params.

        logging.info(f"Simulating loading \'admissions\' with filter_params: {filter_params}")
        # This call internally uses apply_filters
        df_filtered_admissions, _ = loader.load_mimic_table(
            admissions_file_path,
            sample_size   = 5000,  # Reload with filters
            use_dask      = False,
            filter_params = filter_params
        )

        if df_filtered_admissions is not None:
            logging.info(f"After applying filters, {len(df_filtered_admissions)} admissions remain.")
            # You could also call loader.filtering.apply_filters directly if you manage loading related tables
            # manual_filtered_df = loader.filtering.apply_filters(df_admissions.copy(), filter_params, patients_df=None, admissions_df=None, ...)
        else:
            logging.warning("Failed to load admissions with filters.")


    @staticmethod
    def example_get_table_info(loader: DataLoader):
        """Demonstrates getting descriptive information for a table."""
        logging.info("\\n--- Example: Get Table Info ---")
        info_admissions = loader.get_table_info("hosp", "admissions")
        logging.info(f"Info for (\'hosp\', \'admissions\'): {info_admissions}")
        info_chartevents = loader.get_table_info("icu", "chartevents")
        logging.info(f"Info for (\'icu\', \'chartevents\'): {info_chartevents}")
        info_unknown = loader.get_table_info("hosp", "non_existent_table")
        logging.info(f"Info for (\'hosp\', \'non_existent_table\'): {info_unknown}")


    @staticmethod
    def example_convert_to_parquet(loader: DataLoader):
        """Demonstrates converting a loaded DataFrame to Parquet."""
        logging.info("\\n--- Example: Convert to Parquet ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        d_labitems_path = os.path.join(MIMIC_DATA_PATH, "hosp", "d_labitems.csv.gz")
        if not os.path.exists(d_labitems_path):
            d_labitems_path = os.path.join(MIMIC_DATA_PATH, "hosp", "d_labitems.csv")

        if not os.path.exists(d_labitems_path):
            logging.warning(f"d_labitems.csv.gz or .csv not found. Skipping Parquet conversion example.")
            return

        df_labitems, _ = loader.load_mimic_table(d_labitems_path, sample_size=None) # Load full
        if df_labitems is not None and not df_labitems.empty:
            # Create a dummy examples/parquet_files directory for this example
            example_parquet_dir = "mimic_iv_analysis/examples/parquet_files"
            os.makedirs(example_parquet_dir, exist_ok=True)

            # The method expects current_file_path to deduce the parquet_files subdir
            # We'll simulate it by creating a dummy current file path within examples
            dummy_current_file_path = os.path.join("mimic_iv_analysis/examples", "dummy_file.py")

            parquet_file = loader.convert_to_parquet(df_labitems, "d_labitems_example", dummy_current_file_path)
            if parquet_file:
                logging.info(f"Converted \'d_labitems\' to Parquet: {parquet_file}")
                # You can try reading it back:
                # df_from_parquet = pd.read_parquet(parquet_file)
                # logging.info(f"Read back {len(df_from_parquet)} rows from Parquet.")
                if os.path.exists(parquet_file):
                    logging.info(f"Parquet file exists at: {os.path.abspath(parquet_file)}")
                else:
                    logging.warning(f"Parquet file was reportedly created but not found at: {parquet_file}")
            else:
                logging.warning("Failed to convert to Parquet.")
        else:
            logging.warning("Could not load d_labitems for Parquet example.")


    @staticmethod
    def example_load_reference_table(loader: DataLoader):
        """Demonstrates loading a reference/dictionary table."""
        logging.info("\\n--- Example: Load Reference Table (d_icd_diagnoses) ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        df_d_icd_diag = loader.load_reference_table(MIMIC_DATA_PATH, "hosp", "d_icd_diagnoses")
        if not df_d_icd_diag.empty:
            logging.info(f"Loaded \'d_icd_diagnoses\' reference table with {len(df_d_icd_diag)} rows. Columns: {df_d_icd_diag.columns.tolist()}")
        else:
            logging.warning("Failed to load \'d_icd_diagnoses\' reference table.")


    @staticmethod
    def example_load_patient_cohort(loader: DataLoader):
        """Demonstrates loading a sampled patient cohort."""
        logging.info("\\n--- Example: Load Patient Cohort ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        df_patients, subject_ids = loader.load_patient_cohort(MIMIC_DATA_PATH, sample_size=50)
        if not df_patients.empty:
            logging.info(f"Loaded patient cohort with {len(df_patients)} patients. Sampled subject_ids count: {len(subject_ids)}")
        else:
            logging.warning("Failed to load patient cohort.")


    @staticmethod
    def example_load_filtered_table(loader: DataLoader):
        """Demonstrates loading a table and filtering it by a set of values."""
        logging.info("\\n--- Example: Load Filtered Table (admissions for specific patients) ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # First, get some subject_ids from a patient cohort
        _, subject_ids_to_filter = loader.load_patient_cohort(MIMIC_DATA_PATH, sample_size=10)
        if not subject_ids_to_filter:
            logging.warning("Could not get subject_ids for filtering. Skipping.")
            return

        logging.info(f"Will filter \'admissions\' for {len(subject_ids_to_filter)} subject_ids: {list(subject_ids_to_filter)[:3]}...")

        # Load admissions table filtered by these subject_ids
        df_filtered_admissions = loader.load_filtered_table(
            mimic_path=MIMIC_DATA_PATH,
            module="hosp",
            table_name="admissions",
            filter_column="subject_id",
            filter_values=subject_ids_to_filter,
            use_dask=False # Or True
        )
        if not df_filtered_admissions.empty:
            logging.info(f"Loaded {len(df_filtered_admissions)} admissions for the sampled subject_ids. Unique subjects: {df_filtered_admissions['subject_id'].nunique()}")
        else:
            logging.warning("Failed to load filtered admissions or no admissions found for the sample.")


    @staticmethod
    def example_merge_tables(loader: DataLoader):
        """Demonstrates merging two DataFrames."""
        logging.info("\\n--- Example: Merge Tables ---")

        # Create dummy DataFrames for merging
        data1 = {'id': [1, 2, 3], 'value1': ['A', 'B', 'C']}
        df1 = pd.DataFrame(data1)

        data2 = {'id': [2, 3, 4], 'value2': ['X', 'Y', 'Z']}
        df2 = pd.DataFrame(data2)

        logging.info("Merging df1 and df2 on \'id\' (left merge):")
        merged_df = loader.merge_tables(df1, df2, on=['id'], how='left')
        logging.info(f"Merged DataFrame:\\n{merged_df}")

        # Example with actual loaded tables (if available and small)
        if ExampleDataLoader.check_mimic_path():
            df_patients_sample, _ = loader.load_patient_cohort(MIMIC_DATA_PATH, sample_size=5)
            subject_ids = set(df_patients_sample['subject_id'])

            df_admissions_sample = loader.load_filtered_table(MIMIC_DATA_PATH, "hosp", "admissions", "subject_id", subject_ids)

            if not df_patients_sample.empty and not df_admissions_sample.empty:
                logging.info("Merging sample of patients and their admissions:")
                merged_pa = loader.merge_tables(df_patients_sample, df_admissions_sample, on=['subject_id'], how='inner')
                logging.info(f"Merged {len(merged_pa)} rows. Columns: {merged_pa.columns.tolist()}")
            else:
                logging.info("Skipping merge of actual tables as samples were empty or failed to load.")


    @staticmethod
    def example_load_connected_tables(loader: DataLoader):
        """Demonstrates loading and connecting the six main MIMIC-IV tables."""
        logging.info("\\n--- Example: Load Connected Tables ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # 1. Load as a dictionary of tables
        logging.info("Loading connected tables (merged_view=False, sample_size=20)...")
        tables_dict, merged_df_view_false = loader.load_connected_tables(
            mimic_path=MIMIC_DATA_PATH,
            sample_size=20, # Small sample for speed
            use_dask=False,
            merged_view=False
        )
        if tables_dict:
            logging.info(f"Loaded {len(tables_dict)} tables into dictionary: {list(tables_dict.keys())}")
            for name, df in tables_dict.items():
                if df is not None and not df.empty:
                    logging.info(f"  Table '{name}': {len(df)} rows, {df.columns.tolist()[:5]}... cols")
                else:
                    logging.info(f"  Table '{name}': Empty or not loaded.")
            logging.info(f"Merged DataFrame when merged_view=False should be empty: {merged_df_view_false.empty}")
        else:
            logging.warning("Failed to load connected tables into dictionary.")

        # 2. Load as a single merged DataFrame
        logging.info("\\nLoading connected tables (merged_view=True, sample_size=20)...")
        _, merged_df_view_true = loader.load_connected_tables(
            mimic_path=MIMIC_DATA_PATH,
            sample_size=20, # Small sample for speed
            use_dask=False,
            merged_view=True
        )
        if merged_df_view_true is not None and not merged_df_view_true.empty:
            logging.info(f"Loaded a single merged DataFrame with {len(merged_df_view_true)} rows and {len(merged_df_view_true.columns)} columns.")
            logging.info(f"Merged columns (first 5): {merged_df_view_true.columns.tolist()[:5]}")
        else:
            logging.warning("Failed to load connected tables as a single merged DataFrame or it was empty.")


def main():
    ExampleDataLoader.configure_logging()
    data_loader = ExampleDataLoader.setup_loader()

    logging.info(f"--- Starting DataLoader Examples ---")
    logging.info(f"Using MIMIC_DATA_PATH: {MIMIC_DATA_PATH}")
    if not ExampleDataLoader.check_mimic_path():
        logging.error("Please set the MIMIC_DATA_PATH at the top of this script to your local MIMIC-IV dataset path.")
    else:
        ExampleDataLoader.example_scan_directory(data_loader)
        ExampleDataLoader.example_get_table_info(data_loader)
        ExampleDataLoader.example_load_mimic_table(data_loader)
        ExampleDataLoader.example_load_reference_table(data_loader)
        ExampleDataLoader.example_load_patient_cohort(data_loader)
        ExampleDataLoader.example_load_filtered_table(data_loader)
        ExampleDataLoader.example_merge_tables(data_loader)
        ExampleDataLoader.example_convert_to_parquet(data_loader) # Run this after some tables are loaded and files may exist
        ExampleDataLoader.example_load_connected_tables(data_loader) # This is a more comprehensive one
        ExampleDataLoader.example_apply_filters(data_loader) # Shows how filters are used with load_mimic_table

    logging.info("\\n--- DataLoader Examples Finished ---")


if __name__ == "__main__":
    main()
