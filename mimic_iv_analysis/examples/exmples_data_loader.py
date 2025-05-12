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
12. Load data for specific subject IDs

Usage:
    python data_loader_examples.py
"""

import os
import pandas as pd
import logging
import random # For selecting random subjects


# Import our modules
from mimic_iv_analysis.core.data_loader import DataLoader
from mimic_iv_analysis.core.filtering import Filtering # DataLoader uses this



# --- Configuration ---
# ! Replace this with the actual path to your MIMIC-IV dataset !
MIMIC_DATA_PATH = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
MAX_CHUNKS = 5


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
        logging.info("\n--- Example: Scan MIMIC-IV Directory ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        dataset_info_df, dataset_info = loader.scan_mimic_directory(MIMIC_DATA_PATH)

        if dataset_info['available_tables']:
            logging.info(f"Found {len(dataset_info['available_tables'])} modules: {list(dataset_info['available_tables'].keys())}")

            for module, tables in dataset_info['available_tables'].items():
                logging.info(f"  Module '{module}' has {len(tables)} tables: {tables[:3]}...") # Print first 3

            logging.info(f"Example file path for ('hosp', 'patients'): {dataset_info['file_paths'].get(('hosp', 'patients'))}")
            logging.info(f"Example file size for ('hosp', 'patients'): {dataset_info['file_sizes'].get(('hosp', 'patients'))} MB")
            logging.info(f"Example display name for ('hosp', 'patients'): {dataset_info['table_display_names'].get(('hosp', 'patients'))}")

        else:
            logging.info("No tables found. Check your MIMIC_DATA_PATH.")


    @staticmethod
    def example_load_mimic_table(loader: DataLoader):
        """Demonstrates loading a single MIMIC-IV table with various options."""

        logging.info("\n--- Example: Load MIMIC Table (patients) ---")

        # Check if MIMIC_DATA_PATH is set and valid
        if not ExampleDataLoader.check_mimic_path():
            return

        patients_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "patients.csv.gz")

        # Check if file exists
        if not os.path.exists(patients_file_path):
            patients_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "patients.csv") # Try uncompressed

            # Check if uncompressed file exists
            if not os.path.exists(patients_file_path):
                logging.warning(f"patients.csv.gz or patients.csv not found in {os.path.join(MIMIC_DATA_PATH, 'hosp')}. Skipping.")
                return

        # 1. Load with sampling (Pandas)
        logging.info("Loading 'patients' table with sampling (100 rows, Pandas)...")
        df_sampled_pd, total_rows_pd = loader.load_mimic_table(file_path    = patients_file_path,
                                                                sample_size = 100,
                                                                use_dask    = False,
                                                                max_chunks  = MAX_CHUNKS)

        # Check if loading was successful
        if df_sampled_pd is not None:

            logging.info(f"Loaded {len(df_sampled_pd)} rows (sampled) out of {total_rows_pd} total (Pandas). Columns: {df_sampled_pd.columns.tolist()}")

            logging.info(f"Sample data (Pandas):\n{df_sampled_pd.head(3)}")


        else:
            logging.warning("Failed to load sampled (Pandas).")

        # 2. Load with sampling (Dask)
        logging.info("Loading 'patients' table with sampling (100 rows, Dask)...")
        df_sampled_dd, total_rows_dd = loader.load_mimic_table(file_path    = patients_file_path,
                                                                sample_size = 100,
                                                                use_dask    = True,
                                                                max_chunks  = MAX_CHUNKS)

        # Check if loading was successful
        if df_sampled_dd is not None:

            # For Dask, len() triggers computation. df_sampled_dd is already computed by head() in load_mimic_table.
            logging.info(f"Loaded {len(df_sampled_dd)} rows (sampled) out of {total_rows_dd} total (Dask). Columns: {df_sampled_dd.columns.tolist()}")

            logging.info(f"Sample data (Dask):\n{df_sampled_dd.head(3)}") # .head() is fine as it's already computed

        else:
            logging.warning("Failed to load sampled (Dask).")

        # 3. Load full table (Pandas) - careful with large tables
        # For demonstration, we'll try loading a smaller table if patients is too large
        # Or use a known smaller table like d_labitems
        d_labitems_path = os.path.join(MIMIC_DATA_PATH, "hosp", "d_labitems.csv.gz")

        # Check if file exists
        if not os.path.exists(d_labitems_path):
            d_labitems_path = os.path.join(MIMIC_DATA_PATH, "hosp", "d_labitems.csv")

        # Check if file exists
        if os.path.exists(d_labitems_path):
            logging.info("Loading 'd_labitems' table fully (Pandas)...")

            # Use a large number for sample_size to effectively load the full table
            df_full_pd, total_rows_full_pd = loader.load_mimic_table(file_path  = d_labitems_path,
                                                                    sample_size = 10**9,
                                                                    use_dask    = False,
                                                                    max_chunks  = MAX_CHUNKS)

            # Check if loading was successful
            if df_full_pd is not None:
                logging.info(f"Loaded {len(df_full_pd)} rows (full) out of {total_rows_full_pd} total (Pandas).")
            else:
                logging.warning("Failed to load d_labitems full (Pandas).")

        else:
            logging.info("d_labitems.csv.gz or .csv not found, skipping full load example for it.")


    @staticmethod
    def example_apply_filters(loader: DataLoader):
        """Demonstrates applying filters to a loaded DataFrame."""
        logging.info("\n--- Example: Apply Filters ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # First, load a sample of admissions data
        admissions_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "admissions.csv.gz")
        if not os.path.exists(admissions_file_path):
            admissions_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "admissions.csv")

        if not os.path.exists(admissions_file_path):
            logging.warning(f"admissions.csv.gz or admissions.csv not found in {os.path.join(MIMIC_DATA_PATH, 'hosp')}. Skipping apply_filters example.")
            return

        logging.info("Loading initial 'admissions' table for filtering (sample_size=5000)...")
        df_admissions, initial_rows = loader.load_mimic_table(admissions_file_path, sample_size=5000, use_dask=False, max_chunks=MAX_CHUNKS)
        if df_admissions is None or df_admissions.empty:
            logging.warning("Could not load admissions data for filter example.")
            return

        logging.info(f"Loaded {len(df_admissions)} admissions (out of {initial_rows}) before filtering.")

        filter_params = {
            'apply_age_range': True, 'min_age': 30, 'max_age': 60,
            'apply_encounter_timeframe': False, # Needs patients table to be loaded and merged by apply_filters
            'apply_t2dm_diagnosis': False,      # Needs diagnoses_icd table to be loaded and merged by apply_filters
            'apply_valid_admission_discharge': True,
            'apply_inpatient_stay': True, 'admission_types': ['EW EMER.', 'URGENT', 'DIRECT EMER.', 'EU OBSERVATION'],
            'exclude_in_hospital_death': True
        }

        logging.info(f"Applying filters directly using loader.apply_filters with params: {filter_params}")

        # The apply_filters method will load necessary auxiliary tables (like patients, admissions if not already present correctly)
        # based on the MIMIC_DATA_PATH and filter_params.
        df_filtered_admissions = loader.apply_filters(
            df            = df_admissions.copy(), # Pass a copy if you want to keep the original df_admissions unchanged
            filter_params = filter_params,
            mimic_path    = MIMIC_DATA_PATH
        )

        if df_filtered_admissions is not None:
            logging.info(f"After applying filters, {len(df_filtered_admissions)} admissions remain.")
            if len(df_filtered_admissions) < len(df_admissions):
                logging.info("Filtering was successful in reducing the number of rows.")
            else:
                logging.info("Filtering did not reduce the number of rows. Check filter criteria and data.")
        else:
            logging.warning("Filtering resulted in an empty or None DataFrame.")


    @staticmethod
    def example_get_table_info(loader: DataLoader):
        """Demonstrates getting descriptive information for a table."""
        logging.info("\n--- Example: Get Table Info ---")
        info_admissions = loader.get_table_description("hosp", "admissions")
        logging.info(f"Info for ('hosp', 'admissions'): {info_admissions}")
        info_chartevents = loader.get_table_description("icu", "chartevents")
        logging.info(f"Info for ('icu', 'chartevents'): {info_chartevents}")
        info_unknown = loader.get_table_description("hosp", "non_existent_table")
        logging.info(f"Info for ('hosp', 'non_existent_table'): {info_unknown}")


    @staticmethod
    def example_load_reference_table(loader: DataLoader):
        """Demonstrates loading a reference/dictionary table."""
        logging.info("\n--- Example: Load Reference Table (d_icd_diagnoses) ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        df_d_icd_diag = loader.load_reference_table(MIMIC_DATA_PATH, "hosp", "d_icd_diagnoses")
        if not df_d_icd_diag.empty:
            logging.info(f"Loaded 'd_icd_diagnoses' reference table with {len(df_d_icd_diag)} rows. Columns: {df_d_icd_diag.columns.tolist()}")
        else:
            logging.warning("Failed to load 'd_icd_diagnoses' reference table.")


    @staticmethod
    def example_load_patient_cohort(loader: DataLoader):
        """Demonstrates loading a sampled patient cohort."""
        logging.info("\n--- Example: Load Patient Cohort ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        df_patients, subject_ids = loader.load_patient_cohort(MIMIC_DATA_PATH, sample_size=50, max_chunks=MAX_CHUNKS)
        if not df_patients.empty:
            logging.info(f"Loaded patient cohort with {len(df_patients)} patients. Sampled subject_ids count: {len(subject_ids)}")
        else:
            logging.warning("Failed to load patient cohort.")


    @staticmethod
    def example_load_filtered_table(loader: DataLoader):
        """Demonstrates loading a table and filtering it by a set of values."""
        logging.info("\n--- Example: Load Filtered Table (admissions for specific patients) ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # First, get some subject_ids from a patient cohort
        _, subject_ids_to_filter = loader.load_patient_cohort(MIMIC_DATA_PATH, sample_size=10, max_chunks=MAX_CHUNKS)
        if not subject_ids_to_filter:
            logging.warning("Could not get subject_ids for filtering. Skipping.")
            return

        logging.info(f"Will filter 'admissions' for {len(subject_ids_to_filter)} subject_ids: {list(subject_ids_to_filter)[:3]}...")

        # Load admissions table filtered by these subject_ids
        df_filtered_admissions = loader.load_filtered_table(
            mimic_path=MIMIC_DATA_PATH,
            module="hosp",
            table_name="admissions",
            filter_column="subject_id",
            filter_values=subject_ids_to_filter,
            use_dask=False, # Or True
            max_chunks=MAX_CHUNKS
        )
        if not df_filtered_admissions.empty:
            logging.info(f"Loaded {len(df_filtered_admissions)} admissions for the sampled subject_ids. Unique subjects: {df_filtered_admissions['subject_id'].nunique()}")
        else:
            logging.warning("Failed to load filtered admissions or no admissions found for the sample.")


    @staticmethod
    def example_merge_tables(loader: DataLoader):
        """Demonstrates merging two DataFrames."""
        logging.info("\n--- Example: Merge Tables ---")

        # Create dummy DataFrames for merging
        data1 = {'id': [1, 2, 3], 'value1': ['A', 'B', 'C']}
        df1 = pd.DataFrame(data1)

        data2 = {'id': [2, 3, 4], 'value2': ['X', 'Y', 'Z']}
        df2 = pd.DataFrame(data2)

        logging.info("Merging df1 and df2 on 'id' (left merge):")
        merged_df = loader.merge_tables(df1, df2, on=['id'], how='left')
        logging.info(f"Merged DataFrame:\n{merged_df}")

        # Example with actual loaded tables (if available and small)
        if ExampleDataLoader.check_mimic_path():
            df_patients_sample, _ = loader.load_patient_cohort(MIMIC_DATA_PATH, sample_size=5, max_chunks=MAX_CHUNKS)
            subject_ids = set(df_patients_sample['subject_id'])

            df_admissions_sample = loader.load_filtered_table(MIMIC_DATA_PATH, "hosp", "admissions", "subject_id", subject_ids, max_chunks=MAX_CHUNKS)

            if not df_patients_sample.empty and not df_admissions_sample.empty:
                logging.info("Merging sample of patients and their admissions:")
                merged_pa = loader.merge_tables(df_patients_sample, df_admissions_sample, on=['subject_id'], how='inner')
                logging.info(f"Merged {len(merged_pa)} rows. Columns: {merged_pa.columns.tolist()}")
            else:
                logging.info("Skipping merge of actual tables as samples were empty or failed to load.")


    @staticmethod
    def example_load_connected_tables(loader: DataLoader):
        """Demonstrates loading and connecting the six main MIMIC-IV tables."""
        logging.info("\n--- Example: Load Connected Tables ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # 1. Load as a dictionary of tables
        logging.info("Loading connected tables (merged_view=False, sample_size=20)...")
        tables_dict, merged_df_view_false = loader.load_connected_tables(
            mimic_path=MIMIC_DATA_PATH,
            sample_size=20, # Small sample for speed
            use_dask=False,
            merged_view=False,
            max_chunks=MAX_CHUNKS
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
        logging.info("\nLoading connected tables (merged_view=True, sample_size=20)...")
        _, merged_df_view_true = loader.load_connected_tables(
            mimic_path=MIMIC_DATA_PATH,
            sample_size=20, # Small sample for speed
            use_dask=False,
            merged_view=True,
            max_chunks=MAX_CHUNKS
        )
        if merged_df_view_true is not None and not merged_df_view_true.empty:
            logging.info(f"Loaded a single merged DataFrame with {len(merged_df_view_true)} rows and {len(merged_df_view_true.columns)} columns.")
            logging.info(f"Merged columns (first 5): {merged_df_view_true.columns.tolist()[:5]}")
        else:
            logging.warning("Failed to load connected tables as a single merged DataFrame or it was empty.")


    @staticmethod
    def example_load_with_subject_ids(loader: DataLoader):
        """Demonstrates loading data for a specific number of subjects."""
        logging.info("\n--- Example: Load Data by Subject IDs ---")
        if not ExampleDataLoader.check_mimic_path():
            return

        # Ensure directory is scanned so subject IDs are available
        if not loader._data_scan_complete:
            logging.info("Scanning MIMIC directory first to get subject ID list...")
            _, dataset_info = loader.scan_mimic_directory(MIMIC_DATA_PATH)
            if not loader._data_scan_complete:
                logging.warning("Scan failed, cannot proceed with subject ID example.")
                return

        total_subjects = loader.get_total_unique_subjects()
        if total_subjects == 0:
            logging.warning("No subject IDs found (patients.csv might be missing or empty). Skipping subject ID loading example.")
            return
        logging.info(f"Total unique subjects available: {total_subjects}")

        num_subjects_to_load = min(5, total_subjects) # Load for 5 subjects, or fewer if not enough
        logging.info(f"Attempting to load data for {num_subjects_to_load} subjects.")

        target_subject_ids = loader.get_sampled_subject_ids_list(num_subjects_to_load)

        if not target_subject_ids:
            logging.warning(f"Could not retrieve target subject IDs for sampling. Found {len(loader._all_subject_ids if loader._all_subject_ids else [])} subjects in loader state.")
            return

        logging.info(f"Target subject IDs for loading: {target_subject_ids[:10]}... (first 10 if many)")

        # Example: Load 'admissions' table for these subjects
        # admissions_file_path = loader.file_paths.get(('hosp', 'admissions'), None)
        # A more robust way to get path, assuming scan_mimic_directory populates file_paths correctly
        admissions_key = ('hosp', 'admissions') # Using the tuple key
        if admissions_key not in dataset_info['file_paths']:
            logging.warning(f"Admissions table path not found in loader.file_paths after scan. Available keys: {list(dataset_info['file_paths'].keys())}")
            # Fallback to constructing path directly for robustness in example if needed
            admissions_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "admissions.csv.gz")
            if not os.path.exists(admissions_file_path):
                admissions_file_path = os.path.join(MIMIC_DATA_PATH, "hosp", "admissions.csv")
        else:
            admissions_file_path = dataset_info['file_paths'][admissions_key]

        if not admissions_file_path or not os.path.exists(admissions_file_path):
            logging.warning(f"Admissions file path not found or file does not exist: {admissions_file_path}. Skipping 'admissions' load by subject ID.")
        else:
            logging.info(f"Loading 'admissions' table for {len(target_subject_ids)} specific subjects (Pandas)... Path: {admissions_file_path}")
            df_admissions_subject_pd, count_pd = loader.load_mimic_table(
                admissions_file_path,
                target_subject_ids=target_subject_ids,
                use_dask=False
            )
            if df_admissions_subject_pd is not None and not df_admissions_subject_pd.empty:
                logging.info(f"Loaded {len(df_admissions_subject_pd)} admission records for {df_admissions_subject_pd['subject_id'].nunique()} unique subjects (Pandas). Total rows from call: {count_pd}")
                logging.info(f"Data for specific subjects (Pandas - admissions):\n{df_admissions_subject_pd.head()}")
                assert set(df_admissions_subject_pd['subject_id'].unique()).issubset(set(target_subject_ids))
            else:
                logging.warning(f"Failed to load admissions data for specific subjects (Pandas) or data was empty. Found {count_pd} rows.")

        # Example: Load a potentially larger table like 'chartevents' if it's likely to exist, using Dask
        # For this example, we'll try 'outputevents' as it's also common and can be large
        outputevents_key = ('icu', 'outputevents')
        if outputevents_key not in dataset_info['file_paths']:
            logging.info(f"'outputevents' not found in scanned files. Trying direct path construction.")
            outputevents_file_path = os.path.join(MIMIC_DATA_PATH, "icu", "outputevents.csv.gz")
            if not os.path.exists(outputevents_file_path):
                outputevents_file_path = os.path.join(MIMIC_DATA_PATH, "icu", "outputevents.csv")
        else:
            outputevents_file_path = loader.file_paths[outputevents_key]

        if outputevents_file_path and os.path.exists(outputevents_file_path):
            logging.info(f"Loading 'outputevents' table for {len(target_subject_ids)} specific subjects (Dask)... Path: {outputevents_file_path}")
            # Using a small sample_size as a fallback if subject_id filtering isn't efficient or applicable internally
            # The primary filter should be target_subject_ids
            df_output_subject_dd, count_dd = loader.load_mimic_table(
                outputevents_file_path,
                target_subject_ids=target_subject_ids,
                use_dask=True,
                sample_size=10000 # Fallback sample for Dask if needed, though subject_id filtering is preferred
            )
            if df_output_subject_dd is not None and not df_output_subject_dd.empty:
                # df_output_subject_dd is a Pandas DF here due to current Dask implementation returning computed for subject_ids
                logging.info(f"Loaded {len(df_output_subject_dd)} outputevent records for {df_output_subject_dd['subject_id'].nunique()} unique subjects (Dask path). Total rows from call: {count_dd}")
                logging.info(f"Data for specific subjects (Dask - outputevents):\n{df_output_subject_dd.head()}")
                # Check if the loaded subject IDs are a subset of the target ones
                if 'subject_id' in df_output_subject_dd.columns:
                     assert set(df_output_subject_dd['subject_id'].unique()).issubset(set(target_subject_ids))
                else:
                    logging.warning("'subject_id' column not found in Dask-loaded outputevents table, cannot verify subject filtering.")
            else:
                logging.warning(f"Failed to load outputevents data for specific subjects (Dask path) or data was empty. Found {count_dd} rows.")
        else:
            logging.info(f"'outputevents.csv.gz' or '.csv' not found at {outputevents_file_path}. Skipping its load by subject ID example.")

    @staticmethod
    def example_load_mimic_table2(loader: DataLoader):

        # Scan the directory
        dataset_info_df, _ = loader.scan_mimic_directory(mimic_path=MIMIC_DATA_PATH)


        # Load table
        table_name = 'poe'
        module     = 'hosp'

        find_row  = dataset_info_df[(dataset_info_df['module'] == module) & (dataset_info_df['table_name'] == table_name)]
        file_path = dataset_info_df.attrs['mimic_path'] + '/' + find_row['file_path'].values[0]

        print('file_path', file_path)

        patients_df, total_rows = loader.load_mimic_table(file_path=file_path, use_dask=True, sample_size=100)
        return patients_df


    def example_post_processing(loader: DataLoader):
        pass


def main():
    ExampleDataLoader.configure_logging()
    data_loader = ExampleDataLoader.setup_loader()

    logging.info(f"--- Starting DataLoader Examples ---")
    logging.info(f"Using MIMIC_DATA_PATH: {MIMIC_DATA_PATH}")
    if not ExampleDataLoader.check_mimic_path():
        logging.error("Please set the MIMIC_DATA_PATH at the top of this script to your local MIMIC-IV dataset path.")
    else:
        # ExampleDataLoader.example_scan_directory(data_loader)
        # ExampleDataLoader.example_get_table_info(data_loader)
        # ExampleDataLoader.example_load_mimic_table(data_loader)
        # ExampleDataLoader.example_load_reference_table(data_loader)
        # ExampleDataLoader.example_load_patient_cohort(data_loader)
        # ExampleDataLoader.example_load_filtered_table(data_loader)
        # ExampleDataLoader.example_merge_tables(data_loader)
        # ExampleDataLoader.example_convert_to_parquet(data_loader) # Run this after some tables are loaded and files may exist
        # ExampleDataLoader.example_load_connected_tables(data_loader) # This is a more comprehensive one
        # ExampleDataLoader.example_apply_filters(data_loader) # Shows how filters are used with load_mimic_table
        # ExampleDataLoader.example_load_with_subject_ids(data_loader)
        ExampleDataLoader.example_load_mimic_table2(data_loader)

    logging.info("\n--- DataLoader Examples Finished ---")


if __name__ == "__main__":
    main()
