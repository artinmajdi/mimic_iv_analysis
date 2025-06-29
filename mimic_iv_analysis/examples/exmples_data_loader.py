#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating the usage of the DataLoader, ExampleDataLoader, and ParquetConverter classes.

This script shows how to:
1. Initialize the DataLoader
2. Scan the MIMIC-IV directory
3. Load tables with various options
4. Apply filters to loaded DataFrames
5. Get descriptive information for tables
6. Merge tables
7. Use the ExampleDataLoader for simplified data loading
8. Convert tables to Parquet format using ParquetConverter

Usage:
    python exmples_data_loader.py
"""

import os
import pandas as pd
import dask.dataframe as dd
from pathlib import Path

# Import our modules
from mimic_iv_analysis import logger
from mimic_iv_analysis.io.data_loader import DataLoader, ExampleDataLoader, ParquetConverter
from mimic_iv_analysis.configurations.params import TableNames, DataFrameType



# --- Configuration ---
MIMIC_DATA_PATH = Path("/Users/artinmajdi/Documents/GitHubs/RAP/mimic__pankaj/dataset/mimic-iv-3.1")


class DataLoaderExamples:
    """Examples demonstrating the usage of the DataLoader, ExampleDataLoader, and ParquetConverter classes."""

    @staticmethod
    def check_mimic_path():
        """Checks if the MIMIC_DATA_PATH is set and valid."""
        if not os.path.exists(MIMIC_DATA_PATH):
            logger.warning(f"MIMIC_DATA_PATH is not set to a valid path: {MIMIC_DATA_PATH}")
            logger.warning("Examples will not run correctly. Please update it in the script.")
            return False
        return True

    # --- DataLoader Examples ---
    @staticmethod
    def example_scan_directory():
        """Demonstrates scanning the MIMIC-IV directory."""
        logger.info("\n--- Example: Scan MIMIC-IV Directory ---")
        if not DataLoaderExamples.check_mimic_path():
            return

        loader = DataLoader(mimic_path=MIMIC_DATA_PATH)
        loader.scan_mimic_directory()
        tables_info_df, tables_info_dict = loader.tables_info_df, loader.tables_info_dict


        if tables_info_df is None or tables_info_dict is None:
            logger.error(
                f"Failed to scan MIMIC-IV directory. It might not exist or be structured correctly at: {MIMIC_DATA_PATH}."
                " Please check the path and directory structure. Skipping directory scan example."
            )
            return

        if 'available_tables' in tables_info_dict:
            modules = tables_info_dict['available_tables'].keys()
            logger.info(f"Found {len(modules)} modules: {list(modules)}")

            for module, tables in tables_info_dict['available_tables'].items():
                logger.info(f"  Module '{module}' has {len(tables)} tables: {tables[:3]}...")  # Print first 3

            # Print example info for patients
            hosp_patients_key = ('hosp', 'patients')
            if hosp_patients_key in tables_info_dict['file_paths']:
                logger.info(f"Example file path for patients: {tables_info_dict['file_paths'][hosp_patients_key]}")
                logger.info(f"Example file size for patients: {tables_info_dict['file_sizes'][hosp_patients_key]} bytes")
                logger.info(f"Example display name for patients: {tables_info_dict['table_display_names'][hosp_patients_key]}")

        # Print the DataFrame info
        logger.info(f"\nTables info DataFrame columns: {tables_info_df.columns.tolist()}")
        logger.info(f"Tables info DataFrame shape: {tables_info_df.shape}")
        logger.info(f"Example rows:\n{tables_info_df.head(2)}")

    @staticmethod
    def example_load_table():
        """Demonstrates loading a table with various options."""
        logger.info("\n--- Example: Load Table ---")
        if not DataLoaderExamples.check_mimic_path():
            return

        loader = DataLoader(mimic_path=MIMIC_DATA_PATH)
        loader.scan_mimic_directory()

        # 1. Load a table fully
        logger.info("Loading 'patients' table fully...")
        patients_df = loader.load(TableNames.PATIENTS, partial_loading=False)

        # Check if Dask DataFrame and compute if needed for display
        if isinstance(patients_df, dd.DataFrame):
            patients_count = patients_df.shape[0].compute()
            logger.info(f"Loaded {patients_count} patients (full). Sample data:")
            logger.info(patients_df.head())
        else:
            logger.info(f"Loaded {len(patients_df)} patients (full). Sample data:")
            logger.info(patients_df.head())

        # 2. Load with partial loading by subject IDs
        logger.info("\nLoading 'patients' table with partial loading by subject IDs...")
        partial_subject_ids = loader.get_partial_subject_id_list_for_partial_loading(num_subjects=5, random_selection=False, table_name=TableNames.PATIENTS)
        logger.info(f"Selected subject IDs for partial loading: {partial_subject_ids}")

        loader.subject_ids_for_partial_loading = partial_subject_ids
        partial_df = loader.load(table_name=TableNames.PATIENTS, partial_loading=True)

        if isinstance(partial_df, dd.DataFrame):
            partial_count = partial_df.shape[0].compute()
            logger.info(f"Loaded {partial_count} patients with partial loading. Sample data:")
            logger.info(partial_df.head())
        else:
            logger.info(f"Loaded {len(partial_df)} patients with partial loading. Sample data:")
            logger.info(partial_df.head())

    @staticmethod
    def example_merge_tables():
        """Demonstrates merging tables."""
        logger.info("\n--- Example: Merge Tables ---")
        if not DataLoaderExamples.check_mimic_path():
            return

        loader = DataLoader(mimic_path=MIMIC_DATA_PATH)
        loader.scan_mimic_directory()

        # Create dummy DataFrames for demonstration
        data1 = {'id': [1, 2, 3], 'value1': ['A', 'B', 'C']}
        df1 = pd.DataFrame(data1)

        data2 = {'id': [2, 3, 4], 'value2': ['X', 'Y', 'Z']}
        df2 = pd.DataFrame(data2)

        logger.info("Demonstration of merging two DataFrames:")
        merged_df = loader.load(table_name=TableNames.MERGED)

        logger.info("\nMerged dictionary keys:")
        for key in merged_df.columns:
            logger.info(f"  - {key}")

        if isinstance(merged_df, dd.DataFrame):
            logger.info(f"\nSample from (shape: {merged_df.shape[0].compute()}):")
            logger.info(merged_df.head())
        else:
            logger.info(f"\nSample from (shape: {len(merged_df)}):")
            logger.info(merged_df.head())

    @staticmethod
    def example_apply_filters():
        """Demonstrates applying filters to a loaded DataFrame."""
        logger.info("\n--- Example: Apply Filters ---")
        if not DataLoaderExamples.check_mimic_path():
            return

        logger.info("This would show filtering examples using the Filtering class.")
        logger.info("For now, filtering is handled internally by the load_table method.")

    @staticmethod
    def example_get_table_info():
        """Demonstrates getting descriptive information for a table."""
        logger.info("\n--- Example: Get Table Info ---")
        # We can use the TableNames and TableNames Enum classes' description property
        info_admissions = TableNames.ADMISSIONS.description
        logger.info(f"Info for ADMISSIONS: {info_admissions}")

        info_chartevents = TableNames.CHARTEVENTS.description
        logger.info(f"Info for CHARTEVENTS: {info_chartevents}")

    @staticmethod
    def example_parquet_converter():
        """Demonstrates using the ParquetConverter class."""

        logger.info("\n--- Example: Using ParquetConverter ---")
        if not DataLoaderExamples.check_mimic_path():
            return

        # Initialize DataLoader and scan directory
        loader = DataLoader(mimic_path=MIMIC_DATA_PATH)
        loader.scan_mimic_directory()

        # Create ParquetConverter
        converter = ParquetConverter(data_loader=loader)

        # Example: Save a small table as Parquet
        logger.info("Saving d_labitems table as Parquet...")
        try:
            converter.save_as_parquet(TableNames.D_LABITEMS)

            logger.info("Successfully saved d_labitems as Parquet")

        except Exception as e:
            logger.error(f"Error saving d_labitems as Parquet: {e}")

        # Note on saving all tables
        logger.info("\nTo save all tables as Parquet, you would use:")
        logger.info("converter.save_all_tables_as_parquet()")
        logger.info("This operation may take a long time for large datasets, so it's not run in this example.")


def main():
    logger.info("--- Starting DataLoader Examples ---")
    logger.info(f"Using MIMIC_DATA_PATH: {MIMIC_DATA_PATH}")

    if not DataLoaderExamples.check_mimic_path():
        logger.error("Please set the MIMIC_DATA_PATH at the top of this script to your local MIMIC-IV dataset path.")
    else:
        # DataLoader examples
        DataLoaderExamples.example_scan_directory()
        DataLoaderExamples.example_load_table()
        DataLoaderExamples.example_get_table_info()
        DataLoaderExamples.example_merge_tables()
        DataLoaderExamples.example_apply_filters()

        # ParquetConverter examples
        # DataLoaderExamples.example_parquet_converter()

    logger.info("\n--- DataLoader Examples Finished ---")


if __name__ == "__main__":
    main()
