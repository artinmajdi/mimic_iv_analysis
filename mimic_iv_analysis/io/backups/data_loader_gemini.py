# Import necessary libraries
import pandas as pd
import yaml
from functools import reduce
import os
import logging

# Configure logging
# Use a more descriptive logger name if this module is part of a larger package
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a custom exception for data loading errors
class DataLoaderError(Exception):
    """Custom exception for errors encountered during data loading."""
    pass

class MimicIVDataLoader:
    """
    A class to load and merge tables from the MIMIC-IV dataset.

    Attributes:
        data_path (str): The root directory path where MIMIC-IV CSV files are stored.
        config_path (str): The path to the YAML configuration file.
        config (dict): A dictionary holding the loaded configuration.
        tables (dict): A dictionary to store loaded pandas DataFrames, with table names as keys.
    """

    def __init__(self, data_path: str, config_path: str):
        """
        Initializes the MimicIVDataLoader with paths to data and configuration.

        Args:
            data_path (str): The path to the directory containing MIMIC-IV CSV files.
            config_path (str): The path to the YAML configuration file.

        Raises:
            DataLoaderError: If the data_path or config_path is invalid.
        """
        if not os.path.isdir(data_path):
            raise DataLoaderError(f"Invalid data_path: {data_path}. Directory does not exist.")
        if not os.path.isfile(config_path):
            raise DataLoaderError(f"Invalid config_path: {config_path}. File does not exist.")

        self.data_path = data_path
        self.config_path = config_path
        self.config = self._load_config()
        self.tables = {}
        logger.info("MimicIVDataLoader initialized successfully.")

    def _load_config(self) -> dict:
        """
        Loads the YAML configuration file.

        Returns:
            dict: The configuration dictionary.

        Raises:
            DataLoaderError: If the configuration file cannot be loaded or parsed.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config: # Check if config is empty or not loaded correctly
                raise DataLoaderError("Configuration file is empty or could not be parsed.")
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file: {e}")
            raise DataLoaderError(f"Error parsing YAML configuration file: {e}")
        except FileNotFoundError: # Should be caught by __init__, but good for robustness
            logger.error(f"Configuration file not found at {self.config_path}")
            raise DataLoaderError(f"Configuration file not found at {self.config_path}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the config: {e}")
            raise DataLoaderError(f"An unexpected error occurred while loading the config: {e}")


    def load_table(self, table_name: str, columns: list = None, use_config_columns: bool = True, **kwargs) -> pd.DataFrame:
        """
        Loads a specific table from a CSV file into a pandas DataFrame.

        Args:
            table_name (str): The name of the table to load (e.g., 'patients', 'admissions').
                              This name should correspond to a key in the config file and
                              the CSV filename (e.g., 'patients.csv').
            columns (list, optional): A list of specific columns to load.
                                      If None and use_config_columns is True, columns from
                                      the config file are used. Defaults to None.
            use_config_columns (bool, optional): Whether to use columns specified in the
                                                 config file if 'columns' arg is not provided.
                                                 Defaults to True.
            **kwargs: Additional keyword arguments to pass to pandas.read_csv().

        Returns:
            pd.DataFrame: The loaded table as a pandas DataFrame.

        Raises:
            DataLoaderError: If the table name is not found in the configuration,
                             the CSV file is not found, or an error occurs during loading.
        """
        if table_name not in self.config.get('tables', {}):
            logger.error(f"Table '{table_name}' not found in the configuration file.")
            raise DataLoaderError(f"Table '{table_name}' not found in the configuration file.")

        table_info = self.config['tables'][table_name]
        file_name = table_info.get('file_name', f"{table_name}.csv") # Get filename from config or default
        file_path = os.path.join(self.data_path, file_name)

        if not os.path.isfile(file_path):
            logger.error(f"CSV file for table '{table_name}' not found at {file_path}")
            raise DataLoaderError(f"CSV file for table '{table_name}' not found at {file_path}")

        load_columns = columns
        if columns is None and use_config_columns:
            load_columns = table_info.get('columns') # Get columns from config if specified

        try:
            logger.info(f"Loading table '{table_name}' from {file_path}...")
            # Pass specific dtypes if available in config to optimize memory and prevent type issues
            dtype_config = table_info.get('dtypes', None)
            parse_dates_config = table_info.get('parse_dates', None)

            df = pd.read_csv(
                file_path,
                usecols=load_columns,
                dtype=dtype_config, # Add dtype specification
                parse_dates=parse_dates_config, # Add date parsing
                **kwargs
            )
            self.tables[table_name] = df
            logger.info(f"Table '{table_name}' loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except FileNotFoundError: # Should be caught above, but as a safeguard
            logger.error(f"File not found: {file_path}. This should have been caught earlier.")
            raise DataLoaderError(f"File not found: {file_path}")
        except ValueError as ve: # Handles issues like columns not found in CSV
            logger.error(f"ValueError loading table '{table_name}': {ve}")
            raise DataLoaderError(f"ValueError loading table '{table_name}': {ve}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading table '{table_name}': {e}")
            raise DataLoaderError(f"An unexpected error occurred while loading table '{table_name}': {e}")

    def get_table(self, table_name: str) -> pd.DataFrame:
        """
        Retrieves a previously loaded table.

        Args:
            table_name (str): The name of the table to retrieve.

        Returns:
            pd.DataFrame: The requested pandas DataFrame.

        Raises:
            DataLoaderError: If the table has not been loaded yet.
        """
        if table_name not in self.tables:
            logger.error(f"Table '{table_name}' has not been loaded. Call load_table() first.")
            raise DataLoaderError(f"Table '{table_name}' has not been loaded. Call load_table() first.")
        return self.tables[table_name]

    def merge_tables(self, table_names: list, merge_instructions: list = None) -> pd.DataFrame:
        """
        Merges multiple loaded tables based on provided instructions or configuration.

        The function attempts to intelligently merge tables. If merge_instructions are
        provided, they are used. Otherwise, it looks for 'merge_keys' in the config
        for the specified tables.

        Args:
            table_names (list): A list of names of the tables to merge.
                                These tables must be loaded first.
            merge_instructions (list of dict, optional): A list of dictionaries,
                where each dictionary specifies how to merge two tables.
                Each dict should have:
                - 'left_table': name of the left table (str)
                - 'right_table': name of the right table (str)
                - 'left_on': column name(s) from the left table (str or list)
                - 'right_on': column name(s) from the right table (str or list)
                - 'how': type of merge (e.g., 'inner', 'left', 'outer'). Defaults to 'inner'.
                If None, the method will try to use 'merge_keys' from the config.

        Returns:
            pd.DataFrame: The merged pandas DataFrame.

        Raises:
            DataLoaderError: If any of the specified tables are not loaded,
                             if merge keys are ambiguous or not found,
                             or if an error occurs during merging.
        """
        if not table_names or len(table_names) < 2:
            raise DataLoaderError("At least two table names must be provided for merging.")

        # Ensure all tables are loaded
        for table_name in table_names:
            if table_name not in self.tables:
                raise DataLoaderError(f"Table '{table_name}' is not loaded. Load it before merging.")

        if merge_instructions:
            logger.info("Using provided merge_instructions.")
            # Validate merge_instructions structure (basic validation)
            for instr in merge_instructions:
                if not all(k in instr for k in ['left_table', 'right_table', 'left_on', 'right_on']):
                    raise DataLoaderError("Invalid merge_instruction: missing required keys "
                                          "('left_table', 'right_table', 'left_on', 'right_on').")

            # Start with the first table in the first instruction, or determine a base
            # This part needs careful handling if instructions don't form a clear chain
            # For simplicity, let's assume instructions are ordered or we merge sequentially
            # A more robust approach might involve graph-based merging order.

            # Create a dictionary of tables for easy access
            loaded_dfs = {name: self.tables[name].copy() for name in table_names}
            merged_df = None
            processed_tables = set()

            if not merge_instructions:
                 raise DataLoaderError("merge_instructions must be provided if not using config-based merging.")


            # Iteratively merge based on instructions
            # This assumes instructions can be processed in the given order.
            # If instructions are not ordered, a more complex logic to build the merge sequence is needed.
            # For now, let's assume the first left_table is our base.
            current_df_name = merge_instructions[0]['left_table']
            merged_df = loaded_dfs[current_df_name]
            processed_tables.add(current_df_name)

            for instr in merge_instructions:
                left_name = instr['left_table']
                right_name = instr['right_table']
                left_on = instr['left_on']
                right_on = instr['right_on']
                how = instr.get('how', 'inner') # Default to 'inner' merge

                # Ensure the left table in the instruction is our current merged_df or a table we can join to
                # This logic might need refinement depending on how merge_instructions are structured
                if left_name not in processed_tables and left_name != current_df_name :
                    # This case means the instruction's left_table is not the one we are building upon.
                    # This could be an error in instruction order or a more complex merge graph.
                    # For now, we'll assume the instructions imply a sequence.
                    # A more robust solution would check if left_name is part of merged_df.columns
                    # or if we need to switch the base.
                    logger.warning(f"Merge instruction for '{left_name}' and '{right_name}' might be out of sequence "
                                   f"or implies a different base table than current '{current_df_name}'.")
                    # Attempt to merge if the right_table is the one we are building upon (swap logic)
                    if right_name == current_df_name:
                        left_name, right_name = right_name, left_name
                        left_on, right_on = right_on, left_on
                    elif left_name not in merged_df.columns and right_name not in loaded_dfs:
                         raise DataLoaderError(f"Cannot determine merge sequence for instruction: {instr} with current base {current_df_name}")


                if right_name not in loaded_dfs:
                    raise DataLoaderError(f"Right table '{right_name}' in instruction not found in loaded tables.")

                right_df = loaded_dfs[right_name]

                logger.info(f"Merging '{current_df_name}' (or its evolution) with '{right_name}' on "
                            f"left_on={left_on}, right_on={right_on}, how='{how}'")
                try:
                    merged_df = pd.merge(merged_df, right_df,
                                         left_on=left_on, right_on=right_on,
                                         how=how, suffixes=('', f'_{right_name}')) # Add suffixes to avoid column name clashes
                    processed_tables.add(right_name)
                    # The "name" of merged_df effectively becomes a composite, or we track its evolution.
                    # For logging, current_df_name could be updated, e.g., f"{current_df_name}_then_{right_name}"
                except pd.errors.MergeError as me:
                    logger.error(f"Pandas MergeError during merge of '{current_df_name}' and '{right_name}': {me}")
                    raise DataLoaderError(f"Pandas MergeError: {me}")
                except KeyError as ke:
                    logger.error(f"KeyError during merge: {ke}. Check if '{left_on}' or '{right_on}' exist in respective tables.")
                    raise DataLoaderError(f"KeyError during merge: {ke}. Merge columns might be missing.")
                except Exception as e:
                    logger.error(f"Unexpected error during merge of '{current_df_name}' and '{right_name}': {e}")
                    raise DataLoaderError(f"Unexpected error during merge: {e}")

            if merged_df is None: # Should not happen if instructions are valid and tables exist
                raise DataLoaderError("Merging process completed but resulted in an empty DataFrame (merged_df is None).")

            logger.info("All merge instructions processed.")
            return merged_df

        else:
            # Fallback to config-based merging (Original simplified logic)
            # This part of the original logic was a bit simplistic and might lead to issues
            # if merge keys are not straightforward or if multiple merge paths exist.
            # The `merge_instructions` path is more explicit and controllable.
            # For a robust config-based approach, the config would need to define the merge sequence and keys more clearly.
            logger.warning("No merge_instructions provided. Attempting to use 'merge_keys' from config (simplified logic).")
            logger.warning("This config-based merging is basic. For complex merges, provide explicit 'merge_instructions'.")

            # Ensure all tables to be merged are loaded
            dfs_to_merge = [self.tables[name].copy() for name in table_names] # Use copies to avoid modifying original loaded tables

            # Attempt to find common keys from config if not explicitly given
            # This is a very basic approach and might not be robust for all cases.
            # It assumes a chain merge using the first common key found.
            # A more sophisticated approach would build a merge graph or require explicit key pairs in config.

            # Example of how one might try to deduce merge keys (highly simplified):
            # We'd need to iterate through pairs of tables and find their common 'merge_keys' from config.
            # This part is complex to generalize without a clear merge strategy in the config.

            # The original code used reduce with a lambda that merges based on common columns.
            # This is prone to issues if common columns are not the intended merge keys.
            # Let's try to make it slightly more intelligent by looking for 'primary_key' or 'foreign_keys' in config.

            if len(dfs_to_merge) < 2:
                return dfs_to_merge[0] if dfs_to_merge else pd.DataFrame()

            # A more robust config-based merge would require the config to specify pairs and their keys.
            # E.g., config['merges'] = [{'tables': ['patients', 'admissions'], 'on': 'subject_id'}, ...]
            # Without such structure, we rely on pandas' default behavior or simple common columns.

            # For now, let's keep the reduce but with a warning that it's a simple approach.
            # It's better to use explicit merge_instructions.
            try:
                # This merge strategy is implicit. It will merge on common column names.
                # This can be dangerous if not all common columns are intended as join keys.
                # Or, if multiple common columns exist, it uses all of them.
                logger.info(f"Attempting to merge tables: {table_names} on common columns.")
                merged_df = reduce(lambda left, right: pd.merge(left, right, on=None, how='inner'), dfs_to_merge)
                # `on=None` means merge on common columns.
                # `how='inner'` is a default, might need to be configurable.
                logger.info("Tables merged using reduce on common columns.")
                return merged_df
            except pd.errors.MergeError as me:
                logger.error(f"Pandas MergeError during reduce-based merge: {me}. "
                             "This often happens if there are no common columns or ambiguous keys. "
                             "Consider using explicit 'merge_instructions'.")
                raise DataLoaderError(f"Pandas MergeError during reduce-based merge: {me}")
            except Exception as e:
                logger.error(f"Unexpected error during reduce-based merge: {e}")
                raise DataLoaderError(f"Unexpected error during reduce-based merge: {e}")


    def get_merged_table_for_analysis(self, analysis_type: str) -> pd.DataFrame:
        """
        Loads and merges tables required for a specific analysis type,
        as defined in the 'analyses' section of the configuration file.

        Args:
            analysis_type (str): The key for the analysis configuration
                                 (e.g., 'sepsis_prediction').

        Returns:
            pd.DataFrame: The final merged DataFrame for the analysis.

        Raises:
            DataLoaderError: If analysis_type is not in config, or if
                             tables/merge instructions are missing or invalid.
        """
        if 'analyses' not in self.config or analysis_type not in self.config['analyses']:
            logger.error(f"Analysis type '{analysis_type}' not found in configuration.")
            raise DataLoaderError(f"Analysis type '{analysis_type}' not found in configuration.")

        analysis_config = self.config['analyses'][analysis_type]
        tables_to_load = analysis_config.get('tables')
        merge_instructions = analysis_config.get('merge_instructions')

        if not tables_to_load:
            logger.error(f"No tables specified for analysis type '{analysis_type}'.")
            raise DataLoaderError(f"No tables specified for analysis type '{analysis_type}'.")

        # Load all required tables
        for table_name in tables_to_load:
            if table_name not in self.tables: # Load only if not already loaded
                # Get columns specific to this analysis context if defined, else load default
                table_specific_config = self.config.get('tables', {}).get(table_name, {})
                columns_for_analysis = analysis_config.get('table_columns', {}).get(table_name)

                if columns_for_analysis:
                    logger.info(f"Loading table '{table_name}' with specific columns for analysis '{analysis_type}'.")
                    self.load_table(table_name, columns=columns_for_analysis, use_config_columns=False)
                else:
                    # Load with default columns from main table config or all columns
                    self.load_table(table_name, use_config_columns=True)


        # Perform merges
        if not merge_instructions and len(tables_to_load) > 1:
            # If no explicit instructions, and more than one table, we might try the simple reduce merge
            # but it's better to require explicit instructions for analyses.
            logger.warning(f"No explicit 'merge_instructions' for analysis '{analysis_type}'. "
                           "Attempting basic merge on common columns if multiple tables. "
                           "It is highly recommended to provide explicit merge_instructions for analyses.")
            # Ensure the tables_to_load are the ones to be passed to merge_tables if using the simple version
            merged_df = self.merge_tables(table_names=tables_to_load) # This will use the reduce logic
        elif len(tables_to_load) == 1:
            merged_df = self.get_table(tables_to_load[0])
            logger.info(f"Analysis '{analysis_type}' uses a single table: '{tables_to_load[0]}'. No merge needed.")
        elif merge_instructions:
             # Ensure all tables mentioned in merge_instructions are in tables_to_load for clarity
            all_tables_in_instructions = set()
            for instr in merge_instructions:
                all_tables_in_instructions.add(instr['left_table'])
                all_tables_in_instructions.add(instr['right_table'])

            if not all_tables_in_instructions.issubset(set(tables_to_load)):
                missing = all_tables_in_instructions - set(tables_to_load)
                logger.error(f"Tables {missing} in merge_instructions are not listed in 'tables' for analysis '{analysis_type}'.")
                raise DataLoaderError(f"Mismatch between 'tables' and 'merge_instructions' for analysis '{analysis_type}'.")

            merged_df = self.merge_tables(table_names=list(all_tables_in_instructions), merge_instructions=merge_instructions)
        else: # Only one table, no merge needed
            merged_df = self.get_table(tables_to_load[0])


        # Optional: Apply post-merge processing steps if defined in config
        post_processing_steps = analysis_config.get('post_processing')
        if post_processing_steps:
            logger.info(f"Applying post-processing steps for analysis '{analysis_type}'...")
            # Example: merged_df = self._apply_post_processing(merged_df, post_processing_steps)
            # This would require another method to interpret and apply these steps.
            pass # Placeholder for post-processing logic

        logger.info(f"Successfully prepared merged table for analysis '{analysis_type}'.")
        return merged_df


# Example Usage (assuming you have a config.yaml and data files)
if __name__ == '__main__':
    # Create dummy config and data for testing
    DUMMY_DATA_PATH = 'dummy_mimic_data'
    DUMMY_CONFIG_PATH = 'dummy_config.yaml'

    os.makedirs(DUMMY_DATA_PATH, exist_ok=True)

    # Create dummy CSV files
    patients_data = {'subject_id': [1, 2, 3], 'gender': ['M', 'F', 'M'], 'anchor_age': [65, 70, 50]}
    pd.DataFrame(patients_data).to_csv(os.path.join(DUMMY_DATA_PATH, 'patients.csv'), index=False)

    admissions_data = {'subject_id': [1, 2, 2, 3], 'hadm_id': [101, 102, 103, 104], 'admittime': ['2180-05-06', '2191-07-23', '2192-01-01', '2175-10-30'], 'deathtime': [None, '2191-08-10', None, None]}
    pd.DataFrame(admissions_data).to_csv(os.path.join(DUMMY_DATA_PATH, 'admissions.csv'), index=False)

    labevents_data = {'subject_id': [1,1,2,3], 'hadm_id': [101, 101, 102, 104], 'itemid': [50912, 50882, 50912, 50971], 'valuenum': [7.4, 25, 7.35, 130]}
    pd.DataFrame(labevents_data).to_csv(os.path.join(DUMMY_DATA_PATH, 'labevents.csv'), index=False)


    # Create dummy config.yaml
    dummy_config_content = {
        'tables': {
            'patients': {
                'file_name': 'patients.csv',
                'columns': ['subject_id', 'gender', 'anchor_age'],
                'primary_key': 'subject_id'
            },
            'admissions': {
                'file_name': 'admissions.csv',
                'columns': ['subject_id', 'hadm_id', 'admittime', 'deathtime'],
                'foreign_keys': {'subject_id': 'patients'},
                'parse_dates': ['admittime', 'deathtime']
            },
            'labevents':{
                'file_name': 'labevents.csv',
                # 'columns': ['subject_id', 'hadm_id', 'itemid', 'valuenum'], # Load all if not specified
                'foreign_keys': {'subject_id': 'patients', 'hadm_id': 'admissions'}
            }
        },
        'analyses': {
            'patient_admissions_labs': {
                'tables': ['patients', 'admissions', 'labevents'],
                'merge_instructions': [
                    {'left_table': 'patients', 'right_table': 'admissions', 'left_on': 'subject_id', 'right_on': 'subject_id', 'how': 'inner'},
                    # Assuming the result of the first merge is implicitly the left table for the next
                    # A more robust system might name the intermediate result or require explicit chaining.
                    # For this example, merge_tables will sequentially apply these.
                    # The current merge_tables expects the 'left_table' of subsequent merges to be the
                    # evolving merged dataframe. So the next instruction should effectively be on the
                    # result of (patients + admissions) and labevents.
                    # The current implementation of `merge_tables` uses the first table of the *first* instruction
                    # as the initial `merged_df`.
                    # To merge labevents, it should be joined to the result of patients+admissions.
                    # The `left_on` for the second merge would be columns from the (patients+admissions) df.
                    {'left_table': 'admissions', # This implies we are joining to the df that now contains admission columns
                     'right_table': 'labevents',
                     'left_on': ['subject_id', 'hadm_id'], # These columns are now in the merged_df
                     'right_on': ['subject_id', 'hadm_id'],
                     'how': 'left'}
                ]
            },
            'simple_patients': {
                'tables': ['patients']
            }
        }
    }
    with open(DUMMY_CONFIG_PATH, 'w') as f:
        yaml.dump(dummy_config_content, f)

    try:
        # Initialize loader
        loader = MimicIVDataLoader(data_path=DUMMY_DATA_PATH, config_path=DUMMY_CONFIG_PATH)

        # --- Example 1: Load individual tables ---
        # patients_df = loader.load_table('patients')
        # print("\nPatients Table:")
        # print(patients_df.head())

        # admissions_df = loader.load_table('admissions', parse_dates=['admittime', 'deathtime'])
        # print("\nAdmissions Table:")
        # print(admissions_df.head())
        # print(admissions_df.info())


        # --- Example 2: Merge tables using explicit instructions ---
        # loader.load_table('patients')
        # loader.load_table('admissions')
        # merge_instr = [
        #     {'left_table': 'patients', 'right_table': 'admissions',
        #      'left_on': 'subject_id', 'right_on': 'subject_id', 'how': 'inner'}
        # ]
        # merged_patients_admissions = loader.merge_tables(
        #     table_names=['patients', 'admissions'],
        #     merge_instructions=merge_instr
        # )
        # print("\nMerged Patients and Admissions (explicit instructions):")
        # print(merged_patients_admissions.head())

        # --- Example 3: Get merged table for a defined analysis ---
        analysis_df = loader.get_merged_table_for_analysis('patient_admissions_labs')
        print("\nMerged Table for 'patient_admissions_labs' analysis:")
        print(analysis_df.head())
        print(f"Shape of final analysis_df: {analysis_df.shape}")
        print(analysis_df.columns)


        analysis_single_df = loader.get_merged_table_for_analysis('simple_patients')
        print("\nTable for 'simple_patients' analysis:")
        print(analysis_single_df.head())


    except DataLoaderError as e:
        logger.error(f"Data loading/merging failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the example: {e}")
    finally:
        # Clean up dummy files and directory
        if os.path.exists(DUMMY_CONFIG_PATH):
            os.remove(DUMMY_CONFIG_PATH)
        for fname in ['patients.csv', 'admissions.csv', 'labevents.csv']:
            fpath = os.path.join(DUMMY_DATA_PATH, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        if os.path.exists(DUMMY_DATA_PATH):
            os.rmdir(DUMMY_DATA_PATH)
        logger.info("Cleaned up dummy files.")
