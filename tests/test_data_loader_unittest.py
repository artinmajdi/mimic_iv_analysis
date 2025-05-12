import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import dask.dataframe as dd
import os
import sys

# Add the project root to the Python path to allow importing mimic_iv_analysis
# Adjust the path as necessary based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from mimic_iv_analysis.core.data_loader import DataLoader, PATIENTS_TABLE_MODULE, PATIENTS_TABLE_NAME, SUBJECT_ID_COL
from mimic_iv_analysis.core.filtering import Filtering


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader()
        # Mock the Filtering class dependency
        self.mock_filtering_instance = MagicMock(spec=Filtering)
        self.loader.filtering = self.mock_filtering_instance

    def test_format_size(self):
        self.assertEqual(self.loader._format_size(0.0005), "(< 1 KB)")
        self.assertEqual(self.loader._format_size(0.5), "(512.0 KB)")
        self.assertEqual(self.loader._format_size(500), "(500.0 MB)")
        self.assertEqual(self.loader._format_size(1500), "(1.5 GB)")

    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.getsize')
    def test_scan_mimic_directory_valid_path(self, mock_getsize, mock_glob, mock_exists):
        mock_exists.return_value = True
        mock_glob.side_effect = [
            ['/path/to/mimic/hosp/patients.csv.gz', '/path/to/mimic/hosp/admissions.csv'], # hosp files
            ['/path/to/mimic/icu/chartevents.csv.gz']  # icu files
        ]
        mock_getsize.return_value = 1024 * 1024  # 1 MB

        mimic_path = '/path/to/mimic'
        _, dataset_info = self.loader.scan_mimic_directory(mimic_path)

        self.assertIn('hosp',        dataset_info['available_tables'])
        self.assertIn('icu',         dataset_info['available_tables'])
        self.assertIn('patients',    dataset_info['available_tables']['hosp'])
        self.assertIn('admissions',  dataset_info['available_tables']['hosp'])
        self.assertIn('chartevents', dataset_info['available_tables']['icu'])

        self.assertEqual(dataset_info['file_paths'][('hosp', 'patients')], '/path/to/mimic/hosp/patients.csv.gz')
        self.assertEqual(dataset_info['file_sizes'][('hosp', 'patients')], 1.0)
        self.assertEqual(dataset_info['table_display_names'][('hosp', 'patients')], 'patients (1.0 MB)')
        self.assertEqual(self.loader._patients_file_path, '/path/to/mimic/hosp/patients.csv.gz')
        self.assertTrue(self.loader._data_scan_complete)

    @patch('os.path.exists', return_value=False)
    def test_scan_mimic_directory_invalid_path(self, mock_exists):
        _, dataset_info = self.loader.scan_mimic_directory('/invalid/path')
        self.assertEqual(dataset_info['available_tables'], {})
        self.assertEqual(dataset_info['file_paths'], {})
        self.assertEqual(dataset_info['file_sizes'], {})
        self.assertEqual(dataset_info['table_display_names'], {})
        self.assertIsNone(self.loader._patients_file_path) # Should remain None or be reset
        self.assertTrue(self.loader._data_scan_complete) # Scan completes, finds nothing

    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv')
    def test_get_all_subject_ids_successful_load(self, mock_read_csv):
        # Ensure scan_mimic_directory has "run" and set up _patients_file_path
        self.loader._data_scan_complete = True
        self.loader._patients_file_path = '/fake/path/hosp/patients.csv.gz'

        mock_patients_df = pd.DataFrame({SUBJECT_ID_COL: [1, 2, 3, 1, 4]})
        mock_read_csv.return_value = mock_patients_df

        with patch('os.path.exists', return_value=True): # Mock os.path.exists for the patients file
            subject_ids = self.loader.subject_ids_list()

        self.assertListEqual(sorted(subject_ids), [1, 2, 3, 4])
        self.assertEqual(self.loader.get_total_unique_subjects(), 4)
        mock_read_csv.assert_called_once_with(
            '/fake/path/hosp/patients.csv.gz',
            usecols=[SUBJECT_ID_COL],
            compression='gzip',
            encoding='latin-1'
        )

        # Test caching
        mock_read_csv.reset_mock()
        subject_ids_cached = self.loader.subject_ids_list()
        self.assertListEqual(sorted(subject_ids_cached), [1, 2, 3, 4])
        mock_read_csv.assert_not_called()


    def test_get_all_subject_ids_scan_not_complete(self):
        self.loader._data_scan_complete = False
        with self.assertLogs(level='WARNING') as log:
            subject_ids = self.loader.subject_ids_list()
            self.assertIn("MIMIC directory scan has not been performed", log.output[0])
        self.assertEqual(subject_ids, [])
        self.assertEqual(self.loader.get_total_unique_subjects(), 0)

    @patch('os.path.exists', return_value=False)
    def test_get_all_subject_ids_patients_file_not_found(self, mock_path_exists):
        self.loader._data_scan_complete = True
        self.loader._patients_file_path = '/fake/path/hosp/patients.csv.gz' # Path is set but os.path.exists will return False
        with self.assertLogs(level='WARNING') as log:
            subject_ids = self.loader.subject_ids_list()
            self.assertIn("'.csv' or '.csv.gz' not found during scan or path is invalid", log.output[0])
        self.assertEqual(subject_ids, [])

    def test_get_subject_ids_for_sampling(self):
        self.loader._all_subject_ids = [10, 20, 30, 40, 50]
        self.assertIsNone(self.loader.get_sampled_subject_ids_list(None))
        self.assertIsNone(self.loader.get_sampled_subject_ids_list(0))
        self.assertEqual(self.loader.get_sampled_subject_ids_list(3), [10, 20, 30])
        self.assertEqual(self.loader.get_sampled_subject_ids_list(10), [10, 20, 30, 40, 50])

        self.loader._all_subject_ids = None
        self.assertIsNone(self.loader.get_sampled_subject_ids_list(3))

        self.loader._all_subject_ids = []
        self.assertIsNone(self.loader.get_sampled_subject_ids_list(3))


    @patch('os.path.getsize', return_value=10 * 1024 * 1024) # 10MB < LARGE_FILE_THRESHOLD_MB
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv')
    def test_load_mimic_table_pandas_small_file_no_sampling(self, mock_read_csv, mock_getsize):
        mock_df = pd.DataFrame({'colA': [1, 2], SUBJECT_ID_COL: [101, 102]})

        # Mock for header read
        mock_header_df = pd.DataFrame(columns=['colA', SUBJECT_ID_COL])

        # Mock for chunked read
        mock_reader_instance = MagicMock()
        mock_reader_instance.__enter__.return_value = [mock_df] # Simulate one chunk

        mock_read_csv.side_effect = [mock_header_df, mock_reader_instance]

        df, total_rows = self.loader.load_mimic_table('/fake/file.csv', sample_size=None, use_dask=False)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(total_rows, 2)
        self.assertTrue(SUBJECT_ID_COL in df.columns)

        # Check calls: first for header, second for chunking
        self.assertEqual(mock_read_csv.call_count, 2)
        mock_read_csv.assert_any_call('/fake/file.csv', nrows=0, encoding='latin-1', compression=None, low_memory=False)
        mock_read_csv.assert_any_call('/fake/file.csv', chunksize=100000, encoding='latin-1', compression=None, low_memory=False)


    @patch('os.path.getsize', return_value=200 * 1024 * 1024) # 200MB > LARGE_FILE_THRESHOLD_MB
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv')
    def test_load_mimic_table_pandas_large_file_with_sampling(self, mock_read_csv, mock_getsize):
        mock_df_full = pd.DataFrame({'colA': range(200), SUBJECT_ID_COL: range(1000, 1200)})

        # Mock for header read
        mock_header_df = pd.DataFrame(columns=['colA', SUBJECT_ID_COL])

        # Mock for chunked read (to get total_rows)
        mock_reader_instance_total_rows = MagicMock()
        # Simulate two chunks for total_rows calculation
        mock_reader_instance_total_rows.__enter__.return_value = [mock_df_full.iloc[:100], mock_df_full.iloc[100:]]

        # Mock for nrows read (sampling)
        mock_df_sampled = mock_df_full.head(50)

        mock_read_csv.side_effect = [
            mock_header_df,                      # Header read
            mock_reader_instance_total_rows,     # Chunked read for total_rows
            mock_df_sampled                      # Nrows read for sampling
        ]

        df, total_rows = self.loader.load_mimic_table('/fake/large_file.csv', sample_size=50, use_dask=False)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        self.assertEqual(total_rows, 200) # Total rows from chunked read

        self.assertEqual(mock_read_csv.call_count, 3)
        mock_read_csv.assert_any_call('/fake/large_file.csv', nrows=0, encoding='latin-1', compression=None, low_memory=False)
        mock_read_csv.assert_any_call('/fake/large_file.csv', chunksize=100000, encoding='latin-1', compression=None, low_memory=False)
        mock_read_csv.assert_any_call('/fake/large_file.csv', nrows=50, encoding='latin-1', compression=None, low_memory=False)

    @patch('os.path.getsize', return_value=200 * 1024 * 1024) # 200MB > LARGE_FILE_THRESHOLD_MB
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv')
    def test_load_mimic_table_pandas_large_file_with_target_subject_ids(self, mock_read_csv, mock_getsize):
        target_ids = [101, 103]
        chunk1_data = pd.DataFrame({'colA': [1,2,3], SUBJECT_ID_COL: [101, 102, 103]})
        chunk2_data = pd.DataFrame({'colA': [4,5], SUBJECT_ID_COL: [104, 101]})

        mock_header_df = pd.DataFrame(columns=['colA', SUBJECT_ID_COL])

        mock_reader_instance = MagicMock()
        mock_reader_instance.__enter__.return_value = [chunk1_data, chunk2_data] # Simulate chunks

        mock_read_csv.side_effect = [mock_header_df, mock_reader_instance]

        df, total_rows_loaded = self.loader.load_mimic_table(
            '/fake/large_file_subjects.csv',
            target_subject_ids=target_ids,
            use_dask=False
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3) # (101 from chunk1), (103 from chunk1), (101 from chunk2)
        self.assertEqual(total_rows_loaded, 3)
        self.assertTrue(all(sid in target_ids for sid in df[SUBJECT_ID_COL].unique()))

        self.assertEqual(mock_read_csv.call_count, 2)
        mock_read_csv.assert_any_call('/fake/large_file_subjects.csv', nrows=0, encoding='latin-1', compression=None, low_memory=False)
        mock_read_csv.assert_any_call('/fake/large_file_subjects.csv', chunksize=100000, encoding='latin-1', compression=None, low_memory=False)


    @patch('os.path.getsize', return_value=10 * 1024 * 1024) # Small file
    @patch('mimic_iv_analysis.core.data_loader.dd.read_csv')
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv') # For header and dask sample read
    def test_load_mimic_table_dask_small_file(self, mock_pd_read_csv, mock_dd_read_csv, mock_getsize):
        mock_pandas_df = pd.DataFrame({'colA': [1, 2, 3], SUBJECT_ID_COL: [101, 102, 103], 'time_col': ['2020-01-01', '2020-01-02', '2020-01-03']})
        mock_dask_df = dd.from_pandas(mock_pandas_df, npartitions=1)

        # Mock for pd.read_csv (header and sample for Dask)
        mock_header_df = pd.DataFrame(columns=mock_pandas_df.columns)
        mock_sample_df_for_dask = mock_pandas_df.head(5)
        mock_pd_read_csv.side_effect = [mock_header_df, mock_sample_df_for_dask]

        mock_dd_read_csv.return_value = mock_dask_df

        df, total_rows = self.loader.load_mimic_table('/fake/dask_small.csv', sample_size=100, use_dask=True)

        self.assertIsInstance(df, pd.DataFrame) # Dask result is computed
        self.assertEqual(len(df), 3)
        self.assertEqual(total_rows, 3)

        mock_pd_read_csv.assert_any_call('/fake/dask_small.csv', nrows=0, encoding='latin-1', compression=None, low_memory=False)
        mock_pd_read_csv.assert_any_call('/fake/dask_small.csv', nrows=5, encoding='latin-1', compression=None, low_memory=False)
        mock_dd_read_csv.assert_called_once_with(
            '/fake/dask_small.csv',
            encoding='latin-1', compression=None, low_memory=False,
            dtype={'time_col': 'object'} # Dask should try to infer and handle datetime-like cols
        )

    @patch('os.path.getsize', return_value=200 * 1024 * 1024) # Large file
    @patch('mimic_iv_analysis.core.data_loader.dd.read_csv')
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv') # For header and dask sample read
    def test_load_mimic_table_dask_large_file_with_sampling(self, mock_pd_read_csv, mock_dd_read_csv, mock_getsize):
        # Simulate a Dask DataFrame that would be larger than sample_size
        # The actual data doesn't matter as much as the mocked behavior of Dask's head()
        mock_full_pandas_df = pd.DataFrame({'colA': range(2000), SUBJECT_ID_COL: range(2000), 'date_col': pd.date_range('2020-01-01', periods=2000)})
        mock_dask_df = dd.from_pandas(mock_full_pandas_df, npartitions=2)

        mock_header_df = pd.DataFrame(columns=mock_full_pandas_df.columns)
        mock_sample_df_for_dask = mock_full_pandas_df.head(5)
        mock_pd_read_csv.side_effect = [mock_header_df, mock_sample_df_for_dask]

        mock_dd_read_csv.return_value = mock_dask_df

        sample_size_val = 50
        df, total_rows = self.loader.load_mimic_table('/fake/dask_large.csv', sample_size=sample_size_val, use_dask=True)

        self.assertIsInstance(df, pd.DataFrame) # Dask head() computes
        self.assertEqual(len(df), sample_size_val)
        self.assertEqual(total_rows, 2000) # Total rows from Dask df.shape[0].compute()

        mock_dd_read_csv.assert_called_once_with(
            '/fake/dask_large.csv',
            encoding='latin-1', compression=None, low_memory=False,
            dtype={'date_col': 'object'}
        )

    @patch('os.path.getsize', return_value=200 * 1024 * 1024) # Large file
    @patch('mimic_iv_analysis.core.data_loader.dd.read_csv')
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv') # For header
    def test_load_mimic_table_dask_large_file_with_target_ids(self, mock_pd_read_csv, mock_dd_read_csv, mock_getsize):
        target_ids = [10, 15]
        # Mock Dask DataFrame that, when filtered, returns a specific pandas DataFrame
        mock_pandas_data = pd.DataFrame({SUBJECT_ID_COL: [5, 10, 15, 20], 'value': ['a', 'b', 'c', 'd'], 'another_time': ['t1','t2','t3','t4']})

        # This is what filtered_ddf.compute() should return
        mock_computed_filtered_df = mock_pandas_data[mock_pandas_data[SUBJECT_ID_COL].isin(target_ids)]

        # Mock the Dask read_csv to return a Dask DataFrame
        # We need to mock the Dask DataFrame's methods too if they are complex
        mock_dask_df_instance = MagicMock(spec=dd.DataFrame)
        mock_dask_df_instance.columns = mock_pandas_data.columns

        # Mock the filtering and compute call
        # ddf[ddf[SUBJECT_ID_COL].isin(target_subject_ids)]
        mock_filtered_dask_df = MagicMock(spec=dd.DataFrame)
        mock_filtered_dask_df.compute.return_value = mock_computed_filtered_df

        # Make the __getitem__ on the dask_df_instance return our mock_filtered_dask_df
        def dask_getitem_side_effect(condition):
            # Rudimentary check if it's the subject_id filter
            if isinstance(condition, pd.Series) and condition.name == SUBJECT_ID_COL:
                 return mock_filtered_dask_df
            raise NotImplementedError("Dask __getitem__ mock not handling this condition")

        mock_dask_df_instance.__getitem__ = MagicMock(side_effect=dask_getitem_side_effect)

        mock_dd_read_csv.return_value = mock_dask_df_instance

        # Mock for pd.read_csv (header)
        mock_header_df = pd.DataFrame(columns=mock_pandas_data.columns)
        mock_pd_read_csv.return_value = mock_header_df


        df, total_rows_loaded = self.loader.load_mimic_table(
            '/fake/dask_large_subjects.csv',
            target_subject_ids=target_ids,
            use_dask=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(df, mock_computed_filtered_df)
        self.assertEqual(total_rows_loaded, len(mock_computed_filtered_df))

        mock_pd_read_csv.assert_called_once_with('/fake/dask_large_subjects.csv', nrows=0, encoding='latin-1', compression=None, low_memory=False)
        mock_dd_read_csv.assert_called_once_with(
            '/fake/dask_large_subjects.csv',
            dtype={SUBJECT_ID_COL: 'Int64', 'another_time': 'object'}, # Dask should try to infer dtypes
            encoding='latin-1', compression=None, low_memory=False
        )
        # Check that the Dask filtering was attempted
        mock_dask_df_instance.__getitem__.assert_called()


    def test_get_table_info(self):
        self.assertEqual(self.loader.get_table_description('hosp', 'admissions'), "Patient hospital admissions information")
        self.assertEqual(self.loader.get_table_description('icu', 'chartevents'), "Patient charting data (vital signs, etc.)")
        self.assertEqual(self.loader.get_table_description('non_existent_module', 'some_table'), "No description available")


    @patch('os.path.exists')
    @patch('mimic_iv_analysis.core.data_loader.pd.read_csv')
    def test_load_reference_table_success(self, mock_read_csv, mock_exists):
        mock_exists.side_effect = lambda p: p == '/mimic_path/hosp/d_items.csv' # Only uncompressed exists
        mock_df = pd.DataFrame({'itemid': [1,2], 'label': ['Label1', 'Label2']})
        mock_read_csv.return_value = mock_df

        df = self.loader.load_reference_table('/mimic_path', 'hosp', 'd_items')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        mock_read_csv.assert_called_once_with('/mimic_path/hosp/d_items.csv', encoding='latin-1', compression=None)

    @patch('os.path.exists', return_value=False)
    def test_load_reference_table_not_found(self, mock_exists):
        df = self.loader.load_reference_table('/mimic_path', 'hosp', 'non_existent_ref')
        self.assertTrue(df.empty)

    @patch.object(DataLoader, 'load_mimic_table')
    def test_apply_filters_no_filters_enabled(self, mock_load_mimic_table):
        df_input = pd.DataFrame({'a': [1]})
        filter_params = {'apply_age_range': False} # No filters actually on

        df_output = self.loader.apply_filters(df_input, filter_params, mimic_path='/fake/path')

        self.mock_filtering_instance.apply_filters.assert_not_called()
        mock_load_mimic_table.assert_not_called() # No auxiliary tables should be loaded
        pd.testing.assert_frame_equal(df_output, df_input)

    @patch('os.path.exists')
    @patch.object(DataLoader, 'load_mimic_table')
    def test_apply_filters_loads_auxiliary_tables(self, mock_load_mimic_table_dl, mock_os_exists):
        df_input = pd.DataFrame({'subject_id': [1, 2], 'hadm_id': [101, 102]})
        filter_params = {
            'apply_age_range': True, 'min_age': 30, 'max_age': 60, # Needs patients
            'apply_valid_admission_discharge': True, # Needs admissions
            'apply_t2dm_diagnosis': True # Needs diagnoses_icd
        }

        # Mock os.path.exists to control which files are "found"
        def os_exists_side_effect(path):
            if 'patients.csv.gz' in path: return True
            if 'admissions.csv.gz' in path: return True
            if 'diagnoses_icd.csv.gz' in path: return True
            return False
        mock_os_exists.side_effect = os_exists_side_effect

        # Mock return values for loaded auxiliary tables
        mock_patients_df = pd.DataFrame({'subject_id': [1, 2, 3], 'anchor_age': [25, 35, 65]})
        mock_admissions_df = pd.DataFrame({'subject_id': [1,2], 'hadm_id': [101,102], 'admittime': pd.to_datetime(['2020-01-01']), 'dischtime': pd.to_datetime(['2020-01-05'])})
        mock_diagnoses_df = pd.DataFrame({'subject_id': [1], 'hadm_id': [101], 'icd_code': ['250.00'], 'seq_num': [1]})

        # The load_mimic_table on DataLoader instance will be called for aux tables
        def load_mimic_table_side_effect(file_path, sample_size=None, max_chunks=None, **kwargs):
            if 'patients' in file_path: return mock_patients_df, len(mock_patients_df)
            if 'admissions' in file_path: return mock_admissions_df, len(mock_admissions_df)
            if 'diagnoses_icd' in file_path: return mock_diagnoses_df, len(mock_diagnoses_df)
            return pd.DataFrame(), 0
        mock_load_mimic_table_dl.side_effect = load_mimic_table_side_effect

        # Mock the actual filtering logic
        mock_filtered_df_result = pd.DataFrame({'subject_id': [2], 'hadm_id': [102]}) # Dummy result
        self.mock_filtering_instance.apply_filters.return_value = mock_filtered_df_result, ["age_filter"]


        df_output = self.loader.apply_filters(df_input.copy(), filter_params, mimic_path='/fake/mimic_path')

        # Check that load_mimic_table was called for patients, admissions, diagnoses_icd
        expected_calls_load_mimic = [
            call(os.path.join('/fake/mimic_path', 'hosp', 'patients.csv.gz'), sample_size=None, max_chunks=-1),
            call(os.path.join('/fake/mimic_path', 'hosp', 'admissions.csv.gz')), # Default sample_size, max_chunks
            call(os.path.join('/fake/mimic_path', 'hosp', 'diagnoses_icd.csv.gz'))
        ]
        # Allow any order for these calls as their loading order isn't strictly sequential here
        mock_load_mimic_table_dl.assert_has_calls(expected_calls_load_mimic, any_order=True)

        # Check that filtering.apply_filters was called with the loaded aux tables
        # Need to use ANY or a more sophisticated way to check DataFrame equality in mock calls if needed
        self.mock_filtering_instance.apply_filters.assert_called_once()
        args_call = self.mock_filtering_instance.apply_filters.call_args[1] # Get kwargs

        pd.testing.assert_frame_equal(args_call['df'], df_input)
        self.assertEqual(args_call['filter_params'], filter_params)
        pd.testing.assert_frame_equal(args_call['patients_df'], mock_patients_df)
        pd.testing.assert_frame_equal(args_call['admissions_df'], mock_admissions_df)
        pd.testing.assert_frame_equal(args_call['diagnoses_df'], mock_diagnoses_df)
        self.assertIsNone(args_call['transfers_df']) # Not needed by these filters

        pd.testing.assert_frame_equal(df_output, mock_filtered_df_result)

if __name__ == '__main__':
    unittest.main()

