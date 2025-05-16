import pytest
import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from mimic_iv_analysis.io.data_loader import (
    DataLoader, TableNamesHOSP, TableNamesICU,
    DEFAULT_MIMIC_PATH, table_names_enum_converter
)

class TestDataLoader:
    """Test suite for the DataLoader class."""

    @pytest.fixture
    def mock_mimic_dir(self):
        """Create a temporary directory with fake MIMIC data structure."""
        temp_dir = tempfile.mkdtemp()

        # Create module directories
        hosp_dir = Path(temp_dir) / "hosp"
        icu_dir = Path(temp_dir) / "icu"
        hosp_dir.mkdir()
        icu_dir.mkdir()

        # Create sample CSV files
        patients_file = hosp_dir / "patients.csv"
        admissions_file = hosp_dir / "admissions.csv"
        icustays_file = icu_dir / "icustays.csv"

        # Sample patients data
        patients_data = pd.DataFrame({
            'subject_id': [1, 2, 3],
            'gender': ['M', 'F', 'M'],
            'anchor_age': [45, 32, 67],
            'anchor_year': [2120, 2130, 2125]
        })
        patients_data.to_csv(patients_file, index=False)

        # Sample admissions data
        admissions_data = pd.DataFrame({
            'subject_id': [1, 1, 2, 3],
            'hadm_id': [100, 101, 102, 103],
            'admittime': ['2100-01-01', '2101-02-15', '2110-03-22', '2115-05-10'],
            'dischtime': ['2100-01-10', '2101-02-25', '2110-04-01', '2115-05-20'],
            'admission_type': ['emergency', 'elective', 'urgent', 'emergency']
        })
        admissions_data.to_csv(admissions_file, index=False)

        # Sample ICU stays data
        icustays_data = pd.DataFrame({
            'subject_id': [1, 2, 3],
            'hadm_id': [100, 102, 103],
            'stay_id': [1000, 1001, 1002],
            'intime': ['2100-01-02', '2110-03-23', '2115-05-11'],
            'outtime': ['2100-01-08', '2110-03-30', '2115-05-18']
        })
        icustays_data.to_csv(icustays_file, index=False)

        yield Path(temp_dir)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_init(self):
        """Test the initialization of DataLoader."""
        loader = DataLoader()
        assert loader.mimic_path == DEFAULT_MIMIC_PATH
        assert loader.study_table_list == DataLoader.DEFAULT_STUDY_TABLES_LIST
        assert loader._all_subject_ids == []
        assert loader.tables_info_df is None
        assert loader.tables_info_dict is None

        # Test custom params
        custom_path = Path("/custom/path")
        custom_tables = [TableNamesHOSP.PATIENTS, TableNamesHOSP.ADMISSIONS]
        loader = DataLoader(mimic_path=custom_path, study_tables_list=custom_tables)
        assert loader.mimic_path == custom_path
        assert loader.study_table_list == custom_tables

    def test_scan_mimic_directory(self, mock_mimic_dir):
        """Test scanning the MIMIC directory structure."""
        loader = DataLoader(mimic_path=mock_mimic_dir)
        loader.scan_mimic_directory()

        # Verify tables_info_df is populated
        assert loader.tables_info_df is not None
        assert not loader.tables_info_df.empty

        # Check that all expected tables are found
        table_names = loader.tables_info_df['table_name'].tolist()
        assert 'patients' in table_names
        assert 'admissions' in table_names
        assert 'icustays' in table_names

        # Check that file paths are correct
        patients_path = loader.tables_info_df[loader.tables_info_df['table_name'] == 'patients']['file_path'].iloc[0]
        assert str(patients_path).endswith('hosp/patients.csv')

        # Check that module info is correct
        modules = loader.tables_info_df['module'].unique().tolist()
        assert 'hosp' in modules
        assert 'icu' in modules

    def test_study_tables_info(self, mock_mimic_dir):
        """Test retrieving study tables info."""
        loader = DataLoader(
            mimic_path=mock_mimic_dir,
            study_tables_list=[TableNamesHOSP.PATIENTS, TableNamesHOSP.ADMISSIONS]
        )

        # The method should trigger scanning if not done already
        study_info = loader.study_tables_info

        assert not study_info.empty
        assert len(study_info) == 2
        assert set(study_info['table_name'].tolist()) == {'patients', 'admissions'}

    def test_get_file_path(self, mock_mimic_dir):
        """Test retrieving file path for a table."""
        loader = DataLoader(mimic_path=mock_mimic_dir)
        loader.scan_mimic_directory()

        path = loader._get_file_path(TableNamesHOSP.PATIENTS)
        assert path.name == 'patients.csv'
        assert 'hosp' in str(path)

        path = loader._get_file_path(TableNamesICU.ICUSTAYS)
        assert path.name == 'icustays.csv'
        assert 'icu' in str(path)

    # @patch('mimic_iv_analysis.io.data_loader.dd.read_csv')
    # def test_load_csv_table_with_correct_column_datatypes(self, mock_read_csv, mock_mimic_dir):
    #     """Test loading CSV with correct datatypes."""
    #     # Setup mock
    #     mock_df = MagicMock(spec=dd.DataFrame)
    #     mock_read_csv.return_value = mock_df

    #     loader = DataLoader(mimic_path=mock_mimic_dir)
    #     file_path = mock_mimic_dir / "hosp" / "patients.csv"

    #     result = loader.load_csv_table_with_correct_column_datatypes(file_path)

    #     # Verify the mock was called
    #     mock_read_csv.assert_called_once()
    #     assert result == mock_df

    #     # Test file not found
    #     with pytest.raises(FileNotFoundError):
    #         loader.load_csv_table_with_correct_column_datatypes(Path("nonexistent.csv"))

    #     # Test non-CSV file
    #     non_csv = mock_mimic_dir / "not_a_csv.txt"
    #     with open(non_csv, 'w') as f:
    #         f.write("not a csv")

    #     with pytest.warns(UserWarning):
    #         result = loader.load_csv_table_with_correct_column_datatypes(non_csv)
    #         assert result.empty

    @patch('mimic_iv_analysis.io.data_loader.DataLoader.load_table')
    def test_load_unique_subject_ids_for_table(self, mock_load_table, mock_mimic_dir):
        """Test loading unique subject IDs from a table."""
        # Setup mock
        mock_df = MagicMock(spec=dd.DataFrame)
        mock_compute = MagicMock()
        mock_compute.tolist.return_value = [1, 2, 3]
        mock_unique = MagicMock()
        mock_unique.compute.return_value = mock_compute
        mock_df.__getitem__.return_value.unique.return_value = mock_unique
        mock_load_table.return_value = mock_df

        loader = DataLoader(mimic_path=mock_mimic_dir)

        # Test with default table
        result = loader._load_unique_subject_ids_for_table()
        assert result == [1, 2, 3]
        mock_load_table.assert_called_with(table_name=TableNamesHOSP.ADMISSIONS, partial_loading=False)

        # Test with custom table
        mock_load_table.reset_mock()
        result = loader._load_unique_subject_ids_for_table(TableNamesHOSP.PATIENTS)
        assert result == [1, 2, 3]
        mock_load_table.assert_called_with(table_name=TableNamesHOSP.PATIENTS, partial_loading=False)

    # def test_all_subject_ids(self, mock_mimic_dir):
    #     """Test all_subject_ids property."""
    #     loader = DataLoader(mimic_path=mock_mimic_dir)

    #     # Test that the property loads IDs if not already loaded
    #     with patch.object(loader, '_load_unique_subject_ids_for_table') as mock_load:
    #         mock_load.return_value = [1, 2, 3]
    #         assert loader.all_subject_ids == [1, 2, 3]
    #         mock_load.assert_called_once()

    #     # Test that the property uses cached IDs if already loaded
    #     loader._all_subject_ids = [4, 5, 6]
    #     with patch.object(loader, '_load_unique_subject_ids_for_table') as mock_load:
    #         assert loader.all_subject_ids == [4, 5, 6]
    #         mock_load.assert_not_called()

    def test_get_partial_subject_id_list(self):
        """Test getting partial subject ID list."""
        loader = DataLoader()
        loader._all_subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test with num_subjects = 0
        result = loader.get_partial_subject_id_list_for_partial_loading(num_subjects=0)
        assert result == []

        # Test with num_subjects > len(all_subject_ids)
        result = loader.get_partial_subject_id_list_for_partial_loading(num_subjects=20)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test with non-random selection
        result = loader.get_partial_subject_id_list_for_partial_loading(num_subjects=3, random_selection=False)
        assert result == [1, 2, 3]

        # Test with random selection
        with patch('numpy.random.choice') as mock_choice:
            mock_choice.return_value = np.array([3, 1, 7])
            result = loader.get_partial_subject_id_list_for_partial_loading(num_subjects=3, random_selection=True)
            assert result == [3, 1, 7]
            mock_choice.assert_called_with(a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=3)

    @patch('mimic_iv_analysis.io.data_loader.DataLoader.load_table')
    @patch('mimic_iv_analysis.io.data_loader.DataLoader.get_partial_subject_id_list_for_partial_loading')
    def test_load_all_study_tables(self, mock_get_ids, mock_load_table, mock_mimic_dir):
        """Test loading all study tables."""
        # Setup mocks
        mock_get_ids.return_value = [1, 2, 3]
        mock_df = MagicMock(spec=dd.DataFrame)
        mock_load_table.return_value = mock_df

        # Create a minimal DataLoader with just patients table for testing
        loader = DataLoader(
            mimic_path=mock_mimic_dir,
            study_tables_list=[TableNamesHOSP.PATIENTS]
        )

        # Mock study_tables_info
        loader.tables_info_df = pd.DataFrame({
            'module': ['hosp'],
            'table_name': ['patients'],
            'file_path': [mock_mimic_dir / 'hosp' / 'patients.csv'],
            'file_size': [1000],
            'display_name': ['patients 1KB'],
            'suffix': ['.csv'],
            'columns_list': [{'subject_id', 'gender', 'anchor_age', 'anchor_year'}]
        })

        # Test without partial loading
        result = loader.load_all_study_tables(partial_loading=False)
        assert 'patients' in result
        assert result['patients'] == mock_df
        mock_load_table.assert_called_with(
            table_name=TableNamesHOSP.PATIENTS,
            partial_loading=False,
            subject_ids=None,
            use_dask=True
        )

        # Test with partial loading
        mock_load_table.reset_mock()
        result = loader.load_all_study_tables(partial_loading=True, num_subjects=3)
        assert 'patients' in result
        mock_get_ids.assert_called_with(num_subjects=3, random_selection=False)
        mock_load_table.assert_called_with(
            table_name=TableNamesHOSP.PATIENTS,
            partial_loading=True,
            subject_ids=[1, 2, 3],
            use_dask=True
        )

        # Test with use_dask=False
        mock_load_table.reset_mock()
        result = loader.load_all_study_tables(partial_loading=False, use_dask=False)
        assert 'patients' in result
        mock_load_table.assert_called_with(
            table_name=TableNamesHOSP.PATIENTS,
            partial_loading=False,
            subject_ids=None,
            use_dask=False
        )

    # @patch('mimic_iv_analysis.io.data_loader.Filtering')
    # def test_load_table(self, mock_filtering_class, mock_mimic_dir):
    #     """Test loading a table."""
    #     # Setup mocks
    #     mock_filtering = MagicMock()
    #     mock_filtering.render.return_value = dd.from_pandas(
    #         pd.DataFrame({'subject_id': [1, 2, 3], 'column': ['a', 'b', 'c']}),
    #         npartitions=1
    #     )
    #     mock_filtering_class.return_value = mock_filtering

    #     # Create loader with mock mimic dir
    #     loader = DataLoader(mimic_path=mock_mimic_dir)

    #     # Mock _get_file_path and load_csv_table_with_correct_column_datatypes
    #     with patch.object(loader, '_get_file_path') as mock_get_path, \
    #          patch.object(loader, 'load_csv_table_with_correct_column_datatypes') as mock_load_csv:

    #         # Setup mock return values
    #         file_path = mock_mimic_dir / 'hosp' / 'patients.csv'
    #         mock_get_path.return_value = file_path

    #         # Create test DataFrame
    #         df = dd.from_pandas(
    #             pd.DataFrame({'subject_id': [1, 2, 3, 4, 5], 'column': ['a', 'b', 'c', 'd', 'e']}),
    #             npartitions=1
    #         )
    #         mock_load_csv.return_value = df

    #         # Test full loading
    #         result = loader.load_table(TableNamesHOSP.PATIENTS, partial_loading=False)
    #         mock_get_path.assert_called_with(table_name=TableNamesHOSP.PATIENTS)
    #         mock_load_csv.assert_called_with(file_path)
    #         mock_filtering_class.assert_called_with(df=df, table_name=TableNamesHOSP.PATIENTS)

    #         # Test partial loading with subject_ids
    #         mock_get_path.reset_mock()
    #         mock_load_csv.reset_mock()
    #         mock_filtering_class.reset_mock()
    #         subject_ids = [1, 3]

    #         result = loader.load_table(
    #             TableNamesHOSP.PATIENTS,
    #             partial_loading=True,
    #             subject_ids=subject_ids
    #         )

    #         # Check result has only the selected subject_ids
    #         result_df = result.compute()
    #         assert set(result_df['subject_id'].tolist()) == set(subject_ids)

    #         # Test partial loading with sample_size
    #         mock_get_path.reset_mock()
    #         mock_load_csv.reset_mock()
    #         mock_filtering_class.reset_mock()

    #         with patch.object(mock_filtering.render.return_value, 'head') as mock_head:
    #             mock_head.return_value = "sample_df"
    #             result = loader.load_table(
    #                 TableNamesHOSP.PATIENTS,
    #                 partial_loading=True,
    #                 sample_size=2
    #             )
    #             mock_head.assert_called_with(2, compute=False)

    def test_load_with_pandas_chunking(self):
        """Test loading with pandas chunking."""
        # This is a more complex test requiring mocking pd.read_csv with a context manager
        # For brevity, we'll implement a simplified version

        # Create a test CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_csv_path = Path(f.name)

        try:
            # Write test data
            pd.DataFrame({
                'subject_id': [1, 2, 3, 1, 2, 4],
                'value': ['a', 'b', 'c', 'd', 'e', 'f']
            }).to_csv(test_csv_path, index=False)

            loader = DataLoader()

            # Test filtering by subject_ids
            target_subject_ids = [1, 3]
            result_df, row_count = loader.load_with_pandas_chunking(
                file_path=test_csv_path,
                target_subject_ids=target_subject_ids,
                read_params={'dtype': {'subject_id': int}}
            )

            # Check results
            assert row_count == 3  # 2 rows for subject_id 1, 1 row for subject_id 3
            assert set(result_df['subject_id'].tolist()) == set(target_subject_ids)
            assert len(result_df) == 3

            # Test with max_chunks
            result_df, row_count = loader.load_with_pandas_chunking(
                file_path=test_csv_path,
                target_subject_ids=target_subject_ids,
                max_chunks=1,
                read_params={'dtype': {'subject_id': int}}
            )

            # With a small file, we should still get same results
            assert row_count == 3

        finally:
            # Clean up
            test_csv_path.unlink()

    # @patch('mimic_iv_analysis.io.data_loader.DataLoader.load_all_study_tables')
    # def test_merge_tables(self, mock_load_all_tables, mock_mimic_dir):
    #     """Test merging tables."""
    #     # Setup mocks for all required tables
    #     patients_df = dd.from_pandas(
    #         pd.DataFrame({'subject_id': [1, 2, 3]}),
    #         npartitions=1
    #     )

    #     admissions_df = dd.from_pandas(
    #         pd.DataFrame({
    #             'subject_id': [1, 2, 3],
    #             'hadm_id': [100, 101, 102]
    #         }),
    #         npartitions=1
    #     )

    #     diagnoses_icd_df = dd.from_pandas(
    #         pd.DataFrame({
    #             'subject_id': [1, 2, 3],
    #             'hadm_id': [100, 101, 102],
    #             'icd_code': ['A', 'B', 'C'],
    #             'icd_version': [9, 9, 9]
    #         }),
    #         npartitions=1
    #     )

    #     d_icd_diagnoses_df = dd.from_pandas(
    #         pd.DataFrame({
    #             'icd_code': ['A', 'B', 'C'],
    #             'icd_version': [9, 9, 9],
    #             'long_title': ['Disease A', 'Disease B', 'Disease C']
    #         }),
    #         npartitions=1
    #     )

    #     poe_df = dd.from_pandas(
    #         pd.DataFrame({
    #             'subject_id': [1, 2, 3],
    #             'hadm_id': [100, 101, 102],
    #             'poe_id': ['p1', 'p2', 'p3'],
    #             'poe_seq': [1, 1, 1]
    #         }),
    #         npartitions=1
    #     )

    #     poe_detail_df = dd.from_pandas(
    #         pd.DataFrame({
    #             'subject_id': [1, 2],  # Missing subject_id 3 to test left join
    #             'hadm_id': [100, 101],
    #             'poe_id': ['p1', 'p2'],
    #             'poe_seq': [1, 1],
    #             'detail': ['d1', 'd2']
    #         }),
    #         npartitions=1
    #     )

    #     # Set up the mock return value
    #     mock_load_all_tables.return_value = {
    #         'patients': patients_df,
    #         'admissions': admissions_df,
    #         'diagnoses_icd': diagnoses_icd_df,
    #         'd_icd_diagnoses': d_icd_diagnoses_df,
    #         'poe': poe_df,
    #         'poe_detail': poe_detail_df
    #     }

    #     loader = DataLoader(mimic_path=mock_mimic_dir)

    #     # Call load_merged_tables and check result
    #     result = loader.load_merged_tables()

    #     # Assert load_all_study_tables was called with correct tables
    #     mock_load_all_tables.assert_called_once()

    #     # Check results - we should have three dictionaries
    #     assert 'merged_wo_poe' in result
    #     assert 'merged_w_poe' in result
    #     assert 'poe_merged' in result

    #     # Test additional aspects of merges
    #     # For brevity, let's just check a few basic aspects that would indicate
    #     # the merge is working correctly

    #     # Check poe_merged has left join behavior
    #     poe_merged = result['poe_merged'].compute()
    #     assert len(poe_merged) == 3  # Should keep all rows from poe
    #     assert poe_merged[poe_merged['subject_id'] == 3]['detail'].isna().all()  # Missing subject should have NaN detail

    def test_table_names_enum_converter(self):
        """Test the table_names_enum_converter function."""
        # Test hosp module
        result = table_names_enum_converter('patients', 'hosp')
        assert result == TableNamesHOSP.PATIENTS

        # Test icu module
        result = table_names_enum_converter('icustays', 'icu')
        assert result == TableNamesICU.ICUSTAYS

        # Test invalid table name
        with pytest.raises(ValueError):
            table_names_enum_converter('invalid_table', 'hosp')
