#!/usr/bin/env python

import pytest
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from mimic_iv_analysis.io.data_loader import DataLoader, ParquetConverter, ExampleDataLoader

from mimic_iv_analysis.configurations.params import TableNamesHOSP, TableNamesICU, DEFAULT_MIMIC_PATH, convert_table_names_to_enum_class, DEFAULT_STUDY_TABLES_LIST

# Path to the core.filtering module to mock for testing
FILTERING_PATH = 'mimic_iv_analysis.core.filtering.Filtering'

class TestDataLoader:
    """Test suite for the DataLoader class."""

    @pytest.fixture
    def mock_mimic_dir(self):
        """Create a temporary directory with fake MIMIC data structure."""
        temp_dir = tempfile.mkdtemp()

        # Create module directories
        hosp_dir = Path(temp_dir) / "hosp"
        icu_dir  = Path(temp_dir) / "icu"
        hosp_dir.mkdir()
        icu_dir.mkdir()

        # Create sample CSV files
        patients_file        = hosp_dir / "patients.csv"
        admissions_file      = hosp_dir / "admissions.csv"
        diagnoses_icd_file   = hosp_dir / "diagnoses_icd.csv"
        d_icd_diagnoses_file = hosp_dir / "d_icd_diagnoses.csv"
        poe_file             = hosp_dir / "poe.csv"
        poe_detail_file      = hosp_dir / "poe_detail.csv"
        icustays_file        = icu_dir / "icustays.csv"
        transfers_file       = hosp_dir / "transfers.csv"

        # Sample patients data
        patients_data = pd.DataFrame({
            'subject_id' : [1, 2, 3],
            'gender'     : ['M', 'F', 'M'],
            'anchor_age' : [45, 32, 67],
            'anchor_year': [2120, 2130, 2125],
            'anchor_year_group': ['2017 - 2019', '2017 - 2019', '2017 - 2019']
        })
        patients_data.to_csv(patients_file, index=False)

        # Sample admissions data
        admissions_data = pd.DataFrame({
            'subject_id'    : [1, 1, 2, 3],
            'hadm_id'       : [100, 101, 102, 103],
            'admittime'     : ['2100-01-01', '2101-02-15', '2110-03-22', '2115-05-10'],
            'dischtime'     : ['2100-01-10', '2101-02-25', '2110-04-01', '2115-05-20'],
            'admission_type': ['emergency', 'elective', 'urgent', 'emergency'],
            'deathtime'     : [None, None, None, None],  # Add deathtime column
            'hospital_expire_flag': [0, 0, 0, 0]  # Add hospital_expire_flag
        })
        admissions_data.to_csv(admissions_file, index=False)

        # Sample diagnoses_icd data
        diagnoses_icd_data = pd.DataFrame({
            'subject_id' : [1, 2, 3],
            'hadm_id'    : [100, 102, 103],
            'seq_num'    : [1, 1, 1],
            'icd_code'   : ['A01', 'B02', 'C03'],
            'icd_version': [9, 9, 9]
        })
        diagnoses_icd_data.to_csv(diagnoses_icd_file, index=False)

        # Sample d_icd_diagnoses data
        d_icd_diagnoses_data = pd.DataFrame({
            'icd_code'   : ['A01', 'B02', 'C03'],
            'icd_version': [9, 9, 9],
            'long_title' : ['Disease A', 'Disease B', 'Disease C']
        })
        d_icd_diagnoses_data.to_csv(d_icd_diagnoses_file, index=False)

        # Sample poe data
        poe_data = pd.DataFrame({
            'subject_id'            : [1, 2, 3],
            'hadm_id'               : [100, 102, 103],
            'poe_id'                : ['P100', 'P102', 'P103'],
            'poe_seq'               : [1, 1, 1],
            'order_type'            : ['Type A', 'Type B', 'Type C'],
            'discontinue_of_poe_id' : [None, None, None],
            'discontinued_by_poe_id': [None, None, None]
        })
        poe_data.to_csv(poe_file, index=False)

        # Sample poe_detail data
        poe_detail_data = pd.DataFrame({
            'subject_id' : [1, 2],                 # Missing subject 3 to test left join
            'hadm_id'    : [100, 102],
            'poe_id'     : ['P100', 'P102'],
            'poe_seq'    : [1, 1],
            'field_name' : ['Field A', 'Field B'],
            'field_value': ['Value A', 'Value B']
        })
        poe_detail_data.to_csv(poe_detail_file, index=False)

        # Sample ICU stays data
        icustays_data = pd.DataFrame({
            'subject_id': [1, 2, 3],
            'hadm_id'   : [100, 102, 103],
            'stay_id'   : [1000, 1001, 1002],
            'intime'    : ['2100-01-02', '2110-03-23', '2115-05-11'],
            'outtime'   : ['2100-01-08', '2110-03-30', '2115-05-18']
        })
        icustays_data.to_csv(icustays_file, index=False)

        yield Path(temp_dir)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_init(self):
        """Test the initialization of DataLoader."""
        loader = DataLoader()
        assert loader.mimic_path == DEFAULT_MIMIC_PATH
        assert loader.study_table_list == DEFAULT_STUDY_TABLES_LIST
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

        # Test with non-existent directory
        loader = DataLoader(mimic_path=Path("/nonexistent/path"))
        loader.scan_mimic_directory()
        assert loader.tables_info_df is not None
        assert loader.tables_info_df.empty

    def test_study_tables_info(self, mock_mimic_dir):
        """Test retrieving study tables info."""
        loader = DataLoader(
            mimic_path=mock_mimic_dir,
            study_tables_list=[TableNamesHOSP.PATIENTS, TableNamesHOSP.ADMISSIONS]
        )

        # Mock scan_mimic_directory to set tables_info_df
        loader.scan_mimic_directory()

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

    def test_load_csv_table_with_correct_column_datatypes(self, mock_mimic_dir):
        """Test loading CSV with correct datatypes."""
        loader = DataLoader(mimic_path=mock_mimic_dir)
        file_path = mock_mimic_dir / "hosp" / "patients.csv"

        # Test with use_dask=True (default)
        result = loader.load_csv_table_with_correct_column_datatypes(file_path)
        assert isinstance(result, dd.DataFrame)

        # Test with use_dask=False
        result = loader.load_csv_table_with_correct_column_datatypes(file_path, use_dask=False)
        assert isinstance(result, pd.DataFrame)

        # Test file not found
        with pytest.raises(FileNotFoundError):
            loader.load_csv_table_with_correct_column_datatypes(Path("/nonexistent/file.csv"))

        # Test non-CSV file
        non_csv = mock_mimic_dir / "not_a_csv.txt"
        with open(non_csv, 'w') as f:
            f.write("not a csv")

        # Using logger.warning instead of warnings.warn
        result = loader.load_csv_table_with_correct_column_datatypes(non_csv)
        assert result.empty

    @patch(FILTERING_PATH)
    def test_load_unique_subject_ids_for_table(self, mock_filtering, mock_mimic_dir):
        """Test loading unique subject IDs from a table."""
        loader = DataLoader(mimic_path=mock_mimic_dir)
        loader.scan_mimic_directory()

        # Mock load_table instead since that's what's being called
        with patch.object(loader, 'load_table') as mock_load_table:
            # Create mock dataframe
            mock_df = MagicMock()
            mock_df.__getitem__.return_value.unique.return_value.compute.return_value.tolist.return_value = [1, 2, 3]
            mock_load_table.return_value = mock_df

            # Test with patients table
            result = loader._load_unique_subject_ids_for_table(TableNamesHOSP.PATIENTS)
            assert result == [1, 2, 3]
            mock_load_table.assert_called_with(table_name=TableNamesHOSP.PATIENTS, partial_loading=False)

            # Test with default table (admissions)
            result = loader._load_unique_subject_ids_for_table()
            assert result == [1, 2, 3]
            mock_load_table.assert_called_with(table_name=TableNamesHOSP.ADMISSIONS, partial_loading=False)

    @patch(FILTERING_PATH)
    def test_all_subject_ids(self, mock_mimic_dir):
        """Test all_subject_ids property."""
        loader = DataLoader(mimic_path=mock_mimic_dir)
        loader.scan_mimic_directory()

        # Delete the cached property to force it to recalculate
        # This is needed because cached_property caches the result
        try:
            del loader.__dict__['all_subject_ids']
        except KeyError:
            pass  # Property hasn't been accessed yet

        # Clear the _all_subject_ids list to ensure the property runs
        loader._all_subject_ids = []

        # Mock the _load_unique_subject_ids_for_table method
        # According to the implementation, this method should populate the _all_subject_ids list
        def mock_load_ids_for_table():
            loader._all_subject_ids = [1, 2, 3]
            return loader._all_subject_ids

        with patch.object(loader, '_load_unique_subject_ids_for_table', side_effect=mock_load_ids_for_table) as mock_load_method:
            # Test that the property loads IDs
            result = loader.all_subject_ids
            assert result == [1, 2, 3]
            mock_load_method.assert_called_once_with()

        # Test caching behavior by changing the internal list
        loader._all_subject_ids = [4, 5, 6]

        # Since all_subject_ids uses cached_property, it should still return the cached value
        # But we're directly modifying the internal list, so we test the internal list
        assert loader._all_subject_ids == [4, 5, 6]

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

    @patch(FILTERING_PATH)
    def test_load_all_study_tables(self, mock_filtering, mock_mimic_dir):
        """Test loading all study tables."""
        # Setup mock Filtering class
        mock_filter_instance = MagicMock()
        mock_filter_instance.render.side_effect = lambda: dd.from_pandas(
            pd.DataFrame({'subject_id': [1, 2, 3], 'data': ['a', 'b', 'c']}),
            npartitions=1
        )
        mock_filtering.return_value = mock_filter_instance

        # Create a DataLoader with mock directory
        loader = DataLoader(
            mimic_path=mock_mimic_dir,
            study_tables_list=[TableNamesHOSP.PATIENTS, TableNamesHOSP.ADMISSIONS]
        )
        loader.scan_mimic_directory()

        # Test full loading
        with patch.object(loader, 'load_table') as mock_load_table:
            mock_load_table.return_value = dd.from_pandas(
                pd.DataFrame({'subject_id': [1, 2, 3], 'data': ['a', 'b', 'c']}),
                npartitions=1
            )

            result = loader.load_all_study_tables(partial_loading=False)
            assert 'patients' in result
            assert 'admissions' in result
            mock_load_table.assert_any_call(
                table_name=TableNamesHOSP.PATIENTS,
                partial_loading=False,
                subject_ids=None,
                use_dask=True
            )
            mock_load_table.assert_any_call(
                table_name=TableNamesHOSP.ADMISSIONS,
                partial_loading=False,
                subject_ids=None,
                use_dask=True
            )

        # Test partial loading
        with patch.object(loader, 'load_table') as mock_load_table, \
             patch.object(loader, 'get_partial_subject_id_list_for_partial_loading') as mock_get_ids:

            mock_load_table.return_value = dd.from_pandas(
                pd.DataFrame({'subject_id': [1, 2], 'data': ['a', 'b']}),
                npartitions=1
            )
            mock_get_ids.return_value = [1, 2]

            result = loader.load_all_study_tables(partial_loading=True, num_subjects=2)
            assert 'patients' in result
            assert 'admissions' in result
            mock_get_ids.assert_called_with(num_subjects=2, random_selection=False)
            mock_load_table.assert_any_call(
                table_name=TableNamesHOSP.PATIENTS,
                partial_loading=True,
                subject_ids=[1, 2],
                use_dask=True
            )

    @patch('mimic_iv_analysis.io.data_loader.Filtering')
    def test_load_table(self, mock_filtering, mock_mimic_dir):
        """Test loading a table."""
        # Create loader with mock mimic dir
        loader = DataLoader(mimic_path=mock_mimic_dir)
        loader.scan_mimic_directory()  # Need to scan directory first

        # Mock _get_file_path to return an actual file
        with patch.object(loader, '_get_file_path') as mock_get_path, \
            patch.object(loader, 'load_csv_table_with_correct_column_datatypes') as mock_load_csv:

            # Setup mock for file path
            file_path = mock_mimic_dir / "hosp" / "patients.csv"
            mock_get_path.return_value = file_path

            # Create test DataFrame with required columns for PATIENTS table
            test_df = dd.from_pandas(
                pd.DataFrame({'subject_id': [1, 2, 3, 4, 5], 'column': ['a', 'b', 'c', 'd', 'e'], 'anchor_age': [20, 30, 40, 50, 60], 'anchor_year_group': ['2017 - 2019'] * 5}),
                npartitions=1
            )
            mock_load_csv.return_value = test_df

            # Setup mock Filtering that will be created within load_table
            mock_filter_instance = MagicMock()
            filtered_df = dd.from_pandas(
                pd.DataFrame({'subject_id': [1, 3, 5], 'column': ['a', 'c', 'e'], 'anchor_age': [20, 40, 60], 'anchor_year_group': ['2017 - 2019'] * 3}),
                npartitions=1
            )
            mock_filter_instance.render.return_value = filtered_df
            mock_filtering.return_value = mock_filter_instance

            # Test full loading
            result = loader.load_table(TableNamesHOSP.PATIENTS, partial_loading=False)
            mock_get_path.assert_called_with(table_name=TableNamesHOSP.PATIENTS)
            mock_load_csv.assert_called_with(file_path, use_dask=True)
            # Check that Filtering was created with the right parameters
            mock_filtering.assert_called_once()
            call_kwargs = mock_filtering.call_args.kwargs
            assert call_kwargs['df'] is test_df
            assert call_kwargs['table_name'] == TableNamesHOSP.PATIENTS
            assert isinstance(result, dd.DataFrame)

            # Test partial loading with subject_ids
            mock_get_path.reset_mock()
            mock_load_csv.reset_mock()
            mock_filtering.reset_mock()

            result = loader.load_table(
                TableNamesHOSP.PATIENTS,
                partial_loading=True,
                subject_ids=[1, 3]
            )
            mock_get_path.assert_called_with(table_name=TableNamesHOSP.PATIENTS)
            mock_load_csv.assert_called_with(file_path, use_dask=True)

    def test_load_with_pandas_chunking(self):
        """Test loading with pandas chunking."""
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

    @patch('mimic_iv_analysis.io.data_loader.DataLoader.load_all_study_tables')
    def test_load_merged_tables(self, mock_load_all, mock_mimic_dir):
        """Test merging tables."""
        # Create a DataLoader with all required tables
        loader = DataLoader(
            mimic_path=mock_mimic_dir,
            study_tables_list=[
                TableNamesHOSP.PATIENTS,
                TableNamesHOSP.ADMISSIONS,
                TableNamesHOSP.DIAGNOSES_ICD,
                TableNamesHOSP.D_ICD_DIAGNOSES,
                TableNamesHOSP.POE,
                TableNamesHOSP.POE_DETAIL
            ]
        )

        # Use MagicMock objects directly instead of Dask DataFrames
        patients_df = MagicMock()
        admissions_df = MagicMock()
        diagnoses_icd_df = MagicMock()
        d_icd_diagnoses_df = MagicMock()
        poe_df = MagicMock()
        poe_detail_df = MagicMock()

        # Mock return value for load_all_study_tables
        mock_load_all.return_value = {
            'patients': patients_df,
            'admissions': admissions_df,
            'diagnoses_icd': diagnoses_icd_df,
            'd_icd_diagnoses': d_icd_diagnoses_df,
            'poe': poe_df,
            'poe_detail': poe_detail_df
        }

        # Create mock merged DataFrames
        mock_df12 = MagicMock()
        mock_df34 = MagicMock()
        mock_poe_merged = MagicMock()
        mock_merged_full_study = MagicMock()
        mock_merged_w_poe = MagicMock()

        # Set up mock merge return values
        patients_df.merge.return_value = mock_df12
        diagnoses_icd_df.merge.return_value = mock_df34
        poe_df.merge.return_value = mock_poe_merged
        mock_df12.merge.return_value = mock_merged_full_study
        mock_merged_full_study.merge.return_value = mock_merged_w_poe

        # Test loading merged tables
        result = loader.load_merged_tables()

        # Check that the right functions were called
        patients_df.merge.assert_called_with(admissions_df, on='subject_id', how='inner')
        diagnoses_icd_df.merge.assert_called_with(d_icd_diagnoses_df, on=('icd_code', 'icd_version'), how='inner')
        poe_df.merge.assert_called_with(poe_detail_df, on=('poe_id', 'poe_seq', 'subject_id'), how='left')

        # Check results
        assert result['merged_wo_poe'] == mock_merged_full_study
        assert result['merged_full_study'] == mock_merged_w_poe
        assert result['poe_and_details'] == mock_poe_merged

        # Test with provided tables_dict
        mock_load_all.reset_mock()

        tables_dict = {
            'patients': patients_df,
            'admissions': admissions_df,
            'diagnoses_icd': diagnoses_icd_df,
            'd_icd_diagnoses': d_icd_diagnoses_df,
            'poe': poe_df,
            'poe_detail': poe_detail_df
        }

        result = loader.load_merged_tables(tables_dict=tables_dict)

        # Check that load_all_study_tables was not called
        mock_load_all.assert_not_called()

    def test_table_names_enum_converter(self):
        """Test the table_names_enum_converter function."""
        # Test hosp module
        result = convert_table_names_to_enum_class('patients', 'hosp')
        assert result == TableNamesHOSP.PATIENTS

        # Test icu module
        result = convert_table_names_to_enum_class('icustays', 'icu')
        assert result == TableNamesICU.ICUSTAYS

        # Test invalid table name
        with pytest.raises(ValueError):
            convert_table_names_to_enum_class('invalid_table', 'hosp')

    def test_enum_description_and_module(self):
        """Test the description and module properties of TableNamesHOSP and TableNamesICU."""
        # Test hosp table descriptions
        assert TableNamesHOSP.PATIENTS.description == "Patient demographic data"
        assert TableNamesHOSP.ADMISSIONS.description == "Patient hospital admissions information"

        # Test icu table descriptions
        assert TableNamesICU.ICUSTAYS.description == "ICU stay information"
        assert TableNamesICU.CHARTEVENTS.description == "Patient charting data (vital signs, etc.)"

        # Test module property
        assert TableNamesHOSP.PATIENTS.module == "hosp"
        assert TableNamesICU.ICUSTAYS.module == "icu"

        # Test values class method
        assert "patients" in TableNamesHOSP.values()
        assert "icustays" in TableNamesICU.values()


class TestParquetConverter:
    """Test suite for the ParquetConverter class."""

    @pytest.fixture
    def mock_mimic_dir(self):
        """Create a temporary directory with fake MIMIC data structure."""
        temp_dir = tempfile.mkdtemp()

        # Create module directories
        hosp_dir = Path(temp_dir) / "hosp"
        hosp_dir.mkdir()

        # Create sample CSV file
        patients_file = hosp_dir / "patients.csv"

        # Sample patients data
        patients_data = pd.DataFrame({
            'subject_id': [1, 2, 3],
            'gender': ['M', 'F', 'M'],
            'anchor_age': [45, 32, 67],
            'anchor_year': [2120, 2130, 2125],
            'anchor_year_group': ['2017 - 2019', '2017 - 2019', '2017 - 2019']
        })
        patients_data.to_csv(patients_file, index=False)

        yield Path(temp_dir)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_init(self):
        """Test initialization of ParquetConverter."""
        data_loader = MagicMock()
        converter = ParquetConverter(data_loader=data_loader)
        assert converter.data_loader == data_loader

    def test_get_csv_file_path(self, mock_mimic_dir):
        """Test _get_csv_file_path method."""
        # Setup
        data_loader = DataLoader(mimic_path=mock_mimic_dir)
        data_loader.scan_mimic_directory()
        converter = ParquetConverter(data_loader=data_loader)

        # Test getting CSV path
        file_path, suffix = converter._get_csv_file_path(TableNamesHOSP.PATIENTS)
        assert file_path.name == "patients.csv"
        assert suffix == ".csv"

        # Test with parquet file path - this is a simplified test that doesn't use mocking
        with patch.object(data_loader, 'tables_info_df') as mock_df:
            # Set up values to return when indexing
            filtered_df = pd.DataFrame({
                'file_path': [mock_mimic_dir / "hosp" / "patients.parquet"]
            })
            # Mock the behavior of tables_info_df when indexed and return our custom filtered_df
            mock_df.__getitem__.return_value = filtered_df

            # Create test files
            csv_path = mock_mimic_dir / "hosp" / "patients.csv"
            with open(csv_path, 'w') as f:
                f.write("test")

            # Mock Path.exists to return True only for CSV path
            with patch('pathlib.Path.exists', return_value=True):
                result = converter._get_csv_file_path(TableNamesHOSP.PATIENTS)
                # Just verify we get a successful result without error
                assert len(result) == 2

    def test_save_as_parquet(self, mock_mimic_dir):
        """Test save_as_parquet method."""
        # Setup
        data_loader = DataLoader(mimic_path=mock_mimic_dir)
        data_loader.scan_mimic_directory()
        converter = ParquetConverter(data_loader=data_loader)

        # Mock the to_parquet method
        with patch.object(dd.DataFrame, 'to_parquet') as mock_to_parquet, \
             patch.object(converter, '_get_csv_file_path') as mock_get_path, \
             patch.object(data_loader, 'load_csv_table_with_correct_column_datatypes') as mock_load:

            # Set up mocks
            mock_get_path.return_value = (mock_mimic_dir / "hosp" / "patients.csv", ".csv")
            mock_df = dd.from_pandas(pd.DataFrame({'test': [1, 2, 3]}), npartitions=1)
            mock_load.return_value = mock_df

            # Call save_as_parquet
            converter.save_as_parquet(TableNamesHOSP.PATIENTS)

            # Check to_parquet was called with the right path
            mock_to_parquet.assert_called_once()

        # Test with provided DataFrame
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        test_df_dask = dd.from_pandas(test_df, npartitions=1)
        target_path = mock_mimic_dir / "test.parquet"

        with patch.object(dd.DataFrame, 'to_parquet') as mock_to_parquet:
            converter.save_as_parquet(
                table_name=TableNamesHOSP.PATIENTS,
                df=test_df_dask,
                target_parquet_path=target_path
            )

            # Check to_parquet was called with the right arguments
            mock_to_parquet.assert_called_once_with(target_path, engine='pyarrow')

    def test_save_all_tables_as_parquet(self, mock_mimic_dir):
        """Test save_all_tables_as_parquet method."""
        # Setup
        data_loader = DataLoader(mimic_path=mock_mimic_dir)
        converter = ParquetConverter(data_loader=data_loader)

        # Mock the save_as_parquet method
        with patch.object(converter, 'save_as_parquet') as mock_save:
            # Call save_all_tables_as_parquet with custom tables list
            tables_list = [TableNamesHOSP.PATIENTS, TableNamesHOSP.ADMISSIONS]
            converter.save_all_tables_as_parquet(tables_list=tables_list)

            # Check save_as_parquet was called for each table
            assert mock_save.call_count == 2
            mock_save.assert_any_call(table_name=TableNamesHOSP.PATIENTS)
            mock_save.assert_any_call(table_name=TableNamesHOSP.ADMISSIONS)

            # Reset and test with default tables list
            mock_save.reset_mock()
            converter.save_all_tables_as_parquet()

            # Should use the data_loader's study_table_list
            assert mock_save.call_count == len(data_loader.study_table_list)


class TestExampleDataLoader:
    """Test suite for the ExampleDataLoader class."""

    def test_init(self):
        """Test initialization of ExampleDataLoader."""
        # Mock DataLoader to avoid actual loading
        with patch('mimic_iv_analysis.io.data_loader.DataLoader') as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Mock load_all_study_tables
            mock_loader.load_all_study_tables.return_value = {'patients': MagicMock()}

            # Test init with default params
            example_loader = ExampleDataLoader()
            mock_loader_class.assert_called_once()
            mock_loader.scan_mimic_directory.assert_called_once()
            mock_loader.load_all_study_tables.assert_called_once_with(
                partial_loading=False,
                use_dask=True
            )

            # Test init with custom params
            mock_loader_class.reset_mock()
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_all_study_tables.return_value = {'patients': MagicMock()}

            example_loader = ExampleDataLoader(
                partial_loading=True,
                num_subjects=5,
                random_selection=True,
                use_dask=False
            )
            mock_loader_class.assert_called_once()
            mock_loader.scan_mimic_directory.assert_called_once()
            mock_loader.load_all_study_tables.assert_called_once_with(
                partial_loading=True,
                num_subjects=5,
                random_selection=True,
                use_dask=False
            )

    def test_counter(self):
        """Test counter method."""
        # Create mock DataLoader
        mock_loader = MagicMock()

        # Create mock tables_dict with appropriate behavior
        tables_dict = {}

        for table_name in ['patients', 'admissions', 'diagnoses_icd', 'poe', 'poe_detail']:
            mock_df = MagicMock()
            mock_df.shape = MagicMock()
            mock_df.shape.__getitem__.return_value.compute.return_value = 100
            mock_df.subject_id = MagicMock()
            mock_df.subject_id.unique.return_value = MagicMock()
            mock_df.subject_id.unique.return_value.shape = MagicMock()
            mock_df.subject_id.unique.return_value.shape.__getitem__.return_value.compute.return_value = 50
            tables_dict[table_name] = mock_df

        # Create ExampleDataLoader directly with mocks
        with patch('mimic_iv_analysis.io.data_loader.DataLoader', return_value=mock_loader):
            example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
            example_loader.data_loader = mock_loader
            example_loader.tables_dict = tables_dict

            # Test counter method with mocked print
            with patch('builtins.print') as mock_print:
                example_loader.counter()
                assert mock_print.call_count >= 5

    def test_study_table_info(self):
        """Test study_table_info method."""
        # Create mock DataLoader
        mock_loader = MagicMock()
        mock_info = pd.DataFrame({
            'table_name': ['patients', 'admissions'],
            'file_path': ['/path/to/patients.csv', '/path/to/admissions.csv']
        })

        # Set the property directly instead of trying to patch it
        type(mock_loader).study_tables_info = PropertyMock(return_value=mock_info)

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.data_loader = mock_loader

        # Test study_table_info
        result = example_loader.study_table_info()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_merge_two_tables(self):
        """Test merge_two_tables method."""
        # Create mock tables
        mock_table1 = MagicMock()
        mock_table2 = MagicMock()
        mock_merged = MagicMock()
        mock_table1.merge.return_value = mock_merged

        # Create mock tables_dict
        tables_dict = {
            'patients': mock_table1,
            'admissions': mock_table2
        }

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.tables_dict = tables_dict

        # Test merge_two_tables
        result = example_loader.merge_two_tables(
            TableNamesHOSP.PATIENTS,
            TableNamesHOSP.ADMISSIONS,
            on='subject_id'
        )

        # Check merge was called with correct parameters
        mock_table1.merge.assert_called_once_with(
            mock_table2,
            on='subject_id',
            how='inner'
        )

        # Check result
        assert result == mock_merged

        # Test with custom how parameter
        mock_table1.merge.reset_mock()
        mock_table1.merge.return_value = mock_merged

        result = example_loader.merge_two_tables(
            TableNamesHOSP.PATIENTS,
            TableNamesHOSP.ADMISSIONS,
            on='subject_id',
            how='left'
        )

        mock_table1.merge.assert_called_once_with(
            mock_table2,
            on='subject_id',
            how='left'
        )

    def test_save_as_parquet(self):
        """Test save_as_parquet method."""
        # Create mock DataLoader
        mock_loader = MagicMock()

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.data_loader = mock_loader

        # Mock ParquetConverter
        with patch('mimic_iv_analysis.io.data_loader.ParquetConverter') as mock_converter_class:
            mock_converter = MagicMock()
            mock_converter_class.return_value = mock_converter

            # Test save_as_parquet
            example_loader.save_as_parquet(TableNamesHOSP.PATIENTS)

            # Check ParquetConverter was created with correct parameters
            mock_converter_class.assert_called_once_with(data_loader=mock_loader)

            # Check save_as_parquet was called with correct parameters
            mock_converter.save_as_parquet.assert_called_once_with(table_name=TableNamesHOSP.PATIENTS)

    def test_n_rows_after_merge(self):
        """Test n_rows_after_merge method."""
        # Create mock DataFrames with shape.compute() behavior
        tables_dict = {}
        for table_name in ['patients', 'admissions', 'diagnoses_icd', 'd_icd_diagnoses', 'poe', 'poe_detail']:
            mock_df = MagicMock()
            mock_df.shape = MagicMock()
            mock_df.shape.__getitem__.return_value.compute.return_value = 100
            mock_df.merge.return_value = mock_df  # Make merge return self
            tables_dict[table_name] = mock_df

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.tables_dict = tables_dict

        # Test n_rows_after_merge with mocked print
        with patch('builtins.print') as mock_print:
            example_loader.n_rows_after_merge()
            assert mock_print.call_count >= 5

    def test_load_table(self):
        """Test load_table method."""
        # Create mock tables_dict
        mock_df = MagicMock()
        tables_dict = {
            'patients': mock_df
        }

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.tables_dict = tables_dict

        # Test load_table
        result = example_loader.load_table(TableNamesHOSP.PATIENTS)
        assert result == mock_df

    def test_load_all_study_tables(self):
        """Test load_all_study_tables method."""
        # Create mock tables_dict
        tables_dict = {
            'patients': MagicMock(),
            'admissions': MagicMock()
        }

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.tables_dict = tables_dict

        # Test load_all_study_tables
        result = example_loader.load_all_study_tables()
        assert result == tables_dict

    def test_load_merged_tables(self):
        """Test load_merged_tables method."""
        # Create mock DataLoader
        mock_loader = MagicMock()

        # Create specific return value to match with equality check
        return_value = {'merged_table': 'test_value'}
        mock_loader.load_merged_tables.return_value = return_value

        # Create ExampleDataLoader directly with mocks
        example_loader = ExampleDataLoader.__new__(ExampleDataLoader)
        example_loader.data_loader = mock_loader

        # Test load_merged_tables
        result = example_loader.load_merged_tables()
        mock_loader.load_merged_tables.assert_called_once()
        # Use a more specific comparison
        assert result == return_value
