import pytest
import os
import shutil
import pandas as pd
import dask.dataframe as dd
from pandas.testing import assert_frame_equal
import logging

from mimic_iv_analysis.core.data_loader import DataLoader, DEFAULT_MIMIC_PATH, LARGE_FILE_THRESHOLD_MB, PATIENTS_TABLE_NAME, PATIENTS_TABLE_MODULE, SUBJECT_ID_COL

# Configure logging for tests (optional, but can be helpful)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope='session')
def mimic_path_fixture():
    """Provides the path to the MIMIC-IV dataset, defaulting to DEFAULT_MIMIC_PATH."""
    # In a CI environment, you might override this with a path to a small, controlled dataset
    # For local testing, it might point to the actual dataset if available and desired for some tests,
    # but most tests should use temp_mimic_dir for reliability and speed.
    return os.environ.get("TEST_MIMIC_DATA_PATH", DEFAULT_MIMIC_PATH)

@pytest.fixture
def data_loader():
    """Provides an instance of DataLoader."""
    return DataLoader()

@pytest.fixture(scope='module')
def temp_mimic_dir(tmp_path_factory):
    """Creates a temporary MIMIC-like directory structure for testing."""
    temp_dir = tmp_path_factory.mktemp("mimic_data")
    hosp_dir = temp_dir / "hosp"
    icu_dir = temp_dir / "icu"
    hosp_dir.mkdir()
    icu_dir.mkdir()

    # Create dummy patient data with more subjects
    patients_data = {'subject_id': [1, 2, 3, 4, 5, 10, 12, 15, 22, 30],
                       'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
                       'anchor_age': [50, 60, 70, 55, 65, 75, 40, 45, 80, 85]}
    pd.DataFrame(patients_data).to_csv(hosp_dir / f"{PATIENTS_TABLE_NAME}.csv", index=False)

    # Create dummy admissions data
    admissions_data = {'subject_id': [1, 2, 2, 4, 5, 10, 10, 10],
                         'hadm_id': [101, 102, 103, 104, 105, 106, 107, 108],
                         'admittime': ['2180-01-01', '2190-01-01', '2192-01-01', '2185-01-01', '2170-01-01', '2160-01-01', '2161-01-01', '2162-01-01']}
    pd.DataFrame(admissions_data).to_csv(hosp_dir / "admissions.csv.gz", index=False, compression='gzip')

    # Create dummy ICU data
    icustays_data = {'subject_id': [1, 4, 10], 'hadm_id': [101, 104, 106], 'stay_id': [201, 202, 203], 'intime': ['2180-01-01', '2185-01-02', '2160-01-02']}
    pd.DataFrame(icustays_data).to_csv(icu_dir / "icustays.csv", index=False)

    # Create a small dictionary file
    d_items_data = {'itemid': [1, 2], 'label': ['Label1', 'Label2']}
    pd.DataFrame(d_items_data).to_csv(icu_dir / "d_items.csv.gz", index=False, compression='gzip')

    # Create a larger dummy table with subject_id for testing subject_id filtering
    # Simulate largeness by row count relative to a conceptual threshold for tests
    # Actual LARGE_FILE_THRESHOLD_MB is for file size, here we use row count to simplify test setup
    num_large_table_rows = 150 # Assume test threshold is < 150 rows for this file type for large file logic path
    large_table_subjects = [1, 2, 3, 4, 5, 10, 12, 15, 22, 30] # ensure some overlap and some unique
    large_table_data = {'subject_id': [large_table_subjects[i % len(large_table_subjects)] for i in range(num_large_table_rows)],
                          'value1': [i * 10 for i in range(num_large_table_rows)],
                          'value2': [f'text_{i}' for i in range(num_large_table_rows)]}
    pd.DataFrame(large_table_data).to_csv(hosp_dir / "large_table.csv", index=False)

    # Create a table without subject_id for testing fallback
    no_subject_table_data = {'other_id': [100, 200, 300], 'data': ['x', 'y', 'z']}
    pd.DataFrame(no_subject_table_data).to_csv(hosp_dir / "no_subject_table.csv", index=False)


    logger.info(f"Created temp MIMIC directory at {temp_dir}")
    return str(temp_dir)

def test_data_loader_initialization(data_loader):
    """Test DataLoader initialization."""
    assert data_loader is not None
    assert data_loader.mimic_path == DEFAULT_MIMIC_PATH

def test_scan_mimic_directory(data_loader, temp_mimic_dir):
    """Test scanning the MIMIC-IV directory structure."""
    _, dataset_info = data_loader.scan_mimic_directory(temp_mimic_dir)

    assert 'hosp'       in dataset_info['available_tables']
    assert 'icu'        in dataset_info['available_tables']
    assert 'patients'   in dataset_info['available_tables']['hosp']
    assert 'admissions' in dataset_info['available_tables']['hosp']
    assert 'icustays'   in dataset_info['available_tables']['icu']
    assert 'd_items'    in dataset_info['available_tables']['icu']

    assert ('hosp', 'patients') in dataset_info['file_paths']
    assert dataset_info['file_paths'][('hosp', 'patients')].endswith('patients.csv')
    assert ('hosp', 'admissions') in dataset_info['file_sizes']
    assert dataset_info['table_display_names'][('icu', 'd_items')].startswith('d_items (')
    assert data_loader._patients_file_path is not None
    assert data_loader._patients_file_path.endswith(f"{PATIENTS_TABLE_NAME}.csv")
    assert data_loader._data_scan_complete

def test_get_all_subject_ids(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir) # Ensure scan has run
    subject_ids = data_loader._get_all_subject_ids()
    assert isinstance(subject_ids, list)
    assert len(subject_ids) == 10 # From dummy patients_data
    assert set(subject_ids) == {1, 2, 3, 4, 5, 10, 12, 15, 22, 30}

    # Test caching
    data_loader._all_subject_ids = ['cached']
    assert data_loader._get_all_subject_ids() == ['cached']
    data_loader._all_subject_ids = None # Reset for other tests

def test_get_all_subject_ids_no_scan(data_loader, caplog):
    data_loader._data_scan_complete = False # Simulate no scan
    subject_ids = data_loader._get_all_subject_ids()
    assert subject_ids == []
    assert "MIMIC directory scan has not been performed" in caplog.text
    data_loader._data_scan_complete = True # Reset

def test_get_all_subject_ids_patients_missing(data_loader, temp_mimic_dir, caplog):

    data_loader.scan_mimic_directory(temp_mimic_dir) # Scan normally first

    original_patients_path = data_loader._patients_file_path
    data_loader._patients_file_path = "/tmp/nonexistent_patients.csv"
    data_loader._all_subject_ids = None # Clear cache
    subject_ids = data_loader._get_all_subject_ids()

    assert subject_ids == []
    assert f"'{PATIENTS_TABLE_NAME}.csv' or '.csv.gz' not found" in caplog.text

    data_loader._patients_file_path = original_patients_path # Reset
    data_loader._all_subject_ids    = None # Clear cache again

def test_get_total_unique_subjects(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    total_subjects = data_loader.get_total_unique_subjects()
    assert total_subjects == 10

def test_get_subject_ids_for_sampling(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    expected_full_list = sorted([1, 2, 3, 4, 5, 10, 12, 15, 22, 30])

    # Test getting a specific number
    sampled_ids = data_loader.get_subject_ids_for_sampling(3)
    assert len(sampled_ids) == 3
    assert sampled_ids == expected_full_list[:3]

    # Test getting 0 (should return None, meaning no specific subject sampling)
    assert data_loader.get_subject_ids_for_sampling(0) is None

    # Test getting None (should also return None)
    assert data_loader.get_subject_ids_for_sampling(None) is None

    # Test getting more than available (should return all)
    sampled_ids_more = data_loader.get_subject_ids_for_sampling(100)
    assert len(sampled_ids_more) == 10
    assert sorted(sampled_ids_more) == expected_full_list

    # Test when _all_subject_ids is empty
    data_loader._all_subject_ids = []
    assert data_loader.get_subject_ids_for_sampling(5) is None
    data_loader._all_subject_ids = None # Reset

def test_load_mimic_table_pandas_sampled(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir) # To set up patients_file_path etc.
    admissions_path = os.path.join(temp_mimic_dir, "hosp", "admissions.csv.gz") # admissions has 8 rows
    df, total_rows = data_loader.load_mimic_table(admissions_path, sample_size=2, use_dask=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2
    assert total_rows == 8
    assert 'subject_id' in df.columns

def test_load_mimic_table_pandas_full(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    admissions_path = os.path.join(temp_mimic_dir, "hosp", "admissions.csv.gz")
    df, total_rows = data_loader.load_mimic_table(admissions_path, sample_size=100, use_dask=False) # sample_size > total_rows
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert total_rows == 8

def test_load_mimic_table_dask_sampled(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    admissions_path = os.path.join(temp_mimic_dir, "hosp", "admissions.csv.gz")
    df, total_rows = data_loader.load_mimic_table(admissions_path, sample_size=2, use_dask=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2
    assert total_rows == 8
    assert 'subject_id' in df.columns

def test_load_mimic_table_dask_full(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    admissions_path = os.path.join(temp_mimic_dir, "hosp", "admissions.csv.gz")
    df, total_rows = data_loader.load_mimic_table(admissions_path, sample_size=100, use_dask=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert total_rows == 8

def test_load_mimic_table_with_subject_ids_pandas(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    large_table_path = os.path.join(temp_mimic_dir, "hosp", "large_table.csv")
    target_ids = [1, 5, 10] # These are in patients_data and large_table_data

    # Mock LARGE_FILE_THRESHOLD_MB to ensure this path is taken for testing
    # This is a bit of a hack; ideally, the file would be actually large or threshold configurable for tests.
    # For now, rely on the logic path where target_subject_ids implies it might be large file handling path.
    # The `load_mimic_table` uses file_size_mb > LARGE_FILE_THRESHOLD_MB for this path.
    # We'll create a file that *is* large enough by making its size > 0, and test the filtering.
    # The dummy large_table.csv is small, so we are testing the logic flow more than size performance here.

    df, loaded_rows = data_loader.load_mimic_table(large_table_path, target_subject_ids=target_ids, use_dask=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert set(df[SUBJECT_ID_COL].unique()) == set(target_ids)
    # Verify that only rows with target_ids are present
    for sid in df[SUBJECT_ID_COL].unique():
        assert sid in target_ids
    assert loaded_rows == len(df) # For subject_id filtering, total_rows is num loaded

def test_load_mimic_table_with_subject_ids_dask(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    large_table_path = os.path.join(temp_mimic_dir, "hosp", "large_table.csv")
    target_ids = [2, 4, 12]
    df, loaded_rows = data_loader.load_mimic_table(large_table_path, target_subject_ids=target_ids, use_dask=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert set(df[SUBJECT_ID_COL].unique()) == set(target_ids)
    for sid in df[SUBJECT_ID_COL].unique():
        assert sid in target_ids
    assert loaded_rows == len(df)

def test_load_mimic_table_with_subject_ids_no_match(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    large_table_path = os.path.join(temp_mimic_dir, "hosp", "large_table.csv")
    target_ids = [999, 1000] # These are NOT in patients_data
    df, loaded_rows = data_loader.load_mimic_table(large_table_path, target_subject_ids=target_ids, use_dask=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert loaded_rows == 0

def test_load_mimic_table_with_subject_ids_no_subject_col(data_loader, temp_mimic_dir):
    data_loader.scan_mimic_directory(temp_mimic_dir)
    no_subject_table_path = os.path.join(temp_mimic_dir, "hosp", "no_subject_table.csv")
    target_ids = [1, 2]

    # Expect fallback to sample_size (default or specified)
    df, total_rows_original = data_loader.load_mimic_table(no_subject_table_path, target_subject_ids=target_ids, sample_size=2, use_dask=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2 # Should be sampled, as subject_id filtering isn't possible
    assert total_rows_original == 3 # Original total rows of no_subject_table_data
    assert SUBJECT_ID_COL not in df.columns

def test_load_mimic_table_non_existent(data_loader, temp_mimic_dir, caplog):
    """Test loading a non-existent file."""
    non_existent_path = os.path.join(temp_mimic_dir, "hosp", "non_existent.csv")
    df, total_rows = data_loader.load_mimic_table(non_existent_path)
    assert df is None
    assert total_rows == 0
    assert "Error loading data" in caplog.text

@pytest.fixture
def sample_dfs_for_merge():
    """Provides two sample pandas DataFrames for merge tests."""
    left_pd = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'A': ['A0', 'A1', 'A2']})
    right_pd = pd.DataFrame({'key': ['K0', 'K1', 'K3'], 'B': ['B0', 'B1', 'B3']})
    return left_pd, right_pd

def test_merge_tables_pandas(data_loader, sample_dfs_for_merge):
    """Test merging two pandas DataFrames."""
    left_pd, right_pd = sample_dfs_for_merge
    merged_df = data_loader.merge_tables(left_pd, right_pd, on=['key'], how='left')
    expected_df = pd.DataFrame({
        'key': ['K0', 'K1', 'K2'],
        'A': ['A0', 'A1', 'A2'],
        'B': ['B0', 'B1', pd.NA]  # Or np.nan depending on pandas version / types
    }).astype({'B': object}) # Ensure B is object type for NA comparison if needed
    # For newer pandas, NA is preferred. For older, NaN might appear. Let's be flexible.
    if 'B' in merged_df and merged_df['B'].dtype == 'object':
         merged_df['B'] = merged_df['B'].fillna(pd.NA) # Normalize NaNs to NA
    assert_frame_equal(merged_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_dtype=False)

def test_merge_tables_dask(data_loader, sample_dfs_for_merge):
    """Test merging two Dask DataFrames."""
    left_pd, right_pd = sample_dfs_for_merge
    left_dd = dd.from_pandas(left_pd, npartitions=1)
    right_dd = dd.from_pandas(right_pd, npartitions=1)

    merged_dd = data_loader.merge_tables(left_dd, right_dd, on=['key'], how='left')
    assert isinstance(merged_dd, dd.DataFrame)

    computed_merged_df = merged_dd.compute()
    expected_df = pd.DataFrame({
        'key': ['K0', 'K1', 'K2'],
        'A': ['A0', 'A1', 'A2'],
        'B': ['B0', 'B1', pd.NA]
    }).astype({'B': object})
    if 'B' in computed_merged_df and computed_merged_df['B'].dtype == 'object':
         computed_merged_df['B'] = computed_merged_df['B'].fillna(pd.NA)
    assert_frame_equal(computed_merged_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_dtype=False)

def test_merge_tables_mixed_error(data_loader, sample_dfs_for_merge, caplog):
    """Test merging mixed DataFrame types (should return left df and log error)."""
    left_pd, right_pd = sample_dfs_for_merge
    right_dd = dd.from_pandas(right_pd, npartitions=1)

    result_df = data_loader.merge_tables(left_pd, right_dd, on=['key'], how='left')
    assert_frame_equal(result_df, left_pd)
    assert "Cannot merge a Dask DataFrame with a pandas DataFrame" in caplog.text

def test_merge_tables_missing_columns(data_loader, sample_dfs_for_merge, caplog):
    """Test merging with missing join columns (should return left df and log warning)."""
    left_pd, right_pd = sample_dfs_for_merge
    result_df = data_loader.merge_tables(left_pd, right_pd, on=['missing_key'], how='left')
    assert_frame_equal(result_df, left_pd)
    assert "Cannot merge tables: missing_key missing in left table, missing_key missing in right table" in caplog.text


def test_load_reference_table(data_loader, temp_mimic_dir):
    """Test loading a reference table."""
    # d_items is created as gzipped in temp_mimic_dir
    df = data_loader.load_reference_table(temp_mimic_dir, 'icu', 'd_items')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'itemid' in df.columns
    assert 'label' in df.columns
    assert len(df) == 2

def test_get_table_info(data_loader):
    """Test getting table information."""
    info = data_loader.get_table_description('hosp', 'admissions')
    assert "Patient hospital admissions information" in info
    info_unknown = data_loader.get_table_description('foo', 'bar')
    assert "No description available" in info_unknown

# More complex tests for apply_filters might require extensive mocking or a dedicated small dataset
# For now, a simple test to see if it runs with minimal parameters or with mocking

# Example of a very simple apply_filters test (might need adjustment based on Filtering class)
# This test is basic and assumes apply_filters can run with no real filtering if params are minimal.
def test_apply_filters_runs(data_loader, temp_mimic_dir):
    """Test that apply_filters runs without error with a basic DataFrame."""
    df_input = pd.DataFrame({'subject_id': [1, 2, 3]})
    filter_params = {'apply_age_range': False} # Minimal filter

    # Providing mimic_path to allow internal loading if Filtering class tries to access it
    df_filtered = data_loader.apply_filters(df_input.copy(), filter_params, mimic_path=temp_mimic_dir)
    assert isinstance(df_filtered, pd.DataFrame)
    # In this minimal case, expect the dataframe to be returned as is or similar
    assert_frame_equal(df_input, df_filtered)
