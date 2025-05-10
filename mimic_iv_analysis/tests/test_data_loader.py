import pytest
import os
import shutil
import pandas as pd
import dask.dataframe as dd
from pandas.testing import assert_frame_equal
import logging

from mimic_iv_analysis.core.data_loader import DataLoader, DEFAULT_MIMIC_PATH

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

    # Create dummy patient data
    patients_data = {'subject_id': [1, 2, 3], 'gender': ['M', 'F', 'M'], 'anchor_age': [50, 60, 70]}
    pd.DataFrame(patients_data).to_csv(hosp_dir / "patients.csv", index=False)

    # Create dummy admissions data
    admissions_data = {'subject_id': [1, 2, 2], 'hadm_id': [101, 102, 103], 'admittime': ['2180-01-01', '2190-01-01', '2192-01-01']}
    pd.DataFrame(admissions_data).to_csv(hosp_dir / "admissions.csv.gz", index=False, compression='gzip')

    # Create dummy ICU data
    icustays_data = {'subject_id': [1], 'hadm_id': [101], 'stay_id': [201], 'intime': ['2180-01-01']}
    pd.DataFrame(icustays_data).to_csv(icu_dir / "icustays.csv", index=False)

    # Create a small dictionary file
    d_items_data = {'itemid': [1, 2], 'label': ['Label1', 'Label2']}
    pd.DataFrame(d_items_data).to_csv(icu_dir / "d_items.csv.gz", index=False, compression='gzip')

    logger.info(f"Created temp MIMIC directory at {temp_dir}")
    return str(temp_dir)

def test_data_loader_initialization(data_loader):
    """Test DataLoader initialization."""
    assert data_loader is not None
    assert data_loader.mimic_path == DEFAULT_MIMIC_PATH

def test_scan_mimic_directory(data_loader, temp_mimic_dir):
    """Test scanning the MIMIC-IV directory structure."""
    available_tables, file_paths, file_sizes, table_display_names = data_loader.scan_mimic_directory(temp_mimic_dir)

    assert 'hosp' in available_tables
    assert 'icu' in available_tables
    assert 'patients' in available_tables['hosp']
    assert 'admissions' in available_tables['hosp']
    assert 'icustays' in available_tables['icu']
    assert 'd_items' in available_tables['icu']

    assert ('hosp', 'patients') in file_paths
    assert file_paths[('hosp', 'patients')].endswith('patients.csv')
    assert ('hosp', 'admissions') in file_sizes
    assert table_display_names[('icu', 'd_items')].startswith('d_items (')

def test_load_mimic_table_pandas_sampled(data_loader, temp_mimic_dir):
    """Test loading a table with pandas (sampled)."""
    patients_path = os.path.join(temp_mimic_dir, "hosp", "patients.csv")
    df, total_rows = data_loader.load_mimic_table(patients_path, sample_size=2, use_dask=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2  # Sampling might give less if total rows < sample_size
    assert total_rows == 3
    assert 'subject_id' in df.columns

def test_load_mimic_table_pandas_full(data_loader, temp_mimic_dir):
    """Test loading a table with pandas (full)."""
    patients_path = os.path.join(temp_mimic_dir, "hosp", "patients.csv")
    # Use a large sample size to effectively load all rows
    df, total_rows = data_loader.load_mimic_table(patients_path, sample_size=100, use_dask=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert total_rows == 3

def test_load_mimic_table_dask_sampled(data_loader, temp_mimic_dir):
    """Test loading a table with Dask (sampled)."""
    patients_path = os.path.join(temp_mimic_dir, "hosp", "patients.csv")
    df, total_rows = data_loader.load_mimic_table(patients_path, sample_size=2, use_dask=True)
    assert isinstance(df, pd.DataFrame) # Dask load_mimic_table returns computed pandas for sampling now
    assert len(df) <= 2
    assert total_rows == 3
    assert 'subject_id' in df.columns

def test_load_mimic_table_dask_full(data_loader, temp_mimic_dir):
    """Test loading a table with Dask (full)."""
    patients_path = os.path.join(temp_mimic_dir, "hosp", "patients.csv")
    df, total_rows = data_loader.load_mimic_table(patients_path, sample_size=100, use_dask=True)
    assert isinstance(df, pd.DataFrame) # Dask load_mimic_table returns computed pandas
    assert len(df) == 3
    assert total_rows == 3

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

def test_convert_to_parquet_pandas(data_loader, temp_mimic_dir):
    """Test converting a pandas DataFrame to Parquet."""
    df_pd = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    parquet_path = data_loader.convert_to_parquet(df_pd, "test_pd", temp_mimic_dir)
    assert parquet_path is not None
    assert os.path.exists(parquet_path)
    assert parquet_path.endswith("test_pd.parquet")
    df_read = pd.read_parquet(parquet_path)
    assert_frame_equal(df_pd, df_read)
    os.remove(parquet_path) # Clean up

def test_convert_to_parquet_dask(data_loader, temp_mimic_dir):
    """Test converting a Dask DataFrame to Parquet."""
    df_pd = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
    df_dd = dd.from_pandas(df_pd, npartitions=2)
    
    # Dask's to_parquet with single_file=True often creates a file, not a dir
    # If it creates a dir, this test might need adjustment or the function might need to return the dir path
    parquet_output_path = os.path.join(temp_mimic_dir, 'parquet_files', "test_dd.parquet")
    returned_path = data_loader.convert_to_parquet(df_dd, "test_dd", temp_mimic_dir)
    
    assert returned_path is not None
    assert returned_path == parquet_output_path
    
    # Check if the output is a single file or a directory
    if os.path.isfile(returned_path):
        assert os.path.exists(returned_path)
        df_read = pd.read_parquet(returned_path)
        assert_frame_equal(df_pd, df_read.reset_index(drop=True), check_dtype=False)
        os.remove(returned_path)
    elif os.path.isdir(returned_path):
        # If it's a directory, Dask wrote multiple files. Read them back.
        df_read_dd = dd.read_parquet(returned_path)
        assert_frame_equal(df_pd, df_read_dd.compute().reset_index(drop=True), check_dtype=False)
        shutil.rmtree(returned_path) # Clean up directory
    else:
        pytest.fail(f"Parquet output path is neither a file nor a directory: {returned_path}")
    
    # Clean up the parent 'parquet_files' directory if it's empty or only created for this test
    parquet_files_dir = os.path.dirname(returned_path)
    if os.path.exists(parquet_files_dir) and not os.listdir(parquet_files_dir):
        os.rmdir(parquet_files_dir)

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
    info = data_loader.get_table_info('hosp', 'admissions')
    assert "Patient hospital admissions information" in info
    info_unknown = data_loader.get_table_info('foo', 'bar')
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

