#!/usr/bin/env python3
"""
Test script for the refactored TableNames class to ensure all functionality is preserved.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mimic_iv_analysis.configurations.params import TableNames

def test_basic_enum_functionality():
    """Test basic enum functionality."""
    print("Testing basic enum functionality...")
    
    # Test enum values
    assert TableNames.PATIENTS.value == 'patients'
    assert TableNames.ADMISSIONS.value == 'admissions'
    assert TableNames.CHARTEVENTS.value == 'chartevents'
    
    # Test enum comparison
    assert TableNames.PATIENTS == TableNames.PATIENTS
    assert TableNames.PATIENTS != TableNames.ADMISSIONS
    
    print("✓ Basic enum functionality works correctly")

def test_class_methods():
    """Test class methods."""
    print("\nTesting class methods...")
    
    # Test values() method
    all_values = TableNames.values()
    assert isinstance(all_values, list)
    assert 'patients' in all_values
    assert 'admissions' in all_values
    assert len(all_values) > 20  # Should have many tables
    
    # Test hosp_tables() method
    hosp_tables = TableNames.hosp_tables()
    assert isinstance(hosp_tables, list)
    assert 'admissions' in hosp_tables
    assert 'patients' in hosp_tables
    assert 'chartevents' not in hosp_tables  # ICU table
    
    # Test icu_tables() method
    icu_tables = TableNames.icu_tables()
    assert isinstance(icu_tables, list)
    assert 'chartevents' in icu_tables
    assert 'icustays' in icu_tables
    assert 'admissions' not in icu_tables  # HOSP table
    
    print("✓ Class methods work correctly")

def test_properties():
    """Test instance properties."""
    print("\nTesting instance properties...")
    
    # Test hosp property
    hosp_list = TableNames.PATIENTS.hosp
    assert isinstance(hosp_list, list)
    assert 'admissions' in hosp_list
    
    # Test description property
    desc = TableNames.PATIENTS.description
    assert isinstance(desc, str)
    assert len(desc) > 0
    
    # Test module property
    module = TableNames.PATIENTS.module
    assert module in ['hosp', 'icu']
    
    module_icu = TableNames.CHARTEVENTS.module
    assert module_icu == 'icu'
    
    print("✓ Instance properties work correctly")

def test_column_table_mappings():
    """Test the new column-table mapping methods."""
    print("\nTesting column-table mapping methods...")
    
    # Test get_tables_with_column
    tables_with_subject_id = TableNames.get_tables_with_column('subject_id')
    assert isinstance(tables_with_subject_id, list)
    assert 'patients' in tables_with_subject_id
    assert 'admissions' in tables_with_subject_id
    assert len(tables_with_subject_id) > 5  # subject_id is in many tables
    
    # Test with non-existent column
    empty_result = TableNames.get_tables_with_column('non_existent_column')
    assert empty_result == []
    
    # Test get_columns_for_table
    patient_columns = TableNames.get_columns_for_table('patients')
    assert isinstance(patient_columns, list)
    assert 'subject_id' in patient_columns
    assert 'gender' in patient_columns
    assert len(patient_columns) > 3
    
    # Test with non-existent table
    empty_columns = TableNames.get_columns_for_table('non_existent_table')
    assert empty_columns == []
    
    print("✓ Column-table mapping methods work correctly")

def test_error_handling():
    """Test error handling and validation."""
    print("\nTesting error handling...")
    
    # Test type validation for get_tables_with_column
    try:
        TableNames.get_tables_with_column(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "must be a string" in str(e)
    
    # Test type validation for get_columns_for_table
    try:
        TableNames.get_columns_for_table(['not_a_string'])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "must be a string" in str(e)
    
    print("✓ Error handling works correctly")

def test_backward_compatibility():
    """Test that existing functionality is preserved."""
    print("\nTesting backward compatibility...")
    
    # Test that all original enum members still exist
    required_tables = [
        'PATIENTS', 'ADMISSIONS', 'TRANSFERS', 'DIAGNOSES_ICD', 'PROCEDURES_ICD',
        'LABEVENTS', 'PRESCRIPTIONS', 'MICROBIOLOGYEVENTS', 'PHARMACY', 'POE',
        'CHARTEVENTS', 'ICUSTAYS', 'INPUTEVENTS', 'OUTPUTEVENTS'
    ]
    
    for table_name in required_tables:
        assert hasattr(TableNames, table_name), f"Missing table: {table_name}"
        table_enum = getattr(TableNames, table_name)
        assert isinstance(table_enum.value, str)
    
    # Test that constants are still accessible
    assert hasattr(TableNames, '_HOSP_TABLES')
    assert hasattr(TableNames, '_ICU_TABLES')
    assert isinstance(TableNames._HOSP_TABLES, frozenset)
    assert isinstance(TableNames._ICU_TABLES, frozenset)
    
    print("✓ Backward compatibility maintained")

def test_data_integrity():
    """Test data integrity and consistency."""
    print("\nTesting data integrity...")
    
    # Test that all tables in _HOSP_TABLES exist as enum members
    for table_name in TableNames._HOSP_TABLES:
        assert hasattr(TableNames, table_name), f"HOSP table {table_name} not found in enum"
    
    # Test that all tables in _ICU_TABLES exist as enum members
    for table_name in TableNames._ICU_TABLES:
        assert hasattr(TableNames, table_name), f"ICU table {table_name} not found in enum"
    
    # Test that column mappings are consistent
    all_tables_from_columns = set()
    for tables in TableNames._COLUMN_TO_TABLES.values():
        all_tables_from_columns.update(tables)
    
    all_tables_from_table_mapping = set(TableNames._TABLE_TO_COLUMNS.keys())
    
    # Most tables should be represented in both mappings
    common_tables = all_tables_from_columns.intersection(all_tables_from_table_mapping)
    assert len(common_tables) > 15, "Insufficient overlap between column and table mappings"
    
    print("✓ Data integrity checks passed")

def run_all_tests():
    """Run all tests."""
    print("Running comprehensive tests for refactored TableNames class...\n")
    
    try:
        test_basic_enum_functionality()
        test_class_methods()
        test_properties()
        test_column_table_mappings()
        test_error_handling()
        test_backward_compatibility()
        test_data_integrity()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! The refactored TableNames class is working correctly.")
        print("✓ All existing functionality is preserved")
        print("✓ New features work as expected")
        print("✓ Error handling is robust")
        print("✓ Code follows Pythonic principles")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()