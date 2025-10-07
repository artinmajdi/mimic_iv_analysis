#!/usr/bin/env python3
"""
Basic test of the merge function without Dask to isolate core functionality issues.
"""

import pandas as pd
import time
import psutil
import os
from mimic_iv_analysis.io.data_loader import DataLoader

def test_basic_merge():
    """Test the merge function with a small subset to verify core functionality."""
    
    print("🧪 Testing basic merge functionality without Dask...")
    
    # Monitor system resources
    initial_memory = psutil.virtual_memory().used / (1024**3)
    start_time = time.time()
    
    try:
        # Create a DataLoader instance
        loader = DataLoader()
        
        # Load labevents with row limit for testing
        print("📊 Loading small data samples...")
        
        # Load labevents with row limit for testing
        labevents_df = loader.load('labevents', partial_loading=True, num_subjects=100, use_dask=False)
        print(f"✅ Loaded {len(labevents_df)} labevents rows")
        
        # Load admissions
        admissions_df = loader.load('admissions', use_dask=False)
        print(f"✅ Loaded {len(admissions_df)} admissions rows")
        
        # Test the temporal merge function directly
        print("🔄 Testing temporal merge function...")
        
        # Convert to pandas if needed
        if hasattr(labevents_df, 'compute'):
            labevents_pd = labevents_df.compute()
        else:
            labevents_pd = labevents_df
            
        if hasattr(admissions_df, 'compute'):
            admissions_pd = admissions_df.compute()
        else:
            admissions_pd = admissions_df
        
        # Add buffered time columns to admissions
        admissions_pd = admissions_pd.copy()
        admissions_pd['admittime_buffered'] = pd.to_datetime(admissions_pd['admittime']) - pd.Timedelta(hours=24)
        admissions_pd['dischtime_buffered'] = pd.to_datetime(admissions_pd['dischtime']) + pd.Timedelta(hours=24)
        
        # Test the temporal merge partition function
        merged_result = DataLoader._temporal_merge_partition(labevents_pd, admissions_pd)
        
        end_time = time.time()
        final_memory = psutil.virtual_memory().used / (1024**3)
        
        print(f"✅ Basic merge completed successfully!")
        print(f"📊 Results:")
        print(f"   • Input labevents: {len(labevents_pd):,} rows")
        print(f"   • Input admissions: {len(admissions_pd):,} rows") 
        print(f"   • Merged result: {len(merged_result):,} rows")
        print(f"   • Processing time: {end_time - start_time:.2f} seconds")
        print(f"   • Memory used: {final_memory - initial_memory:.2f} GB")
        
        if not merged_result.empty:
            print(f"   • Result columns: {list(merged_result.columns)}")
            print(f"   • Sample data shape: {merged_result.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Basic merge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_merge()
    exit(0 if success else 1)