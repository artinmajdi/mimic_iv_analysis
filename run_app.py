#!/usr/bin/env python
"""
Launch script for the MIMIC-IV Analysis Dashboard.
This script sets up the proper import path before running the app.
"""

import os
import sys
import subprocess

# Add the current directory to Python path for package resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Launch the Streamlit app
if __name__ == "__main__":
    # Confirm Python paths
    print("Python paths:")
    for p in sys.path:
        print(f"  - {p}")

    # Try to import the module to verify it works
    try:
        from mimic_iv_analysis.visualization import app
        print("Successfully imported app module!")
    except ImportError as e:
        print(f"Error importing app module: {e}")
        sys.exit(1)

    # Launch Streamlit
    subprocess.run(["streamlit", "run", os.path.join("src", "visualization", "app.py")])
