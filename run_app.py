#!/usr/bin/env python
"""
Simplified launcher for the MIMIC-IV Analysis Streamlit app.
"""

import os
import sys
import subprocess

# Get the absolute path to the app.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "src", "mimic_iv_analysis", "visualization", "app.py")

if not os.path.exists(app_path):
    print(f"Error: Could not find app at {app_path}")
    sys.exit(1)

print(f"Starting Streamlit app from: {app_path}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {current_dir}")

# Run streamlit with the proper PYTHONPATH
env = os.environ.copy()
env["PYTHONPATH"] = current_dir + ":" + env.get("PYTHONPATH", "")

try:
    # Use the system installed streamlit to run the app
    result = subprocess.run(
        ["streamlit", "run", app_path],
        env=env,
        check=True
    )
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"Error running Streamlit: {e}")
    sys.exit(e.returncode)
except FileNotFoundError:
    print("Streamlit not found. Please install it with 'pip install streamlit'")
    sys.exit(1)
