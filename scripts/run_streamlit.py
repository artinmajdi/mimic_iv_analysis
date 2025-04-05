#!/usr/bin/env python
"""
Launcher script for the MIMIC-IV Analysis Streamlit app.
Sets PYTHONPATH properly before launching.
"""

import os
import sys
import subprocess

def main():
    # Get the project root directory (one level up from scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Make sure the project root is in the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Print environment info
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Setup environment for the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # Create a demo path variable - we'll let the app handle data loading
    demo_path = os.path.join(project_root, "demo_mimic_iv")
    env["DEMO_MIMIC_PATH"] = demo_path

    # App path
    app_path = os.path.join(project_root, "src", "mimic_iv_analysis", "visualization", "app.py")

    # Run Streamlit
    print(f"Starting Streamlit with: {app_path}")
    try:
        result = subprocess.run(
            ["streamlit", "run", app_path],
            env=env,
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Streamlit not found. Please install it with: pip install streamlit")
        return 1

if __name__ == "__main__":
    sys.exit(main())
