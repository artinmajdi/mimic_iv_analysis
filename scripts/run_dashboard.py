#!/usr/bin/env python
"""
Launcher script for the MIMIC-IV Analysis Streamlit app.
Sets PYTHONPATH properly before launching.
Allows user to select which dashboard to run from available options.
"""

import os
import sys
import subprocess
from typing import List, Tuple

def find_dashboard_apps(src_dir: str) -> List[Tuple[str, str]]:
    """
    Find all app.py files in subdirectories of src_dir.
    Returns a list of tuples (dashboard_name, app_path).
    """
    dashboard_apps = []

    # Skip these directories and files
    skip_items = {'.DS_Store', '__pycache__', '__init__.py'}

    for item in os.listdir(src_dir):
        if item in skip_items:
            continue

        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            # Look for app.py in this directory
            potential_app = os.path.join(item_path, 'app.py')
            if os.path.isfile(potential_app):
                dashboard_apps.append((item, potential_app))

    return dashboard_apps

def select_dashboard(dashboard_apps: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Present a menu of available dashboards and get user selection.
    Returns the selected (dashboard_name, app_path).
    """
    if not dashboard_apps:
        print("No dashboards found in src directory!")
        sys.exit(1)

    print("\nAvailable Dashboards:")
    for idx, (name, _) in enumerate(dashboard_apps, 1):
        print(f"{idx}. {name}")

    while True:
        try:
            choice = input("\nSelect a dashboard number to run: ")
            idx = int(choice) - 1
            if 0 <= idx < len(dashboard_apps):
                return dashboard_apps[idx]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # Get the project root directory (one level up from scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    src_dir = os.path.join(project_root, "src")

    # Make sure the project root is in the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Print environment info
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Find available dashboards
    dashboard_apps = find_dashboard_apps(src_dir)

    # Let user select a dashboard
    selected_name, app_path = select_dashboard(dashboard_apps)

    # Setup environment for the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # Create a demo path variable - we'll let the app handle data loading
    demo_path = os.path.join(project_root, "demo_mimic_iv")
    env["DEMO_MIMIC_PATH"] = demo_path

    # Run Streamlit
    print(f"\nStarting {selected_name} dashboard with: {app_path}")
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
