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

def find_dashboard_apps(visualization_dir: str) -> List[Tuple[str, str]]:
    """
    Find all files starting with 'app' in the visualization_dir.
    Returns a list of tuples (dashboard_name, app_path).
    """
    dashboard_apps = []
    skip_items = {'.DS_Store', '__pycache__', '__init__.py'}
    for item in os.listdir(visualization_dir):
        if item in skip_items:
            continue
        if item.startswith('app') and item.endswith('.py'):
            dashboard_name = item.replace('.py', '')
            dashboard_apps.append((dashboard_name, os.path.join(visualization_dir, item)))
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
    visualization_dir = os.path.join(project_root, 'mimic_iv_analysis', 'visualization')

    # Make sure the project root is in the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Print environment info
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Find available dashboards
    dashboard_apps = find_dashboard_apps(visualization_dir)

    # MODIFICATION START
    # Try to auto-select 'app.py' as this is the target specified by the user.
    # The visualization_dir is already calculated and is where app.py should be.
    app_py_target_path = os.path.join(visualization_dir, 'app.py')
    selected_name = None
    app_path = None

    # Check if dashboard_apps is empty first (this case is also handled by select_dashboard if called)
    if not dashboard_apps:
        print(f"No dashboards found in {visualization_dir}. Exiting.")
        sys.exit(1)

    for name_iter, path_iter in dashboard_apps:
        # Compare absolute paths to be certain
        if os.path.abspath(path_iter) == os.path.abspath(app_py_target_path):
            selected_name = name_iter
            app_path = path_iter
            print(f"Automatically selecting dashboard: {selected_name} ({app_path}) as it matches the target app.py.")
            break

    if not selected_name:
        # If app.py was not specifically found by path, and there's only one app, select it.
        if len(dashboard_apps) == 1:
            selected_name, app_path = dashboard_apps[0]
            print(f"Automatically selecting the only available dashboard: {selected_name} ({app_path})")
        else:
            # Fallback to original interactive selection if multiple choices and target 'app.py' not found by path
            print("Multiple dashboards found and the target 'app.py' was not uniquely identified by path, proceeding to manual selection.")
            selected_name, app_path = select_dashboard(dashboard_apps) # Original call
    # MODIFICATION END

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
