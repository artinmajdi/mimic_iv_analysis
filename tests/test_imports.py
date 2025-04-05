#!/usr/bin/env python
"""
Test script to verify that the MIMIC-IV Analysis package is correctly installed and imports work.
"""

import os
import sys
import importlib.util

def check_module(module_name):
    """Check if a module can be imported and print status."""
    try:
        importlib.import_module(module_name)
        print(f"✅ Successfully imported: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {module_name}")
        print(f"   Error: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print("\nChecking system path:")
    for i, path in enumerate(sys.path):
        print(f"  {i+1}. {path}")

    print("\nChecking imports:")
    modules_to_check = [
        "mimic_iv_analysis",
        "mimic_iv_analysis.data",
        "mimic_iv_analysis.core",
        "mimic_iv_analysis.visualization",
        "mimic_iv_analysis.utils"
    ]

    all_passed = all(check_module(module) for module in modules_to_check)

    if all_passed:
        print("\n✅ All imports successful! The package is correctly installed.")
    else:
        print("\n❌ Some imports failed. Please check the installation.")

        print("\nTrying with relative imports:")
        relative_modules = [
            "src",
            "src.data",
            "src.core",
            "src.visualization",
            "src.utils"
        ]
        if all(check_module(module) for module in relative_modules):
            print("\nRelative imports work, but package imports fail.")
            print("To fix this, run: 'pip install -e .' from the project root.")
        else:
            print("\nBoth package and relative imports fail.")
            print("This could be a structure issue with the codebase.")
