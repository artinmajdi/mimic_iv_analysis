# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands

Run unit tests:
```bash
pytest tests/test_data_loader.py::test_data_loader_initialization -v  # Single test
pytest tests/ -v  # All tests
pytest tests/ -k "test_scan" -v  # Tests matching pattern
pytest tests/test_data_loader.py -v  # Specific test file
```

Lint code (requires installing dev extras):
```bash
pip install -e ".[dev]"
flake8 mimic_iv_analysis/
black mimic_iv_analysis/ --line-length 120
mypy mimic_iv_analysis/
```

Run dashboard application:
```bash
python scripts/run_dashboard.py
```

## Code Style Guidelines

1. **Import Order**: Standard library, third-party packages, local imports (each section alphabetically sorted)
2. **Type Hints**: Use type hints from typing module (Dict, Optional, List, Union, Literal)
3. **Constants**: ALL_CAPS with underscores (e.g., DEFAULT_MIMIC_PATH, RANDOM_STATE)
4. **Enums**: Use enum.Enum for table names and module-specific constants
5. **Properties**: Use @property, @cached_property, @lru_cache for expensive operations
6. **Logging**: Use standard logging module (logging.info, logging.warning)
7. **Error Handling**: Graceful failures with informative error messages, return empty DataFrames/lists when appropriate
8. **Line Length**: Maximum 120 characters
9. **Column Types**: Use appropriate dtypes for pandas/dask columns (int64, string, category)
10. **String Formatting**: Use f-strings for clarity
11. **Docstrings**: Clear method documentation explaining parameters, returns, and behaviors
12. **Private Methods**: Prefix with underscore (e.g., _helper_method)
13. **Progress Bars**: Use tqdm for long-running operations
14. **DataFrames**: Support both pandas and dask DataFrames with type checking
15. **File Paths**: Use pathlib.Path whenever possible
16. **Testing**: Use pytest fixtures for setup, assert clear test descriptions
17. **Visualization**: Streamlit for dashboard UI components
18. **NEVER add comments unless explicitly requested by user
19. **Always ignore**: Files inside folders called 'backup' or files with 'backup' in their name