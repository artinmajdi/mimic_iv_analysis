"""Test configuration and fixtures for the MIMIC-IV Analysis package."""

import os
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Fixture to provide path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_config():
    """Fixture to provide sample configuration for testing."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "mimic_iv_test",
            "user": "test_user",
            "password": "test_password"
        },
        "application": {
            "debug": True,
            "log_level": "DEBUG",
            "environment": "testing"
        }
    }

@pytest.fixture
def env_setup():
    """Fixture to set up test environment variables."""
    original_env = dict(os.environ)
    os.environ.update({
        "MIMIC_DB_HOST": "localhost",
        "MIMIC_DB_PORT": "5432",
        "MIMIC_DB_NAME": "mimic_iv_test",
        "ENVIRONMENT": "testing"
    })
    yield
    os.environ.clear()
    os.environ.update(original_env)
