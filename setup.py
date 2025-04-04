"""Setup configuration for MIMIC-IV Analysis package."""

import os
from setuptools import setup, find_packages

def read_requirements(filename: str) -> list[str]:
    """Read requirements from file."""
    requirements = []
    if not os.path.exists(filename):
        return requirements
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

def read_file(filename: str) -> str:
    """Read file contents."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
requirements = read_requirements('setup_config/requirements.txt')

# Read long description from README
long_description = read_file('README.md')

setup(
    name="mimic_iv_analysis",
    version="0.1.0",
    description="A comprehensive toolkit for analyzing MIMIC-IV clinical database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Artin Majdi",
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/yourusername/mimic_iv_analysis",  # Replace with your repo URL
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=requirements,
    python_requires=">=3.10,<3.13",
    entry_points={
        "console_scripts": [
            "mimic-iv=mimic_iv_analysis.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "healthcare",
        "clinical-data",
        "mimic-iv",
        "data-analysis",
        "machine-learning",
        "medical-research",
    ],
    project_urls={
        "Documentation": "https://mimic-iv-analysis.readthedocs.io/",
        "Source": "https://github.com/yourusername/mimic_iv_analysis",
        "Issue Tracker": "https://github.com/yourusername/mimic_iv_analysis/issues",
    },
    include_package_data=True,
    zip_safe=False,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
)
