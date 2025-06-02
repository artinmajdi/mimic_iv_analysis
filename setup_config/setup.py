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

# Package metadata
PACKAGE_NAME = "mimic_iv_analysis"
VERSION      = "1.0.0"
AUTHOR       = "Artin Majdi"
AUTHOR_EMAIL = "msm2024@gmail.com"
DESCRIPTION  = "A comprehensive toolkit for analyzing MIMIC-IV clinical database"
URL          = "https://github.com/artinmajdi/mimic_iv_analysis"
LICENSE      = "MIT License"

CLASSIFIERS = [
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
]

DEV_REQUIREMENTS = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=3.9",
    "mypy>=0.9",
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
]

KEYWORDS = [
    "healthcare",
    "clinical-data",
    "mimic-iv",
    "data-analysis",
    "machine-learning",
    "medical-research",
]


EXTRA_REQUIRES = {
    "dev": DEV_REQUIREMENTS,
    "test": [
        "pytest>=6.0",
        "pytest-cov>=2.0",
    ],
    "docs": [
        "sphinx>=4.0",
        "sphinx-rtd-theme>=1.0",
        "sphinx-autodoc-typehints>=1.12",
    ],
}

# Read requirements
requirements = read_requirements('requirements.txt')

# Read long description from README
long_description = read_file('README.md')

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    python_requires=">=3.10,<3.13",
    install_requires=requirements,
    extras_require=EXTRA_REQUIRES,
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    zip_safe=False,
    keywords=KEYWORDS,
    project_urls={
        "Issue Tracker": f"{URL}/issues",
        "Documentation": f"{URL}/blob/main/docs/README.md",
        "Source": URL,
    },
)
