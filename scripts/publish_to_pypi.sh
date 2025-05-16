#!/bin/bash
# Script to build and publish the project_src package to PyPI or GitHub Packages

set -e  # Exit immediately if a command exits with a non-zero status

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Define a default package name
DEFAULT_PACKAGE_NAME="mimic_iv_analysis"
PACKAGE_NAME=""

# Attempt to get package name from pyproject.toml
PYPROJECT_TOML_PATH="$PROJECT_ROOT/pyproject.toml"
if [ -f "$PYPROJECT_TOML_PATH" ]; then
    PACKAGE_NAME_FROM_FILE=$(grep -m 1 '^name\s*=\s*"[^"]*"' "$PYPROJECT_TOML_PATH" | sed 's/.*name\s*=\s*"\([^"]*\)".*/\1/')
    if [ -n "$PACKAGE_NAME_FROM_FILE" ]; then
        PACKAGE_NAME="$PACKAGE_NAME_FROM_FILE"
        echo "INFO: Using package name from pyproject.toml: $PACKAGE_NAME"
    else
        echo "WARNING: Could not parse package name from $PYPROJECT_TOML_PATH. It may be missing the 'name' field under [project]."
    fi
else
    echo "WARNING: pyproject.toml not found at $PYPROJECT_TOML_PATH."
fi

# If PACKAGE_NAME is still empty, use the default
if [ -z "$PACKAGE_NAME" ]; then
    PACKAGE_NAME="$DEFAULT_PACKAGE_NAME"
    echo "INFO: Using default package name: $PACKAGE_NAME"
fi

# Display banner
echo "============================================================"
echo "Publishing Script for $PACKAGE_NAME"
echo "============================================================"

# Ask user where to publish if not specified as argument
if [[ -z "$1" ]]; then
    echo "Where would you like to publish?"
    echo "1) PyPI (default)"
    echo "2) TestPyPI"
    echo "3) GitHub Packages"
    read -p "Enter your choice [1-3]: " choice

    case $choice in
        2) PUBLISH_TARGET="--test" ;;
        3) PUBLISH_TARGET="--github" ;;
        *) PUBLISH_TARGET="" ;; # Default to PyPI
    esac
else
    PUBLISH_TARGET="$1"
fi

# Check which repository we're publishing to
if [[ "$PUBLISH_TARGET" == "--test" ]]; then
    REPO="testpypi"
    REPO_NAME="TestPyPI"
    INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ $PACKAGE_NAME"
    TWINE_ARGS="--repository testpypi"
    echo "Publishing to TestPyPI"
elif [[ "$PUBLISH_TARGET" == "--github" ]]; then
    REPO="github"
    REPO_NAME="GitHub Packages"
    # Automatically get GitHub repo and branch information
    GITHUB_REPO=$(git config --get remote.origin.url | sed 's/.*github.com:\(.*\).git/\1/' | sed 's/.*github.com\/\(.*\).git/\1/')
    BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # Check for .pypirc file first
    PYPIRC_FILE="$PROJECT_ROOT/.pypirc"
    if [ -f "$PYPIRC_FILE" ]; then
        echo "Found .pypirc file, checking for GitHub credentials..."
        PYPIRC_GITHUB_USER=$(grep -A 2 "\[github\]" "$PYPIRC_FILE" | grep username | sed 's/username = //')
        PYPIRC_GITHUB_TOKEN=$(grep -A 2 "\[github\]" "$PYPIRC_FILE" | grep password | sed 's/password = //')

        # Use credentials from .pypirc if they exist
        if [ -n "$PYPIRC_GITHUB_USER" ]; then
            GITHUB_USER=$PYPIRC_GITHUB_USER
        fi
        if [ -n "$PYPIRC_GITHUB_TOKEN" ]; then
            GITHUB_TOKEN=$PYPIRC_GITHUB_TOKEN
        fi
    else
        echo "No .pypirc file found, checking environment variables..."
    fi

    # If not found in .pypirc, try environment variables
    if [ -z "$GITHUB_USER" ]; then
        GITHUB_USER=${GITHUB_USERNAME:-$GH_USERNAME}
    fi
    if [ -z "$GITHUB_TOKEN" ]; then
        GITHUB_TOKEN=${GITHUB_TOKEN:-$GH_TOKEN}
    fi

    # If still not found, prompt user
    if [ -z "$GITHUB_USER" ]; then
        read -p "Enter GitHub username: " GITHUB_USER
    fi
    if [ -z "$GITHUB_TOKEN" ]; then
        read -sp "Enter GitHub Personal Access Token: " GITHUB_TOKEN
        echo
    fi
    INSTALL_CMD="pip install --index-url https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}/raw/${BRANCH}/dist/ $PACKAGE_NAME"
    TWINE_ARGS="--repository github"
    echo "Publishing to GitHub Packages"
else
    REPO="pypi"
    REPO_NAME="PyPI"
    INSTALL_CMD="pip install $PACKAGE_NAME"
    TWINE_ARGS="--repository pypi"
    echo "Publishing to PyPI"
fi

# Function to get current version from pyproject.toml
get_current_version() {
    grep -m 1 '^version\s*=\s*"[^"]*"' pyproject.toml | sed 's/.*version\s*=\s*"\([^"]*\)".*/\1/'
}

# Clean previous builds first
echo "Cleaning previous build artifacts..."
rm -rf dist build ${PACKAGE_NAME}.egg-info

# Get current version
VERSION=$(get_current_version)
echo "Using version $VERSION from pyproject.toml"
echo "To publish a new version, please update the 'version' in pyproject.toml manually before running this script."

# Install required build tools
echo "Installing/upgrading build tools..."
python -m pip install --upgrade pip build twine

# Build the package
echo "Building distribution packages..."
python -m build

# Upload to repository
echo "Uploading to ${REPO_NAME}..."

if [[ "$REPO" == "github" ]]; then
    echo "Publishing to GitHub Repository..."

    # Create a packages directory if it doesn't exist
    PACKAGES_DIR="$PROJECT_ROOT/packages"
    mkdir -p "$PACKAGES_DIR"

    # Copy the built packages to the packages directory
    cp "$PROJECT_ROOT/dist"/* "$PACKAGES_DIR/"

    # Create a README.md file in the packages directory with installation instructions
    cat > "$PACKAGES_DIR/README.md" << EOF
# $PACKAGE_NAME Package

## Version $VERSION

This directory contains the built packages for the $PACKAGE_NAME Python package.

## Installation

To install directly from this repository:

\`\`\`bash
pip install git+https://github.com/${GITHUB_REPO}.git
\`\`\`

Or download the wheel file and install it locally:

\`\`\`bash
pip install $PACKAGE_NAME-$VERSION-py3-none-any.whl
\`\`\`
EOF

    # Commit and push the changes
    cd "$PROJECT_ROOT"
    git add "packages/"
    git commit -m "Add package version $VERSION"

    # Push to GitHub using HTTPS with token
    REPO_URL="https://$GITHUB_USER:$GITHUB_TOKEN@github.com/${GITHUB_REPO}.git"
    git push "$REPO_URL" HEAD:$BRANCH

    echo "\nPackage published to GitHub Repository!"
    echo "You can install it with:"
    echo "pip install git+https://github.com/${GITHUB_REPO}.git"

    # Or use PyPI for regular publishing
else
    # Set environment variables for twine to use credentials from .pypirc
    # Check for .pypirc and required sections
    PYPIRC_FILE_FULL_PATH="$PROJECT_ROOT/.pypirc"
    if [ ! -f "$PYPIRC_FILE_FULL_PATH" ]; then
        echo "Error: .pypirc file not found at $PYPIRC_FILE_FULL_PATH"
        echo "Please create it with your credentials for $REPO_NAME (section [$REPO])."
        exit 1
    fi

    if ! grep -q "\[$REPO\]" "$PYPIRC_FILE_FULL_PATH"; then
        echo "Error: Section '[$REPO]' not found in $PYPIRC_FILE_FULL_PATH."
        echo "Please add it with 'username = __token__' and 'password = YOUR_API_TOKEN'."
        exit 1
    fi

    # Check if password for the section exists. Note: grep -A 2 finds the section and the next 2 lines.
    if ! grep -A 2 "\[$REPO\]" "$PYPIRC_FILE_FULL_PATH" | grep -q "password"; then
        echo "Error: Password not found for section '[$REPO]' in $PYPIRC_FILE_FULL_PATH."
        echo "Please add 'password = YOUR_API_TOKEN' under the '[$REPO]' section."
        exit 1
    fi

    # Upload to PyPI or TestPyPI using the local .pypirc file with verbose output
    python -m twine upload --verbose --config-file "$PYPIRC_FILE_FULL_PATH" $TWINE_ARGS dist/*

    echo "\nPackage uploaded to ${REPO_NAME}!"
    echo "You can install it with:"
    echo "$INSTALL_CMD"
fi

echo "Package uploaded successfully!"
echo "You can install it with:"
echo "$INSTALL_CMD"

echo "Process completed successfully!"
