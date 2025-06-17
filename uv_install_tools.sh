#!/bin/bash

# Script to install development and documentation tools using uv

set -e

echo "Installing development tools..."

# Development tools as dev dependencies
uv add --dev mypy
uv add --dev black  
uv add --dev flake8
uv add --dev pylint
uv add --dev isort
uv add --dev pytest
uv add --dev pytest-cov
uv add --dev coverage
uv add --dev pre-commit

echo "Installing documentation tools..."

# Documentation tools as dev dependencies
uv add --dev sphinx
uv add --dev sphinx-autodoc-typehints
uv add --dev sphinx-copybutton
uv add --dev sphinx_tabs

echo "Installing additional test tools..."

# Additional test tools as dev dependencies
uv add --dev pydocstyle

echo "All development and documentation tools installed successfully!"
