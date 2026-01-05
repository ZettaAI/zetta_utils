#!/bin/bash
# Script to update pinned dependency requirements
set -e

echo "Installing uv for fast dependency resolution..."
pip install uv

echo "Generating pinned requirements for 'all' extra..."
uv pip compile --upgrade --extra all -o requirements.all.txt pyproject.toml

echo "Generating pinned requirements for 'modules' extra..."
uv pip compile --upgrade --extra modules -o requirements.modules.txt pyproject.toml

echo ""
echo "Pinned requirements files updated successfully!"
echo "  - requirements.all.txt"
echo "  - requirements.modules.txt"
