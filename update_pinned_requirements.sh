#!/bin/bash
# Script to update pinned dependency requirements
set -e

echo "Installing uv for fast dependency resolution..."
pip install uv

echo "Locking dependencies..."
uv lock --upgrade --fork-strategy requires-python --resolution highest

# Build --prune flags from requirements.pruned.txt (one package per line,
# blank lines and # comments ignored).
PRUNE_FLAGS=""
while IFS= read -r pkg; do
    pkg="${pkg%%#*}"             # strip trailing inline comments
    pkg="${pkg//[[:space:]]/}"   # strip whitespace
    [[ -z "$pkg" ]] && continue
    PRUNE_FLAGS+=" --prune $pkg"
done < requirements.pruned.txt

echo "Exporting 'all' extra..."
uv export --format requirements.txt --all-extras --no-emit-project --no-hashes $PRUNE_FLAGS -o requirements.all.txt

echo "Exporting 'modules' extra..."
uv export --format requirements.txt --extra modules --no-emit-project --no-hashes $PRUNE_FLAGS -o requirements.modules.txt

echo ""
echo "Pinned requirements files updated successfully!"
echo "  - requirements.all.txt"
echo "  - requirements.modules.txt"
