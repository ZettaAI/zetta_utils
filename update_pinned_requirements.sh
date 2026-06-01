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
# web_api (CPU torch) conflicts with web_api-gpu (CUDA torch); exclude the CPU
# variant from --all-extras so requirements.all.txt keeps the default CUDA torch.
uv export --format requirements.txt --all-extras --no-extra web-api --no-emit-project --no-hashes $PRUNE_FLAGS -o requirements.all.txt

echo "Exporting 'modules' extra..."
uv export --format requirements.txt --extra modules --no-emit-project --no-hashes $PRUNE_FLAGS -o requirements.modules.txt

echo "Exporting 'web_api' extra..."
# torch resolves to the +cpu build (no nvidia-* CUDA wheels). Installers must
# supply the CPU wheel index, e.g. PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# (the web_api Dockerfile and the web-api-extras CI job set this).
uv export --format requirements.txt --extra web_api --no-emit-project --no-hashes $PRUNE_FLAGS -o requirements.web_api.txt

echo "Exporting 'web_api-gpu' extra..."
uv export --format requirements.txt --extra web_api-gpu --no-emit-project --no-hashes $PRUNE_FLAGS -o requirements.web_api_gpu.txt

echo ""
echo "Pinned requirements files updated successfully!"
echo "  - requirements.all.txt"
echo "  - requirements.modules.txt"
echo "  - requirements.web_api.txt"
echo "  - requirements.web_api_gpu.txt"
