#!/bin/sh

SCRIPT_DIR=$(dirname "$0")
TARGET_DIR="$SCRIPT_DIR/zetta_utils/internal"

cd $TARGET_DIR

if git merge-base --is-ancestor HEAD origin/main; then
    echo "Submodule's commit is on the main branch."
else
    echo "Error: Submodule's commit is not on the main branch."
    exit 1
fi

cd -
