#!/bin/bash

# This script checks the modification date of pyproject.toml
# against a stored date in ~/.config, and if newer, it runs
# the pip command to upgrade all the tools referenced by that
# pyproject file.  It then stores the current date/time so 
# that next time, it will know your tools are already up to date.
#
# The net effect is: you can run this script frequently (e.g.
# upon login), and it will upgrade your stuff when needed.

# Define the path to the pyproject.toml file
PROJECT_FILE="pyproject.toml"

# Define a file under your home directory to store the last checked date
STORED_DATE_FILE="$HOME/.config/zetta/last_pyproject_check"

# Ensure the directory for the stored date exists
mkdir -p "$(dirname "$STORED_DATE_FILE")"

# Get the modification date of pyproject.toml in seconds since epoch
PROJECT_MOD_DATE=$(date -r "$PROJECT_FILE" +%s)

# Check if the stored date file exists
if [ -f "$STORED_DATE_FILE" ]; then
    # Read the stored date from the file
    STORED_DATE=$(cat "$STORED_DATE_FILE")
else
    # If the file doesn't exist, set STORED_DATE to 0
    STORED_DATE=0
fi

# If the stored date is older than the project file's modification date
if [ "$STORED_DATE" -lt "$PROJECT_MOD_DATE" ]; then
    # Run the pip install command
    echo "pyproject.toml has been updated; running pip"
    pip install -e '.[all]' --upgrade
    
    # Store the current date/time in seconds since epoch
    date +%s > "$STORED_DATE_FILE"
    echo "Upgrade complete."
fi
