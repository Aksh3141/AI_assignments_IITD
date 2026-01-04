#!/bin/bash
# Run script for the project
# Usage: ./run.sh hailfinder.bif records.dat

# Exit if any command fails
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh <bif_file> <data_file>"
    exit 1
fi

BIF_FILE=$1
DATA_FILE=$2
OUTPUT_FILE="solved_hailfinder.bif"

# Run the solver
./solver "$BIF_FILE" "$DATA_FILE" "$OUTPUT_FILE"

echo "Run complete. Output written to $OUTPUT_FILE"

