#!/bin/bash

# This script takes a single argument, the base name of the test case.
# e.g., ./run1.sh testcase1
# It will read testcase1.city and produce testcase1.satinput.

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run1.sh <basename>"
    exit 1
fi

BASENAME=$1

# Run the encoder, redirecting stdin from the .city file
# and stdout to the .satinput file.
./encoder < "${BASENAME}.city" > "${BASENAME}.satinput"