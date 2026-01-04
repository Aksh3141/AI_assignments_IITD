#!/bin/bash

# This script takes a single argument, the base name of the test case.
# e.g., ./run2.sh testcase1
# It will read testcase1.city and testcase1.satoutput
# to produce testcase1.metromap.

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run2.sh <basename>"
    exit 1
fi

BASENAME=$1

# The decoder needs to know the problem specs (from .city) and the solution (from .satoutput).
# We pass both files as command-line arguments to the C++ program.
./decoder "${BASENAME}.city" "${BASENAME}.satoutput" > "${BASENAME}.metromap"