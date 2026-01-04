#!/bin/bash
# Compile script for the project

# Exit immediately if any command fails
set -e

# Compile solver.cpp with C++17 standard
g++ -std=c++17 startup_code.cpp -o solver
echo "Compilation successful. Executable 'solver' created."
