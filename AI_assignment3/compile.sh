#!/bin/bash

# This script compiles the encoder and decoder programs.
# The -std=c++17 flag enables modern C++ features.
# The -O2 flag enables optimizations for faster execution.

echo "Compiling encoder..."
g++ -std=c++17 -O2 -o encoder encoder.cpp

echo "Compiling decoder..."
g++ -std=c++17 -O2 -o decoder decoder.cpp

echo "Compilation finished."