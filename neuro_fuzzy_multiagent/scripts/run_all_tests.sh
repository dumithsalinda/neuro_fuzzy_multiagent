#!/bin/bash
# run_all_tests.sh
# Run all test files in sequence and exit on first failure.

set -e

echo "Running test_fuzzy_system.py..."
pytest tests/test_fuzzy_system.py

echo "Running test_neural_network.py..."
pytest tests/test_neural_network.py

echo "Running test_neuro_fuzzy.py..."
pytest tests/test_neuro_fuzzy.py

echo "Running test_environment.py..."
pytest tests/test_environment.py

echo "All tests passed."
