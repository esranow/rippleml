#!/bin/bash
# Run all tests for NeuralWave Core

set -e

echo "Running NeuralWave Core test suite..."
echo "======================================"

# Set PYTHONPATH to parent directory
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Run pytest with verbose output
pytest "$(dirname "${BASH_SOURCE[0]}")" -v --tb=short

echo ""
echo "======================================"
echo "All tests completed successfully!"
