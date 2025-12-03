#!/bin/bash

# Script name: run_inference.sh
# Description: Run the main method of inference/basic-inference.py

# Get the project root directory (where this script is located)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Set Python path (ensure project modules can be imported)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if the Python file exists
INFERENCE_FILE="inference/basic-inference.py"
if [ ! -f "$INFERENCE_FILE" ]; then
    echo "Error: File not found - $INFERENCE_FILE"
    exit 1
fi

# Run the Python script
echo "Running $INFERENCE_FILE ..."
python3 "$INFERENCE_FILE" "$@"

# Capture exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script executed successfully"
else
    echo "Script execution failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE