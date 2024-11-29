#!/bin/bash

# Define the file to monitor
MONITOR_FILE="data/06_models/best_model_uri.txt"

# Function to run the pipeline
run_pipeline() {
    echo "Detected change in $MONITOR_FILE. Triggering pipeline..."

    # Step 1: Run Kedro retrained pipeline
    kedro run -t model_retrained -p ood_detection
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro retrained pipeline. Aborting pipeline."
        return
    fi

    # Step 1: Run Kedro retrained pipeline
    kedro run -t model_retrained -p conformal_prediction
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro retrained pipeline. Aborting pipeline."
        return
    fi

    # Step 2: Build Docker image
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    IMAGE_NAME="saimadhavang/onco-derm-ai:$TIMESTAMP"
    docker build -t $IMAGE_NAME .
    if [[ $? -ne 0 ]]; then
        echo "Error in Docker build. Aborting pipeline."
        return
    fi

    # Step 3: Run Docker container
    docker run -p 8000:8000 --gpus all $IMAGE_NAME
    if [[ $? -ne 0 ]]; then
        echo "Error in running Docker container. Aborting pipeline."
        return
    fi

    echo "Deploy pipeline executed successfully."
}

# Ensure inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
    echo "inotifywait could not be found. Please install inotify-tools."
    exit 1
fi

# Monitor the file for changes
echo "Monitoring changes in $MONITOR_FILE..."
while inotifywait -e close_write "$MONITOR_FILE"; do
    run_pipeline
done
