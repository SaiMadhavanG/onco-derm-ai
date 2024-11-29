#!/bin/bash
# Define the file to monitor
MONITOR_FILE="data/monitoring/current.pkl"

# Function to run the pipeline
run_pipeline() {
    echo "Detected change in $MONITOR_FILE. Triggering pipeline..."

    # Step 1: Run the Python script
    python3 src/onco_derm_ai/pipelines/retrain/replace_train.py
    if [[ $? -ne 0 ]]; then
        echo "Error in running replace_train.py. Aborting pipeline."
        return
    fi

    # Step 2: Run Kedro commands
    kedro run -p data_preprocessing
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro data_preprocessing pipeline. Aborting pipeline."
        return
    fi

    mlflow ui --port 5000 &

    kedro run -p model_training
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro model_training pipeline. Aborting pipeline."
        return
    fi

    echo "Retrain pipeline executed successfully."
}

# Ensure inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
    echo "inotifywait could not be found. Please install inotify-tools."
    exit 1
fi

# Monitor the file for changes
echo "Monitoring changes in $MONITOR_FILE..."
while inotifywait -e close_write "$MONITOR_FILE"; do
    echo "Detected change in $MONITOR_FILE. Triggering pipeline..."
    run_pipeline
done
