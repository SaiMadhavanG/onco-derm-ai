#!/bin/bash

# Directory to monitor
MONITOR_DIR="data/02_intermediate"
EXTENSION=".pkl"

# Function to run the model_eval Kedro pipeline
run_model_eval_pipeline() {
    echo "Detected a new $EXTENSION file in $MONITOR_DIR. Triggering model_eval pipeline..."

    # Step 1: Run Kedro pipeline
    kedro run -p model_eval
    if [[ $? -ne 0 ]]; then
        echo "Error in running Kedro model_eval pipeline. Aborting."
        return
    fi

    echo "model_eval pipeline executed successfully."
}

# Ensure inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
    echo "inotifywait could not be found. Please install inotify-tools."
    exit 1
fi

# Monitor the directory for new files with the specified extension
echo "Monitoring directory $MONITOR_DIR for new $EXTENSION files..."
inotifywait -m -e create "$MONITOR_DIR" --format "%f" | while read -r new_file; do
    if [[ $new_file == *$EXTENSION ]]; then
        run_model_eval_pipeline
    fi
done
