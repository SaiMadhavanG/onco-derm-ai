#!/bin/bash

# File to monitor
MONITOR_FILE="data/06_models/best_model_uri.txt"

# Docker Hub repository details
DOCKER_USERNAME="saimadhavang"
DOCKER_REPO="onco-derm-ai"

# Function to run both retrain and deploy pipelines
run_pipeline() {
    echo "Change detected in $MONITOR_FILE. Triggering retrain and deploy pipelines..."

    # Step 1: Run the Python script
    python3 src/onco_derm_ai/pipelines/retrain/replace_train.py
    if [[ $? -ne 0 ]]; then
        echo "Error in running replace_train.py. Aborting pipeline."
        return
    fi

    # Step 2: Run Kedro commands for data preprocessing
    kedro run -p data_preprocessing
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro data_preprocessing pipeline. Aborting pipeline."
        return
    fi

    # Optional: Start MLflow UI (can be removed if not required)
    mlflow ui --port 5000 &

    # Step 3: Run Kedro commands for model training
    kedro run -p model_training
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro model_training pipeline. Aborting pipeline."
        return
    fi

    # Step 4: Run Kedro retrained pipelines
    kedro run -t model_retrained -p ood_detection
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro retrained pipeline (OOD detection). Aborting pipeline."
        return
    fi

    kedro run -t model_retrained -p conformal_prediction
    if [[ $? -ne 0 ]]; then
        echo "Error in Kedro retrained pipeline (Conformal prediction). Aborting pipeline."
        return
    fi

    # Step 5: Build Docker image
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    IMAGE_NAME="$DOCKER_USERNAME/$DOCKER_REPO:$TIMESTAMP"
    docker build -t $IMAGE_NAME .
    if [[ $? -ne 0 ]]; then
        echo "Error in Docker build. Aborting pipeline."
        return
    fi

    # Step 6: Push Docker image to Docker Hub
    echo "Pushing Docker image to Docker Hub: $IMAGE_NAME"
    docker push $IMAGE_NAME
    if [[ $? -ne 0 ]]; then
        echo "Error in pushing Docker image to Docker Hub. Aborting pipeline."
        return
    fi

    # Step 7: Run Docker container
    docker run -p 8000:8000 --gpus all $IMAGE_NAME
    if [[ $? -ne 0 ]]; then
        echo "Error in running Docker container. Aborting pipeline."
        return
    fi

    echo "Pipeline executed successfully."
}

# Ensure inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
    echo "inotifywait could not be found. Please install inotify-tools."
    exit 1
fi

# Monitor the file for any changes (write, attribute, etc.)
echo "Monitoring changes in $MONITOR_FILE..."
inotifywait -m -e modify -e attrib -e close_write "$MONITOR_FILE" | while read -r path event file; do
    if [[ "$path$file" == "$MONITOR_FILE" ]]; then
        run_pipeline
    fi
done
