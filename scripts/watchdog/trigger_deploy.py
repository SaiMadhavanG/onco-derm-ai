import subprocess
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("data/06_models/best_model_uri.txt"):
            run_pipeline()


def run_pipeline():
    try:
        # # Step 1: Run the Python script
        # subprocess.run(["python3", "src/onco_derm_ai/pipelines/retrain/replace_train.py"], check=True)

        # # Step 2: Run Kedro commands
        # subprocess.run(["kedro", "run", "-p", "data_preprocessing"], check=True)
        # subprocess.run(["kedro", "run", "-p", "model_training"], check=True)
        subprocess.run(["kedro", "run", "-t", "retrained"], check=True)

        # Step 3: Build Docker image
        timestamp = time.strftime("%Y%m%d%H%M%S")
        image_name = f"saimadhavang/onco-derm-ai:{timestamp}"
        subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

        # Step 4: Run Docker container
        subprocess.run(
            ["docker", "run", "-p", "8000:8000", "--gpus", "all", image_name],
            check=True,
        )

    except subprocess.CalledProcessError:
        pass


if __name__ == "__main__":
    path = "data/06_models"
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
