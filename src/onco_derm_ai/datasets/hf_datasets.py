import os
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from kedro.io import AbstractDataset


class HFImageFolderDataSet(AbstractDataset):
    """
    A Kedro dataset for loading and saving datasets in the Hugging Face ImageFolder format.

    This dataset is designed to handle image datasets organized into folders by class, as expected by Hugging Face's `ImageFolder` format.

    Example directory structure:

    root/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image3.jpg
            image4.jpg

    Parameters
    ----------
    data_dir: str
        Path to the root directory of the dataset.
    split_config: Optional[Dict[str, float]]
        Optional dictionary specifying splits for train/test/validation.
        Example: {"train": 0.8, "test": 0.2}. If provided, splits the dataset accordingly.
    include_metadata: bool, default=False
        Whether to include additional metadata about the dataset (e.g., metadata.csv or metadata.jsonl).

    Usage in Kedro Catalog
    ----------------------
    ```yaml
    my_image_folder_dataset:
      type: path.to.HuggingFaceImageFolderDataSet
      data_dir: data/01_raw/images
      split_config:
        train: 0.8
        test: 0.2
      include_metadata: true
    ```

    Returns
    -------
    A `DatasetDict` containing splits (if split_config is provided) or a single dataset.
    Metadata is included if `include_metadata` is set to True.

    Methods
    -------
    load():
        Loads the dataset and optionally splits it.
    save(data):
        Saves the dataset back to the specified directory in ImageFolder format.
    exists():
        Checks if the dataset directory exists.
    describe():
        Returns a description of the dataset configuration.
    """

    def __init__(
        self,
        data_dir: str,
        split_config: Optional[Dict[str, float]] = None,
        include_metadata: bool = False,
    ):
        self._data_dir = data_dir
        self._split_config = split_config
        self._include_metadata = include_metadata

    def _load(self) -> Any:
        # Load the dataset from the specified directory
        dataset = load_dataset("imagefolder", data_dir=self._data_dir)

        # Handle splits if specified
        if self._split_config:
            dataset = dataset.train_test_split(
                test_size=self._split_config.get("test", 0.2)
            )

            # Further split the train set if validation is included
            if "train" in self._split_config and "validation" in self._split_config:
                train_val_split = dataset["train"].train_test_split(
                    test_size=self._split_config["validation"]
                    / (1 - self._split_config["test"])
                )
                dataset = DatasetDict(
                    {
                        "train": train_val_split["train"],
                        "validation": train_val_split["test"],
                        "test": dataset["test"],
                    }
                )

        if self._include_metadata:
            # Include metadata if a metadata.csv or metadata.jsonl file exists
            metadata_path_csv = os.path.join(self._data_dir, "metadata.csv")
            metadata_path_jsonl = os.path.join(self._data_dir, "metadata.jsonl")

            if os.path.exists(metadata_path_csv):
                import pandas as pd

                metadata = pd.read_csv(metadata_path_csv)
                dataset = dataset.add_column(
                    "metadata", metadata.to_dict(orient="records")
                )

            elif os.path.exists(metadata_path_jsonl):
                import jsonlines

                with jsonlines.open(metadata_path_jsonl) as reader:
                    metadata = list(reader)
                dataset = dataset.add_column("metadata", metadata)

        return dataset

    def _save(self, data: Any) -> None:
        """
        Save the dataset back to the specified directory in ImageFolder format.

        Parameters
        ----------
        data: DatasetDict
            The Hugging Face dataset to save. Assumes the dataset has the same structure
            as it was loaded.
        """
        from pathlib import Path

        output_dir = Path(self._data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split, dataset in data.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            metadata = []

            for i, example in enumerate(dataset):
                label = dataset.features["label"].names[example["label"]]
                class_dir = split_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)

                # Save image to class directory
                image = example["image"]
                image_name = f"image_{i}.jpg"
                image.save(class_dir / image_name)

                # Collect metadata if available
                metadata.append(
                    {"file_name": image_name, **example.get("metadata", {})}
                )

            # Save metadata if available
            if metadata:
                import pandas as pd

                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_csv(split_dir / "metadata.csv", index=False)

    def _exists(self) -> bool:
        return os.path.exists(self._data_dir)

    def _describe(self) -> Dict[str, Any]:
        return {
            "data_dir": self._data_dir,
            "split_config": self._split_config,
            "include_metadata": self._include_metadata,
        }


class HFDatasetWrapper(AbstractDataset):
    def __init__(
        self,
        dataset_path: str,
    ):
        self.dataset_path = dataset_path

    def _load(self) -> Dataset:
        """Loads the dataset from the specified path."""
        return load_from_disk(
            self.dataset_path,
        )

    def _save(self, data: Dataset) -> None:
        """Saves the dataset to disk using Hugging Face's `save_to_disk` method."""
        data.save_to_disk(self.dataset_path)

    def _exists(self) -> bool:
        """Checks if the dataset exists locally by verifying the presence of files in the cache directory."""
        cache_dir = self._load_args.get("cache_dir")
        if cache_dir:
            return os.path.exists(cache_dir)
        return False

    def _describe(self) -> Dict[str, Any]:
        """Returns a description of the dataset configuration."""
        return {
            "dataset_path ": self.dataset_path,
        }
