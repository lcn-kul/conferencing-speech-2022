from typing import List
from datasets import Audio, load_dataset
import os
from pathlib import Path
from torch.utils.data import DataLoader

from src import constants


def _fix_path(example):
    # If the path does not start with "data/", then it's already finished.
    audio_path = Path(example["audio_path"])
    # feat_path = Path(example["feat_path"])
    if audio_path.parts[0] != constants.DIR_DATA.name:
        return example

    # Fix relative audio path.
    full_path = constants.DIR_PROJECT.joinpath(audio_path)
    new_path = os.path.relpath(full_path, os.getcwd())
    example["audio_path"] = new_path
    example["audio"] = new_path

    # # Fix relative feature path.
    # full_path = constants.DIR_PROJECT.joinpath(feat_path)
    # new_path = os.path.relpath(full_path, os.getcwd())
    # example["feat_path"] = new_path

    # Return example.
    return example


def create_train_data_loader(sampling_rate: int, use_example: bool):

    # Create list of CSV paths.
    csv_paths: List[Path] = []
    csv_paths.append(constants.DATASET_TRAIN_EG.csv_path)
    # csv_type = "example_csvs" if use_example else "processed_csvs"
    # csv_paths.extend(constants.IU_BLOOMINGTON_TRAIN[csv_type])
    # csv_paths.extend(constants.NISQA_TRAIN[csv_type])
    # csv_paths.extend(constants.PSTN_TRAIN[csv_type])
    # csv_paths.extend(constants.TENCENT_TRAIN[csv_type])

    # Make sure files exist.
    for csv_path in csv_paths:
        assert csv_path.exists()

    # Convert Posix paths to strings.
    csv_paths = map(str, csv_paths)

    # Create dataset.
    dataset = load_dataset("csv", data_files=csv_paths, sep=",")

    # Fix relative paths.
    dataset["train"] = dataset["train"].map(_fix_path)

    # Set up audio loading.
    dataset["train"] = dataset["train"].cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate),
    )

    # Create data loader.
    dl = DataLoader(dataset["train"])
    return dl
