import csv
import glob
import io
import librosa
import numpy as np
import os
import soundfile as sf
import torch
import webdataset as wds

from src import constants
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.run_once import run_once
from src.utils.split import Split, ALL_SPLITS, DEV_SPLITS
from src.utils.full_path import full_path


def _decode_non_mp3_file_like(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio

    array, sampling_rate = sf.read(file)
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=new_sr,
            res_type="kaiser_best"
        )
        sampling_rate = new_sr
    return array, sampling_rate


def load_audio(file_path: str, sampling_rate: int) -> torch.Tensor:
    array, _ = _decode_non_mp3_file_like(file_path, sampling_rate)
    array = np.float32(array)
    return torch.from_numpy(array)


def serialize_pt(x: torch.Tensor) -> bytes:
    stream = io.BytesIO()
    torch.save(x, stream)
    return stream.getvalue()


def _create_shards(split: Split, example: bool):

    if example:
        max_size = 50e6  # 50 MB
    else:
        max_size = 1e9  # 1 GB

    # 100k items
    if example:
        max_count = 1e3  # 1k items
    else:
        max_count = 100e3  # 100k items

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example)

    # Load CSV.
    rows = []
    with open(dataset.csv_path, mode="r", encoding="utf8") as csv_in:
        csv_reader = csv.reader(csv_in)
        for idx, row in enumerate(csv_reader):
            # Skip header row & empty rows.
            if idx == 0 or len(row) == 0:
                continue
            rows.append(row)

    # Write shards for all input types.
    types = {
        'audio': STANDARDIZED_CSV_INFO.col_audio_path,  # 1st column
        'mfcc': STANDARDIZED_CSV_INFO.col_mfcc_path,  # 2nd column
        'mfcc_ext': STANDARDIZED_CSV_INFO.col_mfcc_ext_path,  # 3rd column
        'xlsr': STANDARDIZED_CSV_INFO.col_xlsr_path,  # 4th column
    }
    for key in types:
        print(f"Creating '{key}' shards...")

        # Remove existing shards.
        glob_path = str(dataset.shards_dir.joinpath(f"{key}.*.tar"))
        existing_paths = glob.glob(glob_path)
        if len(existing_paths) > 0:
            print(f"Removing {len(existing_paths)} existing shards.")
            for existing_path in existing_paths:
                os.remove(existing_path)

        # This is the output pattern under which we write shards.
        pattern = os.path.join(dataset.shards_dir, f"{key}.%06d.tar")
        with wds.ShardWriter(pattern, maxsize=int(max_size), maxcount=int(max_count)) as sink:
            for idx, row in enumerate(rows):
                col = types[key]

                # Load features and convert to Tensor.
                file_path: str = row[col]
                if key == "audio":
                    features = load_audio(
                        full_path(file_path), sampling_rate=16_000)
                else:
                    features = torch.load(full_path(file_path))

                # Load normalized MOS and convert to Tensor.
                norm_mos = torch.tensor(
                    float(row[STANDARDIZED_CSV_INFO.col_norm_mos]))

                # Construct sample.
                xkey = "%07d" % idx
                sample = {
                    "__key__": xkey,
                    "features.pth": serialize_pt(features),
                    "norm_mos.pth": serialize_pt(norm_mos),
                }

                # Write the sample to the sharded tar archives.
                sink.write(sample)


def create_shards(split: Split, example: bool):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"created_shards_{split_name}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            print(f"{example_str}Creating shards for {split_name} split.")
            _create_shards(split, example)
        else:
            print(f"{example_str}Shards already created for {split_name} split.")


if __name__ == "__main__":
    example: bool = True
    for split in DEV_SPLITS:
        create_shards(split, example)
