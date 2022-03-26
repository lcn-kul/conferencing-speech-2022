import csv
import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple

from src import constants
from src.model.config import Config, Input
from src.utils.split import Split
from src.utils.csv_info import STANDARDIZED_CSV_INFO
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


class MyCrop(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.input == Input.AUDIO:
            center_crop = transforms.CenterCrop((config.feat_seq_len, 1,))
        elif config.input == Input.MFCC:
            center_crop = transforms.CenterCrop((config.feat_seq_len, 40,))
        elif config.input == Input.MFCC_EXT:
            center_crop = transforms.CenterCrop((config.feat_seq_len, 120,))
        elif config.input == Input.XLSR:
            center_crop = transforms.CenterCrop((config.feat_seq_len, 1024,))
        self.center_crop = center_crop

    def forward(self, x):
        out = self.center_crop(x)
        return out


class MyNormalize(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = var.sqrt()

    def forward(self, x: torch.Tensor):
        out = (x - self.mean) / self.std
        return out


def make_transform(
    config: Config,
    mean: torch.Tensor,
    var: torch.Tensor,
):
    return transforms.Compose([
        MyCrop(config),
        MyNormalize(mean, var),
    ])


class CsvDataset(Dataset):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        split = Split.TEST
        self.split = split
        normalization_split = Split.TRAIN_SUBSET
        self.normalization_split = normalization_split
        example = False
        self.example = example

        # For printing...
        split_name = str(split).lower().split(".")[1]
        example_str = "(example) " if example else ""
        print(f"{example_str}Creating submission dataloader for {split_name} set.")

        # Select train, val, trainval or test dataset.
        normalization_dataset = constants.get_dataset(normalization_split, example)
        file_name = str(config.input).lower().split(".")[1]  # audio, mfcc, ...

        # Load mean/var.
        mu_path = normalization_dataset.norm_dir.joinpath(f"{file_name}.mu.pt")
        var_path = normalization_dataset.norm_dir.joinpath(f"{file_name}.var.pt")
        if not mu_path.exists() or not var_path.exists():
            msg = f"Cannot find {file_name}.mu.pt and {file_name}.var.pt in {normalization_dataset.norm_dir}."
            raise Exception(msg)
        mean = torch.load(mu_path)
        var = torch.load(var_path)

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        csv_path = constants.TEST_CSV_PATH_FOR_EVAL
        with open(csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # Save feature_path, norm_mos
                file_path: str = in_row[0]
                self.csv_data.append(file_path)

        # Create transform.
        self.transform = make_transform(config, mean, var)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[index]
        if self.config.name == "audio":
            features = load_audio(full_path(file_path), sampling_rate=16_000)
        else:
            features = torch.load(full_path(file_path))
        features = self.transform(features)

        return features
