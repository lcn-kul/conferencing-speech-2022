import csv
import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from typing import Tuple
from src import constants
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.full_path import full_path
from src.utils.run_once import run_once
from src.utils.split import ALL_SPLITS, Split


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


class CsvDataset(Dataset):

    def __init__(self, split: Split, example: bool, col_key: int) -> None:
        super().__init__()
        self.split = split
        self.example = example
        self.col_key = col_key

        # Select train, val, trainval or test dataset.
        dataset = constants.get_dataset(split, example)

        # Type to CSV column.
        types = {
            'audio': STANDARDIZED_CSV_INFO.col_audio_path,  # 1st column
            'mfcc': STANDARDIZED_CSV_INFO.col_mfcc_path,  # 2nd column
            'mfcc_ext': STANDARDIZED_CSV_INFO.col_mfcc_ext_path,  # 3rd column
            'xlsr': STANDARDIZED_CSV_INFO.col_xlsr_path,  # 4th column
        }
        col_path = types[col_key]

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # Save feature_path, norm_mos
                file_path: str = in_row[col_path]
                self.csv_data.append(file_path)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[index]
        if self.col_key == "audio":
            features = load_audio(full_path(file_path), sampling_rate=16_000)
        else:
            features = torch.load(full_path(file_path))

        return features


def _calculate_norm_csv(split: Split, example: bool):

    # For printing...
    split_name = str(split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Calculating norm for {split_name} set from CSV.")

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example)

    for key in {"audio", "mfcc", "mfcc_ext", "xlsr"}:

        # Create dataloader.
        csv_dataset = CsvDataset(split, example, key)
        dataloader = DataLoader(csv_dataset, batch_size=None, shuffle=False)

        # Iterate through data and calculate running x and x^2.
        x1 = None
        x2 = None
        N = 0
        for feature in dataloader:
            if feature.dim() == 1:
                feature = feature.unsqueeze(1)
            if feature.dim() != 2:
                raise Exception(
                    f"Expected feature dimensionality of 2 for {key}.")

            if x1 is None:
                x1 = torch.zeros((feature.shape[1]),)
                x2 = torch.zeros((feature.shape[1],))

            # Update running x, x^2 and count.
            x1 += feature.sum(dim=0)
            x2 += feature.pow(2).sum(dim=0)
            N += feature.shape[0]

        # Calculate mean and variance from x, x^2 and N.
        #   μ = 1/N.∑_{i=1..N} (xᵢ)
        #   σ² = 1/N.∑_{i=1..N} (xᵢ²) - μ²
        mu = x1/N
        var = x2/N - mu*mu

        # Save results.
        mu_path = dataset.norm_dir.joinpath(f"{key}.mu.pt")
        torch.save(mu, str(mu_path))
        var_path = dataset.norm_dir.joinpath(f"{key}.var.pt")
        torch.save(var, str(var_path))
        N_path = dataset.norm_dir.joinpath(f"{key}.N.pt")
        torch.save(N, str(N_path))


def _calculate_norm_wds(split: Split, example: bool):

    # For printing...
    split_name = str(split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Calculating norm for {split_name} set from WDS.")

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example)

    for key in {"audio", "mfcc", "mfcc_ext", "xlsr"}:

        # Find shards.
        shard_paths = dataset.shards_dir.glob(f"{key}.*.tar")
        shard_paths = list(map(str, shard_paths))
        if len(shard_paths) == 0:
            raise Exception(f"{example_str}No shards found for {key}.")

        # Create DataLoader.
        web_dataset = (
            wds.WebDataset(shard_paths)
            .shuffle(1000)
            .decode()
            .to_tuple("features.pth norm_mos.pth")
        )
        web_loader = wds.WebLoader(
            web_dataset, batch_size=None, shuffle=False
        )

        # Iterate through data and calculate running x and x^2.
        x1 = None
        x2 = None
        N = 0
        for feature, label in web_loader:
            if feature.dim() == 1:
                feature = feature.unsqueeze(1)
            if feature.dim() != 2:
                raise Exception(
                    f"Expected feature dimensionality of 2 for {key}.")

            if x1 is None:
                x1 = torch.zeros((feature.shape[1]),)
                x2 = torch.zeros((feature.shape[1],))

            # Update running x, x^2 and count.
            x1 += feature.sum(dim=0)
            x2 += feature.pow(2).sum(dim=0)
            N += feature.shape[0]

        # Calculate mean and variance from x, x^2 and N.
        #   μ = 1/N.∑_{i=1..N} (xᵢ)
        #   σ² = 1/N.∑_{i=1..N} (xᵢ²) - μ²
        mu = x1/N
        var = x2/N - mu*mu

        # Save results.
        mu_path = dataset.norm_dir.joinpath(f"{key}.mu.pt")
        torch.save(mu, str(mu_path))
        var_path = dataset.norm_dir.joinpath(f"{key}.var.pt")
        torch.save(var, str(var_path))
        N_path = dataset.norm_dir.joinpath(f"{key}.N.pt")
        torch.save(N, str(N_path))


def _calculate_norm_trainval(example: bool):

    # Print split name.
    example_str = "(example) " if example else ""
    print(f"{example_str}Calculating norm for split: trainval.")

    # Make sure train/val CSVs have been processed.
    train = constants.get_dataset(Split.TRAIN, example)
    val = constants.get_dataset(Split.VAL, example)
    trainval = constants.get_dataset(Split.TRAINVAL, example)

    for key in {"audio", "mfcc", "mfcc_ext", "xlsr"}:
        # Construct and check paths.
        train_mu_path = train.norm_dir.joinpath(f"{key}.mu.pt")
        train_var_path = train.norm_dir.joinpath(f"{key}.var.pt")
        train_N_path = train.norm_dir.joinpath(f"{key}.N.pt")
        val_mu_path = val.norm_dir.joinpath(f"{key}.mu.pt")
        val_var_path = val.norm_dir.joinpath(f"{key}.var.pt")
        val_N_path = val.norm_dir.joinpath(f"{key}.N.pt")
        trainval_mu_path = trainval.norm_dir.joinpath(f"{key}.mu.pt")
        trainval_var_path = trainval.norm_dir.joinpath(f"{key}.var.pt")
        trainval_N_path = trainval.norm_dir.joinpath(f"{key}.N.pt")
        if not train_mu_path.exists():
            raise Exception(f"train mu path does not exist: {train_mu_path}")
        if not train_var_path.exists():
            raise Exception(f"train var path does not exist: {train_var_path}")
        if not val_mu_path.exists():
            raise Exception(f"val mu path does not exist: {val_mu_path}")
        if not val_var_path.exists():
            raise Exception(f"val var path does not exist: {val_var_path}")

        # Load train mean, variance, sample_count.
        mu1 = torch.load(str(train_mu_path))
        var1 = torch.load(str(train_var_path))
        N1 = torch.load(str(train_N_path))

        # Load val mean, variance, sample_count.
        mu2 = torch.load(str(val_mu_path))
        var2 = torch.load(str(val_var_path))
        N2 = torch.load(str(val_N_path))

        # Combined mean.
        #    μₘₙ = 1/(M+N).(M.μₘ + N.μₙ)
        mu = 1/(N1+N2) * (N1*mu1 + N2*mu2)

        # Combined variance.
        #   σ²ₘₙ = 1/(M+N).(M.(σ²ₘ+μₘ²) + N.(σ²ₙ+μₙ²)) - μₘₙ²
        X1 = N1*(var1 + mu1.pow(2))
        X2 = N2*(var2 + mu2.pow(2))
        var = 1/(N1+N2) * (X1 + X2) - mu.pow(2)

        # Combined samples.
        N = N1 + N2

        # Save resuts.
        torch.save(mu, str(trainval_mu_path))
        torch.save(var, str(trainval_var_path))
        torch.save(N, str(trainval_N_path))

        # TODO: save values

        # Derivation.
        # Source: https://stats.stackexchange.com/a/43183
        #
        # def mean: μ = 1/N.∑_{i=1..N} (xᵢ)
        # def var: σ² = 1/N.∑_{i=1..N} (xᵢ - μ)²
        #
        # Required derivations:
        #  - ∑_{i=1..N} (xᵢ) = N.μ
        #  - ∑_{i=1..N} (xᵢ²) = N.(σ² + μ²)
        #    since:
        #     - N.σ² = ∑_{i=1..N} (xᵢ - μ)²
        #            = ∑_{i=1..N} (xᵢ² - 2.xᵢμ + μ²)
        #            = ∑_{i=1..N} (xᵢ²) - 2.μ.∑_{i=1..N} (xᵢ) + N.μ²
        #            = ∑_{i=1..N} (xᵢ²) - 2.μ.(N.μ) + N.μ²
        #            = ∑_{i=1..N} (xᵢ²) - N.μ²
        #
        # Combination of two sets
        #  - set m: 1..M
        #  - set n: M+1..M+N
        #
        # combined mean:
        #    μₘₙ = 1/(M+N).∑_{i=1..M+N} (xᵢ)
        #        = 1/(M+N).( ∑_{i=1..M} (xᵢ) +  ∑_{i=M+1..M+N} (xᵢ) )
        #        = 1/(M+N).( M.μₘ + N.μₙ )
        #
        # combined variance:
        #   1. first calculate: X := (M+N)(σ²ₘₙ + μₘₙ²)
        #      X = ∑_{i=1..M+N} (xᵢ²)
        #        = ∑_{i=1..M} (xᵢ²) +  ∑_{i=M+1..M+N} (xᵢ²)
        #        = M.(σ²ₘ + μₘ²) + N.(σ²ₙ + μₙ²)
        #   2. solving algebraically for σ²ₘₙ:
        #   σ²ₘₙ = 1/(M+N).(M.(σ²ₘ+μₘ²) + N.(σ²ₙ+μₙ²)) - μₘₙ²
        #


def calculate_norm(split: Split, example: bool):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"calculated_norm_{split_name}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            if split == split.TRAINVAL:
                # Special case: calculate using train+val mean/variance.
                _calculate_norm_trainval(example)
            else:
                try:
                    _calculate_norm_wds(split, example)
                except Exception as e:
                    print(str(e))
                    print("trying norm calculation using csv")
                    _calculate_norm_csv(split, example)
        else:
            print(f"{example_str}Norm already calculated for {split_name} split.")


if __name__ == "__main__":
    example: bool = True
    for split in ALL_SPLITS:
        calculate_norm(split, example)
