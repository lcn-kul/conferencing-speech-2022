import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List

from src import constants
from src.model.config import Config, ALL_CONFIGS
from src.model.model import Model
from src.utils.run_once import run_once
from src.predict.csv_dataset import CsvDataset
from src.utils.split import Split


def make_dataloader(config: Config, split: Split, normalization_split: Split, example: bool, cpus: int):

    # For printing...
    split_name = str(split).lower().split(".")[1]
    norm_split_name = str(normalization_split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Creating dataloader for {split_name} set using {norm_split_name} as normalization.")

    # Create dataset.
    dataset = CsvDataset(config, split, normalization_split, example)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=cpus-1)

    return dataloader


def _find_lowest_idx(ckpt_paths: List[str]):
    # Find lowest validation loss.
    stems = [os.path.splitext(os.path.basename(str(x)))[0] for x in ckpt_paths]
    parts_per_stem = [x.split("-") for x in stems]
    dict_per_stem = [{p.split("=")[0]: p.split("=")[1]
                      for p in parts if "=" in p} for parts in parts_per_stem]
    val_loss_per_stem = [x["val_loss"] for x in dict_per_stem]
    lowest_idx = min(range(len(val_loss_per_stem)),
                     key=val_loss_per_stem.__getitem__)
    return lowest_idx


def _predict_model(config: Config, example: bool, use_subset: bool, split: Split, normalization_split: Split, cpus: int, gpus: int):
    split_name = str(split).lower().split(".")[1]
    norm_split_name = str(normalization_split).lower().split(".")[1]
    example_name = "_example" if example else ""
    subset_str = "_subset" if use_subset else ""

    if gpus > 0:
        device = 'cuda'
    else:
        device = 'cpu'

    # Load best model.
    model_name = f"trained_model_{config.name}{example_name}{subset_str}"
    model_dir = constants.MODELS_DIR.joinpath(model_name)
    ckpt_paths = list(model_dir.glob("best*.ckpt"))
    if len(ckpt_paths) == 0:
        raise Exception(f"No checkpoint path found in {model_dir}.")
    lowest_val_idx = _find_lowest_idx(ckpt_paths)
    ckpt_path = ckpt_paths[lowest_val_idx]
    model = Model.load_from_checkpoint(str(ckpt_path)).to(device)

    # Create dataloader.
    dl = make_dataloader(config, split, normalization_split, example, cpus)

    # Output path.
    dataset = constants.get_dataset(split, example)
    out_file = f"prediction_{config.name}_{split_name}_norm_{norm_split_name}{example_name}{subset_str}.csv"
    out_path = dataset.predictions_dir.joinpath(out_file)

    # Iterate through data.
    model.eval()
    with open(out_path, mode="w", encoding="utf8") as f:
        f.write("prediction" + "\n")
        for features, mos_norm in tqdm(dl):
            out: torch.Tensor = model(features.to(device)).cpu()
            out_denorm = out * 4.0 + 1.0  # Range 0-1 --> 1-5
            f.write("%0.7f" % out_denorm.item() + "\n")


def predict_model(config: Config, example: bool, use_subset: bool, split: Split, normalization_split: Split, cpus: int, gpus: int):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    subset_str = "_subset" if use_subset else ""
    split_name = str(split).lower().split(".")[1]
    norm_split_name = str(normalization_split).lower().split(".")[1]
    flag_name = f"predicted_model_{config.name}_{split_name}_norm_{norm_split_name}{example_name}{subset_str}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _predict_model(config, example, use_subset, split,
                           normalization_split, cpus, gpus)
        else:
            print(
                f"{example_str}Prediction already made for {config.name} on split {split_name} using {norm_split_name} normalization.")


if __name__ == "__main__":
    example: bool = True
    cpus: int = 1
    gpus: int = 0
    for config in ALL_CONFIGS:
        for use_subset in [True, False]:
            if use_subset:
                train_split = Split.TRAIN_SUBSET
            else:
                train_split = Split.TRAIN
            normalization_split = train_split
            for split in [Split.VAL, Split.VAL_SUBSET]:
                predict_model(config, example, use_subset, split,
                                normalization_split, cpus, gpus)
