import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List

from src import constants
from src.model.config import Config, ALL_CONFIGS, XLSR_BLSTM_CONFIG
from src.model.model import Model
from src.utils.run_once import run_once
from src.predict.csv_dataset_submission import CsvDataset
from src.utils.split import Split


def make_dataloader(config: Config, cpus: int):

    # Create dataset.
    dataset = CsvDataset(config)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=cpus-1)

    return dataloader



def _predict_model(example: bool, modelpath:str, split: Split, cpus: int):
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"{example_str}Using: %s" % device)

    # Load best model.
    # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/924#issuecomment-591108496
    ckpt_path = constants.MODELS_DIR.joinpath(modelpath)
    model = Model.load_from_checkpoint(str(ckpt_path)).to(device)

    # Create dataloader.
    dl = make_dataloader(XLSR_BLSTM_CONFIG, cpus)

    # Output path.
    dirname = modelpath.split("/")[0]
    dataset = constants.get_dataset(split, example)
    out_file = f"final_prediction_submission_{dirname}_{split_name}{example_name}.csv"
    out_path = dataset.predictions_dir.joinpath(out_file)

    # Iterate through data.
    model.eval()
    with open(out_path, mode="w", encoding="utf8") as f:
        f.write("prediction" + "\n")
        for features in tqdm(dl):
            out: torch.Tensor = model(features.to(device)).cpu()
            out_denorm = out * 4.0 + 1.0  # 0-1 --> 1-5
            f.write("%0.7f" % out_denorm.item() + "\n")


def predict_final_model_submission(example: bool, modelpath: str, split: Split, cpus: int):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    split_name = str(split).lower().split(".")[1]
    dirname = modelpath.split("/")[0]
    flag_name = f"predicted_final_model_submission_{dirname}_{split_name}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _predict_model(example, modelpath, split, cpus)
        else:
            print(
                f"{example_str}Final prediction already made on split {split_name}.")


if __name__ == "__main__":
    example: bool = False
    modelpath = "final_model_17mar_xlsr_blstm/best-epoch=012-val_loss=0.014164.ckpt"
    cpus: int = 4
    split = Split.TEST
    predict_final_model_submission(example, modelpath, split, cpus)
