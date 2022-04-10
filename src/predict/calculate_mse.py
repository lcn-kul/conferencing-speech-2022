import csv

from src import constants
from src.model.config import Config, ALL_CONFIGS
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.run_once import run_once
from src.utils.split import Split


def _calculate_mse(config: Config, example: bool, use_subset: bool, split: Split, normalization_split: Split):
    split_name = str(split).lower().split(".")[1]
    norm_split_name = str(normalization_split).lower().split(".")[1]
    example_name = "_example" if example else ""
    subset_str = "_subset" if use_subset else ""


    # Load ground truth.
    dataset = constants.get_dataset(split, example)
    gt_values = []
    with open(dataset.csv_path, mode="r", encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            # Skip header row & empty rows.
            if idx == 0 or len(row) == 0:
                continue
            gt_values.append(float(row[STANDARDIZED_CSV_INFO.col_mos]))

    # Load prediction values.
    pred_file = f"prediction_{config.name}_{split_name}_norm_{norm_split_name}{example_name}{subset_str}.csv"
    pred_path = dataset.predictions_dir.joinpath(pred_file)
    pred_values = []
    with open(pred_path, mode="r", encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            # Skip header row & empty rows.
            if idx == 0 or len(row) == 0:
                continue
            pred_values.append(float(row[0]))

    # Calculate MSE.
    if len(gt_values) != len(pred_values):
        raise Exception(
            "Ground truth values and predicted values have different lengths.")
    se = 0.0
    N = len(gt_values)
    for idx in range(N):
        se += (pred_values[idx] - gt_values[idx])**2
    mse = se / N

    # Write MSE to output file.
    out_file = f"mse_{config.name}_{split_name}_norm_{norm_split_name}{example_name}.csv"
    out_path = dataset.predictions_dir.joinpath(out_file)
    with open(out_path, mode="w", encoding="utf8") as f:
        f.write("mos_mse" + "\n")
        f.write("%0.7f" % mse)


def calculate_mse(config: Config, example: bool, use_subset: bool, split: Split, normalization_split: Split):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    subset_str = "_subset" if use_subset else ""
    split_name = str(split).lower().split(".")[1]
    norm_split_name = str(normalization_split).lower().split(".")[1]
    flag_name = f"calculated_mse_{config.name}_{split_name}_norm_{normalization_split}{example_name}{subset_str}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _calculate_mse(config, example, use_subset, split, normalization_split)
        else:
            print(
                f"{example_str}MSE already calculated for already made for {config.name} on split {split_name} using {norm_split_name} normalization ({subset_str}).")


if __name__ == "__main__":
    example: bool = True
    for config in ALL_CONFIGS:
        for use_subset in [True,]:
            if use_subset:
                train_split = Split.TRAIN_SUBSET
            else:
                train_split = Split.TRAIN
            for split in [Split.VAL, Split.VAL_SUBSET]:
                for normalization_split in [split, train_split]:
                    calculate_mse(config, example, use_subset,
                                  split, normalization_split)
