import csv
import math
from random import Random
from src import constants
from src.data.process_raw_csvs.utils.transform_csv import transform_csv
from src.utils.csv_info import STANDARDIZED_CSV_INFO, STANDARDIZED_CSV_HEADER
from src.utils.run_once import run_once
from src.utils.split import Split, ALL_SPLITS, DEV_SPLITS



def _process_raw_csvs(split: Split, example: bool = False):

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example)

    # Select appropriate CSV infos.
    if split in [Split.TRAIN, Split.TRAIN_SUBSET, Split.VAL, Split.VAL_SUBSET]:
        csv_infos = constants.TRAIN_CSVS
    elif split == Split.TEST:
        csv_infos = constants.TEST_CSVS
    else:
        raise Exception(f"Unknown split: {split}")

    # Print split name.
    split_name = str(split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Processing raw CSVs for split: {split_name}.")

    # Load all CSVs.
    rows = []
    for csv_info in csv_infos:
        csv_path = csv_info.csv_path
        if split == Split.VAL_SUBSET:
            out_dir = constants.get_dataset(Split.VAL, example).features_dir
        elif split == Split.TRAIN_SUBSET:
            out_dir = constants.get_dataset(Split.TRAIN, example).features_dir
        else:
            out_dir = dataset.features_dir
        print(f"{example_str}Processing raw CSV: {csv_path}")
        new_rows = transform_csv(
            in_path=csv_path,
            out_dir=out_dir,
            csv_info=csv_info,
        )
        new_rows.pop(0)  # Remove header
        rows.extend(new_rows)

    # Shuffle rows before making split.
    rdm = Random(42)  # Reproducible random number generation.
    rdm.shuffle(rows)

    # Construct example rows.
    if example:

        # Calculate fraction of rows to keep for example.
        frac_to_keep = 1 / 300  # 1 GB of 300 GB

        # Create example CSV by shuffling rows and keeping top X %.
        rows_to_keep = math.ceil(frac_to_keep * len(rows))
        assert rows_to_keep > 0 and rows_to_keep <= len(rows)
        rows = rows[:rows_to_keep]

    # Train/val split.
    val_rows = math.ceil(constants.VAL_SPLIT * len(rows))
    train_rows = len(rows) - val_rows
    if split == Split.TRAIN or split == Split.TRAIN_SUBSET:
        rows = rows[:train_rows]
    if split == Split.VAL or split == Split.VAL_SUBSET:
        rows = rows[train_rows:]

    # Val subset?
    if split in [Split.TRAIN_SUBSET, Split.VAL_SUBSET]:
        col_subset = STANDARDIZED_CSV_INFO.col_in_subset
        rows = [row for row in rows if row[col_subset] == "True"]

    # Add header row.
    rows.insert(0, STANDARDIZED_CSV_HEADER)

    # Write to output CSV.
    with open(dataset.csv_path, mode="w", encoding="utf8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows)

    print(f"{example_str}Finished.")


def process_raw_csvs(split: Split, example: bool):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"processed_csv_{split_name}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _process_raw_csvs(split, example)
        else:
            print(f"{example_str}Raw CSVs already processed for {split_name} split.")


if __name__ == "__main__":
    example: bool = True
    for split in DEV_SPLITS:
        for example in [True, False]:
            process_raw_csvs(split, example)
