
import csv
import os
from pathlib import Path
import shutil
from typing import List

from src import constants
from src.utils.cp import cp
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.run_once import run_once


def _copy_csv_files(
    in_csv: Path,
    out_dir: Path,

):
    """Copy the files in the given CSV to the output directory.

    Args:
        in_csv (Path): Path to input (standardized) CSV file.
        out_dir (Path): Root directory where data will be copied.
    """

    # Make sure input path exists.
    if not in_csv.exists():
        raise Exception("Path does not exist: %s" % str(in_csv))

    # Create output directory.
    out_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # Copy the CSV file itself.

    # - Calculate relative path from DIR_PROJECT to in_csv.
    in_csv_rel_path = os.path.relpath(in_csv, constants.DIR_PROJECT)

    # - Copy the file.
    dst_path = out_dir.joinpath(in_csv_rel_path)
    cp(in_csv, dst_path)

    # CSV info for standardized format.
    csv_info = STANDARDIZED_CSV_INFO

    # Read CSV rows.
    with open(in_csv, encoding="utf8", mode="r") as in_csv:
        csv_reader = csv.reader(in_csv)
        for idx, in_row in enumerate(csv_reader):

            # Skip header row.
            if idx == 0:
                continue

            # Copy files associated with this data row.

            # - Audio file.
            audio_path = in_row[csv_info.col_audio_path]
            src_path = constants.DIR_PROJECT.joinpath(audio_path)
            dst_path = out_dir.joinpath(audio_path)
            cp(src_path, dst_path)

            # - Feature file.
            # TODO: fix me
            feat_path = in_row[csv_info.col_feat_path]
            src_path = constants.DIR_PROJECT.joinpath(feat_path)
            dst_path = out_dir.joinpath(feat_path)
            cp(src_path, dst_path)


def _create_example_zip():
    """Create a ZIP containing the example training dataset."""

    print("Creating example training data ZIP file...")

    # Create list of CSV paths.
    csv_paths: List[Path] = []
    csv_type = "example_csvs"
    csv_paths.extend(constants.IU_BLOOMINGTON_TRAIN[csv_type])
    csv_paths.extend(constants.NISQA_TRAIN[csv_type])
    csv_paths.extend(constants.PSTN_TRAIN[csv_type])
    csv_paths.extend(constants.TENCENT_TRAIN[csv_type])

    # Make sure files exist.
    for csv_path in csv_paths:
        assert csv_path.exists()

    # Output directory.
    out_dir = constants.DIR_PROJECT.joinpath("data_example")

    # Copy files.
    print("Copying files to %s" % str(out_dir))
    for csv_path in csv_paths:
        _copy_csv_files(csv_path, out_dir)

    # Create ZIP.
    print("Creating ZIP: %s.zip" % str(out_dir))
    basename = out_dir
    shutil.make_archive(basename, 'zip', out_dir)

    print("Finished.")


def create_example_zip():

    # Run feature extraction exactly once.
    flag_name = "created_example_zip"
    with run_once(flag_name) as should_run:
        if should_run:
            _create_example_zip()
        else:
            print("Example data ZIP already created.")


if __name__ == "__main__":
    create_example_zip()
