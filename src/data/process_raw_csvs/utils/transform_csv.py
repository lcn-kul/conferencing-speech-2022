import csv
import os
from pathlib import Path

from src import constants
from src.utils.csv_info import STANDARDIZED_CSV_HEADER, CsvInfo
from src.utils.mos_transform import MosTransform


def transform_csv(in_path: Path, out_dir: Path, csv_info: CsvInfo):
    """Transform the given CSV file to the standardized format found in
    src/utils/csv_info.py.

    Args:
        in_path (Path): Path to input CSV file.
        out_dir (Path): Where to the processed CSV will be stored.
        csv_info (CsvInfo): CSV metadata.
    """

    # Make sure input path exists.
    in_dir = in_path.parent
    if not in_path.exists():
        raise Exception("Path does not exist: %s" % str(in_path))

    # Calculate relative path from DIR_PROJECT to in_dir/out_dir.
    in_dir_rel_path = os.path.relpath(in_dir, constants.DIR_PROJECT)
    out_dir_rel_path = os.path.relpath(out_dir, constants.DIR_PROJECT)

    # Object for MOS normalization from [1,5] to [0,1].
    mos_normalizer = MosTransform(
        in_mos_min=1,
        in_mos_max=5,
        out_mos_min=0,
        out_mos_max=1,
    )

    # Open file.
    with open(in_path, encoding="utf8", mode="r") as in_csv:

        # Create CSV reader/writer.
        csv_reader = csv.reader(in_csv)

        # Output rows.
        out_rows = []
        for idx, in_row in enumerate(csv_reader):

            # Write header row.
            if idx == 0:
                # Write to output file.
                out_rows.append(STANDARDIZED_CSV_HEADER)
                continue

            # Skip empty row.
            if len(in_row) == 0:
                continue

            # Process row...
            out_row = []

            # 1. Feature paths.

            # 1.1. Audio path:
            #      These files will be kept in the "raw" directory (in_dir).
            audio_path = in_row[csv_info.col_audio_path]
            audio_base, audio_ext = os.path.splitext(audio_path)
            if audio_ext != ".wav":
                msg = "Expected .wav file but got %s file!" % audio_ext
                raise Exception(msg)
            audio_path = os.path.join(in_dir_rel_path, audio_path)
            audio_path = os.path.relpath(audio_path)  # resolve path
            out_row.append(audio_path)

            # 1.2. Other feature paths:
            #      These files will eventually be saved in the "processed"
            #      directory (out_dir).
            other_features = ["mfcc", "mfcc_ext", "xlsr"]
            for name in other_features:
                feat_path = audio_base + f".{name}.pt"
                feat_path = os.path.join(out_dir_rel_path, feat_path)
                feat_path = os.path.relpath(feat_path)  # resolve path
                out_row.append(feat_path)

            # 2. Labels.

            # 2.1. MOS
            mos = in_row[csv_info.col_mos]
            if csv_info.mos_transform is not None:
                mos = csv_info.mos_transform.transform_str(mos)
            out_row.append(mos)

            # 2.2. Normalized MOS.
            norm_mos = mos_normalizer.transform_str(mos)
            out_row.append(norm_mos)

            # 3. In PSTN/Tencent subset?
            out_row.append(str(csv_info.in_subset))

            # Append to output rows.
            out_rows.append(out_row)

    return out_rows
