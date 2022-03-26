import csv
import h5py
import numpy as np
import os
from pathlib import Path
import torch
from typing import List

from src import constants


def convert_h5(use_example: bool):

    # Load input csvs.
    csv_paths: List[Path] = []
    csv_type = "example_csvs" if use_example else "processed_csvs"
    csv_paths.extend(constants.IU_BLOOMINGTON_TRAIN[csv_type])
    csv_paths.extend(constants.NISQA_TRAIN[csv_type])
    csv_paths.extend(constants.PSTN_TRAIN[csv_type])
    csv_paths.extend(constants.TENCENT_TRAIN[csv_type])

    # Make sure files exist.
    for csv_path in csv_paths:
        assert csv_path.exists()

    csv_paths = map(str, csv_paths)

    for csv_path in csv_paths:

        if os.path.exists(csv_path + ".old"):
            csv_path_in = csv_path + ".old"
            csv_path_out = csv_path
            save_old = False
        else:
            csv_path_in = csv_path
            csv_path_out = csv_path
            save_old = True

        # Go through input csv.
        with open(csv_path_in, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            new_rows = []
            for idx, row in enumerate(csv_reader):
                # Header row....
                if idx == 0:
                    new_rows.append(
                        ["audio_path", "mfcc_path", "mfcc_ext_path",
                            "xlsr_path", "mos", "norm_mos"]
                    )
                    continue
                if len(row) == 0:
                    new_rows.append([])
                    continue

                # Extract variables.
                audio_path, feat_path, *rest = row

                # Skip if already processed.
                if not os.path.exists(feat_path):
                    continue

                # Calculate new paths.
                base_feat_path, _ = os.path.splitext(feat_path)
                mfcc_path = base_feat_path + ".mfcc.pt"
                mfcc_ext_path = base_feat_path + ".mfcc_ext.pt"
                xlsr_path = base_feat_path + ".xlsr.pt"

                # Create new files.
                with h5py.File(feat_path) as f_in:
                    pt_mfcc = torch.from_numpy(f_in['mfcc'][:])
                    pt_mfcc_ext = torch.from_numpy(f_in['mfcc_ext'][:])
                    pt_xlsr = torch.from_numpy(f_in['xlsr'][:])
                    torch.save(pt_mfcc, mfcc_path)
                    torch.save(pt_mfcc_ext, mfcc_ext_path)
                    torch.save(pt_xlsr, xlsr_path)

                # Delete old file.
                # os.remove(feat_path)

                new_row = [audio_path, mfcc_path,
                           mfcc_ext_path, xlsr_path, *rest]
                new_rows.append(new_row)

        # Replace with new csv.
        if save_old:
            os.rename(csv_path, csv_path + ".old")
        with open(csv_path_out, encoding="utf8", mode="w") as out_csv:
            csv_writer = csv.writer(out_csv)
            csv_writer.writerows(new_rows)


if __name__ == "__main__":
    convert_h5(use_example=True)
