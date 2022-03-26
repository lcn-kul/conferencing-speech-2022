import csv
from pathlib import Path

from src.utils.csv_info import STANDARDIZED_CSV_INFO

def combine_csvs():
    script_dir = Path(__file__).parent
    ground_truth_dir = script_dir.joinpath("ground_truths")
    predictions_dir = script_dir.joinpath("predictions")
    out_dir = script_dir.joinpath("eval_input")


    for gt_path in ground_truth_dir.iterdir():
        if gt_path.suffix != ".csv":
            continue 

        col_gt_mos = STANDARDIZED_CSV_INFO.col_mos

        # Load ground truth csv.
        gt_rows = []
        with open(gt_path, mode="r", encoding="utf8") as f:
            csv_reader = csv.reader(f)
            for idx, row in enumerate(csv_reader):
                if idx == 0 or len(row) == 0:
                    continue
                gt_rows.append(row[col_gt_mos])

        # Go through prediction subfolder with same stem as ground truth CSV.
        stem = gt_path.stem
        pred_dir = predictions_dir.joinpath(stem)
        if not pred_dir.exists():
            print(f"Cannot find prediction dir: {pred_dir}")
            continue
        for pred_path in pred_dir.iterdir():
            if pred_path.suffix != ".csv":
                continue

            # Load prediction csv.
            pred_rows = []
            with open(pred_path, mode="r", encoding="utf8") as f:
                csv_reader = csv.reader(f)
                for idx, row in enumerate(csv_reader):
                    if idx == 0 or len(row) == 0:
                        continue
                    pred_rows.append(row[0])

            if len(pred_rows) != len(gt_rows):
                print(f"rows do not match for {pred_path}")
                continue

            # Combine.
            out_dir_i = out_dir.joinpath(stem)
            out_dir_i.mkdir(mode=0o755, parents=True, exist_ok=True)
            out_path = out_dir_i.joinpath(pred_path.name)
            with open(out_path, mode="w", encoding="utf8") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["mos", "mos_pred"])
                for idx in range(len(pred_rows)):
                    csv_writer.writerow([gt_rows[idx], pred_rows[idx]])

if __name__ == "__main__":
    combine_csvs()