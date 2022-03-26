import csv
from src import constants
from src.utils.run_once import run_once


def _process_raw_test_csv():

    msg = "Converting raw test CSV into PSTN, Tencent and Tub CSVs"
    msg += " for pipeline compatibility."
    print(msg)

    OUTPUT_HEADER = ["deg_wav", "predict_mos"]

    # Load CSV rows.
    # - Columns: dataset_name, audio_file, empty prediction column
    raw_csv_path = constants.TEST_RAW_CSV_PATH
    rows_pstn = []
    rows_tencent = []
    rows_tub = []
    with open(raw_csv_path, encoding="utf8", mode="r") as in_csv:
        csv_reader = csv.reader(in_csv)
        for idx, in_row in enumerate(csv_reader):

            # Write header row.
            if idx == 0:
                rows_pstn.append(OUTPUT_HEADER)
                rows_tencent.append(OUTPUT_HEADER)
                rows_tub.append(OUTPUT_HEADER)
                continue

            # Skip empty row.
            if len(in_row) == 0:
                continue

            # Process row... Fill in temporary MOS value of 5 to be compatible
            # with the rest of the pipeline.
            tmp_mos = 5
            out_row = [in_row[1], str(tmp_mos)]
            if in_row[0] == constants.PSTN_TEST_ZIP_FOLDER:
                rows_pstn.append(out_row)
            elif in_row[0] == constants.TENCENT_TEST_ZIP_FOLDER:
                rows_tencent.append(out_row)
            elif in_row[0] == constants.TUB_TEST_ZIP_FOLDER:
                rows_tub.append(out_row)
            else:
                raise Exception(f"Unknown test dataset name: {in_row[0]}")

    # Write to output CSVs.
    pstn_csv_path = constants.PSTN_TEST_CSVS[0].csv_path
    with open(pstn_csv_path, mode="w", encoding="utf8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows_pstn)
    tencent_csv_path = constants.TENCENT_TEST_CSVS[0].csv_path
    with open(tencent_csv_path, mode="w", encoding="utf8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows_tencent)
    tub_csv_path = constants.TUB_TEST_CSVS[0].csv_path
    with open(tub_csv_path, mode="w", encoding="utf8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows_tub)

    print("Finished.")


def process_raw_test_csv():

    # Flag name. Make sure this operation is only performed once.
    flag_name = "processed_raw_test_csv"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _process_raw_test_csv()
        else:
            print("Raw test CSV already processed.")


if __name__ == "__main__":
    process_raw_test_csv()
