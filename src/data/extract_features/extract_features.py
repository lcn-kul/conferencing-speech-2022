import csv
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf
import torch
from torchaudio.transforms import ComputeDeltas, MFCC
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel
from tqdm.auto import tqdm

from src import constants
from src.utils.run_once import run_once
from src.utils.split import Split, ALL_SPLITS, DEV_SPLITS
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.full_path import full_path


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


def _extraction_progress_path(split: Split, example: bool, partition_idx: int, num_partitions: int) -> Path:
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    if num_partitions == 1:
        partition_name = ""
    else:
        partition_name = f"_{partition_idx}-{num_partitions}"
    progress_file_name = f"extract_features_progress_{split_name}{example_name}{partition_name}"
    return constants.DIR_DATA_FLAGS.joinpath(progress_file_name)


def _get_extraction_progress(split: Split, example: bool, partition_idx: int, num_partitions: int) -> int:
    """Returns the most recent finished index. If no progress."""
    path = _extraction_progress_path(
        split, example, partition_idx, num_partitions)
    if path.exists():
        with open(path, mode="r", encoding="utf8") as f:
            finished_idx = int(f.readline())
        return finished_idx
    else:
        return -1


def _write_extraction_progress(split: Split, example: bool, partition_idx: int, num_partitions: int, finished_idx: int):
    path = _extraction_progress_path(
        split, example, partition_idx, num_partitions)
    with open(path, mode='w', encoding="utf8") as f:
        f.write(str(finished_idx))


def _extract_features(split: Split, example: bool, partition_idx: int = 0, num_partitions: int = 1):

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example)

    # Load progress for this partition and for the single partition.
    finished_idx = _get_extraction_progress(
        split, example, partition_idx, num_partitions)
    single_idx = _get_extraction_progress(split, example, 0, 1)

    # For printing...
    split_name = str(split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Extracting features for {split_name} set.")

    # MODEL REQUIRES 16 kHz SAMPLING RATE.
    SAMPLING_RATE = 16_000

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"{example_str}Using: %s" % device)

    # Create model.
    print(f"{example_str}Loading model...")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    if constants.XLSR_DIR.exists():
        model = AutoModel.from_pretrained(str(constants.XLSR_DIR))
    else:
        model = Wav2Vec2Model.from_pretrained(f"facebook/{constants.XLSR_NAME}")
    model = model.to(device)

    # Create MFCC calculator.
    print(f"{example_str}Creating MFCC components...")
    calculate_mfcc = MFCC(sample_rate=SAMPLING_RATE)

    # Create MFCC Delta calculator.
    compute_deltas = ComputeDeltas(win_length=5, mode='replicate')

    # Load CSV rows.
    rows = []
    with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
        csv_reader = csv.reader(in_csv)
        for idx, row in enumerate(csv_reader):

            # Skip header row & empty rows.
            if idx == 0 or len(row) == 0:
                continue

            # Append row.
            rows.append(row)

    # Which rows should be processed by the partition.
    if single_idx == -1:
        partition_start = int(partition_idx/num_partitions * len(rows))
        partition_end = int((partition_idx+1)/num_partitions * len(rows))
    else:
        new_len = len(rows) - (single_idx+1)
        partition_start = int(partition_idx/num_partitions * new_len)
        partition_start += single_idx+1
        partition_end = int((partition_idx+1)/num_partitions * new_len)
        partition_end += single_idx+1
    print(f"Processing sample {partition_start}..{partition_end-1}")

    # ======================================================================= #
    #                           CALCULATE FEATURES                            #
    # ======================================================================= #

    print(f"{example_str}Calculating features for {len(rows)} audio files...")
    write_progress_freq = 10
    for idx, row in enumerate(tqdm(rows)):

        # Skip indices that are not for this partition.
        if idx < partition_start:
            continue
        if idx >= partition_end:
            continue

        # These have already been completed.
        if idx <= finished_idx:
            continue

        # Extract paths.
        audio_path = row[STANDARDIZED_CSV_INFO.col_audio_path]
        mfcc_path = row[STANDARDIZED_CSV_INFO.col_mfcc_path]
        mfcc_ext_path = row[STANDARDIZED_CSV_INFO.col_mfcc_ext_path]
        xlsr_path = row[STANDARDIZED_CSV_INFO.col_xlsr_path]

        # Load audio.
        audio_data_np, _ = _decode_non_mp3_file_like(
            full_path(audio_path), SAMPLING_RATE)
        audio_data_np = np.float32(audio_data_np)
        audio_data_pt = torch.from_numpy(audio_data_np)

        # Calculate wav2vec2 vector.
        inputs = feature_extractor(
            audio_data_pt,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt"
        )
        input = inputs["input_values"].to(device)
        with torch.no_grad():
            output = model(input)
        xlsr: torch.Tensor = output.last_hidden_state.squeeze().cpu()

        # MFCC (and deltas)
        mfcc: torch.Tensor = calculate_mfcc(audio_data_pt)
        mfcc_d: torch.Tensor = compute_deltas(mfcc)
        mfcc_d2: torch.Tensor = compute_deltas(mfcc_d)
        mfcc_ext = torch.concat((mfcc, mfcc_d, mfcc_d2))

        # Transpose MFCC from (n_mfcc, T) to (T, n_mfcc).
        # This will match the wav2vec2 size of (T, 1024).
        mfcc = mfcc.permute((1, 0))
        mfcc_ext = mfcc_ext.permute((1, 0))

        # Save results to .pt files.
        torch.save(mfcc, full_path(mfcc_path))
        torch.save(mfcc_ext, full_path(mfcc_ext_path))
        torch.save(xlsr, full_path(xlsr_path))

        # Finished this index.
        if idx % write_progress_freq == write_progress_freq - 1:
            _write_extraction_progress(
                split, example, partition_idx, num_partitions, idx)

    print("")
    print(f"{example_str}Finished.")


def extract_features(split: Split, example: bool, partition_idx: int = 0, num_partitions: int = 1):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"extracted_features_{split_name}{example_name}"

    # Special case: subset uses features from main datasets, otherwise
    # we would have to extract the features twice.
    if split == Split.TRAIN_SUBSET or split == Split.VAL_SUBSET:
        print(f"{example_str}Feature extraction not needed for {split_name} split.")
        return

    # Run exactly once.
    with run_once(flag_name, partition_idx, num_partitions) as should_run:
        if should_run:
            _extract_features(split, example, partition_idx, num_partitions)
        else:
            print(f"{example_str}Features already extracted for {split_name} split.")


if __name__ == "__main__":
    example: bool = True
    for split in DEV_SPLITS:
        extract_features(split, example, partition_idx=0, num_partitions=1)
