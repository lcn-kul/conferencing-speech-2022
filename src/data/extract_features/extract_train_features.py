import h5py
import os
import torch
from torchaudio.transforms import ComputeDeltas, MFCC
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm.auto import tqdm

from src import constants
from src.data.extract_features.create_train_data_loader import (
    create_train_data_loader,
)
from src.utils.run_once import run_once


def _extract_train_features(use_example: bool):

    # MODEL REQUIRES 16 kHz SAMPLING RATE.
    SAMPLING_RATE = 16_000

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Using: %s" % device)

    # Create model.
    print("Loading model...")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    model = Wav2Vec2Model.from_pretrained(f"facebook/{constants.XLSR_NAME}")
    model = model.to(device)

    # Create MFCC calculator.
    print("Creating MFCC components...")
    calculate_mfcc = MFCC(sample_rate=SAMPLING_RATE)

    # Create MFCC Delta calculator.
    compute_deltas = ComputeDeltas(win_length=5, mode='replicate')

    # Create data loader.
    print("Creating data loader...")
    dl = create_train_data_loader(SAMPLING_RATE, use_example)

    # ======================================================================= #
    #                           CALCULATE FEATURES                            #
    # ======================================================================= #

    print("Calculating features for %i audio files..." % len(dl))
    for example in tqdm(dl):
        data = example["audio"]["array"].squeeze()

        # Model.
        inputs = feature_extractor(
            data,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt"
        )
        input = inputs["input_values"].to(device)
        with torch.no_grad():
            output = model(input)
        xlsr = output.last_hidden_state.squeeze()

        # MFCC (and deltas)
        mfcc: torch.Tensor = calculate_mfcc(data)
        mfcc_d: torch.Tensor = compute_deltas(mfcc)
        mfcc_d2: torch.Tensor = compute_deltas(mfcc_d)
        mfcc_ext = torch.concat((mfcc, mfcc_d, mfcc_d2))

        # TODO: transpose and use .npy, not .h5

        # Transpose MFCC from (n_mfcc, T) to (T, n_mfcc).
        # This will match the wav2vec2 size of (T, 1024).
        # mfcc = mfcc.permute((1, 0))
        # mfcc_ext = mfcc_ext.permute((1, 0))

        # Save results to npy files.


        # Save results to h5py file.
        h5py_path = example["feat_path"][0]
        h5py_dir = os.path.dirname(h5py_path)
        if not os.path.isdir(h5py_dir):
            os.makedirs(h5py_dir)

        with h5py.File(h5py_path, "w") as f:
            f.create_dataset("xlsr", data=xlsr.cpu().numpy())
            f.create_dataset("mfcc", data=mfcc.cpu().numpy())
            f.create_dataset("mfcc_ext", data=mfcc_ext.cpu().numpy())

    print("\nFinished.")


def extract_train_features(use_example: bool = False):

    # Paths for "flag" files. This is used to make sure the operation is only
    # performed once.
    example_flag_name = "extracted_train_features_example"
    full_flag_name = "extracted_train_features"

    # Run feature extraction exactly once.
    flag_name = example_flag_name if use_example else full_flag_name
    with run_once(flag_name) as should_run:
        if should_run:
            _extract_train_features(use_example)
        else:
            print("Feature extraction already performed.")


if __name__ == "__main__":
    extract_train_features(True)
