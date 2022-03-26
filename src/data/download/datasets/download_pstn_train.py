from pathlib import Path

from src import constants
from src.data.download.utils.download_dataset_zip import download_dataset_zip


def download_pstn_train(
    tmp_dir: Path = None,
    tqdm_name: str = None,
    tqdm_idx: int = None,
):
    """Download the PSTN Corpus and extract it to the appropriate directory."""

    download_dataset_zip(
        name="pstn",
        data_url=constants.PSTN_TRAIN_URL,
        output_dir=constants.PSTN_TRAIN_DIR,
        extracted_name=constants.PSTN_TRAIN_ZIP_FOLDER,
        tmp_dir=tmp_dir,
        tqdm_name=tqdm_name,
        tqdm_idx=tqdm_idx,
    )


if __name__ == "__main__":
    download_pstn_train(tqdm_name="pstn", tqdm_idx=0)
