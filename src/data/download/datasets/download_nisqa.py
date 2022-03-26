from pathlib import Path

from src import constants
from src.data.download.utils.download_dataset_zip import download_dataset_zip


def download_nisqa(
    tmp_dir: Path = None,
    tqdm_name: str = None,
    tqdm_idx: int = None,
):
    """Download the NISQA Corpus and extract it to the appropriate
    directory."""

    download_dataset_zip(
        name="nisqa",
        data_url=constants.NISQA_TRAIN_URL,
        output_dir=constants.NISQA_TRAIN_DIR,
        extracted_name=constants.NISQA_TRAIN_ZIP_FOLDER,
        tmp_dir=tmp_dir,
        tqdm_name=tqdm_name,
        tqdm_idx=tqdm_idx,
    )


if __name__ == "__main__":
    download_nisqa(tqdm_name="nisqa", tqdm_idx=0)
