from pathlib import Path

from src import constants
from src.data.download.utils.download_dataset_zip import download_dataset_zip


def download_tencent_test(
    tmp_dir: Path = None,
    tqdm_name: str = None,
    tqdm_idx: int = None,
):
    """Download the test set of the Tencent Corpus and extract it to the
    appropriate directory."""

    download_dataset_zip(
        name="tencent_test",
        data_url=constants.TENCENT_TEST_URL,
        output_dir=constants.TENCENT_TEST_DIR,
        extracted_name=constants.TENCENT_TEST_ZIP_FOLDER,
        tmp_dir=tmp_dir,
        tqdm_name=tqdm_name,
        tqdm_idx=tqdm_idx,
    )


if __name__ == "__main__":
    download_tencent_test(tqdm_name="tencent", tqdm_idx=0)
