from pathlib import Path

from src import constants
from src.data.download.utils.gdrive import GoogleDriveDownloader
from src.utils.run_once import run_once


def download_iu_bloomington(
    tmp_dir: Path = None,
):
    """Download the IU Bloomington Corpus and extract it to the appropriate
    directory."""

    # Temporary download directory.
    if tmp_dir is None:
        tmp_dir = constants.DIR_DATA.joinpath("tmp")
        tmp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # Create GoogleDriveDownloader object.
    flag_name = "iu_bloomington_downloaded"
    with run_once(flag_name) as should_run:
        if should_run:
            print("Preparing to download IU Bloomington Corpus...")
            downloader = GoogleDriveDownloader(constants.GDRIVE_CRED_PATH)
            downloader.download_folder(
                folder_id=constants.IU_BLOOMINGTON_TRAIN_ID,
                output_dir=constants.IU_BLOOMINGTON_TRAIN_DIR,
            )

    print("Finished IU Bloomington Corpus.")


if __name__ == "__main__":
    download_iu_bloomington()
