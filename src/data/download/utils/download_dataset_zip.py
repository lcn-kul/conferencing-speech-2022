from pathlib import Path
import wget
import zipfile

from src import constants
from src.utils.mv import mv
from src.data.download.utils.tqdm_get import tqdm_get
from src.data.download.utils.tqdm_write import tqdm_print
from src.utils.run_once import run_once


def download_dataset_zip(
    name: str,
    data_url: Path,
    output_dir: Path,
    extracted_name: str,
    tmp_dir: Path = None,
    tqdm_name: str = None,
    tqdm_idx: int = None,
):
    """Download the ZIP file and extract it to the appropriate directory."""

    # Whether we are using tqdm for printing.
    # Note: if tqdm_name/tqdm_idx is None, then tqdm_print will just print().
    use_tqdm = (tqdm_name is not None) and (tqdm_idx is not None)

    # Temporary download directory.
    if tmp_dir is None:
        tmp_dir = constants.DIR_DATA.joinpath("tmp")
        tmp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # Download ZIP to `zip_path`.
    zip_path = tmp_dir.joinpath(name + ".zip")
    flag_name = name + "_downloaded"
    with run_once(flag_name) as should_run:
        if should_run:
            if use_tqdm:
                tqdm_get(data_url, zip_path, tqdm_name, tqdm_idx)
            else:
                print("Downloading %s ZIP file." % name)
                wget.download(data_url, out=str(zip_path), )
                print("\r\n")

    # Extract ZIP.
    flag_name = name + "_extracted"
    with run_once(flag_name) as should_run:
        if should_run:
            tqdm_print("Extracting ZIP file...", tqdm_name, tqdm_idx)
            with zipfile.ZipFile(str(zip_path), mode="r") as f:
                f.extractall(tmp_dir)

    # Move files to correct directory.
    flag_name = name + "_moved"
    with run_once(flag_name) as should_run:
        if should_run:
            tqdm_print("Moving files...", tqdm_name, tqdm_idx)
            extract_dir = tmp_dir.joinpath(extracted_name)
            dst_path = output_dir
            mv(extract_dir, dst_path)

    # Remove ZIP file.
    if zip_path.exists():
        zip_path.unlink()

    tqdm_print("Finished %s." % name, tqdm_name, tqdm_idx)
