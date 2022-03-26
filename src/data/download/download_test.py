from typing import List

from src import constants

from src.data.download.datasets import (
    download_pstn_test,
    download_tencent_test,
    download_tub_test,
)
from src.data.download.utils.tqdm_write import (
    tqdm_run_parallel,
    tqdm_printer,
)


def _pad_name(name: str, max_len: int):
    return name + " "*(max_len-len(name))


def _format_names(names: List[str]):
    max_len = max(map(len, names))
    return [_pad_name(name, max_len) for name in names]


def download_test():

    # Create tmp dir.
    tmp_dir = constants.DIR_DATA.joinpath("tmp")
    tmp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # NISQA, PSTN and Tencent can be downloaded in parallel.
    funcs = [download_pstn_test, download_tencent_test, download_tub_test]
    N = len(funcs)
    args = [(tmp_dir,) for _ in funcs]
    names = _format_names(["pstn", "tencent", "tub"])

    print("Downloading PSTN, Tencent and TUB...")
    with tqdm_printer(N):
        tqdm_run_parallel(funcs, args, names)


if __name__ == "__main__":
    download_test()
