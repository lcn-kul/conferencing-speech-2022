import time
from tqdm import tqdm

from src.data.download.utils.tqdm_write import (
    tqdm_print,
    tqdm_printer,
    tqdm_run_parallel,
)


def func0(tqdm_name, tqdm_idx):
    N = 100
    with tqdm(desc=tqdm_name, total=N, position=tqdm_idx, leave=False) as pbar:
        for i in range(N):
            pbar.update()
            time.sleep(0.05)
    tqdm_print("done", tqdm_name, tqdm_idx)


def func1(tqdm_name, tqdm_idx):
    N = 200
    with tqdm(desc=tqdm_name, total=N, position=tqdm_idx, leave=False) as pbar:
        for i in range(N):
            pbar.update()
            time.sleep(0.05)
    tqdm_print("done", tqdm_name, tqdm_idx)


def func2(tqdm_name, tqdm_idx):
    N = 600
    with tqdm(desc=tqdm_name, total=N, position=tqdm_idx, leave=False) as pbar:
        for i in range(N):
            pbar.update()
            time.sleep(0.02)
    tqdm_print("done", tqdm_name, tqdm_idx)


def test_tqdm():

    funcs = [func0, func1, func2]
    args = [tuple() for _ in funcs]
    names = ["func0", "func1", "func2"]

    print("Testing tqdm...")
    time.sleep(1)
    with tqdm_printer(3):
        tqdm_run_parallel(funcs, args, names)
    print("done")


if __name__ == "__main__":
    test_tqdm()
