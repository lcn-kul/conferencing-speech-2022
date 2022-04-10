from contextlib import contextmanager
import os
from pathlib import Path
from src import constants


@contextmanager
def run_once(flag_name: str, partition_idx: int = 0, num_partitions: int = 1):
    """Run a section of code exactly one time.

    This works by creating an empty "flag" file after the code is successfully
    executed. If the flag file exists, then we know not to run the code again.

    Usage:
    ```
    flag_name = "xyz_finished"
    with run_once(flag_name) as should_run:
        if should_run:
            # EXECUTE CODE
            print("xyz")
    ```

    Args:
        flag_name (str): Unique name of flag file.

    Yields:
        bool: Should run.
    """



    # Construct flag path.
    flag_dir = constants.DIR_DATA_FLAGS
    flag_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    flag_path = flag_dir.joinpath(flag_name + ".txt")
    if num_partitions == 1:
        flag_path_part = flag_path
    else:
        flag_name_part = f"{flag_name}_{partition_idx}_{num_partitions}"
        flag_path_part = flag_dir.joinpath(flag_name_part + ".txt")

    try:
        if flag_path.exists() or flag_path_part.exists():
            should_run = False
            yield should_run
        else:
            should_run = True
            yield should_run
            with open(str(flag_path_part), "w") as f:
                f.write("")
    finally:
        # Search for complete partition.
        parts = list(flag_dir.glob(f"{flag_name}_[0-9]*_[0-9]*.txt"))
        split_parts = [str(p.stem).split("_") for p in parts]
        split_parts = [["_".join(p[:-2]), int(p[-2]), int(p[-1])] for p in split_parts]
        d = {}
        for _, idx, N in split_parts:
            if N not in d:
                d[N] = set()
            d[N].add(idx)

        complete = False
        for N in d:
            all_present = True
            for i in range(N):
                if i not in d[N]:
                    all_present = False
                    break
            if all_present:
                complete = True
                break
        
        # Replace the partition flags with a single flag representing all
        # partitions complete.
        if complete:
            with open(str(flag_path), "w") as f:
                f.write("")
            for path in parts:
                os.remove(path)


