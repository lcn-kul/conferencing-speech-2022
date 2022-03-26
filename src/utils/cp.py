from pathlib import Path
import shutil


def cp(src_path: Path, dst_path: Path):
    """Copy a file or directory from the source path to the destination path.

    This will create any required directories and delete the existing path.

    Args:
        src_path (Path): Source path
        dst_path (Path): Destination path
    """
    # Make directories (including parents).
    try:
        dst_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    except FileExistsError:
        dst_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    # Remove the existing final directory.
    shutil.rmtree(str(dst_path), ignore_errors=True)
    # Copy the file/directory.
    shutil.copy(str(src_path), str(dst_path))
