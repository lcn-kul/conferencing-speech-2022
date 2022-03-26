from src import constants


def full_path(path: str, make_dir: bool = True):
    full_path = constants.DIR_PROJECT.joinpath(path)
    if make_dir:
        full_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    return str(full_path)
