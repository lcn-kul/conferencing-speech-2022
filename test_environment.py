from pathlib import Path
import sys

REQUIRED_PYTHON = "python3"


def main():

    ROOT_DIR = Path(__file__).resolve().parent
    VENV_DIR = ROOT_DIR.joinpath("venv")
    if not VENV_DIR.exists():
        msg = ">>> venv/ does not exist! Please run 'make create_environment' first."
        print(msg)
        exit(1)

    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
