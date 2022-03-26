from contextlib import contextmanager
from multiprocessing import Pool, RLock
import sys
from tqdm import tqdm
from typing import Tuple

# stdout tips: https://tldp.org/HOWTO/Bash-Prompt-HOWTO/x361.html
# part 2: http://www.climagic.org/mirrors/VT100_Escape_Codes.html


def _tqdm_write(x):
    sys.stdout.write(x)
    sys.stdout.flush()


def _clear():
    clearall = "\033[2J"
    goto_zero = "\033[0;0H"
    _tqdm_write(clearall)
    _tqdm_write(goto_zero)


def _init_print(N: int):
    # Make N lines.
    _tqdm_write("\n"*(N))

    # Move to top-left.
    up = "\033[%iA" % N
    _tqdm_write(up)
    _tqdm_write("\r")


def _end_print(N: int):
    # Move to bottom-left.
    down = "\033[%iB" % N
    _tqdm_write(down)
    _tqdm_write("\r")


@contextmanager
def tqdm_printer(N: int = None):
    try:
        # Initialize with clear.
        if N is None:
            _clear()
        else:
            _init_print(N)

        # Do stuff...
        yield
    finally:
        # Finalize with clear.
        if N is None:
            _clear()
        else:
            _end_print(N)


def tqdm_print(x, name, idx):
    # Use normal print if name or idx is nOne
    if name is None or idx is None:
        print(x)
        return

    msg = name + ": " + x
    tqdm.get_lock().acquire()
    down = "\033[%iB" % idx
    save = "\033[s"
    restore = "\033[u"
    clear_line = "\033[2K"
    _tqdm_write(save)
    if idx > 0:
        _tqdm_write(down)
    _tqdm_write(clear_line + "\r")
    _tqdm_write(msg)
    _tqdm_write(restore)
    tqdm._lock.release()


def _run(f, args: Tuple, tqdm_name: str, tqdm_idx: int):
    f(*args, tqdm_name=tqdm_name, tqdm_idx=tqdm_idx)


def tqdm_run_parallel(funcs, args_per_func, names):
    assert len(funcs) == len(names)
    N = len(funcs)
    run_args_list = [(funcs[i], args_per_func[i], names[i], i, )
                     for i in range(N)]

    pool = Pool(processes=N, initargs=(RLock(),), initializer=tqdm.set_lock)

    jobs = [pool.apply_async(_run, args=x)
            for x in run_args_list]
    pool.close()
    result_list = [job.get() for job in jobs]

    return result_list
