import os
import sys
import contextlib


@contextlib.contextmanager
def ignore_warning():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


class hide_print:
    def __init__(self, hide=True):
        self.hide = hide
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        if (self.hide):
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (self.hide):
            sys.stdout.close()
            sys.stdout = self._original_stdout
