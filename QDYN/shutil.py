from __future__ import print_function, division, absolute_import, \
                       unicode_literals
from shutil import *
import contextlib
import os
from os import rmdir

def mkdir(name, mode=0750):
    """
    Implementation of 'mkdir -p': Creates folder with the given `name` and the
    given permissions (`mode`)

    * Create missing parents folder
    * Do nothing if the folder with the given `name` already exists
    * Raise OSError if there is already a file with the given `name`
    """
    if os.path.isdir(name):
        pass
    elif os.path.isfile(name):
        raise OSError("A file with the same name as the desired " \
                      "dir, '%s', already exists." % name)
    else:
        os.makedirs(name, mode)


# 'chdir' context manager
@contextlib.contextmanager
def chdir(dirname=None):
    """
    Change directory. Use as
        >>> mkdir('dir')
        >>> with chdir('dir'):
        ...     pass
        >>> rmdir('dir')
    """
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)

def tail(file, n):
    """
    Print the last n lines of the given file
    """
    with open(file) as in_fh:
        lines = in_fh.readlines()
        print("".join(lines[-n:]))
