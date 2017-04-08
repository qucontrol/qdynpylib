"""Auxilliary routines for testing either the QDYN python package or the QDYN
Fortran library"""
from __future__ import print_function, division, absolute_import

import os
import re

from distutils import dir_util
import subprocess


FEATURES = ['check-cheby', 'no-check-cheby', 'check-newton', 'no-check-newton',
            'parallel-ham', 'no-parallel-ham', 'use-mpi', 'use-mkl',
            'parallel-oct', 'no-parallel-oct', 'backtraces', 'no-backtraces',
            'debug', 'no-debug']


def qdyn_feature(configure_log, feature):
    """Check whether QDYN was configured with the given feature (e.g.
    'use_mpi', 'no-parallel-oct', ect), given the path to configure.log"""
    if feature not in FEATURES:
        raise ValueError("Unknown feature: %s. Valid features are: %s"
                         % (feature, ", ".join(FEATURES)))
    with open(configure_log) as in_fh:
        for line in in_fh:
            if 'no-%s' % feature in line:
                return False
            elif feature in line:
                return True
    return False


def get_mpi_implementation(configure_log):
    """Return the name of the MPI implementation that QDYN was configured with
    (e.g. 'openmpi', or None if QDYN was compiled without MPI support"""
    with open(configure_log) as in_fh:
        for line in in_fh:
            if 'use-mpi' in line:
                m = re.search(r'use-mpi=(\w+)', line)
                if m:
                    implementation = m.group(1)
                    return implementation
    return None


def get_qdyn_compiler(configure_log):
    """Return the name the Fortran compiler that QDYN was compiled with"""
    with open(configure_log) as in_fh:
        for line in in_fh:
            if line.startswith("FC"):
                m = re.search(r'FC\s*:\s*(.*)', line)
                if m:
                    fc = m.group(1)
                    return fc
    return None


def mpirun(cmd, procs=1, implementation='openmpi', hostfile=None):
    """Return a modified `cmd` that runs the given `cmd` (list) using MPI.

    If `hostfile` is given, it will be overwritten and used in such a manner as
    to force the use of the given number of processes.

    Args:
        cmd (list): list of command args, cf. `subprocess.call`
        procs (int): Number of MPI processes that should be used
        implementation (str): name of MPI implementation
        hostfile (str): Path to file that should be used as a "hostfile",
            forcing MPI to use the specified number of processes even if the
            MPI environment would not ordinarily allow for it.
    """
    if implementation is None:
        return cmd
    elif implementation == 'openmpi':
        new_cmd = ['mpirun', '-n', str(procs), ]
        if hostfile is not None:
            hostfile = os.path.abspath(hostfile)
            with (open(hostfile, 'w')) as out_fh:
                out_fh.write("localhost slots=%d\n" % procs)
            new_cmd += ['--hostfile', hostfile]
        new_cmd += cmd
        return new_cmd
    else:
        raise ValueError("Unknown MPI implementation")


def datadir(tmpdir, request):
    '''Proto-fixture responsible for searching a folder with the same name
    as a test module and, if available, moving all contents to a temporary
    directory so tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return str(tmpdir)


def make_qdyn_utility(util='qdyn_prop_traj', procs=1, threads=1):
    """Generate a proto-fixture that returns a wrapper around
    the utility `util` in '../utils/', relative to the test directory.

    If `procs` > 1 and QDYN was compiled with MPI support, then call the
    utility through mpirun (or equivalent), to run `procs` simultaneous copies
    of the program. If `threads` is > 1, the program will run with multiple
    OpenMP threads.

    The wrapper returned by the fixture behaves like `subprocess.call`, with a
    predefined `cmd` and environment.
    """
    def qdyn_utility(request, tmpdir):
        """Wrapper for the {util} utility""".format(util=util)
        test_module = request.module.__file__
        testdir = os.path.split(test_module)[0]
        units_files = os.path.join(testdir, '..', 'units_files')
        configure_log = os.path.join(testdir, '..', 'configure.log')
        mpi_implementation = get_mpi_implementation(configure_log)
        exe = os.path.abspath(os.path.join(testdir, '..', 'utils', util))
        env = os.environ.copy()
        env['QDYN_UNITS'] = units_files
        env['OMP_NUM_THREADS'] = str(threads)
        cmd = [exe, ]
        if (mpi_implementation is not None) and (procs > 1):
            cmd = mpirun(cmd, procs=procs, implementation=mpi_implementation,
                         hostfile=str(tmpdir.join('hostfile')))

        def run_cmd(*args):
            full_cmd = cmd + list(args)
            return subprocess.call(full_cmd, env=env, universal_newlines=True)

        return run_cmd

    return qdyn_utility
