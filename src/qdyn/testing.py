"""Auxilliary routines for testing either the QDYN python package or the QDYN
Fortran library"""
import os
import re
import subprocess
from distutils import dir_util
from pathlib import Path


FEATURES = [
    'check-cheby',
    'no-check-cheby',
    'check-newton',
    'no-check-newton',
    'parallel-ham',
    'no-parallel-ham',
    'use-mpi',
    'use-mkl',
    'parallel-oct',
    'no-parallel-oct',
    'backtraces',
    'no-backtraces',
    'debug',
    'no-debug',
]


def qdyn_feature(configure_log, feature):
    """Check whether QDYN was configured with the given feature (e.g.
    'use_mpi', 'no-parallel-oct', ect), given the path to configure.log"""
    if feature not in FEATURES:
        raise ValueError(
            "Unknown feature: %s. Valid features are: %s"
            % (feature, ", ".join(FEATURES))
        )
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
        cmd (list): list of command args, cf. `subprocess.run`
        procs (int): Number of MPI processes that should be used
        implementation (str): name of MPI implementation
        hostfile (str): Path to file that should be used as a "hostfile",
            forcing MPI to use the specified number of processes even if the
            MPI environment would not ordinarily allow for it.
    """
    if implementation is None:
        return cmd
    elif implementation in ['openmpi', 'intel']:
        new_cmd = ['mpirun', '-n', str(procs)]
        if implementation == 'openmpi' and hostfile is not None:
            hostfile = os.path.abspath(hostfile)
            with (open(hostfile, 'w')) as out_fh:
                out_fh.write("localhost slots=%d\n" % procs)
            new_cmd += ['--hostfile', hostfile]
        new_cmd += cmd
        return new_cmd
    else:
        raise ValueError("Unknown MPI implementation: %s" % implementation)


def datadir(tmpdir, request):
    """Proto-fixture responsible for searching a folder with the same name
    as a test module and, if available, moving all contents to a temporary
    directory so tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return str(tmpdir)


def make_qdyn_utility(util='qdyn_prop_traj', procs=1, threads=1):
    """Generate a proto-fixture to wrap around the tiven `util`.

    Returns a callable that takes any numer of positional `args` and any number
    of `kwargs`, such that calling it is equivalent to

    ::

        subprocess.run([cmd, *args], **kwargs)

    where ``cmd`` is the absolute path of the compiled QDYN utility (in the
    ``utils`` subfolder of the project root, found by traversing up from the
    directory in which the test is defined).

    If `procs` > 1 and QDYN was compiled with MPI support, then ``cmd`` will be
    ``mpirun`` (or an equivalent suitable MPI runner based on how QDYN was
    compiled), to run `procs` simultaneous copies of the `util`. If `threads`
    is > 1, the program will run with multiple OpenMP threads, by setting the
    ``OMP_NUM_THREADS`` environment variable.

    The `util` will also use the development units file by setting
    ``QDYN_UNITS`` to the ``units_file`` folder in the project root.
    """

    def qdyn_utility(request, tmpdir):
        """Wrapper for the {util} utility""".format(util=util)
        test_module = Path(request.module.__file__)
        root_dir = test_module.parent.absolute()
        while len(root_dir.parts) > 1:  # go to filesystem root
            root_dir = root_dir.parent
            if (root_dir / 'qdyn.f90').is_file():
                break

        units_files = root_dir / 'units_files'
        if not units_files.is_dir():
            raise IOError(f"Cannot find units_files folder in {root_dir}")
        configure_log = root_dir / 'configure.log'
        if not configure_log.is_file():
            raise IOError(f"Cannot find configure.log folder in {root_dir}")
        mpi_implementation = get_mpi_implementation(configure_log)
        exe = root_dir / 'utils' / util
        if not exe.is_file():
            raise IOError(f"Cannot find executable {exe}")
        cmds = [str(exe)]
        if (mpi_implementation is not None) and (procs > 1):
            cmds = mpirun(
                cmds,
                procs=procs,
                implementation=mpi_implementation,
                hostfile=str(tmpdir.join('hostfile')),
            )

        def run_cmd(*args, **kwargs):
            env = kwargs.get('env', os.environ.copy())
            if 'QDYN_UNITS' not in env:
                env['QDYN_UNITS'] = units_files
            if 'OMP_NUM_THREADS' not in env:
                env['OMP_NUM_THREADS'] = str(threads)
            kwargs['env'] = env
            return subprocess.run([*cmds, *args], **kwargs)

        return run_cmd

    return qdyn_utility
