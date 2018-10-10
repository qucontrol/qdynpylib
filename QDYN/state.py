"""Utiltiy routines for working with states (wave functions or density
matrices)
"""
from __future__ import print_function, division, absolute_import

import numpy as np
from numpy import pi
from six.moves import xrange

from .linalg import norm
from .io import read_real


def write_psi_amplitudes(psi, filename):
    """Write the wave function to file in the same format as the
    `write_psi_amplitudes` Fortran routine

    Parameters:
        psi (numpy array): Array of complex probability amplitudes
        filename (str): Name of file to which to write
    """
    with open(filename, 'w') as out_fh:
        is_complex = np.any(np.iscomplex(psi))
        if is_complex:
            out_fh.write("#%9s%25s%25s\n"
                         % ('index', 'Re[Psi]', 'Im[Psi]'))
        else:
            out_fh.write("#%9s%25s\n" % ('index', 'Re[Psi]'))
        for i, val in enumerate(psi):
            if np.abs(val) > 1e-16:
                if is_complex:
                    out_fh.write("%10d%25.16E%25.16E\n"
                                 % (i+1, val.real, val.imag))
                else:
                    out_fh.write("%10d%25.16E\n" % (i+1, val.real))


def read_psi_amplitudes(filename, n, block=1, normalize=True):
    """Read the wave function of size `n` from file. For 'block=1', inverse to
    `write_psi_amplitudes`. Returns complex or real numpy array

    By specifying `blocks`, data may be read from a file that contains multiple
    wave functions, in the format generated e.g. by the
    ``qdyn_prop_traj --write-all-states`` utility

    Parameters:
        filename (str): Name of file from which to read data
        n (int): dimension of the Hilbert space (i.e. size of returned vector)
        block (int): One-based index of the block to read from
            `filename`, if the file contains multiple block. Blocks must be
            separated by exactly two empty lines
        normalize (bool): Whether or not to normalize the wave function
    """
    if block < 1:
        raise ValueError("Invalid block %d, block must be >= 1" % block)
    return next(iterate_psi_amplitudes(filename, n, block, normalize))


def iterate_psi_amplitudes(filename, n, start_from_block=1, normalize=True):
    """Iterate over blocks of wave functions stored in `filename`

    This is equivalent to the following pseudocode::

        iter([read_psi_amplitudes(filename, n, block, normalize)
              for block >= start_from_block])

    Hoever, the data file is only traversed once, and memory is only required
    for one block.

    Parameters:
        filename (str): Name of file from which to read data
        n (int): dimension of the Hilbert space (i.e. size of returned vector)
        start_from_block (int): One-based index of block from which to
            start the iterator.
        normalize (bool): Whether or not to normalize the wave function
    """
    psi = np.zeros(n, dtype=np.complex128)
    i_block = 1
    blanks = 0

    def normalized(psi):
        if normalize:
            nrm = norm(psi)
            if nrm > 1e-15:
                return psi/norm(psi)
            else:
                return psi * 0.0
        else:
            return psi

    with open(filename, 'r') as in_fh:
        for line_nr, line in enumerate(in_fh):
            line = line.strip()
            if line == '':
                blanks += 1
                if blanks >= 2:
                    if i_block >= start_from_block:
                        yield normalized(psi)
                        psi = np.zeros(n, dtype=np.complex128)
                    blanks = 0
                    i_block += 1
                continue
            if i_block < start_from_block:
                continue
            if not line.startswith("#"):
                vals = line.split()[:3]
                try:
                    i = int(vals[0]) - 1
                    psi[i] = read_real(vals[1])
                    if len(vals) == 3:
                        psi[i] += 1j*read_real(vals[2])
                except (ValueError, TypeError) as exc_info:
                    raise ValueError("Invalid format: %s" % str(exc_info))
    if start_from_block > i_block:
        raise ValueError("Requested block %d, file only has %d blocks"
                         % (start_from_block, i_block))
    yield normalized(psi)


def random_density_matrix(N):
    """
    Return a random N x N density matrix

    >>> rho = random_density_matrix(10)

    The resulting density matrix is normalized, positive semidefinite, and
    Hermitian

    >>> assert( abs(np.trace(rho) - 1.0) <= 1.0e-14 )
    >>> assert( np.min(np.linalg.eigvals(rho).real) > 0.0 )
    >>> assert( np.max(np.abs(rho.H - rho)) <= 1.0e-14 )
    """
    rho = np.matrix(np.zeros(shape=(N,N)), dtype=np.complex128)
    for i in xrange(N):
        for j in xrange(N):
            r   = np.random.rand()
            phi = np.random.rand()
            rho[i,j] = r * np.exp(2.0j*pi*phi)
    # make hermitian
    rho = 0.5 * (rho + rho.H)
    # make positive-semidifinite by squaring
    rho = rho * rho
    # normalize
    rho = rho / (rho.trace()[0,0])
    return rho
