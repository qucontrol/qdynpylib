"""Utiltiy routines for working with states (wave functions or density
matrices)
"""
from __future__ import print_function, division, absolute_import

import numpy as np
from numpy import pi
from six.moves import xrange

from .linalg import norm


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


def read_psi_amplitudes(filename, n):
    """Read the wave function of size `n` from file. Inverse to
    `write_psi_amplitudes`. Returns complex or real numpy array.

    Paramters:
        filename (str): Name of file from which to read data
        n(int): dimension of the Hilbert space (i.e. size of returned vector)
    """
    psi = np.zeros(n, dtype=np.complex128)
    with open(filename, 'r') as in_fh:
        for line in in_fh:
            line = line.strip()
            if not line.startswith("#"):
                vals = line.split()[:3]
                try:
                    i = int(vals[0]) - 1
                    psi[i] = float(vals[1])
                    if len(vals) == 3:
                        psi[i] += 1j*float(vals[2])
                except (ValueError, TypeError) as exc_info:
                    raise ValueError("Invalid format: %s" % str(exc_info))
    return psi/norm(psi)


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
