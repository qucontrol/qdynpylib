"""Moule containing useful utilities for scattering calculations."""

import numpy as np
from scipy.special import riccati_jn, riccati_yn


def scattering_phase_from_psi(psi, r_grid, E, mu):
    """Return the scattering phase of the given wavefunction.
    """
    k = np.sqrt(2.0 * mu * E)

    # choose grid values for expansion coefficients
    i_r1 = int(0.8 * len(r_grid))
    i_r2 = int(0.8 * len(r_grid)) - 1
    r1 = r_grid[i_r1]
    r2 = r_grid[i_r2]

    R1 = np.asscalar(np.real(psi[i_r1])) * r1
    R2 = np.asscalar(np.real(psi[i_r2])) * r2

    ## use asymptotic approximation for scattering wavefunctions
    # j1 = np.sin(k*r1 - l*0.5*np.pi)
    # j2 = np.sin(k*r2 - l*0.5*np.pi)
    # y1 = np.cos(k*r1 - l*0.5*np.pi)
    # y2 = np.cos(k*r2 - l*0.5*np.pi)

    # use exact ricatti and hankel functions for scattering wave
    j1 = riccati_jn(l, k * r1)[0][l]
    j2 = riccati_jn(l, k * r2)[0][l]
    y1 = -riccati_yn(l, k * r1)[0][l]
    y2 = -riccati_yn(l, k * r2)[0][l]

    A = k * (R2 * y1 - R1 * y2) / (j2 * y1 - j1 * y2)
    B = k * (R1 / y1 - j1 / y1 * (R2 * y1 - R1 * y2) / (j2 * y1 - j1 * y2))

    # renormalise expansion coefficients
    norm = np.sqrt(A ** 2 + B ** 2)
    A = A / norm
    B = B / norm

    # calculate phase shift
    # Note: to obtain the scattering phase, we can use either the arctan or
    # the arctan2 function. The difference is, that arctan gives the phase
    # on the interval [-pi/2, pi/2], while arctan2 gives the phase on the
    # the interval [-pi, pi]. Hence, the two possibilities give rise to changes
    # of pi in the scattering phase.
    # For the calculation of the scattering MATRIX this is not important, since
    # in the exponential function delta is multiplied by two. The scattering
    # matrix is therefore independent of the scattering phase up to modulo pi.
    # Only for smoothing the phase, one has to keep track, with functions was
    # used.
    phase = np.arctan(B / A)
    # phase = np.arctan2(B,A)

    return phase
