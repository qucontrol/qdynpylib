"""Moule containing useful utilities for scattering calculations."""

import numpy as np
from scipy.special import riccati_jn, riccati_yn
from scipy.special import eval_legendre


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


def partial_wave_scattering_amplitude(l, k,  S_l):
    """Return the legendre coefficient in the expansion of the scattering
    amplitude for the given partial wave quantum number `l`.
    """
    #f_l = (S_l - 1)
    #for l in l_vals:
    #    f_l[l] = f_l[l] * (2*l+1)/(2j*k)
    f_l = (S_l - 1) * (2*l+1)/(2j*k)
    return f_l


def scattering_amplitude(theta, l_vals, f_l):
    """Calculate the scattering amplitude for a single energy.
    """
    f_k = np.zeros(len(theta))
    for l in l_vals:
        f_k = f_k + f_l[l] * eval_legendre(l, np.cos(theta))
    return f_k


def differential_cross_section(f_k):
    """
    """
    d_sigma = np.abs(f_k)**2
    return d_sigma


def partial_wave_cross_section(k, l, delta_l):
    """
    """
    sigma_l = np.zeros(len(l_vals))
    for l in l_vals:
        sigma_l[l] = 4*np.pi*(2*l+1)/(k**2) * np.sin(delta_l[l])**2
    return sigma_l


def integral_cross_section(sigma_l):
    """
    """
    sigma = np.sum(sigma_l, axis=0)
    return sigma


def integrate_differential_cross_section(d_sigma, theta, range):
    """
    """
    Ntheta = len(theta)

    if range == 'full':
        x = theta
        y = d_sigma * np.sin(theta)
    elif range == 'backward hemisphere':
        x = theta[Ntheta//2:]
        y = d_sigma[Ntheta//2:] * np.sin(x)
    elif range == 'forward hemisphere':
        x = theta[:Ntheta//2]
        y = d_sigma[:Ntheta//2] * np.sin(x)

    integral = 2*np.pi * integrate.trapz(y, x)
    return integral
