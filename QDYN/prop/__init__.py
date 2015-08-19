#!/usr/bin/env python
"""
Propagation of states and density matrices. Calculates the result of the
function

    exp[-1j*O*dt].state

For a constant operator O or the corresponding product of exponentials for a
time-dependent (piece-wise constant) Operator
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
from . import newton
from . import exact
from six.moves import xrange

def generate_apply_H(H0, H1, pulse):
    """
    Generate an apply_H routine that applies the Hamiltonian
    H0 + pulse(t)*H1

    Arguments
    ---------

    H0: ndarray, matrix
        Drift Hamiltonian

    H1: ndarray, matrix
        Control Hamiltonian

    pulse: function
        pulse(t) must return a complex control value

    Example
    -------

    >>> H0 = np.matrix(np.array([[1,0],[0,2]]))
    >>> H1 = np.matrix(np.array([[0,1],[1,0]]))
    >>> def pulse(t):
    ...     return t * 0.1
    ...
    >>> e1 = np.array([1,0])
    >>> e2 = np.array([0,1])
    >>> apply_H = generate_apply_H(H0, H1, pulse)
    >>> apply_H(e1, 2.0)
    matrix([[ 1. ,  0.2]])
    >>> apply_H(e2, 2.0)
    matrix([[ 0.2,  2. ]])

    """
    N = H0.shape[0]

    def apply_H(state, t):
        assert state.shape == (N,), \
        "state of shape %s must be %d-dimensional Hilbert space vector" \
        % (str(state.shape), N)
        H = np.matrix(H0 + pulse(t) * H1)
        return H.dot(state)

    return apply_H


def generate_apply_L(H0, H1, pulse, dissipator=None):
    """
    Generate an apply_H routine that applies the Hamiltonian
    H0 + pulse(t)*H1

    Arguments
    ---------

    H0: ndarray, matrix
        Drift Hamiltonian

    H1: ndarray, matrix
        Control Hamiltonian

    pulse: function
        pulse(t) must return a complex control value

    dissipator: function, optional
        dissipator(state) returns the dissipator part of the master equation

    """
    N = H0.shape[0]

    def apply_L(state, t):
        assert state.shape == (N,N), \
        "state must be %d x %d density matrix" % (N, N)
        H = np.matrix(H0 + pulse(t) * H1)
        result = H.dot(state) - state.dot(H)
        if dissipator is not None:
            result += dissipator(state)
        return result

    return apply_L



def propagate(apply_op, state, tgrid, method='exact', info_hook=None,
    storage=None, m=3, maxrestart=20, tol=1.0e-12):
    """
    Propagate state over the given time grid

    Arguments
    ---------

    apply_op: routine implementing O(t).state:
        >>> def apply_op(state,t):
        ...     # state -> O(t) applied to state
        ...     return state

    state: ndarray
        Data structure containing state description. 1D-array for wave
        functions, 2D array or matrix for density matrices

    tgrid: ndarray
        Array of time grid points

    method: str
        Name of propagation method to use

    info_hook: routine, optional
        Routine that is called after each propagation step
        >>> def info_hook(state, t):
        ...     pass

    storage: array, optional
        Array to which all propagated states will be appended

    m: int, optional
        For Newton propagator, Arnoldi order

    maxrestart: int, optional
        For Newton propagator, number of restarts

    tol: float, optional
        Desired precision of propagation result

    Returns
    -------

    Propagated state at `tgrid[-1]`

    """
    assert(tgrid[0] == 0.0), "time grid must start at t = 0"
    dt = tgrid[1] - tgrid[0]
    if (state.dtype != np.complex128):
        raise TypeError("State must be of dtype 'complex128'")
    if info_hook is not None:
        info_hook(state, t=tgrid[0])
    for step in xrange(1, len(tgrid)):
        t = 0.5*(tgrid[step-1] + tgrid[step]) # time to use for L(t)
        if method == 'exact':
            state = exact.prop_step_exact(apply_op, state, t, dt)
        elif method == 'newton':
            state = newton.newton_step(apply_op, state, t, dt, m, maxrestart,
                                       tol)
        else:
            raise ValueError("Unknown method: %s", method)
        if info_hook is not None:
            info_hook(state, t=(t+0.5*dt))
        if storage is not None:
            storage.append(state)
    return state

