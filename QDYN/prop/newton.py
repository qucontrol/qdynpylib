#!/usr/bin/env python
"""
Newton Propagator
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import logging
from ..linalg import norm, inner
import numpy as np
from six.moves import xrange

VERBOSE = False

def Arnoldi(apply_A, t, dt, v0, m_max):
    """
    Calculate the (extended) Hessenberg matrix of an operator A(t) and return
    it together with the  m+1 (m <= m_max) Arnoldi vectors (orthonormlized
    Krylov vectors).  Also return the combined Ritz values (all eigenvalues of
    the Hessenberg matrices of size 1..m )

    Arguments
    ---------
    apply_A : function
        Function encoding the operator A(t). Calling `apply_A(v0, t)` must
        return the vector A(t).v0
    t : float
        Time at which to evaluate A
    dt : float
        Time step
    v0 : ndarray
        Initial vector or density matrix
    m_max : integer
        Maximal Krylov dimension

    Returns
    -------
    arnoldi_vecs : array of ndarray(complex128)
        `arnoldi_vecs` has size m+1 and contains the orthonormalized Krylov
        vectors u_1 .. u_{m+1}
    Hess : matrix(complex128)
        Extended Hessenberg matrix, of size (m+1) x (m+1). The top left
        sub-matrix of size m x m is the proper Hessenberg matrix. The m+1st row
        and m+1 column are zero, except for the (m+1, m+1) element, which
        contains the h_{m+1, m} (sic!) Hessenberg value
    Ritz : ndarray(complex128)
        Set of the eigenvalues of all Hessenberg matrices Hess_j=1..m
    m : integer
        Size of returned Hessenberg matrix
    """
    m = m_max

    # Hessenberg matrix (at maximum size)
    Hess = np.matrix(np.zeros(shape=(m_max+1,m_max+1), dtype=np.complex128))

    # Eigenvalues of all Hess
    Ritz = []

    arnoldi_vecs = []

    beta = norm(v0)
    if (abs(beta-1.0) > 1.0e-10):
        print("beta = ", beta)
        raise AssertionError("v0 must have norm 1.0")
    v = v0 / beta
    arnoldi_vecs.append(v)
    for j in xrange(m):
        v = apply_A(v, t) # v_{j+1}
        for i, v_i in enumerate(arnoldi_vecs):
            Hess[i,j] = dt * inner(v_i, v)
            v = v - np.dot((Hess[i,j]/dt), v_i)
        # At this point, we have finished the (j+1) x (j+1) Hessenberg matrix
        Ritz.extend(np.linalg.eigvals(Hess[:j+1,:j+1]))
        h_next = norm(v)
        Hess[j+1,j] = h_next * dt
        if h_next <= 1.0e-14: # abort early at convergence
            m = j
            break
        v *= 1 / h_next # normalize
        arnoldi_vecs.append(v)
    # At this point, arnoldi_vecs contains m+1 elements
    Ritz = np.array(Ritz, dtype=np.complex128)
    return arnoldi_vecs, Hess[:m+1,:m+1], Ritz, m


def normalize_points(z):
    """
    Given a set of complex points z, return the normalization radius
    """
    r = 0.0
    for z_i in z:
        r_i = abs(z_i)
        if r_i > r:
            r = r_i
    # we need to enlarge the radius a little bit to account for points that
    # will be added in later iterations
    r *= 1.2 # arbitary factor
    assert(r > 0.0), "Radius is zero"
    return r


def extend_newton_coeffs(old_a, new_leja, center, radius):
    """
    Extend a set of Newton coefficients, by using a set of new_leja points
    which are normalized with the given center and radius
    """
    n_old = len(old_a)
    m = len(new_leja) - n_old
    a = np.zeros(n_old+m, dtype=np.complex128)
    a[:n_old] = old_a
    n0 = n_old

    if n_old == 0:
        a[0] =   np.exp(-1j * new_leja[0])
        n0 = 1

    for k in xrange(n0, n_old+m):
        d  = 1.0
        pn = 0.0
        for n in xrange(1,k): # 1..k-1
            zd  = new_leja[k] - new_leja[n-1]
            d  *= zd / radius
            pn += a[n] * d
        zd = new_leja[k] - new_leja[k-1]
        d *= zd / radius
        assert(abs(d) > 1.0e-200), "Divided differences too small"
        a[k] =  (np.exp(-1j * new_leja[k])
                 -a[0] - pn ) / d
    return a


def extend_leja(old_leja, new_points, n_use):
    """
    Given a set of normalized (ordered) Leja points, extract n_use points from
    the (normalized) new_points, and append them to the set of leja points
    """
    n_old = len(old_leja)
    new_leja = np.zeros(n_old + n_use, dtype=np.complex128)
    new_leja[:n_old] = old_leja[:]
    i_add_start = 0
    if n_old == 0:
        # At the very beginning, start with the point that has largest absolute
        # value
        for i in xrange(len(new_points)-1): # 0 .. n_old - 2
            if (abs(new_points[i]) > abs(new_points[-1])):
                temp = new_points[i]
                new_points[i] = new_points[-1]
                new_points[-1] = temp
        new_leja[0] = new_points[-1]
        i_add_start = 1
    # find the best point for new_leja[n_old+n]
    n_added = i_add_start
    ex = 1.0/(n_old + n_use)
    for i_add in xrange(i_add_start, n_use):
        p_max = 0.0
        i_max = 0
        # the new leja are defined with index  0 .. (n_old-1)+n
        # the new candidates are defined with index 0 .. len(new_points)-1+n
        for i in xrange(len(new_points)-i_add): # trial points (candidates)
            p = 1.0
            for j in xrange(n_old + i_add): # existing leja points
                p *= np.abs(new_points[i] - new_leja[j])**ex
            # at this point p is the divided difference denominator for the
            # candidate with index i
            if p > p_max:
                p_max = p
                i_max = i
        # XXX if p_max below limit: abort
        new_leja[n_old+i_add] = new_points[i_max]
        n_added += 1
        # remove the used point by moving in the last point
        new_points[i_max] = new_points[len(new_points)-1-i_add]
    return new_leja, n_added


def newton_step(apply_op, state, t, dt, m_max, maxrestart, tol):
    """
    Perform a propagation step using the Restarted Newton algorithm

    Arguments
    ---------

    apply_op: function
        Function that applies the operator (Liouvillian/Hamiltonian) at a given
        time to a given state (`apply_op(state, t) -> state`)

    state: ndarray
        Data structure containing state description. 1D-array for wave
        functions, 2D array or matrix for density matrices

    t: float
        Time argument to pass to `apply_op`

    dt: float
        Time grid step

    m_max: int
        Maximal order of Arnoldi procedure

    maxrestart: int
        Maximal number of Newton restarts

    tol: float
        Desired precision of propagation result
    """

    logger = logging.getLogger(__name__)
    N = state.shape[0]
    assert(m_max <= N), "m_max must be smaller than the system dimension"
    def gen_zero_vec(v0):
        """ Generator for generation routine that produces zero-vector """
        def zero_vec():
            result = np.zeros(shape=v0.shape, dtype=v0.dtype)
            if isinstance(v0, np.matrix):
                return np.matrix(result)
            else:
                return result
        return zero_vec
    zero_vec = gen_zero_vec(state)
    w = zero_vec()                                     # result vector
    Z = np.zeros(0, dtype=np.complex128)               # Leja points
    a = np.zeros(0, dtype=np.complex128)               # Newton coeffs

    beta = norm(state)
    v = state / beta

    for s in xrange(maxrestart):

        #if s > 0: # check convergence
            #if ((abs(a[-1])  * beta_prev) / norm(w)) < tol:
                #print("Converged at restart ", s-1)
                #print("norm of wp     : ", norm(wp))
                #print("norm of w      : ", norm(w))
                #print("beta           : ", beta)
                #print("beta*a[-1]     : ", beta * a[-1])
                #print("max Leja radius: ", np.max(np.abs(Z)))
                #break

        arnoldi_vecs, Hess, Ritz, m = Arnoldi(apply_op, t, dt, v, m_max)
        if m < m_max:
            logger.warn("Arnoldi only returned order %d instead of the "
                        "requested %d", m, m_max)
        if m == 0 and s == 0:
            # The input state must be an eigenstate
            eig_val = beta * Hess[0,0]
            phase = np.exp(-1.0j * eig_val) # dt is absorbed in eig_val
            w = phase * state
            break

        # normalize Ritz points
        if s == 0:
            radius = normalize_points(Ritz)
            center = 0.0
        assert(radius > 0.0), "Radius is zero"

        # get Leja points (i.e. Ritz points in the proper order)
        n_s = len(Z)
        Z, m = extend_leja(Z, Ritz, m) # Z now contains m new Leja points
        assert(m > 0), "No new Leja points"
        a = extend_newton_coeffs(a, Z, center, radius)

        R = np.matrix(np.zeros(shape=(m+1,1), dtype=np.complex128))
        R[0,0] = beta
        P = a[n_s] * R
        for k in xrange(1, m): # 1..m-1
            R = (np.dot(Hess, R) - Z[n_s+k-1] * R) / radius
            P += a[n_s+k] * R

        wp = zero_vec()
        for i in xrange(m): # 0 .. m-1
            wp += P[i,0] * arnoldi_vecs[i]

        w += wp

        # starting vector for next iteration
        R = (np.dot(Hess, R) - Z[n_s+m-1] * R) / radius
        beta = norm(R)
        R /= beta
        # beta would be the norm of v, with the above normalization, v will now
        # be normalized
        v = zero_vec()
        for i in xrange(m+1): # 0 .. m
            v += R[i,0] * arnoldi_vecs[i]

        #if (norm(wp) / norm(w) < tol):
#        if (norm(wp) < tol):
        if (beta*abs(a[-1])/(1+norm(w)) < tol):
            logger.debug("Converged at restart %s", s)
            logger.debug("norm of wp     : %s", norm(wp))
            logger.debug("norm of w      : %s", norm(w))
            logger.debug("beta           : %s", beta)
            logger.debug("|R*a[-1]|/|w|  : %s", norm(R) * a[-1] / norm(w))
            logger.debug("max Leja radius: %s", np.max(np.abs(Z)))
            break

        assert(not np.isnan(np.sum(v))), "v contains NaN"
        assert(not np.isnan(np.sum(w))), "w contains NaN"

        if s == maxrestart - 1:
            logger.warn("DID NOT REACH CONVERGENCE")
            logger.warn("increase number of restarts")

    return w


