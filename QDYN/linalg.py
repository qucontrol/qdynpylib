#!/usr/bin/env python
"""
Linear algebra helper routines
"""
from __future__ import print_function, division, absolute_import

import warnings
import logging

import numpy as np
import scipy.linalg
import scipy.sparse
from six.moves import xrange


def inner(v1, v2):
    """Calculate the inner product of the two vectors or matrices v1, v2

    For vectors, the inner product is the standard Euclidian inner product

    For matrices, the innner product is the Hilbert-Schmidt overlap.

    Note that the inner product of the vectorization of two matrices is the
    same as the inner product of the original matrices, and that two m x 1
    matrices have the same inner product as the corresponding m-dimensional
    vectors

    The `inner` routine corresponds to `overlap` in QDYN.

    Examples
    --------

    >>> v1 = np.array([1.0, 1.0j, 1.0, 1.0j])
    >>> v2 = np.array([1.0j, 1.0j, 1.0j, 1.0j])
    >>> inner(v1, v2)
    (2+2j)
    >>> m1 = np.matrix(v1.reshape((2,2)))
    >>> m2 = np.matrix(v2.reshape((2,2)))
    >>> inner(m1, m2)
    (2+2j)
    >>> m1 = v1.reshape((2,2))
    >>> m2 = v2.reshape((2,2))
    >>> inner(m1, m2)
    (2+2j)
    >>> m1 = v1.reshape((4,1))
    >>> m2 = v2.reshape((4,1))
    >>> inner(m1, m2)
    (2+2j)
    >>> m1 = v1.reshape((4,1), order='F')
    >>> m2 = v2.reshape((4,1), order='F')
    >>> inner(m1, m2)
    (2+2j)
    >>> m1 = v1.reshape((1,4), order='F')
    >>> m2 = v2.reshape((1,4), order='F')
    >>> inner(m1, m2)
    (2+2j)
    """
    assert (type(v1) is type(v2)), \
    "v1 and v2 must be of the same type: types are %s vs %s" \
    % (type(v1), type(v2))
    # numpy matrices are sub-types of ndarray
    assert isinstance(v1, np.ndarray), \
    "v1, v2 must be numpy matrices, or 1D/2D numpy arrays"
    if isinstance(v1, np.matrix):
        return trace(np.dot(v1.H, v2))
    else:
        assert (len(v1.shape) <= 2), "v1, v2 must be matrix or vector"
        if len(v1.shape) == 1:  # vector
            return np.vdot(v1, v2)
        else: # matrix as 2D array
            return trace(np.dot(v1.conjugate().transpose(), v2))


def trace(m):
    """Return the trace of the given matrix"""
    return (np.asarray(m)).trace()


def norm(v):
    """Calculate the norm of a vector or matrix `v`, matching the inner product
    defined in the `inner` routine. An algorithm like
    Gram-Schmidt-Orthonormalization will only work if the choice of norm and
    inner product are compatible.

    If `v` is a vector, the norm is the 2-norm (i.e. the standard Euclidian
    vector norm).

    If `v` is a matrix, the norm is the Hilbert-Schmidt (aka Frobenius) norm.
    Note that the HS norm of a matrix is identical to the 2-norm of any
    vectorization of that matrix (e.g. writing the columns of the matrix
    underneat each other). Also, the HS norm of the m x 1 matrix is the same as
    the 2-norm of the equivalent m-dimensional vector.

    """
    # scipy.linalg.norm does the right thing in all instances.
    return scipy.linalg.norm(v)


def get_op_matrix(apply_op, state_shape, t):
    """Return the explicit matrix for the operator encoded in `apply_op`,
    assuming that `apply_op` takes a numpy array or matrix of the given
    `state_shape` as its first argument

    Arguments
    ---------

    apply_op: routine
        apply_op(state, t) must return application of O(t) to state

    state_shape: tuple of ints
        Shape of states that apply_op understands

    t: float
        Time to pass to apply_op

    Returns
    -------

    Numpy matrix of size N x N, where N is the product of the entries of
    state_shape

    Example
    -------
    >>> from . prop import generate_apply_H, generate_apply_L
    >>> H0 = np.matrix(np.array([[1,0],[0,2]]))
    >>> H1 = np.matrix(np.array([[0,1],[1,0]]))
    >>> def pulse(t):
    ...     return t * 0.1
    ...
    >>> apply_H = generate_apply_H(H0, H1, pulse)
    >>> get_op_matrix(apply_H, state_shape=(2,), t=2.0)
    matrix([[ 1.0+0.j,  0.2+0.j],
            [ 0.2+0.j,  2.0+0.j]])
    >>> apply_L = generate_apply_L(H0, H1, pulse)
    >>> get_op_matrix(apply_L, state_shape=(2,2), t=2.0)
    matrix([[ 0.0+0.j,  0.2+0.j, -0.2+0.j,  0.0+0.j],
            [ 0.2+0.j,  1.0+0.j,  0.0+0.j, -0.2+0.j],
            [-0.2+0.j,  0.0+0.j, -1.0+0.j,  0.2+0.j],
            [ 0.0+0.j, -0.2+0.j,  0.2+0.j,  0.0+0.j]])
    """
    assert 1 <= len(state_shape) <= 2, \
    "dimension of shape must be 1 or 2"
    N = state_shape[0]
    reshape = lambda v: v # keep vector as-is
    if len(state_shape) == 2:
        N *= state_shape[1]
        reshape = lambda v: np.matrix(v.reshape(state_shape, order='F'))
    O = np.matrix(np.zeros(shape=(N,N), dtype=np.complex128))
    # Columns of L = results of applying the Liouville operator on basis
    # elements of the Liouville space (rho)
    for j in xrange(N):
        v = np.zeros(N)
        v[j] = 1.0 # j'th basis vector
        ov = vectorize(apply_op(reshape(v), t))
        for i in xrange(N):
            O[i,j] = ov[i]
    return O


def vectorize(a, order='F'):
    """Return vectorization of multi-dimensional numpy array or matrix `a`

    Examples
    --------

    >>> a = np.array([1,2,3,4])
    >>> vectorize(a)
    array([1, 2, 3, 4])

    >>> a = np.array([[1,2],[3,4]])
    >>> vectorize(a)
    array([1, 3, 2, 4])

    >>> a = np.matrix(np.array([[1,2],[3,4]]))
    >>> vectorize(a)
    array([1, 3, 2, 4])

    >>> vectorize(a, order='C')
    array([1, 2, 3, 4])
    """
    N = a.size
    return np.squeeze(np.asarray(a).reshape((1,N), order=order))


def is_hermitian(matrix):
    """Return True if matrix is Hermitian, False otherwise. The `matrix` can be
    a numpy array or matrix, a scipy sparse matrix, or a `qutip.Qobj` instance.

    >>> m = np.matrix([[0, 1j], [-1j, 1]])
    >>> is_hermitian(m)
    True

    >>> m = np.matrix([[0, 1j], [-1j, 1j]])
    >>> is_hermitian(m)
    False

    >>> m = np.array([[0, -1j], [-1j, 1]])
    >>> is_hermitian(m)
    False

    >>> from scipy.sparse import coo_matrix
    >>> m  = coo_matrix(np.matrix([[0, 1j], [-1j, 0]]))
    >>> is_hermitian(m)
    True
    """
    if hasattr(matrix, 'isherm'): # qutip.Qobj
        return matrix.isherm
    else: # any numpy matrix/array or scipy sparse matrix)
        #pylint: disable=simplifiable-if-statement
        if (abs(matrix - matrix.conjugate().transpose())).max() < 1e-14:
            # If we were to "simplify the if statement" and return the above
            # expression directly, we might get an instance of numpy.bool_
            # instead of the builtin bool that we want.
            return True
        else:
            return False


def iscomplexobj(x):
    """Check whether the (multidimensional `x` object) has a type that
    allows for complex entries.

    >>> iscomplexobj(1)
    False
    >>> iscomplexobj(1+0j)
    True
    >>> iscomplexobj([3, 1+0j, True])
    True
    >>> iscomplexobj(np.array([3, 1j]))
    True
    >>> iscomplexobj(scipy.sparse.csr_matrix([[1, 2], [4, 5]]))
    False
    >>> iscomplexobj(scipy.sparse.csr_matrix([[1, 2], [4, 5j]]))
    True
    """
    # This is a workaround for numpy bug #7924
    try:
        dtype = x.dtype
    except AttributeError:
        dtype = np.asarray(x).dtype
    try:
        return issubclass(dtype.type, np.core.numeric.complexfloating)
    except AttributeError:
        return False


def choose_sparsity_model(matrix):
    """Return one of 'full', 'banded', 'dia', or 'indexed', depending on an
    estimate of white might be the best storage format for the given `matrix`.

    >>> m = scipy.sparse.random(100, 100, 0.01)
    >>> choose_sparsity_model(m)
    'indexed'

    >>> m = scipy.sparse.random(100, 100, 0.5)
    >>> choose_sparsity_model(m)
    'full'

    >>> m = np.diag(np.ones(20))
    >>> choose_sparsity_model(m)
    'banded'

    >>> m = np.diag(np.zeros(20))
    >>> m[2,2] = m[10,10] = m[11,11] = 1
    >>> choose_sparsity_model(m)
    'indexed'

    >>> td_data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> off = np.array([0, -1, 1])
    >>> m = scipy.sparse.dia_matrix((td_data, off), shape=(5,5)).todense()
    >>> choose_sparsity_model(m) # should eventually be 'dia'
    'indexed'

    >>> m = scipy.sparse.dia_matrix((td_data, off), shape=(20,20)).todense()
    >>> m[19,19] = 1
    >>> choose_sparsity_model(m)
    'indexed'
    """
    if repr(matrix).startswith('Quantum object'):  # qutip.Qobj
        matrix = matrix.data
    n, m = matrix.shape
    assert n == m
    size = n * m
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            dia_matrix = scipy.sparse.dia_matrix(matrix)
            if len(dia_matrix.data) == 1: # diagonal matrix
                nnz = (dia_matrix.data != 0).sum()
                if nnz < n / 4:
                    return 'indexed'
                else:
                    return 'banded'
            elif len(dia_matrix.data) <= 5:
                nnz = (dia_matrix.data != 0).sum()
                if nnz < dia_matrix.data.size / 4:
                    return 'indexed'
                else:
                    #return 'dia' # 'dia' is not yet fully implemented in QDYN
                    return 'indexed'
        except scipy.sparse.SparseEfficiencyWarning:
            pass # continue on to coo_matrix
    coo_matrix = scipy.sparse.coo_matrix(matrix)
    if coo_matrix.nnz <= size / 10:
        return 'indexed'
    else:
        return 'full'



def reg_diff(data, itern, alph, u0=None, ep=1e-6, dx=None):
    """Perform regularized numerical differentiation for a large data array,
    based on the method outlined in Rick Chartrand, "Numerical differentiation
    of noisy, nonsmooth data, ISRN Applied Mathematics, Vol. 2011, Article ID
    164564, 2011. We use the variant of the algorithm for large `data` arrays.

    Args:
        data (numpy array): Vector of data to be differentiated.
        itern (int): Number of iterations to run the main loop.  A stopping
            condition based on the norm of the gradient vector g below would be
            an easy modification.  No default value.
        alph (float): Regularization parameter.  This is the main parameter to
            fiddle with.  Start by varying by orders of magnitude until
            reasonable results are obtained.  A value to the nearest power of
            10 is usually adequate.  No default value.  Higher values increase
            regularization strength and improve conditioning.
        u0 (numpy array): Initialization of the iteration.  Default value is
            the naive derivative (without scaling), of appropriate length (this
            being different for the two methods).  Although the solution is
            theoretically independent of the initialization, a poor choice can
            exacerbate conditioning issues when the linear system is solved.
        ep (float): Parameter for avoiding division by zero.  Default value is
            1e-6.  Results should not be very sensitive to the value.  Larger
            values improve conditioning and therefore speed, while smaller
            values give more accurate results with sharper jumps.
        dx (float): Grid spacing, used in the definition of the derivative
            operators.  Default is the reciprocal of the data size.

    Returns:
        u (numpy array): Estimate of the regularized derivative of data.
    """

    logger = logging.getLogger(__name__)

    # Make sure we have a column vector
    data = np.array(data)
    if len(data.shape) != 1:
        logger.error("data is not a column vector")
        return
    # Get the data size.
    n = len(data)

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Construct antidifferentiation operator and its adjoint.
    A = lambda v: np.cumsum(v)
    AT = lambda w: (sum(w) * np.ones(len(w))
                    - np.transpose(np.concatenate(([0.0], np.cumsum(w[:-1])))))
    # Construct differentiation matrix.
    c = np.ones(n)
    D = scipy.sparse.spdiags([-c, c], [0, 1], n, n) / dx
    mask = np.ones((n, n))
    mask[-1, -1] = 0.0
    D = scipy.sparse.dia_matrix(D.multiply(mask))
    DT = D.transpose()
    # Since Au(0) = 0, we need to adjust.
    data = data - data[0]
    # Default initialization is naive derivative.
    if u0 is None:
        u0 = np.concatenate(([0], np.diff(data)))
    u = u0
    # Precompute.
    ATd = AT(data)

    def linop_matvec(v):
        return alph * L * v + AT(A(v))
    linop = scipy.sparse.linalg.LinearOperator((n, n), linop_matvec)

    # Main loop.
    for ii in range(1, itern + 1):

        # Diagonal matrix of weights, for linearizing E-L equation.
        Q = scipy.sparse.spdiags(1.0/np.sqrt((D*u)**2.0+ep), 0, n, n)
        # Linearized diffusion matrix, also approximation of Hessian.
        L = DT*Q*D
        # Gradient of functional.
        g = AT(A(u)) - ATd
        g = g + alph * L * u
        # Build preconditioner.
        c = np.cumsum(range(n, 0, -1))
        B = alph * L + scipy.sparse.spdiags(c[::-1], 0, n, n)
        # droptol = 1.0e-2
        R = scipy.sparse.dia_matrix(np.linalg.cholesky(B.todense()))
        # Prepare to solve linear equation.
        tol = 1.0e-4
        maxit = 100

        [s, info_i] = scipy.sparse.linalg.cg(linop, -g, None, tol, maxit, None,
                                             np.dot(R.transpose(), R))
        logger.debug("iteration %4d: relative change = %.3e, gradient "
                     "norm = %.3e", ii,
                     np.linalg.norm(s[0])/np.linalg.norm(u),
                     np.linalg.norm(g))
        if info_i > 0:
            logger.debug("WARNING - convergence to tolerance not achieved!")
        elif info_i < 0:
            logger.debug("WARNING - illegal input or breakdown")
        else:
            [s, info_i] = scipy.sparse.linalg.cg(linop, -g, None, tol, maxit,
                                                None, np.dot(R.transpose(), R))
        # Update current solution
        u = u + s
        u = u/dx

    return u
