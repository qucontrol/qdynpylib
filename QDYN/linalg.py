#!/usr/bin/env python
"""
Linear algebra helper routines
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
import scipy.linalg
from six.moves import xrange


def inner(v1, v2):
    """
    Calculate the inner product of the two vectors or matrices v1, v2

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
    assert (type(v1) == type(v2)), \
    "v1 and v2 must be of the same type: types are %s vs %s" \
    % (type(v1), type(v2))
    # numpy matrices are sub-types of ndarray
    assert isinstance(v1, np.ndarray), \
    "v1, v2 must be numpy matrices, or 1D/2D numpy arrays"
    if isinstance(v1, np.matrix):
        return trace(np.dot(v1.H, v2))
    else:
        assert (len(v1.shape) <= 2), "v1, v2 must be matrix or vector"
        if (len(v1.shape) == 1): # vector
            return np.vdot(v1, v2)
        else: # matrix as 2D array
            return trace(np.dot(v1.conjugate().transpose(), v2))


def trace(m):
    """
    Return the trace of the given matrix
    """
    return (np.asarray(m)).trace()


def norm(v):
    """
    Calculate the norm of a vector or matrix v, matching the inner product
    defined in the `inner` routine. An algorithm like
    Gram-Schmidt-Orthonormalization will only work if the choice of norm and
    inner product are compatible.

    If v is a vector, the norm is the 2-norm (i.e. the standard Euclidian
    vector norm).

    If v is a matrix, the norm is the Hilbert-Schmidt (aka Frobenius) norm.
    Note that the HS norm of a matrix is identical to the 2-norm of any
    vectorization of that matrix (e.g. writing the columns of the matrix
    underneat each other). Also, the HS norm of the m x 1 matrix is the same as
    the 2-norm of the equivalent m-dimensional vector.

    """
    # scipy.linalg.norm does the right thing in all instances.
    return scipy.linalg.norm(v)


def get_op_matrix(apply_op, state_shape, t):
    """
    Return the explicit matrix for the operator encoded in apply_op, assuming
    that apply_op takes a numpy array or matrix of the given `state_shape` as
    its first argument

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
    """
    Return vectorization of multi-dimensional numpy array or matrix a

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
    """
    Return True if matrix is Hermitian, False otherwise
    """
    n, m = matrix.shape
    for i in xrange(n):
        if (matrix[i,i].imag != 0.0):
            return False
            raise ValueError("Matrix has complex entries on diagonal")
        for j in xrange(i+1, m):
            if (abs(matrix[i,j] - matrix[j,i].conjugate()) > 1.0e-15):
                return False
    return True
