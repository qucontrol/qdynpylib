#!/usr/bin/env python
"""Linear algebra helper routines"""
import warnings

import numpy as np
import scipy.linalg
import scipy.sparse


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
    assert type(v1) is type(v2), (
        "v1 and v2 must be of the same type: types are %s vs %s"
        % (type(v1), type(v2))
    )
    # numpy matrices are sub-types of ndarray
    assert isinstance(
        v1, np.ndarray
    ), "v1, v2 must be numpy matrices, or 1D/2D numpy arrays"
    if hasattr(np, 'matrix') and isinstance(v1, np.matrix):
        # np.matrix might not exist in future versions of numpy
        return trace(np.dot(v1.H, v2))
    else:
        assert len(v1.shape) <= 2, "v1, v2 must be matrix or vector"
        if len(v1.shape) == 1:  # vector
            return np.vdot(v1, v2)
        else:  # matrix as 2D array
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
    if repr(v).startswith('Quantum object'):  # qutip.Qobj
        v = v.data
    if isinstance(v, scipy.sparse.spmatrix):
        return scipy.sparse.linalg.norm(v)
    else:
        return scipy.linalg.norm(v)


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
    if repr(a).startswith('Quantum object'):  # qutip.Qobj
        a = a.data.todense()
    N = a.size
    return np.squeeze(np.asarray(a).reshape((1, N), order=order))


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
    if hasattr(matrix, 'isherm'):  # qutip.Qobj
        return matrix.isherm
    else:  # any numpy matrix/array or scipy sparse matrix)
        # pylint: disable=simplifiable-if-statement
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
    # This is a workaround for numpy bug #7924. It also works for qutip objects
    try:
        dtype = x.dtype
    except AttributeError:
        try:
            # qutip.Qobj
            dtype = x.data.dtype
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
            if len(dia_matrix.data) == 1:  # diagonal matrix
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
                    # return 'dia' # 'dia' is not yet fully implemented in QDYN
                    return 'indexed'
        except scipy.sparse.SparseEfficiencyWarning:
            pass  # continue on to coo_matrix
    coo_matrix = scipy.sparse.coo_matrix(matrix)
    if coo_matrix.nnz <= size / 10:
        return 'indexed'
    else:
        return 'full'


def triu(matrix):
    """Return the upper triangle of the given `matrix`, which can be a numpy
    object or scipy sparse matrix. The returned matrix will have the same type
    as the input `matrix`. The input `matrix` can also be a QuTiP operator,
    but in this case, the type is *not* preserved: the result is equivalent to
    ``triu(matrix.data)``"""
    if repr(matrix).startswith('Quantum object'):  # qutip.Qobj
        matrix = matrix.data
    if isinstance(matrix, np.ndarray):
        return np.triu(matrix)
    elif isinstance(matrix, scipy.sparse.spmatrix):
        return scipy.sparse.triu(matrix)
    else:
        raise TypeError(
            "matrix must be numpy object, sparse matrix, or " "QuTiP operator"
        )


def tril(matrix):
    """Like `triu`, but return the lower triangle"""
    if repr(matrix).startswith('Quantum object'):  # qutip.Qobj
        matrix = matrix.data
    if isinstance(matrix, np.ndarray):
        return np.tril(matrix)
    elif isinstance(matrix, scipy.sparse.spmatrix):
        return scipy.sparse.tril(matrix)
    else:
        raise TypeError(
            "matrix must be numpy object, sparse matrix, or " "QuTiP operator"
        )


def banded_to_full(banded, n, kl, ku, mode):
    """Convert a rectangular matrix in the Lapack "banded" format
    (http://www.netlib.org/lapack/lug/node124.html) into a (square) full matrix

    Args:
        banded (numpy array): Rectangular matrix in banded format
        n (int): The dimension of the (full) matrix
        kl (int):  The number of lower diagonals (kl=0 for
            diagonal/upper-triangular matrix)
        ku (int):  The number of upper diagonals (ku=0 for
            diagonal/lower-triangular matrix)
        mode (str): On of 'g', 'h', 's', 't' corresponding to "general",
        "Hermitian", "symmetric", and 'triangular'. The values 'g' and 's' are
        dequivalent, except that for 's' iether kl or ku must be zero. For
        Hermitian or symmetric storage, exactly one of `kl`, `ku` must be zero.
        Which one determines whether `banded` is assumed to contain the data
        for the upper or lower triangle

    Returns:
        full: numpy array of same type as `banded`
    """
    full = np.zeros(shape=(n, n), dtype=banded.dtype)
    if mode in ['g', 't']:
        if mode == 't':
            assert kl == 0 or ku == 0
        for j in range(n):
            for i in range(max(0, j - ku), min(n, j + kl + 1)):
                full[i, j] = banded[ku + i - j, j]
    else:  # Hermitian or symmetric
        if kl == 0:  # upper triangle
            kd = ku
            for j in range(n):
                for i in range(max(0, j - kd), j + 1):
                    full[i, j] = banded[kd + i - j, j]
                    if i != j:
                        if mode == 'h':
                            full[j, i] = banded[kd + i - j, j].conjugate()
                        elif mode == 's':
                            full[j, i] = banded[kd + i - j, j]
                        else:
                            raise ValueError("Invalid mode %s" % mode)
        elif ku == 0:  # lower triangle
            kd = kl
            for j in range(n):
                for i in range(j, min(n, j + kd + 1)):
                    full[i, j] = banded[i - j, j]
                    if i != j:
                        if mode == 'h':
                            full[j, i] = banded[i - j, j].conjugate()
                        elif mode == 's':
                            full[j, i] = banded[i - j, j]
                        else:
                            raise ValueError("Invalid mode %s" % mode)
        else:
            raise ValueError(
                "For mode %s, either kl or ku must be zero" % mode
            )
    return full
