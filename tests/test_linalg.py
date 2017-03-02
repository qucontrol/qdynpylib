"""Test linear algebra module"""
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from QDYN.linalg import triu, tril, norm


def test_norm():
    """Test calculation of norm for matrix in varying formats"""
    A = np.diag([1, 1, 1, 1])
    assert abs(norm(A) - 2.0) < 1e-12
    assert abs(norm(np.matrix(A)) - 2.0) < 1e-12
    assert abs(norm(scipy.sparse.coo_matrix(A)) - 2.0) < 1e-12


def test_triu_tril():
    """Test obtaining upper and lower triangle of matrix"""
    A = np.array([
        [ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 16, 16]])
    A_u = np.array([
        [ 1,  2,  3,  4],
        [ 0,  6,  7,  8],
        [ 0,  0, 11, 12],
        [ 0,  0,  0, 16]])
    A_l = np.array([
        [ 1,  0,  0,  0],
        [ 5,  6,  0,  0],
        [ 9, 10, 11,  0],
        [13, 14, 16, 16]])
    assert norm(A_u - triu(A)) < 1e-12
    assert norm(A_l - tril(A)) < 1e-12
    assert norm(np.matrix(A_u) - triu(np.matrix((A)))) < 1e-12
    assert norm(np.matrix(A_l) - tril(np.matrix((A)))) < 1e-12
    assert norm(
        scipy.sparse.coo_matrix(A_u) - triu(scipy.sparse.coo_matrix((A)))
    ) < 1e-12
    assert norm(
        scipy.sparse.coo_matrix(A_l) - tril(scipy.sparse.coo_matrix((A)))
    ) < 1e-12

