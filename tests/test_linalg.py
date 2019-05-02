"""Test linear algebra module"""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from qdyn.linalg import banded_to_full, norm, tril, triu


def test_norm():
    """Test calculation of norm for matrix in varying formats"""
    A = np.diag([1, 1, 1, 1])
    assert abs(norm(A) - 2.0) < 1e-12
    assert abs(norm(scipy.sparse.coo_matrix(A)) - 2.0) < 1e-12


def test_triu_tril():
    """Test obtaining upper and lower triangle of matrix"""
    A = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 16, 16]]
    )
    A_u = np.array([[1, 2, 3, 4], [0, 6, 7, 8], [0, 0, 11, 12], [0, 0, 0, 16]])
    A_l = np.array(
        [[1, 0, 0, 0], [5, 6, 0, 0], [9, 10, 11, 0], [13, 14, 16, 16]]
    )
    assert norm(A_u - triu(A)) < 1e-12
    assert norm(A_l - tril(A)) < 1e-12
    assert (
        norm(scipy.sparse.coo_matrix(A_u) - triu(scipy.sparse.coo_matrix((A))))
        < 1e-12
    )
    assert (
        norm(scipy.sparse.coo_matrix(A_l) - tril(scipy.sparse.coo_matrix((A))))
        < 1e-12
    )


def test_banded_to_full():
    """Test conversion between banded and full matrix format"""
    A_diag = np.diag([11, 22, 33, 44, 55])
    A_diag_band = np.array([[11, 22, 33, 44, 55]])
    A = banded_to_full(A_diag_band, 5, 0, 0, 'g')
    assert norm(A - A_diag) == 0
    A = banded_to_full(A_diag_band, 5, 0, 0, 't')
    assert norm(A - A_diag) == 0
    A = banded_to_full(A_diag_band, 5, 0, 0, 'h')
    assert norm(A - A_diag) == 0
    A = banded_to_full(A_diag_band, 5, 0, 0, 's')
    assert norm(A - A_diag) == 0

    A_gen = np.array(
        [
            [11, 12, 13, 0, 0],
            [21, 22, 23, 24, 0],
            [0, 32, 33, 34, 35],
            [0, 0, 43, 44, 45],
            [0, 0, 0, 54, 55],
        ]
    )
    A_gen_band = np.array(
        [
            [0, 0, 13, 24, 35],
            [0, 12, 23, 34, 45],
            [11, 22, 33, 44, 55],
            [21, 32, 43, 54, 0],
        ]
    )
    A = banded_to_full(A_gen_band, 5, kl=1, ku=2, mode='g')
    assert norm(A - A_gen) == 0

    A_sym = np.array(
        [
            [11, 12, 13, 0, 0],
            [12, 22, 23, 24, 0],
            [13, 23, 33, 34, 35],
            [0, 24, 34, 44, 45],
            [0, 0, 35, 45, 55],
        ]
    )
    A_sym_band_u = np.array(
        [[0, 0, 13, 24, 35], [0, 12, 23, 34, 45], [11, 22, 33, 44, 55]]
    )
    A_sym_band_l = np.array(
        [[11, 22, 33, 44, 55], [12, 23, 34, 45, 0], [13, 24, 35, 0, 0]]
    )
    A = banded_to_full(A_sym_band_u, 5, kl=0, ku=2, mode='s')
    assert norm(A - A_sym) == 0
    A = banded_to_full(A_sym_band_l, 5, kl=2, ku=0, mode='s')
    assert norm(A - A_sym) == 0

    A_herm = np.array(
        [
            [11, 12j, 13j, 0, 0],
            [-12j, 22, 23j, 24j, 0],
            [-13j, -23j, 33, 34j, 35j],
            [0, -24j, -34j, 44, 45j],
            [0, 0, -35j, -45j, 55],
        ]
    )
    A_herm_band_u = np.array(
        [[0, 0, 13j, 24j, 35j], [0, 12j, 23j, 34j, 45j], [11, 22, 33, 44, 55]]
    )
    A_herm_band_l = np.array(
        [
            [11, 22, 33, 44, 55],
            [-12j, -23j, -34j, -45j, 0],
            [-13j, -24j, -35j, 0, 0],
        ]
    )
    A = banded_to_full(A_herm_band_u, 5, kl=0, ku=2, mode='h')
    assert norm(A - A_herm) < 1e-14
    A = banded_to_full(A_herm_band_l, 5, kl=2, ku=0, mode='h')
    assert norm(A - A_herm) < 1e-14

    A_triu = np.array(
        [
            [11, 12, 13, 0, 0],
            [0, 22, 23, 24, 0],
            [0, 0, 33, 34, 35],
            [0, 0, 0, 44, 45],
            [0, 0, 0, 0, 55],
        ]
    )
    A_triu_band = A_sym_band_u
    A = banded_to_full(A_triu_band, 5, kl=0, ku=2, mode='t')
    assert norm(A - A_triu) == 0
    A = banded_to_full(A_triu_band, 5, kl=0, ku=2, mode='g')
    assert norm(A - A_triu) == 0

    A_tril = np.array(
        [
            [11, 0, 0, 0, 0],
            [12, 22, 0, 0, 0],
            [13, 23, 33, 0, 0],
            [0, 24, 34, 44, 0],
            [0, 0, 35, 45, 55],
        ]
    )
    A_tril_band = A_sym_band_l
    A = banded_to_full(A_tril_band, 5, kl=2, ku=0, mode='t')
    assert norm(A - A_tril) == 0
    A = banded_to_full(A_tril_band, 5, kl=2, ku=0, mode='g')
    assert norm(A - A_tril) == 0
