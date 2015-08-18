from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import tempfile
import os
import QDYN
import numpy as np
import scipy.sparse
from six.moves import xrange


def tempfilename():
    file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    filename = file.name
    file.close()
    return filename


def test_print_matrix():

    print("*** Running test_print_matrix")

    M = np.matrix([[1.0, 2.0, 0.0], [-1.0j, 2.0, 1.0e-20],
                   [1+1j, 1.0e-9, -1.0]])

    expected = [
    'M = [',
    '{ 1.00E+00,      0.0}( 2.00E+00,      0.0)(        0,        0)',
    '(      0.0,-1.00E+00){ 2.00E+00,      0.0}(      0.0,      0.0)',
    '( 1.00E+00, 1.00E+00)( 1.00E-09,      0.0){-1.00E+00,      0.0}',
    ']']

    # write to already open file
    filename = tempfilename()
    with open(filename, 'w') as out_fh:
        QDYN.io.print_matrix(M, matrix_name='M', out=out_fh)
        QDYN.io.print_matrix(M, matrix_name='M', out=out_fh)
    with open(filename) as in_fh:
        for i, line in enumerate(in_fh):
            assert line.rstrip() == expected[i%5]
    os.unlink(filename)


def identical_matrices(A, B):
    if isinstance(A, scipy.sparse.spmatrix):
        A = A.todense()
    if isinstance(B, scipy.sparse.spmatrix):
        B = B.todense()
    return QDYN.linalg.norm(A-B) < 1.0e-14

def print_file(file):
    with open(file) as in_fh:
        for line in in_fh:
            print(line, end="")

def make_hermitian(A):
    n = A.shape[0]
    for i in xrange(n):
        for j in xrange(i):
            A[i,j] = A[j,i].conjugate()
    return A


def test_read_write_indexed_matrix():

    print("Simple real sparse matrix")
    M = np.matrix([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]])
    Id = scipy.sparse.eye(3, format='coo')
    M = scipy.sparse.kron(M, Id)
    filename = tempfilename()
    QDYN.io.write_indexed_matrix(M, filename)
    print_file(filename)
    O = QDYN.io.read_indexed_matrix(filename)
    assert identical_matrices(M, O)
    os.unlink(filename)
    print("")

    print("Complex sparse matrix")
    filename = tempfilename()
    M2 = np.matrix(make_hermitian((M + 0.5j * M)).todense(),
                   dtype=np.complex128)
    QDYN.io.write_indexed_matrix(M2, filename)
    print_file(filename)
    O2 = QDYN.io.read_indexed_matrix(filename)
    assert identical_matrices(M2, O2)
    os.unlink(filename)
    print("")

    print("Complex non-Hermitian sparse matrix")
    filename = tempfilename()
    M3 = np.matrix((M + 0.5j * M).todense(),
                   dtype=np.complex128)
    QDYN.io.write_indexed_matrix(M3, filename, hermitian=False)
    print_file(filename)
    O3 = QDYN.io.read_indexed_matrix(filename, expand_hermitian=False)
    assert identical_matrices(M3, O3)
    os.unlink(filename)
    print("")

