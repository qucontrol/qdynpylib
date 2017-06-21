from __future__ import print_function, division, absolute_import
import tempfile
import os
import filecmp

import numpy as np
import scipy.sparse
from six.moves import xrange

import QDYN

# buitin fixtures: request, tmpdir


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
    """Check if matrices A, B are identical up to a precision of 1.0e-14"""
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
    """For matrix A to be Hermitian"""
    n = A.shape[0]
    for i in xrange(n):
        for j in xrange(i):
            A[i,j] = A[j,i].conjugate()
    return A


def test_read_write_indexed_matrix():
    """Test reading and writing of sparse matrices"""

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


def test_single_val_read_indexed_matrix(request):
    """Test that we can read an indexed matrix with onl one entry"""
    datadir = os.path.splitext(request.module.__file__)[0]
    filename = os.path.join(datadir, 'single_val_matrix.dat')
    matrix = QDYN.io.read_indexed_matrix(filename, expand_hermitian=False)
    assert matrix.nnz == 1


def test_read_write_cmplx_array(request, tmpdir):
    """Test that we can read and write a complex array from file"""
    datadir = os.path.splitext(request.module.__file__)[0]
    infile = os.path.join(datadir, 'v0.dat')
    outfile = str(tmpdir.join('v0.dat'))
    v0 = QDYN.io.read_cmplx_array(infile)
    assert len(v0) == 100
    z = complex(-3.52976827605130314E-02, -2.08251339037964119E-02)
    assert abs(v0[1] - z) < 1e-14
    QDYN.io.write_cmplx_array(v0, outfile, fmtstr='%25.17E')
    assert filecmp.cmp(outfile, infile, shallow=False)
    QDYN.io.write_cmplx_array(v0, outfile, append=True,
                              comment="# second block")
    v0_double = QDYN.io.read_cmplx_array(outfile)
    assert len(v0_double) == 200
    with open(outfile) as fh:
        assert "# second block" in fh.read()


def test_datablock(request):
    """Test that we can extacts blocks from a file"""
    datadir = os.path.splitext(request.module.__file__)[0]
    infile = os.path.join(datadir, 'blocks.dat')
    assert len(list(QDYN.io.datablock(infile, -2))) == 0
    assert len(list(QDYN.io.datablock(infile, 0))) == 0
    assert len(list(QDYN.io.datablock(infile, 1))) == 16
    assert len(list(QDYN.io.datablock(infile, 2))) == 16
    assert len(list(QDYN.io.datablock(infile, 4))) == 16
    assert len(list(QDYN.io.datablock(infile, 5))) == 0
    assert (list(QDYN.io.datablock(infile, 1))[1] ==
            b' -1.42014783459907938E-03  2.01139166550349301E-16\n')
    assert (list(QDYN.io.datablock(infile, 1, 'ascii'))[1] ==
            ' -1.42014783459907938E-03  2.01139166550349301E-16\n')
