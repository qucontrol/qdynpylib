"""
Module containing routines for reading and writing files compatible with QDYN
"""
import numpy as np
import re
import os.path
import sys
import re
from StringIO import StringIO


def read_U(file, must_be_file=True):
    """ Read in complex 4x4 matrix from file (as written by the QDYN
        write_cmplx_matrix routine). As an alternative to giving the file, you
        may give the contents of the file as a multiline string.

        Return a 4x4 double precision complex Numpy matrix

        Assumes the propagation is in the canonical basis
    """
    U = np.zeros(shape=(4,4), dtype=np.complex128)
    if os.path.isfile(file):
        fh = open(file)
    else:
        # if file is not actually a file, we assume that it's the contents of
        # a file a a string
        if must_be_file:
            raise IOError("%s does not exits" % file)
        else:
            fh = StringIO(file)
    try:
        i = 0
        for line in fh:
            items = re.split("[(){}]+", line.strip())[1:-1]
            if len(items) != 4: continue
            j = 0
            for item in items:
                if "," in item:
                    x, y = item.split(",")
                    z = complex(float(x), float(y))
                elif item.strip() == '0':
                    z = complex(0.0, 0.0)
                U[i,j] = z
                j += 1
            i += 1
    finally:
        fh.close()
    return np.matrix(U)


def print_2q_gate(U):
    """
    Print a complex 4x4 matrix to the screen
    """
    for i in xrange(4):
        row_str = "%5f+%5fj  %5f+%5fj  %5f+%5fj  %5f+%5fj " % (
                  U[i,0].real, U[i,0].imag,
                  U[i,1].real, U[i,1].imag,
                  U[i,2].real, U[i,2].imag,
                  U[i,3].real, U[i,3].imag)
        print row_str


def read_indexed_matrix(filename, format='coo', shape=None,
expand_hermitian=True, val_real=False):
    """
    Read in a matrix from the file with the given filename

    The file must contain a description in indexed format, like this:

        # row  col  re(val) im(val)
            0    1  1.0     0.0
            1    0  0.0     1.0

    The fourth column is optional, if not present, the result will be real.

    Return a matrix in any of the numpy/scipy sparse (or non-sparse) formats.
    See the documentation of scipy.sparse for information about the different
    sparse formats

    Arguments
    ---------

    filename: string
        Name of file from which to read the matrix
    format: string, optional
        Result type:
        * 'coo' (default): scipy.sparse.coo.coo_matrix
        * 'array': numpy.ndarray
        * 'dense': numpy.matrixlib.defmatrix.matrix
        * 'bsr': scipy.sparse.bsr.bsr_matrix
        * 'csc': scipy.sparse.csc.csc_matrix
        * 'csr': scipy.sparse.csr.csr_matrix
        * 'dia': scipy.sparse.dia.dia_matrix
        * 'dok': scipy.sparse.dok.dok_matrix
        * 'lil': scipy.sparse.lil.lil_matrix
    shape: int or sequence of two ints, optional
        If given, shape of the resulting matrix. If not given, will be
        determined from largest occurring index in the data from the input file
    expand_hermitian: boolean, optional
        By default, the matrix to be read in is assumed to be Hermitian, and
        the input file must only contain data for the upper or lower triangle
        of the Matrix. The other triangle is filled automatically with the
        complex conjugate values. With `expand_hermitian=False`, the input file
        must contain *all* entries of the matrix.
    val_real: boolean, optional
        If True, only read 3 columns from the input file (i, j, value), even if
        more columns are present in the file, and return a real matrix.
    """
    from scipy.sparse import coo_matrix
    file_row, file_col \
    = np.genfromtxt(filename, usecols=(0,1), unpack=True, dtype=np.int)
    file_real_val \
    = np.genfromtxt(filename, usecols=(2,), unpack=True, dtype=np.float64)
    val_is_real = False
    if not val_real:
        try:
            file_imag_val = np.genfromtxt(filename, usecols=(3,), unpack=True,
                                          dtype=np.float64)
        except ValueError:
            # File does not contain a fourth column
            val_is_real = True
    # check data consistency, count number of non-zero elements (nnz)
    nnz = 0
    upper = None # all vals in upper triangle (True) or lower triangle (False)?
    for k in xrange(len(file_real_val)):
        i = file_row[k]
        j = file_col[k]
        assert i > 0, "Row-indices in file must be one-based"
        assert j > 0, "Column-indices in file must be one-based"
        if i == j:
            nnz += 1
        else:
            if expand_hermitian:
                if upper is None:
                    upper = (j > i)
                assert (j > i) == upper, \
                "If expand_hermitian is True, file must contain data only " \
                "for the upper or only for the lower triangle"
                nnz += 2
            else:
                nnz += 1
    row = np.zeros(nnz, dtype=np.int)
    col = np.zeros(nnz, dtype=np.int)
    if val_is_real:
        val = np.zeros(nnz, dtype=np.float64)
    else:
        val = np.zeros(nnz, dtype=np.complex128)
    l = 0
    for k in xrange(len(file_real_val)):
        i = file_row[k] - 1 # adjust for zero-based indexing in Python
        j = file_col[k] - 1
        v = file_real_val[k]
        if not val_is_real:
            v += 1.0j * file_imag_val[k]
        row[l] = i
        col[l] = j
        val[l] = v
        l += 1
        if (i != j) and (expand_hermitian):
            row[l] = j
            col[l] = i
            val[l] = v.conjugate()
            l += 1
    m = coo_matrix((val, (row, col)), shape=shape)
    if format == 'coo':
        return m
    else:
        return getattr(m, 'to'+format)() # e.g. format='dense' -> m.todense()


def print_matrix(M, matrix_name=None, limit=1.0e-14, fmt="%9.2E",
    outfile=None):
    """
    Print a numpy complex matrix to screen. Values below the given limit
    are printed as zero
    """
    m, n = M.shape
    if outfile is not None:
        out = open(outfile, 'w')
    else:
        out = sys.stdout
    fmt_rx = re.compile(r'%[#0i +-]?(?P<width>\d+)\.\d+[hlL]?[diouxXeEfFgG]')
    fmt_m = fmt_rx.match(fmt)
    width = 9
    if fmt_m:
        width = int(fmt_m.group('width'))
        zero_fmt   = '%' + ("%dd" % width)
        zero_fmt   = "%s,%s" % (zero_fmt, zero_fmt)
        zero = zero_fmt % (0,0)
        small_fmt = '%' + ("%d.1f" % width)
        small = small_fmt % 0
    else:
        raise ValueError("fmt must match '%[conversion flags]w.d<type>'")
    try:
        if matrix_name is not None:
            print >> out, "%s = [" % matrix_name
        for i in xrange(m):
            for j in xrange(n):
                if M[i,j] == 0.0:
                    entry = zero
                else:
                    x = M[i,j].real
                    if abs(x) < limit:
                        x = 0.0
                    y = M[i,j].imag
                    if abs(y) < limit:
                        y = 0.0
                    if x == 0.0:
                        entry = small
                    else:
                        entry = fmt % x
                    entry += ","
                    if y == 0.0:
                        entry += small
                    else:
                        entry += fmt % y
                if i == j:
                    out.write("{" + entry + "}")
                else:
                    out.write("(" + entry + ")")
            out.write("\n")
        if matrix_name is not None:
            print >> out, "]"
    finally:
        if outfile is not None:
            out.close()


def fix_fortran_exponent(num_str):
    """
    In 3-digit exponents, Fortran drops the 'E'. Return a string with the 'E'
    restored.
    """
    if not 'E' in num_str:
        return re.sub('(\d)([+-]\d)', r'\1E\2', num_str)
    return num_str


def read_complex(str):
    """
    Convert a string to a complex number
    """
    real_part, imag_part = str.split()
    real_part = fix_fortran_exponent(real_part)
    imag_part = fix_fortran_exponent(imag_part)
    return float(real_part) + 1.0j*float(imag_part)
