"""
Module containing routines for reading and writing files compatible with QDYN
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
import sys
import re
import os
import scipy.sparse
from click.utils import open_file
# import for doctests
from contextlib import contextmanager
import tempfile
from six.moves import xrange


@contextmanager
def tempinput(data, binary=False):
    """Context manager providing a temporary filename for a file containing the
    given data. If binary is True, the data will be written as-is, and must be
    suitable for writing in binary mode. Otherwise, if encoding the given data
    to utf-8 is at all possible, the temporary file will be
    a text file with utf-8 encoding. The file is deleted on leaving the
    context.

    >>> test_str = '''
    ... In the world of the very small, where particle and wave
    ... aspects of reality are equally significant, things do not
    ... behave in any way that we can understand from our experience
    ... of the everyday world...all pictures are false, and there is
    ... no physical analogy we can make to understand what goes on
    ... inside atoms. Atoms behave like atoms, nothing else.'''
    >>> with tempinput(test_str) as filename:
    ...     with open_file(filename) as in_fh:
    ...         for line in in_fh:
    ...             print(line.strip())
    <BLANKLINE>
    In the world of the very small, where particle and wave
    aspects of reality are equally significant, things do not
    behave in any way that we can understand from our experience
    of the everyday world...all pictures are false, and there is
    no physical analogy we can make to understand what goes on
    inside atoms. Atoms behave like atoms, nothing else.
    """
    # see http://stackoverflow.com/questions/11892623/python-stringio-and-compatibility-with-with-statement-context-manager
    temp = tempfile.NamedTemporaryFile(delete=False)
    if not binary:
        try:
            # Python 3 str data can be encoded, as well as Python 2 unicode
            # data
            data = data.encode('utf-8')
        except (AttributeError, UnicodeDecodeError):
            # Python 3 bytes data is already encoded and will raise an
            # AttributeError; standard Python 2 str data
            # tends to raise UnicodeDecodeError for non-ascii strings. Consider
            # importing unicode_literals from __future__ to fix this
            pass
    temp.write(data)
    temp.close()
    yield temp.name
    os.unlink(temp.name)


def write_indexed_matrix(matrix, filename, comment=None, line_formatter=None,
header=None, hermitian=True):
    """
    Write the given matrix to file in indexed format (1-based indexing)

    Arguments
    ---------

    matrix: numpy matrix, 2D ndarray, or any scipy sparse matrix
        Matrix to write to file

    filename: str
        Name of file to write to

    comment: str of array of strings, optional
        Comment line, or array of comment lines to write to the top of the
        file. Each line that does not start with '#' will have "# "
        prepended.

    line_formatter: callable, optional
        Function that takes three arguments i, j, v (row index, column index,
        and complex value matrix[i,j]) and returns a line to be written to
        file. If the function returns None for any input data, no line will be
        written to file. If not given, defaults to

            lambda i, j, v: "%8d%8d%25.16E" % (i, j, v.real)

        if matrix is real and

            lambda i, j, v:  "%8d%8d%25.16E%25.16E" % (i, j, v.real, v.imag)

        if matrix is complex.

    header: str, optional
        Header line to be written before any data. Must start with either '#'
        or a space, in which case the leading space will be replaced with '#'.
        Defaults to a header line suitable for the default line_formatter

    hermitian: boolean, optional
        If True, write only entries from the upper triangle
    """

    # set line formatter
    def real_formatter(i, j, v):
        return "%8d%8d%25.16E" % (i, j, v.real)
    def complex_formatter(i, j, v):
        return "%8d%8d%25.16E%25.16E" % (i, j, v.real, v.imag)
    if line_formatter is None:
        if np.iscomplexobj(matrix):
            line_formatter = complex_formatter
        else:
            line_formatter = real_formatter

    # set header
    if header is None:
        if np.iscomplexobj(matrix):
            header = "# %6s%8s%25s%25s\n" \
                     % ('row', 'column', 'Re(val)', 'Im(val)')
        else:
            header = "# %6s%8s%25s\n" % ('row', 'column', 'Re(val)')
    else:
        if not header.startswith("#"):
            if header.startswith(" "):
                header = "#" + header[1:]
            else:
                header = "#" + header

    with open_file(filename, 'w') as out_fh:

        # write comment(s)
        if comment is not None:
            if isinstance(comment, (list, tuple)):
                comment = "\n".join(comment)
            if len(comment) > 0:
                for line in comment.split("\n"):
                    if not line.startswith("#"):
                        line = "# " + line
                    out_fh.write(line + "\n")

        # write header
        out_fh.write(header)

        # write data
        sparse_h = scipy.sparse.coo_matrix(matrix)
        for i_val in xrange(sparse_h.nnz):
            i = sparse_h.row[i_val] + 1 # 1-based indexing
            j = sparse_h.col[i_val] + 1
            v = sparse_h.data[i_val]
            if (not hermitian) or (j >= i):
                line = line_formatter(i, j, v)
                if line is not None:
                    out_fh.write(line)
                    if not line.endswith("\n"):
                        out_fh.write("\n")


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
    out=None):
    """
    Print a numpy complex matrix to screen, or to a file if outfile is given.
    Values below the given limit are printed as zero

    Arguments
    ---------

    M: numpy matrix, 2D ndarray
        Matrix to print. In addition to a standard dense matrix, may also be
        any scipy sparse matrix in a format where M[i,j] is defined.
    matrix_name: str, optional
        Name of matrix
    limit: float, optional
       Any number (real or imaginary part) whose absolute value is smaller than
       this limit will be printed as 0.0.
    fmt: str, optional
        Format of each entry (both for real and imaginary part)
    out: open filehandle. If None, print to stdout

    Examples
    --------

    >>> import numpy as np
    >>> M = np.matrix([[1.0, 2.0, 0.0], [-1.0j, 2.0, 1.0e-20],
    ... [1+1j, 1.0e-9, -1.0]])

    >>> print_matrix(M)
    { 1.00E+00,      0.0}( 2.00E+00,      0.0)(        0,        0)
    (      0.0,-1.00E+00){ 2.00E+00,      0.0}(      0.0,      0.0)
    ( 1.00E+00, 1.00E+00)( 1.00E-09,      0.0){-1.00E+00,      0.0}

    >>> print_matrix(M, limit=1.0e-5)
    { 1.00E+00,      0.0}( 2.00E+00,      0.0)(        0,        0)
    (      0.0,-1.00E+00){ 2.00E+00,      0.0}(      0.0,      0.0)
    ( 1.00E+00, 1.00E+00)(      0.0,      0.0){-1.00E+00,      0.0}

    >>> M[2,1] = 1.0
    >>> print_matrix(M, fmt="%5.1f")
    {  1.0,  0.0}(  2.0,  0.0)(    0,    0)
    (  0.0, -1.0){  2.0,  0.0}(  0.0,  0.0)
    (  1.0,  1.0)(  1.0,  0.0){ -1.0,  0.0}

    >>> print_matrix(M, matrix_name="M", fmt="%5.1f")
    M = [
    {  1.0,  0.0}(  2.0,  0.0)(    0,    0)
    (  0.0, -1.0){  2.0,  0.0}(  0.0,  0.0)
    (  1.0,  1.0)(  1.0,  0.0){ -1.0,  0.0}
    ]

    >>> import scipy.sparse
    >>> print_matrix(scipy.sparse.csr_matrix(M), matrix_name="M", fmt="%5.1f")
    M = [
    {  1.0,  0.0}(  2.0,  0.0)(    0,    0)
    (  0.0, -1.0){  2.0,  0.0}(  0.0,  0.0)
    (  1.0,  1.0)(  1.0,  0.0){ -1.0,  0.0}
    ]
    """
    m, n = M.shape
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
    if out is None:
        out = sys.stdout
    if matrix_name is not None:
        out.write("%s = [\n" % matrix_name)
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
        out.write("]\n")


def fix_fortran_exponent(num_str):
    """
    In 3-digit exponents, Fortran drops the 'E'. Return a string with the 'E'
    restored.

    >>> print(fix_fortran_exponent("1.0-100"))
    1.0E-100
    >>> print(fix_fortran_exponent("1.0E-99"))
    1.0E-99
    """
    if not 'E' in num_str:
        return re.sub('(\d)([+-]\d)', r'\1E\2', num_str)
    return num_str


def read_complex(str):
    """
    Convert a string to a complex number

    >>> read_complex("1.0 -2.0-100")
    (1-2e-100j)
    """
    real_part, imag_part = str.split()
    real_part = fix_fortran_exponent(real_part)
    imag_part = fix_fortran_exponent(imag_part)
    return float(real_part) + 1.0j*float(imag_part)


def matrix_to_latex(M):
    r"""
    Return matrix M as LaTeX Code

    >>> from . gate2q import CNOT
    >>> print(matrix_to_latex(CNOT))
    \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 0 \\
    \end{pmatrix}
    """
    lines = []
    from sympy.printing.latex import latex
    from sympy import nsimplify
    lines.append(r'\begin{pmatrix}')
    for line in np.asarray(M):
        entries = [latex(nsimplify(v)) for v in line]
        lines.append(" & ".join(entries) + r' \\')
    lines.append(r'\end{pmatrix}')
    return "\n".join(lines)


def mathematica_number(val):
    r"""
    Format a number in a way that can be pasted into Mathematica

    >>> print(mathematica_number(0.0))
    0

    >>> print(mathematica_number(1.0))
    1.0

    In [36]: print(mathematica_number(1.0j))
    1.0I

    In [37]: print(mathematica_number(1.0j+1.0))
    1.0+1.0I

    In [38]: print(mathematica_number(-1.0j+1.0))
    1.0-1.0I
    """
    result = ""
    if val == 0.0:
        return "0"
    if val.real != 0.0:
        result = str(val.real)
    if val.imag != 0.0:
        if result != "" and val.imag > 0.0:
            result += "+"
        result += str(val.imag) + "I"
    return result


def matrix_to_mathematica(M):
    r"""
    Print matrix M as a string that can be pasted into Mathematica

    >>> from . gate2q import CNOT, identity
    >>> print(matrix_to_mathematica(CNOT))
    {{1.0, 0, 0, 0}, {0, 1.0, 0, 0}, {0, 0, 0, 1.0}, {0, 0, 1.0, 0}}
    >>> print(matrix_to_mathematica(1.0j*CNOT+identity))
    {{1.0+1.0I, 0, 0, 0}, {0, 1.0+1.0I, 0, 0}, {0, 0, 1.0, 1.0I}, {0, 0, 1.0I, 1.0}}
    """
    lines = []
    for row in np.asarray(M):
        entries = [mathematica_number(v) for v in row]
        lines.append('{' + ", ".join(entries) + '}')
    return '{' + ", ".join(lines) + '}'
