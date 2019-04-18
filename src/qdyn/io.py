"""Routines for reading and writing files compatible with QDYN"""
from __future__ import absolute_import, division, print_function

import io
import logging
import os
import re
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import scipy.sparse

from .linalg import iscomplexobj, norm


@contextmanager
def open_file(file, mode='r', **kwargs):
    """Wrapper around :func:`io.open`, allowing `file` to be a file handle.

    Any `kwargs` are passed to :func:`io.open`. The `file` parameter may also
    be given as the string '-', which indicates :obj:`sys.stdin` for
    ``mode='r'`` and :obj:`sys.stdout` for ``mode='w'``. If `file` is an open
    file handle, its `mode` and `encoding` are checked against the arguments;
    the file handle will not be closed one exit.

    In Python 2, if no `kwargs` beyond `mode` are given and `file` is a string
    giving a pathname or a file descriptor, the built-in :func:`open` function
    is used instead of :func:`io.open`. As a consequence, regular (unencoded)
    strings can be read or written directly from or to the resulting file
    handle, without having to open it in binary mode. This behavior provides
    maximal compatibility between Python 2 and 3.

    This function must be used as a context manager::

    >>> with open_file('-', 'w') as out_fh:
    ...      written_bytes = out_fh.write("Hello World")
    Hello World

    Raises:
        IOError: If the file cannot be opened (see :func:`io.open`), if `file`
        is open file handle with the incorrect `mode` or `encoding`, or if
        `file` is '-' and `mode` is neither 'r' nor 'w'.
    """
    fh_attribs = ('seek', 'close', 'read', 'write')
    is_fh = all(hasattr(file, attr) for attr in fh_attribs)
    if is_fh:
        if hasattr(file, 'mode'):  # e.g. sys.stdout doesn't have mode attr
            if file.mode != mode:
                raise IOError(
                    "File handle %s aleady open in mode %s; cannot open in "
                    "mode %s" % (file, file.mode, mode)
                )
        if 'encoding' in kwargs and hasattr(file, 'encoding'):
            if kwargs['encoding'] != file.encoding:
                raise IOError(
                    "Open file handle %s has enconding %s; cannot open with "
                    "encoding %s" % (file, file.encoding, kwargs['encoding'])
                )
        yield file
    else:  # str or descriptor or path-like object
        if file == '-':
            if mode == 'r':
                yield sys.stdin
            elif mode == 'w':
                yield sys.stdout
            else:
                raise IOError(
                    "File '-' can only be opened in 'r' mode (stdin) or "
                    "'w' mode (stdout)"
                )
        else:
            if len(kwargs) > 0:
                with io.open(file, mode, **kwargs) as fh:
                    yield fh
            else:
                # on Python 3, `io.open` and `open` are identical
                with open(file, mode, **kwargs) as fh:
                    yield fh


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
    # see http://stackoverflow.com/questions/11892623
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


def write_indexed_matrix(
    matrix,
    filename,
    comment=None,
    line_formatter=None,
    header=None,
    hermitian=False,
    limit=0.0,
):
    """
    Write the given matrix to file in indexed format (1-based indexing)

    Arguments
    ---------

    matrix: numpy matrix, 2D ndarray, qutip.Qobj, or any scipy sparse matrix
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

    limit: float, optional
        Only values with an absolute value greater than `limit` are written
    """

    # set line formatter

    def real_formatter(i, j, v):
        return "%8d%8d%25.16E" % (i, j, v.real)

    def complex_formatter(i, j, v):
        return "%8d%8d%25.16E%25.16E" % (i, j, v.real, v.imag)

    if repr(matrix).startswith('Quantum object'):
        # handle qutip Qobj (without importing the qutip package)
        matrix = matrix.data
    if iscomplexobj(matrix):
        is_real = False
        if line_formatter is None:
            line_formatter = complex_formatter
    else:
        is_real = True
        if line_formatter is None:
            line_formatter = real_formatter

    # set header
    if header is None:
        if iscomplexobj(matrix):
            header = "# %6s%8s%25s%25s\n" % (
                'row',
                'column',
                'Re(val)',
                'Im(val)',
            )
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
        for i_val in range(sparse_h.nnz):
            i = sparse_h.row[i_val] + 1  # 1-based indexing
            j = sparse_h.col[i_val] + 1
            v = sparse_h.data[i_val]
            if (not hermitian) or (j >= i):
                if abs(v) > limit:
                    if is_real:
                        assert v.imag == 0
                    line = line_formatter(i, j, v)
                    if line is not None:
                        out_fh.write(line)
                        if not line.endswith("\n"):
                            out_fh.write("\n")


def read_indexed_matrix(
    filename, format='coo', shape=None, expand_hermitian=False, val_real=False
):
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

    filename: str
        Name of file from which to read the matrix
    format: str, optional
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
        If True, the input file must contain data only for the upper or lower
        triangle. The oterh triangle will be set with the complex conjugate
        values.
    val_real: boolean, optional
        If True, only read 3 columns from the input file (i, j, value), even if
        more columns are present in the file, and return a real matrix.
    """
    from scipy.sparse import coo_matrix

    file_row, file_col = np.genfromtxt(
        filename, usecols=(0, 1), unpack=True, dtype=np.int
    )
    file_real_val = np.genfromtxt(
        filename, usecols=(2,), unpack=True, dtype=np.float64
    )
    # numpy doesn't generate arrays if there is only one value -- force it!
    file_row = file_row.reshape(file_row.size)
    file_col = file_col.reshape(file_col.size)
    file_real_val = file_real_val.reshape(file_real_val.size)
    val_is_real = False
    if val_real:
        val_is_real = True
    else:
        try:
            file_imag_val = np.genfromtxt(
                filename, usecols=(3,), unpack=True, dtype=np.float64
            )
            file_imag_val = file_imag_val.reshape(file_imag_val.size)
        except ValueError:
            # File does not contain a fourth column
            val_is_real = True
    # check data consistency, count number of non-zero elements (nnz)
    nnz = 0
    upper = (
        None
    )  # all vals in upper triangle (True) or lower triangle (False)?
    for k in range(len(file_real_val)):
        i = file_row[k]
        j = file_col[k]
        assert i > 0, "Row-indices in file must be one-based"
        assert j > 0, "Column-indices in file must be one-based"
        if i == j:
            nnz += 1
        else:
            if expand_hermitian:
                if upper is None:
                    upper = j > i
                assert (j > i) == upper, (
                    "If expand_hermitian is True, file must contain data only "
                    "for the upper or only for the lower triangle"
                )
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
    for k in range(len(file_real_val)):
        i = file_row[k] - 1  # adjust for zero-based indexing in Python
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
        return getattr(
            m, 'to' + format
        )()  # e.g. format='dense' -> m.todense()


def _compress_str(s, spaces_to_drop):
    """Remove `spaces_to_drop` spaces from `s`, alternating between left and
    right"""
    assert s.count(" ") >= spaces_to_drop
    from_left = True
    l = 0
    r = len(s)
    drop = set()
    remaining_spaces = spaces_to_drop
    while remaining_spaces > 0:
        if from_left:
            l = s.find(" ", l)
            drop.add(l)
            l += 1  # since `s.find` is inclusive, but we need exclusive
        else:
            r = s.rfind(" ", 0, r)
            drop.add(r)
        from_left = not from_left
        remaining_spaces -= 1
    assert len(drop) == spaces_to_drop
    return ''.join([l for (i, l) in enumerate(s) if i not in drop])


def print_matrix(
    M,
    matrix_name=None,
    limit=1.0e-14,
    fmt="%9.2E",
    compress=False,
    zero_as_blank=False,
    out=None,
):
    """Print a numpy complex matrix to screen, or to a file if outfile is given.
    Values below the given limit are printed as zero

    Arguments
    ---------

    M: numpy matrix, 2D ndarray, sparse matrix
        Matrix to print. In addition to a standard dense matrix, may also be
        any scipy sparse matrix in a format where M[i,j] is defined.
    matrix_name: str, optional
        Name of matrix
    limit: float, optional
       Any number (real or imaginary part) whose absolute value is smaller than
       this limit will be printed as 0.0.
    fmt: str or callable, optional
        Format of each entry (both for real and imaginary part). If str, must
        be an "old-style" format string the formats a single floating value. If
        a callable, the callable must return a string of fixed length when
        passed a number. The string returned by ``fmt(0)`` will be used for
        values that are exactly zero, whereas the string returned by
        ``fmt(0.0)`` will be used for values that are below `limit`.
    compress: bool, optional
        If True, remove spaces to compress the output to be narrower. Real and
        imaginary parts will no longer be aligned.
    zero_as_blank: bool, optional
        If True, represent entries that are exactly zero as blank strings
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

    >>> print_matrix(M, compress=True)
    {1.00E+00,     0.0}(2.00E+00,     0.0)(       0,       0)
    (    0.0,-1.00E+00){2.00E+00,     0.0}(     0.0,     0.0)
    (1.00E+00,1.00E+00)(1.00E-09,     0.0){-1.00E+00,    0.0}

    >>> print_matrix(M, compress=True, zero_as_blank=True)
    {1.00E+00,     0.0}(2.00E+00,     0.0)(                 )
    (    0.0,-1.00E+00){2.00E+00,     0.0}(     0.0,     0.0)
    (1.00E+00,1.00E+00)(1.00E-09,     0.0){-1.00E+00,    0.0}

    >>> M[2,1] = 1.0
    >>> print_matrix(M, fmt="%5.1f")
    {  1.0,  0.0}(  2.0,  0.0)(    0,    0)
    (  0.0, -1.0){  2.0,  0.0}(  0.0,  0.0)
    (  1.0,  1.0)(  1.0,  0.0){ -1.0,  0.0}

    >>> def compact_exp_fmt(x):
    ...     if x == 0:
    ...         return '%7d' % 0
    ...     else:  # single-digit exponent
    ...         s = "%8.1e" % x
    ...         base, exp = s.split('e')
    ...         return base + 'e%+d' % int(exp)
    >>> print_matrix(M, compress=True, zero_as_blank=True, fmt=compact_exp_fmt)
    {1.0e+0,     0}(2.0e+0,     0)(             )
    (    0,-1.0e+0){2.0e+0,     0}(     0,     0)
    (1.0e+0,1.0e+0)(1.0e+0,     0){-1.0e+0,    0}

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
    if callable(fmt):
        zero = "%s,%s" % (fmt(0), fmt(0))
        small = fmt(0.0)
        formatter = fmt
    else:
        fmt_rx = re.compile(
            r'%[#0i +-]?(?P<width>\d+)\.\d+[hlL]?[diouxXeEfFgG]'
        )
        fmt_m = fmt_rx.match(fmt)
        width = 9
        if fmt_m:
            width = int(fmt_m.group('width'))
            zero_fmt = '%' + ("%dd" % width)
            zero_fmt = "%s,%s" % (zero_fmt, zero_fmt)
            zero = zero_fmt % (0, 0)
            small_fmt = '%' + ("%d.1f" % width)
            small = small_fmt % 0
        else:
            raise ValueError("fmt must match '%[conversion flags]w.d<type>'")
        formatter = lambda x: fmt % x
    if zero_as_blank:
        zero = " " * len(zero)
    if out is None:
        out = sys.stdout
    if matrix_name is not None:
        out.write("%s = [\n" % matrix_name)
    entries = [[]]
    for i in range(m):
        for j in range(n):
            if M[i, j] == 0.0:
                entry = zero
            else:
                x = M[i, j].real
                if abs(x) < limit:
                    x = 0.0
                y = M[i, j].imag
                if abs(y) < limit:
                    y = 0.0
                if x == 0.0:
                    entry = small
                else:
                    entry = formatter(x)
                entry += ","
                if y == 0.0:
                    entry += small
                else:
                    entry += formatter(y)
            entries[-1].append(entry)
        entries.append([])
    if compress:
        spaces_to_drop = min(
            [entry.count(' ') for row in entries for entry in row]
        )
    for i in range(m):
        for j in range(n):
            entry = entries[i][j]
            if compress:
                entry = _compress_str(entry, spaces_to_drop)
            if i == j:
                out.write("{" + entry + "}")
            else:
                out.write("(" + entry + ")")
        out.write("\n")
    if matrix_name is not None:
        out.write("]\n")


def fix_fortran_exponent(num_str):
    """In 3-digit exponents, Fortran drops the 'E'. Return a string with the
    'E' restored.

    >>> print(fix_fortran_exponent("1.0-100"))
    1.0E-100
    >>> print(fix_fortran_exponent("1.0E-99"))
    1.0E-99
    """
    if 'E' not in num_str:
        return re.sub(r'(\d)([+-]\d)', r'\1E\2', num_str)
    return num_str


def read_complex(s):
    """Convert a string to a complex number

    >>> read_complex("1.0 -2.0-100")
    (1-2e-100j)
    """
    real_part, imag_part = s.split()
    real_part = fix_fortran_exponent(real_part)
    imag_part = fix_fortran_exponent(imag_part)
    return float(real_part) + 1.0j * float(imag_part)


def read_real(s):
    """Convert a string to a real number

    This works for Fortran-formatted numbers with a missing 'E' sign

    >>> read_real("-2.0-100")
    -2e-100
    """
    try:
        return float(s)
    except ValueError:
        return float(fix_fortran_exponent(s))


def read_real(str):
    """Convert a string to a real number

    This works for Fortran-formatted numbers with a missing 'E' sign

    >>> read_real("-2.0-100")
    -2e-100
    """
    try:
        return float(str)
    except ValueError:
        return float(fix_fortran_exponent(str))


def read_cmplx_array(filename, **kwargs):
    """Read a complex array from a file. The file must contain two columns
    (real and imaginary part). This routine is equivalent to the Fortran QDYN
    ``read_cmplx_array`` routine

    Args:
        filename (file, str, pathlib.Path, list of str, generator): File,
            filename, list, or generator to read. Cf. `fname` in
            :func:`numpy.genfromtxt`.
        kwargs: All keyword arguments are passed to :func:`numpy.genfromtxt`

    Notes:
        You may use :func:`datablock` as a wrapper for `fileanme` in order to
        read a specific block from a file that contains multiple blocks
    """
    x, y = np.genfromtxt(filename, usecols=(0, 1), unpack=True, **kwargs)
    return x + 1j * y


def read_cmplx_matrix(filename):
    """Read in complex matrix from file (as written by the Fortran QDYN
    ``print_cmplx_matrix`` routine).

    Return a two-dimensional, double precision complex Numpy array

    Args:
        filename: str or file-like object from which to read gate, or file-like
            object with equivalent content
    """
    U = []
    with open_file(filename) as fh:
        for line in fh:
            items = re.split("[(){}]+", line.strip())[1:-1]
            U.append([])
            for item in items:
                if "," in item:
                    x, y = item.split(",")
                    z = complex(float(x), float(y))
                elif item.strip() == '0':
                    z = complex(0.0, 0.0)
                U[-1].append(z)
    return np.array(U, dtype=np.complex128)


def datablock(filename, block, decoding=None):
    """Iterator over the lines inside the file with the given `filename` inside
    the given `block` only. Blocks must be separated by two empty lines.

    Args:
        filename (str): Name of file from which to read
        block (int): One-based Index of the desired block. A value of less than
            1 or more than the number of block available in the file will not
            cause an error, but result in an empty iterator
        decoding (None or str): By default, the resulting lines are byte
            strings (so that :func:`datablock` can wrap `fname` in
            :func:`np.genfromtxt`). If `decoding` is given different from None,
            the resulting strings will be decoded instead.
    """
    in_block = 0
    blank = 0
    with open_file(filename, 'rb') as in_fh:
        for line in in_fh:
            if len(line.strip()) == 0:
                blank += 1
            elif not line.strip().startswith(b'#'):
                if in_block == 0 or blank >= 2:
                    in_block += 1
                    blank = 0  # reset counter
                if in_block == block:
                    if decoding is None:
                        yield line
                    else:
                        yield line.decode(decoding)
                elif in_block > block:
                    break


def write_cmplx_array(
    carray, filename, header=None, fmtstr='%25.16E', append=False, comment=None
):
    """Write a complex array to file. Equivalent to the Fortran QDYN
    `write_cmplx_array`  routine. Two columns will be written to the output
    file (real and imaginary part of `carray`)

    Args:
        filename (str): Name of file to which to write the array
        header (str or None): A header line, to be written immediately before
            the data. Should start with a '#'
        fmtstr (str or None): The format to use for reach of the two columns
        append (bool): If True, append to existing files, with a separator of
            two blank lines
        comment (str or None): A comment line, to be written at the top of the
            file (before the header). Should start with a '#'
    """
    header_lines = []
    if comment is not None:
        header_lines.append(comment)
    if header is not None:
        header_lines.append(header)
    if append:
        with open_file(filename, 'a') as out_fh:
            out_fh.write("\n\n")
            writetotxt(
                out_fh, carray, fmt=(fmtstr, fmtstr), header=header_lines
            )

    else:
        writetotxt(filename, carray, fmt=(fmtstr, fmtstr), header=header_lines)


def writetotxt(fname, *args, **kwargs):
    """Inverse function to numpy.genfromtxt and similar to `numpy.savetxt`,
    but allowing to write *multiple* numpy arrays as columns to a text file.
    Also, handle headers/footers more intelligently.

    The first argument is the filename or handler to which to write, followed
    by an arbitrary number of numpy arrays, to be written as columns in the
    file (real arrays will produce once column, complex arrays two). The
    remaining keyword arguments are passed directly to `numpy.savetxt` (with
    fixes to the header/footer lines, as described below)

    Args:
        fname (str): filename or file handle
        args (ndarray): Numpy arrays to write to fname. All arrays must have
            the same length
        fmt (str, list(str)): A single format (e.g. '%10.5f'), a sequence
            of formats, or a multi-format string, e.g.
            'Iteration %d -- %10.5f', in which case `delimiter` is ignored. For
            a complex array in `*args`, a format for the real and imaginary
            parts must be given.  Defaults to '%25.16E'.
        delimiter (str): Character separating columns. Defaults to ''
        header (str, list(str)): String that will be written at the beginning
            of the file. If sequence of strings, multiple lines will be
            written.
        footer (str, list(str)): String that will be written at the end of the
            file. If sequence of strings, multiple lines will be written
        comments (str): String that will be prepended to the each line of the
            ``header`` and ``footer`` strings, to mark them as comments.
            Defaults to '# '

    Note:

        The `header` and `footer` lines are handled more intelligently than by
        the `numpy.savetxt` routine.  First, header and footer may be an array
        of lines instead of just a (multiline) string.  Second, each line in
        the header may or may not already include the `comments` prefix. If a
        line does not include the `comments` prefix yet, but starts with a
        sufficient number of spaces, the `comments` prefix will not be
        prepended to the line in output, but will overwrite the beginning of
        the line, so as not to change the line length. E.g. giving
        `header="   time [ns]"` will result in a header line of
        `#  time [ns]` in the output, not `#    time [ns]`.

        Further explanation of the `fmt` parameter
        (``%[flag]width[.precision]specifier``):

        flags:
            ``-`` : left justify

            ``+`` : Forces to precede result with + or -.

            ``0`` : Left pad the number with zeros instead of space (see
                    width).

        width:
            Minimum number of characters to be printed. The value is not
            truncated if it has more characters.

        precision:
            - For integer specifiers (eg. ``d,i``), the minimum number of
            digits.
            - For ``e, E`` and ``f`` specifiers, the number of digits to print
            after the decimal point.
            - For ``g`` and ``G``, the maximum number of significant digits.

        specifiers (partial list):
            ``c`` : character

            ``d`` or ``i`` : signed decimal integer

            ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

            ``f`` : decimal floating point

            ``g,G`` : use the shorter of ``e,E`` or ``f``

        For more details, see `numpy.savetxt`
    """
    fmt = kwargs.get('fmt', '%25.16E')
    delimiter = kwargs.get('delimiter', '')
    header = kwargs.get('header', '')
    footer = kwargs.get('footer', '')
    comments = kwargs.get('comments', "# ")
    l_comments = len(comments)

    # open file
    own_fh = False
    if isinstance(fname, str):
        own_fh = True
        if fname.endswith('.gz'):
            import gzip

            fh = gzip.open(fname, 'wb')
        else:
            fh = open(fname, 'w')
    elif hasattr(fname, 'write'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    try:

        # write header
        if isinstance(header, (list, tuple)):
            header = "\n".join(header)
        if len(header) > 0:
            for line in header.split("\n"):
                if not line.startswith(comments):
                    if line.startswith(" " * l_comments):
                        line = comments + line[l_comments:]
                    else:
                        line = comments + line
                fh.write(line)
                fh.write("\n")

        # check input data and prepare row format
        n_cols = 0
        n_rows = 0
        row_fmt = ""
        completed_row_fmt = False
        if isinstance(fmt, (list, tuple)):
            row_fmt = delimiter.join(fmt)
            completed_row_fmt = True
        elif isinstance(fmt, str):
            if fmt.count('%') > 1:
                row_fmt = fmt
                completed_row_fmt = True
        for a in args:
            if n_rows == 0:
                n_rows = len(a)
            else:
                if n_rows != len(a):
                    raise ValueError("All arrays must be of same length")
            if np.iscomplexobj(a):
                n_cols += 2
                if not completed_row_fmt:
                    row_fmt += "%s%s%s%s" % (fmt, delimiter, fmt, delimiter)
            else:
                n_cols += 1
                if not completed_row_fmt:
                    row_fmt += "%s%s" % (fmt, delimiter)
        if row_fmt.count('%') != n_cols:
            raise ValueError(
                'fmt has wrong number of %% formats:  %s' % row_fmt
            )
        if not completed_row_fmt:
            if len(delimiter) > 0:
                row_fmt = row_fmt[: -len(delimiter)]
            completed_row_fmt = True

        # write out data
        for i_row in range(n_rows):
            row_data = []
            for a in args:
                if np.iscomplexobj(a):
                    row_data.append(a[i_row].real)
                    row_data.append(a[i_row].imag)
                else:
                    row_data.append(a[i_row])
            try:
                fh.write(row_fmt % tuple(row_data))
            except TypeError:
                raise TypeError(
                    "Cannot format row data %s with format %s"
                    % (repr(tuple(row_data)), repr(fmt))
                )
            fh.write("\n")

        # write footer
        if isinstance(footer, (list, tuple)):
            footer = "\n".join(footer)
        if len(footer) > 0:
            for line in footer.split("\n"):
                if not line.startswith(comments):
                    if line.startswith(" " * l_comments):
                        line = comments + line[l_comments:]
                    else:
                        line = comments + line
                fh.write(line)
                fh.write("\n")

    finally:
        if own_fh:
            fh.close()


def read_ascii_dump(filename, convert_boolean=True, flatten=False):
    """Read a file in the format written by the QDYN `dump_ascii_*` routines
    and return its data as a nested OrderedDict. The QDYN type of the dumped
    data structure is recorded in the `typename` attribute of the result.

    Args:
        filename (str): name of file from which to read data
        convert_boolean (bool): Convert strings 'T', 'F' to Boolean values True
            and False
        flatten (bool):  If True, numerical array data is returned flattend
            (one-dimensional). If False, it is reshaped to the shape defined in
            the dump file.
    """
    with open_file(filename) as in_fh:
        try:
            return _read_ascii_dump(
                in_fh, convert_boolean=convert_boolean, flatten=flatten
            )
        except StopIteration:
            # If the file properly ended in '# END ASCII DUMP', we should have
            # returned cleanly instead of running into the end of the file
            raise ValueError("Premature file end")


def _read_ascii_dump(data, convert_boolean=True, flatten=False, typename=None):
    """Recursive implementation of `read_ascci_dump`

    Args:
        data (iterator): An iterator of lines (e.g., an open file handle).
        convert_boolean (bool): Convert 'T', 'F' to boolean values?
        flatten (bool):  If True, numerical array data is returned flattend
            (one-dimensional). If False, it is reshaped to the shape defined in
            the dump file.
        typename (str or None): The name of the type that is being read. If
            None, extract it from the first line.
    """
    logger = logging.getLogger(__name__)
    logger.debug("ENTERING READ_ASCII_DUMP")
    result = OrderedDict([])
    rx_field = re.compile(r'^# (?P<field>\w+)$')
    rx_arraybounds = re.compile(r'^(\s+\d+){2,}$')
    rx_item = re.compile(r'^# item\s+\d+')
    rx_int = re.compile(r'^\s*(?P<val>[0-9+-]+)$')
    rx_float = re.compile(r'^\s*(?P<val>([+-]?\d+\.[\dEe+-]+))$')
    rx_complex = re.compile(
        r'^\s*(?P<re>([+-]?\d+\.[\dEe+-]+))\s+'
        r'(?P<im>([+-]?\d+\.[\dEe+-]+))$'
    )
    rx_boolean = re.compile(r'^(?P<val>[T|F])$')
    rx_str = re.compile(r'^(?P<val>.*)$')
    bool_map = {'T': True, 'F': False}

    def ftfloat(num_str):
        return float(fix_fortran_exponent(num_str))

    def get_shape(line):
        bounds = [int(i) for i in line.split()]
        lbounds = np.array(bounds[: (len(bounds) // 2)])
        ubounds = np.array(bounds[(len(bounds) // 2) :])
        return tuple(ubounds - lbounds + 1)

    rx_converters = [
        (rx_int, lambda m: int(m.group('val'))),
        (rx_float, lambda m: ftfloat(m.group('val'))),
        (
            rx_complex,
            lambda m: complex(ftfloat(m.group('re')), ftfloat(m.group('im'))),
        ),
    ]
    if convert_boolean:
        rx_converters.append((rx_boolean, lambda m: bool_map[m.group('val')]))
    rx_converters.append((rx_str, lambda m: m.group('val')))

    if typename is None:
        first_line = next(data)
        logger.debug("FIRST LINE: %s", first_line.rstrip())
        if not first_line.startswith("# ASCII DUMP"):
            raise ValueError("Invalid dump format")
        typename = first_line[13:].strip()
    result.typename = typename

    field = None
    value = None
    array_size = 0
    array = []
    looking_for_field_name = True

    while True:
        line = next(data)
        logger.debug("LINE: %s", line.rstrip())
        if rx_item.match(line):
            logger.debug("  (skipped as item line)")
            continue  # skip line
        if looking_for_field_name:
            m = rx_field.match(line)
        else:
            m = None
        if m:  # field name line
            field = m.group('field')
            logger.debug("  (matched as field-name line, %s)", field)
            value = None
            result[field] = None  # so we can detect fixed-sized arrays
            array_size = 0
            array = []
            looking_for_field_name = False
            # Not looking for two consecutive field name lines means we could
            # have string values that start with '#'
        else:  # value-line, array bound line, or end of dump
            if line.startswith('# END ASCII DUMP'):
                logger.debug("  (end dump -> return)")
                return result
            # array bounds?
            if rx_arraybounds.match(line):
                array_size = int(value)  # what we thought was a simple int ...
                result[field] = None  # ... value is actually the array_size
                shape = get_shape(line)
                logger.debug(
                    "  (array bound line -> value will be array "
                    "of size %d)",
                    array_size,
                )
                # note that only allocatable components have an array bound
                # line. Fixed-sized arrays are handled separately (below)
                continue
            # obtain value
            if line.startswith('# ASCII DUMP'):
                logger.debug("  (value is sub-dump, recurse)")
                typename = line[13:].strip()
                value = _read_ascii_dump(
                    data,
                    convert_boolean=convert_boolean,
                    flatten=flatten,
                    typename=typename,
                )
            else:
                matched = False
                for rx, conv in rx_converters:
                    m = rx.match(line)
                    if m:
                        matched = True
                        value = conv(m)
                        logger.debug("  (value: %s)", value)
                        break
                assert matched
            # append value to array, or set result
            if array_size > 0:
                array.append(value)
                array_size -= 1
                logger.debug(
                    "  (appended value %s, %d remaining)", value, array_size
                )
                if array_size == 0:
                    if np.isscalar(array[0]):
                        if flatten:
                            logger.debug(
                                "  (set result for %s as flat " "np.array)",
                                field,
                            )
                            result[field] = np.array(array)
                        else:
                            logger.debug(
                                "  (set result for %s as reshaped "
                                "np.array)",
                                field,
                            )
                            result[field] = np.array(array).reshape(
                                shape, order='F'
                            )
                    else:
                        result[field] = array
                        logger.debug("  (set result for %s as list)", field)
            else:  # not an array (or at least not an allocatable array)
                if result[field] is None:
                    logger.debug(
                        "  (set result for %s as simple value %s)",
                        field,
                        value,
                    )
                    result[field] = value
                elif isinstance(result[field], list):
                    logger.debug(
                        "  (append simple value %s to fixed-sized "
                        "array for %s)",
                        value,
                        field,
                    )
                    result[field].append(value)
                else:
                    logger.debug(
                        "  (append simple value %s to fixed-sized "
                        "array for %s [new])",
                        value,
                        field,
                    )
                    result[field] = [result[field], value]
            if array_size == 0:
                looking_for_field_name = True  # next line may be a field name


def write_psi_amplitudes(psi, filename):
    """Write the wave function to file in the same format as the
    `write_psi_amplitudes` Fortran routine

    Parameters:
        psi (numpy array): Array of complex probability amplitudes
        filename (str): Name of file to which to write
    """
    with open_file(filename, 'w') as out_fh:
        is_complex = np.any(np.iscomplex(psi))
        if is_complex:
            out_fh.write("#%9s%25s%25s\n" % ('index', 'Re[Psi]', 'Im[Psi]'))
        else:
            out_fh.write("#%9s%25s\n" % ('index', 'Re[Psi]'))
        for i, val in enumerate(psi):
            if np.abs(val) > 1e-16:
                if is_complex:
                    out_fh.write(
                        "%10d%25.16E%25.16E\n" % (i + 1, val.real, val.imag)
                    )
                else:
                    out_fh.write("%10d%25.16E\n" % (i + 1, val.real))


def read_psi_amplitudes(filename, n, block=1, normalize=True):
    """Read the wave function of size `n` from file. For 'block=1', inverse to
    `write_psi_amplitudes`. Returns complex or real numpy array

    By specifying `blocks`, data may be read from a file that contains multiple
    wave functions, in the format generated e.g. by the
    ``qdyn_prop_traj --write-all-states`` utility

    Parameters:
        filename (str): Name of file from which to read data
        n (int): dimension of the Hilbert space (i.e. size of returned vector)
        block (int): One-based index of the block to read from
            `filename`, if the file contains multiple block. Blocks must be
            separated by exactly two empty lines
        normalize (bool): Whether or not to normalize the wave function
    """
    if block < 1:
        raise ValueError("Invalid block %d, block must be >= 1" % block)
    return next(iterate_psi_amplitudes(filename, n, block, normalize))


def iterate_psi_amplitudes(filename, n, start_from_block=1, normalize=True):
    """Iterate over blocks of wave functions stored in `filename`

    This is equivalent to the following pseudocode::

        iter([read_psi_amplitudes(filename, n, block, normalize)
              for block >= start_from_block])

    Hoever, the data file is only traversed once, and memory is only required
    for one block.

    Parameters:
        filename (str): Name of file from which to read data
        n (int): dimension of the Hilbert space (i.e. size of returned vector)
        start_from_block (int): One-based index of block from which to
            start the iterator.
        normalize (bool): Whether or not to normalize the wave function
    """
    psi = np.zeros(n, dtype=np.complex128)
    i_block = 1
    blanks = 0

    def normalized(psi):
        if normalize:
            nrm = norm(psi)
            if nrm > 1e-15:
                return psi / norm(psi)
            else:
                return psi * 0.0
        else:
            return psi

    with open_file(filename, 'r') as in_fh:
        for line_nr, line in enumerate(in_fh):
            line = line.strip()
            if line == '':
                blanks += 1
                if blanks >= 2:
                    if i_block >= start_from_block:
                        yield normalized(psi)
                        psi = np.zeros(n, dtype=np.complex128)
                    blanks = 0
                    i_block += 1
                continue
            if i_block < start_from_block:
                continue
            if not line.startswith("#"):
                vals = line.split()[:3]
                try:
                    i = int(vals[0]) - 1
                    psi[i] = read_real(vals[1])
                    if len(vals) == 3:
                        psi[i] += 1j * read_real(vals[2])
                except (ValueError, TypeError) as exc_info:
                    raise ValueError("Invalid format: %s" % str(exc_info))
    if start_from_block > i_block:
        raise ValueError(
            "Requested block %d, file only has %d blocks"
            % (start_from_block, i_block)
        )
    yield normalized(psi)
