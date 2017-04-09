"""Routines for reading and writing files compatible with QDYN"""
from __future__ import print_function, division, absolute_import

import sys
import re
import os
from collections import OrderedDict
import logging
# import for doctests
from contextlib import contextmanager
import tempfile
import json

import scipy.sparse
import numpy as np
import six
from six.moves import xrange
from click.utils import open_file

from .linalg import iscomplexobj


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
        header=None, hermitian=False, limit=0.0):
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
            i = sparse_h.row[i_val] + 1  # 1-based indexing
            j = sparse_h.col[i_val] + 1
            v = sparse_h.data[i_val]
            if (not hermitian) or (j >= i):
                if (abs(v) > limit):
                    if is_real:
                        assert v.imag == 0
                    line = line_formatter(i, j, v)
                    if line is not None:
                        out_fh.write(line)
                        if not line.endswith("\n"):
                            out_fh.write("\n")


def read_indexed_matrix(filename, format='coo', shape=None,
        expand_hermitian=False, val_real=False):
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
        If True, the input file must contain data only for the upper or lower
        triangle. The oterh triangle will be set with the complex conjugate
        values.
    val_real: boolean, optional
        If True, only read 3 columns from the input file (i, j, value), even if
        more columns are present in the file, and return a real matrix.
    """
    from scipy.sparse import coo_matrix
    file_row, file_col \
    = np.genfromtxt(filename, usecols=(0,1), unpack=True, dtype=np.int)
    file_real_val \
    = np.genfromtxt(filename, usecols=(2,), unpack=True, dtype=np.float64)
    # numpy doesn't generate arrays if there is only one value -- force it!
    file_row = file_row.reshape(file_row.size)
    file_col = file_col.reshape(file_col.size)
    file_real_val = file_real_val.reshape(file_real_val.size)
    val_is_real = False
    if val_real:
        val_is_real = True
    else:
        try:
            file_imag_val = np.genfromtxt(filename, usecols=(3,), unpack=True,
                                          dtype=np.float64)
            file_imag_val = file_imag_val.reshape(file_imag_val.size)
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
        return getattr(m, 'to'+format)()  # e.g. format='dense' -> m.todense()


def print_matrix(M, matrix_name=None, limit=1.0e-14, fmt="%9.2E",
        out=None):
    """Print a numpy complex matrix to screen, or to a file if outfile is given.
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
        zero_fmt = '%' + ("%dd" % width)
        zero_fmt = "%s,%s" % (zero_fmt, zero_fmt)
        zero = zero_fmt % (0, 0)
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
    """In 3-digit exponents, Fortran drops the 'E'. Return a string with the
    'E' restored.

    >>> print(fix_fortran_exponent("1.0-100"))
    1.0E-100
    >>> print(fix_fortran_exponent("1.0E-99"))
    1.0E-99
    """
    if 'E' not in num_str:
        return re.sub('(\d)([+-]\d)', r'\1E\2', num_str)
    return num_str


def read_complex(str):
    """Convert a string to a complex number

    >>> read_complex("1.0 -2.0-100")
    (1-2e-100j)
    """
    real_part, imag_part = str.split()
    real_part = fix_fortran_exponent(real_part)
    imag_part = fix_fortran_exponent(imag_part)
    return float(real_part) + 1.0j*float(imag_part)


def read_cmplx_array(filename, **kwargs):
    """Read a complex array from a file. The file must contain two columns
    (real and imaginary part). This routine is equivalent to the Fortran QDYN
    `read_cmplx_array` routine

    Args:
        filename (str): Name of the file from which to read the array
        kwargs: All keyword arguments are passed to `numpy.genfromtxt`
    """
    x, y = np.genfromtxt(filename, usecols=(0, 1), unpack=True, **kwargs)
    return x + 1j*y


def write_cmplx_array(carray, filename, header=None, fmtstr='%25.16E',
                      append=False, comment=None):
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
        with open(filename, 'a') as out_fh:
            out_fh.write("\n\n")
            writetotxt(out_fh, carray, fmt=(fmtstr, fmtstr),
                       header=header_lines)

    else:
        writetotxt(filename, carray, fmt=(fmtstr, fmtstr),
                   header=header_lines)


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
    fmt       = kwargs.get('fmt', '%25.16E')
    delimiter = kwargs.get('delimiter', '')
    header    = kwargs.get('header', '')
    footer    = kwargs.get('footer', '')
    comments  = kwargs.get('comments', "# ")
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
                    if line.startswith(" "*l_comments):
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
            raise ValueError('fmt has wrong number of %% formats:  %s'
                             % row_fmt)
        if not completed_row_fmt:
            if len(delimiter) > 0:
                row_fmt = row_fmt[:-len(delimiter)]
            completed_row_fmt = True

        # write out data
        for i_row in xrange(n_rows):
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
                raise TypeError("Cannot format row data %s with format %s" %
                                (repr(tuple(row_data)), repr(fmt)))
            fh.write("\n")

        # write footer
        if isinstance(footer, (list, tuple)):
            footer = "\n".join(footer)
        if len(footer) > 0:
            for line in footer.split("\n"):
                if not line.startswith(comments):
                    if line.startswith(" "*l_comments):
                        line = comments + line[l_comments:]
                    else:
                        line = comments + line
                fh.write(line)
                fh.write("\n")

    finally:
        if own_fh:
            fh.close()


def split_sup_sub(name):
    """Split a (pseudo-) LaTeX string at top-level
    subscripts/superscripts/groups

    >>> assert split_sup_sub('CNOT') == ['CNOT', ]
    >>> assert split_sup_sub('CNOT_system') == ['CNOT', '_system']
    >>> assert split_sup_sub('A_1^2') == ['A', '_1', '^2']
    >>> assert split_sup_sub('A_{1^2}^2') == ['A', '_{1^2}', '^2']
    >>> assert split_sup_sub('^1A_1{B}') == ['^1A', '_1', '{B}']
    >>> assert split_sup_sub('^1A_{1}B{C}') == ['^1A', '_{1}', 'B', '{C}']
    >>> assert split_sup_sub('{A}B{C}') == ['{A}', 'B', '{C}']
    """
    parts = []
    part = ''
    bracelevel = 0
    prev_letter = None
    for letter in name:
        if letter in ['^', '_']:
            if (bracelevel == 0) and (len(part) > 0):
                parts.append(part)
                part = ''
            part += letter
        elif letter == "{":
            if ((bracelevel == 0) and (len(part) > 0)
                    and (prev_letter not in ['^', '_'])):
                parts.append(part)
                part = ''
            bracelevel += 1
            part += '{'
        elif letter == "}":
            bracelevel -= 1
            if (bracelevel == 0) and (len(part) > 0):
                parts.append(part+letter)
                part = ''
            else:
                part += letter
        else:
            part += letter
        prev_letter = letter
    # rest of string
    if len(part) > 0:
        parts.append(part)
    return parts


def matrix_to_latex(M, name=None):
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

    >> print(matrix_to_latex(CNOT, name='CNOT'))
    CNOT = \begin{pmatrix}
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
    if name is None:
        return "\n".join(lines)
    else:
        return name + " = " + "\n".join(lines)


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


def obj_to_data(obj, classkey=None, attr_filter=None, depth=10):
    """Convert the given object recursively into a simple data structure of
    dicts and lists. This facilitates easy introspection into complicated
    objects. See `obj_to_json` for examples.

    Arguments
    ---------

    obj: object
        The object to convert
    classkey: str or None
        When converting an object into a dictionary, include the name of the
        object's class using `classkey` as the key
    attr_filter: dict or None
        Dictionary where each key is a class, and each value is a list of
        regular experessions (as strings, or compiled). When converting an
        object into a dictionary, only consider attributes that match one of
        the regular expressions in the list. If None, assume ('[^_].*', ), that
        is all attributes that do not start with an underscore
    depth: int
        Recursion depth
    """
    if depth < 0:
        return '...'
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = obj_to_data(v, classkey=classkey,
                                  attr_filter=attr_filter, depth=depth-1)
        return data
    elif hasattr(obj, "_ast"):
        return obj_to_data(obj._ast(), classkey=classkey,
                           attr_filter=attr_filter, depth=depth-1)
    elif isinstance(obj, six.string_types):
        return obj
    elif hasattr(obj, "__iter__"):
        return [obj_to_data(v, classkey=classkey, attr_filter=attr_filter,
                            depth=depth-1)
                for v in obj]
    elif hasattr(obj, "__dict__"):
        data = {}
        try:
            allowed_key_rxs = []
            for pattern in attr_filter[obj.__class__]:
                allowed_key_rxs.append(re.compile(pattern))
        except (AttributeError, KeyError, TypeError):
            allowed_key_rxs = [re.compile(r'[^_].*'), ]
        for k, v in six.iteritems(obj.__dict__):
            for rx in allowed_key_rxs:
                if rx.match(k):
                    data[k] = obj_to_data(v, classkey=classkey,
                                          attr_filter=attr_filter,
                                          depth=depth-1)
                    break
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def obj_to_json(obj, classkey=None, attr_filter=None, depth=10, sort_keys=True,
        indent=4, separators=(',', ': ')):
    """Return a json string for the data returned by obj_to_data for the given
    obj. The parameters `obj`, `classkey`, `attr_filter`, and `depth` are
    passed to `obj_to_data`; the remaining arguments are passed to `json.dumps`

    >>> import numpy as np
    >>> import matplotlib
    >>> from matplotlib.figure import Figure as figure
    >>> from collections import defaultdict
    >>> fig = figure()
    >>> ax = fig.add_subplot(111)
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> l1 = ax.plot(x, np.sin(x), label='sin')
    >>> l2 = ax.plot(x, np.cos(x), label='cos')
    >>> F = defaultdict(lambda:[r'^(?!(_|axes|figure|axis|callbacks))'])
    >>> F[matplotlib.figure.Figure]          = ['_axstack']
    >>> F[matplotlib.figure.AxesStack]       = ['_elements']
    >>> F[matplotlib.axes._subplots.Subplot] = ['lines']
    >>> F[matplotlib.lines.Line2D]           = [r'_label']
    >>> print(obj_to_json(fig, attr_filter=F, classkey='class'))
    {
        "_axstack": {
            "_elements": [
                [
                    [
                        [
                            1,
                            1,
                            1
                        ],
                        []
                    ],
                    [
                        1,
                        {
                            "class": "AxesSubplot",
                            "lines": [
                                {
                                    "_label": "sin",
                                    "class": "Line2D"
                                },
                                {
                                    "_label": "cos",
                                    "class": "Line2D"
                                }
                            ]
                        }
                    ]
                ]
            ],
            "class": "AxesStack"
        },
        "class": "Figure"
    }
    """
    return json.dumps(
        obj_to_data(obj, classkey=classkey, attr_filter=attr_filter,
                    depth=depth),
        sort_keys=sort_keys, indent=indent, separators=separators)


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
    with open(filename) as in_fh:
        try:
            return _read_ascii_dump(in_fh, convert_boolean=convert_boolean,
                                    flatten=flatten)
        except StopIteration:
            # If the file properly ended in '# END ASCII DUMP', we should have
            # returned cleanly instead of running into the end of the file
            raise ValueError("Premature file end")


def _read_ascii_dump(data, convert_boolean=True, flatten=False,
                     typename=None):
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
    rx_complex = re.compile(r'^\s*(?P<re>([+-]?\d+\.[\dEe+-]+))\s+'
                            r'(?P<im>([+-]?\d+\.[\dEe+-]+))$')
    rx_boolean = re.compile(r'^(?P<val>[T|F])$')
    rx_str = re.compile(r'^(?P<val>.*)$')
    bool_map = {'T': True, 'F': False}

    def ftfloat(num_str):
        return float(fix_fortran_exponent(num_str))

    def get_shape(line):
        bounds = [int(i) for i in line.split()]
        lbounds = np.array(bounds[:(len(bounds)//2)])
        ubounds = np.array(bounds[(len(bounds)//2):])
        return tuple(ubounds - lbounds + 1)

    rx_converters = [
            (rx_int, lambda m: int(m.group('val'))),
            (rx_float, lambda m: ftfloat(m.group('val'))),
            (rx_complex, lambda m: complex(ftfloat(m.group('re')),
                                           ftfloat(m.group('im')))),
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
                array_size = int(value) #  what we thought was a simple int ...
                result[field] = None    #  ... value is actually the array_size
                shape = get_shape(line)
                logger.debug("  (array bound line -> value will be array "
                             "of size %d)", array_size)
                # note that only allocatable components have an array bound
                # line. Fixed-sized arrays are handled separately (below)
                continue
            # obtain value
            if line.startswith('# ASCII DUMP'):
                logger.debug("  (value is sub-dump, recurse)")
                typename = line[13:].strip()
                value = _read_ascii_dump(data,
                                         convert_boolean=convert_boolean,
                                         flatten=flatten,
                                         typename=typename)
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
                logger.debug("  (appended value %s, %d remaining)",
                             value, array_size)
                if array_size == 0:
                    if np.isscalar(array[0]):
                        if flatten:
                            logger.debug("  (set result for %s as flat "
                                         "np.array)", field)
                            result[field] = np.array(array)
                        else:
                            logger.debug("  (set result for %s as reshaped "
                                         "np.array)", field)
                            result[field] = np.array(array).reshape(shape,
                                                                    order='F')
                    else:
                        result[field] = array
                        logger.debug("  (set result for %s as list)",
                                     field)
            else:  # not an array (or at least not an allocatable array)
                if result[field] is None:
                    logger.debug("  (set result for %s as simple value %s)",
                                 field, value)
                    result[field] = value
                elif isinstance(result[field], list):
                    logger.debug("  (append simple value %s to fixed-sized "
                                 "array for %s)", value, field)
                    result[field].append(value)
                else:
                    logger.debug("  (append simple value %s to fixed-sized "
                                 "array for %s [new])", value, field)
                    result[field] = [result[field], value]
            if array_size == 0:
                looking_for_field_name = True  # next line may be a field name
