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
