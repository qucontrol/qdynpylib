"""
Module that contains the Gate2Q class for working with two-qubit gates.

Also defines common two-qubit gates as Gate2Q objects.
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
import scipy
import cmath
import os
from numpy import pi, cos, sin
import re
from warnings import warn
from QDYN.io import open_file, matrix_to_latex, matrix_to_mathematica
from QDYN.linalg import inner

class Gate2Q(np.matrixlib.defmatrix.matrix):
    """
    Subclass of numpy.matrix that contains a two-qubit gate. It must therefore
    be a 4 x 4 complex matrix.

    Instantiation is the same as for a numpy matrix, i.e. a gate can be
    constructed from another 2D-Array:

    >>> CNOT = np.matrix(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'))
    >>> # Note: the 'str' wrapper is only required in context of doctest
    >>> CNOT = Gate2Q(CNOT)
    >>> CNOT
    matrix([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j]])
    >>> type(CNOT)
    <class 'QDYN.gate2q.Gate2Q'>

    Alternatively, it can be create from a string:

    >>> Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'))
    matrix([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j]])

    However, we also allow to create a zero gate as

    >>> Gate2Q()
    matrix([[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]])

    Lastly, we may construct a gate directly from a file or file-like object,
    (as written by the QDYN write_cmplx_matrix routine)

        >>> gate = '''
        ... U = [
        ... { 1.00000000E+00,            0.0}(              0,              0)(              0,              0)(              0,              0)
        ... (              0,              0){ 5.72735140E-01, 8.19740483E-01}(              0,              0)(              0,              0)
        ... (              0,              0)(              0,              0){ 2.12007110E-01,-9.77268124E-01}(              0,              0)
        ... (              0,              0)(              0,              0)(              0,              0){ 9.99593327E-01,-2.85163130E-02}
        ... ]
        ... '''
        >>> from StringIO import StringIO
        >>> Gate2Q(file=StringIO(gate))
        matrix([[ 1.00000000+0.j        ,  0.00000000+0.j        ,
                  0.00000000+0.j        ,  0.00000000+0.j        ],
                [ 0.00000000+0.j        ,  0.57273514+0.81974048j,
                  0.00000000+0.j        ,  0.00000000+0.j        ],
                [ 0.00000000+0.j        ,  0.00000000+0.j        ,
                  0.21200711-0.97726812j,  0.00000000+0.j        ],
                [ 0.00000000+0.j        ,  0.00000000+0.j        ,
                  0.00000000+0.j        ,  0.99959333-0.02851631j]])

    The Gate2Q object can also be supplied a name that will be used for output.

    >>> CNOT = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'), name='CNOT')
    >>> print(CNOT.name)
    CNOT
    """

    def __new__(cls, *args, **kwargs):
        """
        Return a new instance of Gate2Q
        """

        file = None

        # We allow an empty constructor to create a zero-matrix
        if len(args) == 0:
            args = [(np.zeros((4,4), dtype=np.complex128)), ]
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, np.ndarray):
                # it appears that a numpy matrix will not change its type to
                # Gate2Q. So, we make sure to downcast it to a normal array
                args = [np.asarray(arg), ]
            if isinstance(arg, str):
                if os.path.isfile(arg):
                    kwargs['file'] = arg
                    args = [(np.zeros((4,4), dtype=np.complex128)), ]

        # Make sure that dtype is np.complex128
        if 'dtype' in kwargs:
            if kwargs['dtype'] is not np.complex128:
                warn('dtype is ignored (must be numpy.complex128)')
        kwargs['dtype'] = np.complex128

        # Allow to read from file
        if 'file' in kwargs:
            file = kwargs['file'] # delay until after construction
            del kwargs['file']

        # Allow to set name
        name = None
        if 'name' in kwargs:
            name = kwargs['name'] # delay until after construction
            del kwargs['name']

        gate = np.matrixlib.defmatrix.matrix.__new__(cls, *args, **kwargs)

        # read from file (delayed)
        if file is not None:
            gate.read(file, name=None)

        # set name (delayed)
        gate.name = name

        return gate

    def __array_finalize__(self, obj):
        """
        Finalize the creation of the matrix, also allows for view-casting,
        see <http://docs.scipy.org/doc/numpy/user/basics.subclassing.html>
        """
        if 'name' not in self.__dict__:
            self.name = None

    def __str__(self):
        """
        Return a string representation of the two-qubit gate, as valid Python
        code.

        >>> CNOT = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'),
        ...               name='CNOT')
        >>> # Note: the 'str' wrapper is only required in context of doctest
        >>> print(CNOT)
        CNOT = Gate2Q([
        [1.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j],
        [0.000000+0.000000j, 1.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j],
        [0.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j, 1.000000+0.000000j],
        [0.000000+0.000000j, 0.000000+0.000000j, 1.000000+0.000000j, 0.000000+0.000000j]])

        >>> CNOT = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'))
        >>> print(CNOT)
        [[1.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j],
         [0.000000+0.000000j, 1.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j],
         [0.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j, 1.000000+0.000000j],
         [0.000000+0.000000j, 0.000000+0.000000j, 1.000000+0.000000j, 0.000000+0.000000j]]
        """
        if self.name is not None:
            result = "%s = Gate2Q([\n" % self.name
        else:
            result = ""
        for i in xrange(4):
            fmt = "[%5f+%5fj, %5f+%5fj, %5f+%5fj, %5f+%5fj],\n"
            if self.name is None:
                if i == 0:
                    fmt = "[" + fmt
                else:
                    fmt = " " + fmt
            if i == 3:
                if self.name is None:
                    fmt = fmt[:-2] + "]"
                else:
                    fmt = fmt[:-2] + "])"
            row_str = fmt % (
                    self[i,0].real, self[i,0].imag,
                    self[i,1].real, self[i,1].imag,
                    self[i,2].real, self[i,2].imag,
                    self[i,3].real, self[i,3].imag)
            result += row_str
        return result

    def _repr_latex_(self):
        """Alias for self.to_latex; used for IPython Notebook display"""
        return self.to_latex()

    def to_latex(self):
        r"""
        Return a LaTeX representation of the gate

        >>> CNOT = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'))
        >>> print(CNOT.to_latex())
        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0 \\
        \end{pmatrix}
        """
        return matrix_to_latex(self)

    def to_mathematica(self):
        """
        Return a Mathematica representation of the gate
        >>> print(cphase(0.25*pi).to_mathematica())
        {{0.707106781187+0.707106781187I, 0, 0, 0}, {0, 1.0, 0, 0}, {0, 0, 1.0, 0}, {0, 0, 0, 1.0}}
        """
        return matrix_to_mathematica(self)

    def to_latex_arrows(self, env='equation*'):
        r"""
        Return a LaTeX representation where the complex numbers in the gate are
        represented by pointer arrows.

        >>> CNOT = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'),
        ...               name='CNOT')
        >>> print(CNOT.to_latex_arrows())
        \begin{equation*}
        \text{CNOT} = \begin{pmatrix}
        \scalebox{1.000000}{\rotatebox{0}{$\rightarrow$}} &
        &
        &
        \\
        &
        \scalebox{1.000000}{\rotatebox{0}{$\rightarrow$}} &
        &
        \\
        &
        &
        &
        \scalebox{1.000000}{\rotatebox{0}{$\rightarrow$}} \\
        &
        &
        \scalebox{1.000000}{\rotatebox{0}{$\rightarrow$}} &
        \\
        \end{pmatrix}
        \end{equation*}

        Note: The LaTeX code will *not* render in the IPython notebook
        """
        lines = []
        if env is not None:
            if env == '$':
                lines.append(r'$')
            else:
                lines.append("\\begin{%s}" % env)
        name = r'\text{%s}' % self.name
        if name is None:
            name = r'U'
        lines.append(name + r' = \begin{pmatrix}')
        n = 4
        for i in xrange(n):
            for j in xrange(n):
                if j==n-1:
                    eol = r'\\'
                else:
                    eol = r'&'
                r, phi = cmath.polar(self[i,j])
                phi = int(phi * 114.5915590262)
                if (r > 1.0e-3):
                    lines.append(
                    "\\scalebox{%f}{\\rotatebox{%d}{$\\rightarrow$}} %s"
                    % (r, phi, eol))
                else:
                    lines.append(eol)
        lines.append(r'\end{pmatrix}')
        if env is not None:
            if env == '$':
                lines.append(r'$')
            else:
                lines.append("\\end{%s}" % env)
        return "\n".join(lines)

    def arrow_plot(self, ax=None, figsize=None, phi0=0.5*pi, title=None):
        """
        Render a graphical represenation of the gate where the complex entries
        are represented by arrows, using matplotlib.

        The plot can be rendered onto an existing axis, or alternatively, a
        simple new figure will be created and displayed.

        Arguments
        ---------

        ax: matplotlib Axes instance
            If given, render onto the given axis, but don't display anything.
            Otherwise, create a new axes in a new matplotlib.pylab figure, and
            display it
        figsize: tuple of floats
            If `ax` is not given, the size of the matplotlib.pylab figure that
            should be created
        phi0: float
            Angle in radians that a phase of zero corresponds to. This
            defaults to pi/2, so real numbers are represented by a vertical
            arrow
        title: str
            Title for the axes. Defaults to `self.name`. Pass an empty string
            in order to suppress the title.

        Example
        -------

        Rendering onto an existing axes allows to show multiple gates in one
        figure. For example, in an IPython notebook, we could compare four
        standard quantum gates in a single row:

            fig = plt.figure(figsize=(9,2))
            arrow_plot(sqrt_iSWAP, ax=fig.add_subplot(141, aspect=1))
            arrow_plot(iSWAP, ax=fig.add_subplot(142, aspect=1))
            arrow_plot(BGATE, ax=fig.add_subplot(143, aspect=1))
            arrow_plot(CNOT, ax=fig.add_subplot(144, aspect=1))
            plt.show(fig)
        """
        if ax is None:
            import matplotlib.pylab as plt
            fig = plt.figure(figsize=figsize, facecolor='w', edgecolor='w')
            ax = fig.add_axes([0, 0, 1, 1], aspect='equal')
        if title is None:
            if self.name is not None:
                ax.set_title("%s:" % self.name, loc='left')
        else:
            if title != '':
                ax.set_title(title, loc='left')
        ax.set_xlim(0,4)
        ax.set_ylim(0,4)
        ax.xaxis.set_ticks([1,2,3])
        ax.yaxis.set_ticks([1,2,3])
        ax.tick_params(length=0, width=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        def arrow_pos(row, col, length, phi, head_length=0.0):
            """Return x, y, dx, dy"""
            center = ( col+0.5, 4.0-row-0.5)
            x = center[0] - 0.5* length * np.cos(phi)
            y = center[1] - 0.5* length * np.sin(phi)
            dy = (length-head_length) * np.sin(phi)
            dx = (length-head_length) * np.cos(phi)
            return x, y, dx, dy
        for row in xrange(4):
            for col in xrange(4):
                length = np.abs(self[row, col])
                if length > 1.0e-3:
                    phi = np.angle(self[row,col]) + phi0
                    x, y, dx, dy \
                    = arrow_pos(row, col, length, phi, head_length=0.1)
                    ax.arrow(x, y, dx, dy, width=0.05, head_width=0.1,
                            head_length=0.1, fc='black', ec='black')
        if ax is None:
            plt.show(fig)

    def read(self, file, name='U'):
        """
        Read in complex 4x4 matrix from file (as written by the QDYN
        write_cmplx_matrix routine).

        Return a 4x4 double precision complex Numpy matrix

        Assumes the propagation is in the canonical basis

        Arguments
        ---------

        file: str or file-like object
            Filename of file from which to read gate, or file-like object with
            equivalent content

        name: str
            Value for self.name. If name is None, self.name will remain
            unchanged
        """
        with open_file(file) as fh:
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
                    self[i,j] = z
                    j += 1
                i += 1
        if name is not None:
            self.name = name

    def closest_unitary(self, get_distance=False):
        """
        Return the closest unitary matrix to the given gate (which may be
        non-unitary due to loss from the logical subspace)

        If get_dinstance is True, return a tuple (U_unitary, d) where U_unitary
        is the closest unitary and d is the 2-norm operator distance between
        U_unitary and the original gate.

        >>> CNOT = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0'))
        >>> from QDYN.linalg import norm
        >>> norm(CNOT - CNOT.closest_unitary()) < 1.0e-15
        True

        >>> d = 0.01
        >>> U = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -0.99'))
        >>> U_unitary, dist = U.closest_unitary(get_distance=True)
        >>> CPHASE = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1'))
        >>> from QDYN.linalg import norm
        >>> norm(U_unitary - CPHASE) < 1.0e-15
        True
        >>> abs(dist - d) < 1.0e-15
        True
        """
        V, S, Wh = scipy.linalg.svd(self)
        U_unitary = Gate2Q(V.dot(Wh))
        if not get_distance:
            return U_unitary
        d = 0.0
        for s in S:
            if abs(s - 1.0) > d:
                d = abs(s-1.0)
        return U_unitary, d

    def pop_loss(self):
        """
        Return the loss of population from the two-qubit logical subspace
        """
        return pop_loss(self)

    def concurrence(self):
        """
        Return the concurrence of the gate
        >>> round(SWAP.concurrence(), 2)
        0.0
        >>> round(CNOT.concurrence(), 2)
        1.0
        >>> round(identity.concurrence(), 2)
        0.0
        """
        from QDYN.weyl import concurrence as concurrence_c1c2c3
        from QDYN.weyl import c1c2c3
        return concurrence_c1c2c3(*c1c2c3(self))

    def weyl_coordinates(self):
        """
        Return the Weyl chamber coordinates (c1, c2, c3) for the current gate,
        in units of pi

        >>> print("%.2f %.2f %.2f" % CNOT.weyl_coordinates())
        0.50 0.00 0.00
        """
        from QDYN.weyl import c1c2c3
        return c1c2c3(self)

    def local_invariants(self):
        """
        Return the local invariants (g1, g2, g3) for the current gate

        >>> print("%.2f %.2f %.2f" % CNOT.local_invariants())
        0.00 0.00 1.00
        """
        from QDYN.weyl import g1g2g3
        return g1g2g3(self)


    def F_avg(self, O, closest_unitary=False):
        """
        Return the average gate fidelity of the current gate with respect to
        the gate O.

        >>> U = 0.95*CNOT
        >>> abs(U.F_avg(CNOT) - 0.9025) < 1e-15
        True

        If `closest_unitary=True`, calculate the fidelity for the closest
        unitary from the current gate

        >>> U = 0.95*CNOT
        >>> abs(U.F_avg(CNOT, closest_unitary=True) - 1.0) < 1e-15
        True
        """
        if closest_unitary:
            return F_avg(self.closest_unitary(), O)
        else:
            return F_avg(self, O)

    def cartan_decomposition(self):
        """
        Return the Cartan decomposition of the closest unitary for the current
        gate. See `QDYN.weyl.cartan_decomposition` for details

        Returns
        -------

        k1 : Gate2Q
            left local operations in SU(2) x SU(2)
        A  : Gate2Q
            non-local operations, in SU(4)
        k2 : Gate2Q
            right local operations in SU(2) x SU(2)

        Example
        -------

        >>> k1, A, k2 = CNOT.cartan_decomposition()
        >>> abs(k1.concurrence()) < 1.0e-15
        True
        >>> abs(k2.concurrence()) < 1.0e-15
        True
        >>> abs(A.concurrence() - 1.0) < 1.0e-15
        True
        """
        from QDYN.weyl import cartan_decomposition
        k1, A, k2 = cartan_decomposition(self.closest_unitary())
        return Gate2Q(k1), Gate2Q(A), Gate2Q(k2)


def pop_loss(A):
    """
    Assuming that A is a matrix obtained from projecting a unitary matrix
    to a smaller subspace, return the loss of population from the subspace
    (averaged over all basis states).

    A can be a matrix of arbitrary dimension N. The loss of population is
    calculated as

    1 - \Tr[A^\dagger A] / N

    >>> U = Gate2Q(str('1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -0.9'))
    >>> abs( (1 - (0.9 * 0.9 + 3) / 4) - pop_loss(U) ) < 1.0e-15
    True
    """
    N = A.shape[0]
    A = np.asarray(A)
    A_H = A.conjugate().transpose()
    return 1.0 - (A_H.dot(A)).trace().real / N + 0.0
    # + 0.0 is for numerical stability (avoids result -0.0)


def F_avg(U, O):
    """
    Calculate the average gate fidelity of of the gate U with respect to the
    optimal gate O.
    """
    n = O.shape[0]
    F_U_avg = abs(inner(O,U))**2
    Udagger = U.conjugate().transpose()
    Udagger_O = Udagger.dot(O)
    F_TW_avg = inner(Udagger_O, Udagger_O)
    return (F_TW_avg + F_U_avg).real / (n*(n+1))


###############################################################################
# Standard two-qubit gates

CNOT   = Gate2Q([[ 1,  0,  0,  0],
                 [ 0,  1,  0,  0],
                 [ 0,  0,  0,  1],
                 [ 0,  0,  1,  0]], name='CNOT')

CPHASE  = Gate2Q([[ 1,  0,  0,   0],
                  [  0,  1,  0,   0],
                  [  0,  0,  1,   0],
                  [  0,  0,  0,  -1]], name='CPHASE')


def cphase(phi=pi, state='00'):
    """
    Construct a controlled phasegate, where the phase phi is on the given state

    >>> cphase(pi/4.0, '00')
    matrix([[ 0.70710678+0.70710678j,  0.00000000+0.j        ,
              0.00000000+0.j        ,  0.00000000+0.j        ],
            [ 0.00000000+0.j        ,  1.00000000+0.j        ,
              0.00000000+0.j        ,  0.00000000+0.j        ],
            [ 0.00000000+0.j        ,  0.00000000+0.j        ,
              1.00000000+0.j        ,  0.00000000+0.j        ],
            [ 0.00000000+0.j        ,  0.00000000+0.j        ,
              0.00000000+0.j        ,  1.00000000+0.j        ]])
    """
    logical_basis = {'00':0, '01':1, '10':2, '11':3}
    gate  = Gate2Q(np.identity(4))
    if state not in logical_basis:
        raise ValueError("state must be in [%s]"
                         % ", ".join(logical_basis.keys()))
    i = logical_basis[state]
    gate[i,i] = np.exp(1j*phi)
    gate.name = "CPHASE_{%s}(%f pi)" % (state, phi/pi)
    return gate


SWAP    = Gate2Q([[ 1,  0,  0,   0],
                  [  0,  0,  1,   0],
                  [  0,  1,  0,   0],
                  [  0,  0,  0,   1]], name='CPHASE')

DCNOT   = Gate2Q([[ 1,  0,  0,   0],
                  [  0,  0, 1j,   0],
                  [  0, 1j,  0,   0],
                  [  0,  0,  0,   1]], name='DCNOT')

iSWAP = DCNOT.copy()
iSWAP.name = 'iSWAP'

sqrt_iSWAP   = Gate2Q(
    [[ 1,                  0,          0,        0],
     [ 0,  1.0 /np.sqrt(2.0), 1.0j/np.sqrt(2.0), 0],
     [ 0,  1.0j/np.sqrt(2.0), 1.0 /np.sqrt(2.0), 0],
     [ 0,                  0,                 0, 1]],
    name='sqrt_iSWAP')

sqrt_SWAP = Gate2Q(
    [[ 1,           0,          0, 0],
     [ 0,  0.5*(1-1j), 0.5*(1+1j), 0],
     [ 0,  0.5*(1+1j), 0.5*(1-1j), 0],
     [ 0,           0,          0, 1]],
    name='sqrt_SWAP')

BGATE = Gate2Q(
    [[    cos(pi/8),               0,              0, 1j*sin(pi/8)],
     [            0,     cos(3*pi/8), 1j*sin(3*pi/8),            0],
     [            0,  1j*sin(3*pi/8),    cos(3*pi/8),            0],
     [ 1j*sin(pi/8),               0,              0,   cos(pi/8)]],
    name='BGATE')

identity  = Gate2Q(np.identity(4), name='identity')