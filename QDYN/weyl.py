"""
Routines for calculating local invariants, concurrence, and related quantities
for two-qubit gates
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
import scipy.linalg

# TODO: routine to obtain canonical gate from weyl-coordinates
# TODO: allow to obtain gates for names points in Weyl chamber

Qmagic = (1.0/np.sqrt(2.0)) * np.matrix(
                   [[ 1,  0,  0,  1j],
                    [ 0, 1j,  1,  0],
                    [ 0, 1j, -1,  0],
                    [ 1,  0,  0, -1j]], dtype=np.complex128)


SxSx   =  np.matrix( # sigma_x * sigma_x
                   [[ 0,  0,  0,  1],
                    [ 0,  0,  1,  0],
                    [ 0,  1,  0,  0],
                    [ 1,  0,  0,  0]], dtype=np.complex128)

SySy   =  np.matrix( # sigma_y * sigma_y
                   [[ 0,  0,  0, -1],
                    [ 0,  0,  1,  0],
                    [ 0,  1,  0,  0],
                    [-1,  0,  0,  0]], dtype=np.complex128)

SzSz   =  np.matrix( # sigma_x * sigma_x
                   [[ 1,  0,  0,  0],
                    [ 0, -1,  0,  0],
                    [ 0,  0, -1,  0],
                    [ 0,  0,  0,  1]], dtype=np.complex128)


def g1g2g3(U):
    """
    Given Numpy matrix U, calculate local invariants g1,g2,g3
    U must be in the canonical basis

    >>> from . gate2q import CNOT
    >>> print("%.2f %.2f %.2f" % g1g2g3(CNOT))
    0.00 0.00 1.00
    """
    # mathematically, the determinant of U and to_magic(U) is the same, but
    # we seem to get better numerical accuracy if we calculate detU with
    # the rotated U
    detU = np.linalg.det(to_magic(U))
    m = to_magic(U).T * to_magic(U)
    g1_2 = (np.trace(m))**2 / (16.0 * detU)
    g3   = (np.trace(m)**2 - np.trace(m*m)) / ( 4.0 * detU)
    g1 = g1_2.real + 0.0 # adding 0.0 turns -0.0 result into +0.0
    g2 = g1_2.imag + 0.0
    g3 = g3.real   + 0.0
    return (g1, g2, g3)


def c1c2c3(U):
    """
    Given U (canonical basis), calculate the Weyl Chamber coordinates
    c1,c2,c3

    Algorithm from Childs et al., PRA 68, 052311 (2003).

    >>> from . gate2q import CNOT
    >>> print("%.2f %.2f %.2f" % c1c2c3(CNOT))
    0.50 0.00 0.00
    """
    U_tilde = SySy * U.transpose() * SySy
    ev = np.linalg.eigvals((U * U_tilde)/np.sqrt(complex(np.linalg.det(U))))
    two_S = np.angle(ev) / np.pi
    for i in range(len(two_S)):
        if two_S[i] <= -0.5: two_S[i] += 2.0
    S = np.sort(two_S / 2.0)[::-1] # sort decreasing
    n = int(round(sum(S)))
    S -= np.r_[np.ones(n), np.zeros(4-n)]
    S = np.roll(S, -n)
    M = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    c1, c2, c3 = np.dot(M, S[:3])
    if c3 < 0:
        c1 = 1 - c1
        c3 = -c3
    return (c1+0.0, c2+0.0, c3+0.0) # adding 0.0 turns -0.0 result into +0.0


def g1g2g3_from_c1c2c3(c1, c2, c3):
    """
    Calculate the local invariants from the Weyl-chamber coordinates (c1, c2,
    c3, in units of pi)

    >>> from . gate2q import CNOT
    >>> print("%.2f %.2f %.2f" % g1g2g3_from_c1c2c3(*c1c2c3(CNOT)))
    0.00 0.00 1.00
    """
    c1 *= np.pi
    c2 *= np.pi
    c3 *= np.pi
    g1 = np.cos(c1)**2 * np.cos(c2)**2 * np.cos(c3)**2 \
       - np.sin(c1)**2 * np.sin(c2)**2 * np.sin(c3)**2     + 0.0
    g2 = 0.25 * np.sin(2*c1) * np.sin(2*c2) * np.sin(2*c3) + 0.0
    g3 = 4*g1 - np.cos(2*c1) * np.cos(2*c2) * np.cos(2*c3) + 0.0
    return g1, g2, g3


def point_in_weyl_chamber(c1, c2, c3):
    """
    Return True if the coordinates c1, c2, c3 are inside the Weyl chamber

    >>> from . gate2q import BGATE, identity
    >>> point_in_weyl_chamber(*c1c2c3(BGATE))
    True
    >>> point_in_weyl_chamber(*c1c2c3(identity))
    True
    """
    return ( ((c1 < 0.5)  and (c2 <= c1)   and (c3 <= c2))
          or ((c1 >= 0.5) and (c2 <= 1-c1) and (c3 <= c2)) )


def point_in_PE(c1, c2, c3):
    """
    Return True if the coordinates c1, c2, c3 are inside the perfect-entangler
    polyhedron

    >>> from QDYN.gate2q import BGATE
    >>> point_in_PE(*c1c2c3(BGATE))
    True
    >>> from QDYN.gate2q import identity
    >>> point_in_PE(*c1c2c3(identity))
    False
    """
    if point_in_weyl_chamber(c1, c2, c3):
        return ( ((c1 + c2) >= 0.5) and (c1-c2 <= 0.5) and ((c2+c3) <= 0.5))
    else:
        return False


def concurrence(c1, c2, c3):
    """
    Calculate the concurrence directly from the Weyl Chamber coordinates c1,
    c2, c3

    >>> from . gate2q import SWAP, CNOT, identity
    >>> round(concurrence(*c1c2c3(SWAP)), 2)
    0.0
    >>> round(concurrence(*c1c2c3(CNOT)), 2)
    1.0
    >>> round(concurrence(*c1c2c3(identity)), 2)
    0.0
    """
    if ((c1 + c2) >= 0.5) and (c1-c2 <= 0.5) and ((c2+c3) <= 0.5):
        # if we're inside the perfect-entangler polyhedron in the Weyl chamber
        # the concurrence is 1 by definition. the "regular" formula gives wrong
        # results in this case.
        return 1.0
    else:
        c1_c2_c3 = np.array([c1, c2, c3])
        c3_c1_c2 = np.roll(c1_c2_c3, 1)
        m = np.concatenate((c1_c2_c3 - c3_c1_c2, c1_c2_c3 + c3_c1_c2))
        return np.max(abs(np.sin(np.pi * m)))


def to_magic(A):
    """ Convert a matrix A that is represented in the canonical basis to a
        representation in the Bell basis
    """
    return Qmagic.conj().T * A * Qmagic


def from_magic(A):
    """ The opposite of to_magic """
    return Qmagic * A * Qmagic.conj().T


def J_T_LI(O, U, form='g'):
    """
    Given Numpy matrices O (optimal gate), U (obtained gate), calculate the
    value of the Local invariants-functional
    """
    if form == 'g':
        return np.sum(np.abs(np.array(g1g2g3(O)) - np.array(g1g2g3(U)))**2)
    elif form=='c':
        delta_c = np.array(c1c2c3(O)) - np.array(c1c2c3(U))
        return np.prod(np.cos(np.pi * (delta_c) / 2.0))
    else:
        raise ValueError("Illegal value for 'form'")


def F_PE(g1, g2, g3):
    """
    Evaluate the Perfect-Entangler Functional

    >>> from . gate2q import CNOT
    >>> F_PE(*g1g2g3(CNOT))
    0.0
    >>> from . gate2q import identity
    >>> F_PE(*g1g2g3(identity))
    2.0
    """
    return g3 * np.sqrt(g1**2 + g2**2) - g1 + 0.0


def cartan_decomposition(U):
    """
    Calculate the Cartan Decomposition of the given U in U(4)

    U = k1 * A * k2

    up to a global phase

    Parameters
    ----------

    U : numpy matrix
        Two-qubit quantum gate. Must be unitary

    Returns
    -------

    k1 : numpy matrix
       left local operations in SU(2) x SU(2)
    A  : numpy matrix
        non-local operations, in SU(4)
    k2 : numpy matrix
       right local operations in SU(2) x SU(2)

    Notes
    -----

    If you are working with a logical subspace, you should unitarize U before
    calculating the Cartan decomposition

    References
    ----------

    * D. Reich. Optimising the nonlocal content of a two-qubit gate. Diploma
      Thesis. FU Berlin, 2010. Appendix E

    * Zhang et al. PRA 67, 042313 (2003)
    """
    U = np.matrix(U)                    # in U(4)
    Utilde = U / np.linalg.det(U)**0.25 # U in SU(4)

    found_branch = False
    # The fourth root has four branches; the correct solution could be in
    # any one of them
    for branch in xrange(4):

        UB = to_magic(Utilde) # in Bell basis
        m = UB.T * UB

        # The F-matrix can be calculated according to Eq (21) in PRA 67, 042313
        # It is a diagonal matrix containing the squares of the eigenvalues of
        # m
        c1, c2, c3 = c1c2c3(Utilde)
        F1 = np.exp( np.pi * 0.5j * ( c1 - c2 + c3) )
        F2 = np.exp( np.pi * 0.5j * ( c1 + c2 - c3) )
        F3 = np.exp( np.pi * 0.5j * (-c1 - c2 - c3) )
        F4 = np.exp( np.pi * 0.5j * (-c1 + c2 + c3) )
        Fd = np.array([F1, F2, F3, F4])
        F  = np.matrix(np.diag(Fd))

        # Furthermore, we have Eq (22), giving the eigen-decomposition of the
        # matrix m. This gives us the matrix O_2.T of the eigenvectors of m
        Fsq, O_2_transposed = np.linalg.eig(m)

        ord1 = np.argsort(np.angle(Fd**2)) # sort according to complex phase
        ord2 = np.argsort(np.angle(Fsq))   # ... (absolute value is 1)
        diff = np.sum( np.abs( (Fd**2)[ord1] - Fsq[ord2] ) )
        # Do Fd**2 and Fsq contain the same values (irrespective of order)?
        if  diff < 1.0e-12:
            found_branch = True
            break
        else:
            Utilde *= 1.0j

    # double check that we managed to find a branch (just to be 100% safe)
    assert(found_branch), \
    "Couldn't find correct branch of fourth root in mapping U(4) -> SU(4)"

    # Getting the entries of F from Eq (21) instead of by taking the square
    # root of Fsq has the benefit that we don't have to worry about whether we
    # need to take the positive or negative root.
    # However, we do need to make sure that the eigenstates are ordered to
    # correspond to F1, F2, F3, F4
    # After reordering, we need to transpose to get O_2 itself
    reordered = np.matrix(np.zeros((4,4)), dtype=np.complex128)
    order = []
    for i in xrange(4):
        for j in xrange(4):
            if (abs(Fd[i]**2 - Fsq[j]) < 1.0e-12):
                if not j in order:
                    order.append(j)
    assert len(order) == 4, "Couldn't order O_2" # should not happen
    # Reorder using the order we just figured out, and transpose
    for i in xrange(4):
        reordered[:,i] = O_2_transposed[:,order[i]]
    O_2 = reordered.T

    # Now that we have O_2 and F, completing the Cartan decomposition is
    # straightforward, following along Appendix E of Daniel's thesis
    k2 = from_magic(O_2)
    O_1 = UB * O_2.T * F.H
    k1 = from_magic(O_1)
    A = np.matrix(scipy.linalg.expm(np.pi*0.5j * (c1*SxSx +c2*SySy + c3*SzSz)))

    # Check our results
    from . gate2q import identity
    assert( np.max(np.abs(O_1*O_1.T - identity)) < 1.0e-12 ), \
    "O_1 not orthogonal"
    assert( np.max(np.abs(O_2*O_2.T - identity)) < 1.0e-12 ), \
    "O_2 not orthogonal"
    assert( np.max(np.abs((k1*A*k2 - Utilde))) < 1.0e-12 ), \
    "Cartan Decomposition Failed"

    return k1, A, k2
