"""
Routines for calculating local invariants, concurrence, and related quantities
for two-qubit gates
"""
import numpy as np
import scipy.linalg

Qmagic = (1.0/np.sqrt(2.0)) * np.matrix(
                   [[ 1,  0,  0,  1j],
                    [ 0, 1j,  1,  0],
                    [ 0, 1j, -1,  0],
                    [ 1,  0,  0, -1j]], dtype=np.complex128)

CNOT   = np.matrix([[ 1,  0,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0,  0,  0,  1],
                    [ 0,  0,  1,  0]], dtype=np.complex128)

CPHASE  = np.matrix([[ 1,  0,  0,   0],
                    [  0,  1,  0,   0],
                    [  0,  0,  1,   0],
                    [  0,  0,  0,  -1]], dtype=np.complex128)

SWAP    = np.matrix([[ 1,  0,  0,   0],
                    [  0,  0,  1,   0],
                    [  0,  1,  0,   0],
                    [  0,  0,  0,   1]], dtype=np.complex128)

DCNOT   = np.matrix([[ 1,  0,  0,   0],
                    [  0,  0, 1j,   0],
                    [  0, 1j,  0,   0],
                    [  0,  0,  0,   1]], dtype=np.complex128)
iSWAP = DCNOT

unity  = np.matrix([[ 1,  0,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0,  0,  1,  0],
                    [ 0,  0,  0,  1]], dtype=np.complex128)

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

SQRT2 = np.sqrt(2.0)

QGATE   = np.matrix([[ 1,           0,          0, 0],
                    [  0,  1.0 /SQRT2, 1.0j/SQRT2, 0],
                    [  0,  1.0j/SQRT2, 1.0 /SQRT2, 0],
                    [  0,           0,          0, 1]], dtype=np.complex128)
sqrt_iSWAP = QGATE

COSPI8 = np.cos(np.pi / 8.0)
SINPI8 = np.sin(np.pi / 8.0)
BGATE = (1.0/np.sqrt(2.0)) * np.matrix(
  [[ COSPI8+SINPI8*1j,                 0,                 0, SINPI8-COSPI8*1j],
   [                0,  COSPI8-SINPI8*1j,  SINPI8+COSPI8*1j,                0],
   [                0,  SINPI8+COSPI8*1j,  COSPI8-SINPI8*1j,                0],
   [ SINPI8-COSPI8*1j,                 0,                 0, COSPI8+SINPI8*1j]
  ], dtype=np.complex128)

BGATE2 = np.matrix( # alternative BGATE
  [[ COSPI8  ,           0,         0, SINPI8*1j],
   [         0,     SINPI8, COSPI8*1j,         0],
   [         0,  COSPI8*1j,    SINPI8,         0],
   [ SINPI8*1j,          0,         0,    COSPI8]
  ], dtype=np.complex128)

sqrt_SWAP = np.matrix(np.zeros(shape=(4,4)), dtype=np.complex128)
sqrt_SWAP[0,0] = np.exp(1j * np.pi / 8.0)
sqrt_SWAP[1,1] = 0.5 * ((COSPI8+SINPI8) + 1j * (SINPI8-COSPI8))
sqrt_SWAP[1,2] = 0.5 * ((COSPI8-SINPI8) + 1j * (SINPI8+COSPI8))
sqrt_SWAP[2,1] = sqrt_SWAP[1,2]
sqrt_SWAP[2,2] = sqrt_SWAP[1,1]
sqrt_SWAP[3,3] = sqrt_SWAP[0,0]

NGATE = np.matrix(
  [[ 0.5*(COSPI8+SINPI8 + 1j*(COSPI8-SINPI8)),                 0,                 0, 0.5*(SINPI8-COSPI8 + 1j*(SINPI8+COSPI8))],
   [                                        0,                 0,  SINPI8+COSPI8*1j,                                        0],
   [                                        0,  SINPI8+COSPI8*1j,                 0,                                        0],
   [ 0.5*(SINPI8-COSPI8 + 1j*(SINPI8+COSPI8)),                 0,                 0, 0.5*(COSPI8+SINPI8 + 1j*(COSPI8-SINPI8))]
  ], dtype=np.complex128)

NGATE2    = np.matrix([[ 1,           0,           0,   0],
                      [  0,  0.5*(1+1j),  0.5*(1-1j),   0],
                      [  0,  0.5*(1-1j),  0.5*(1+1j),   0],
                      [  0,           0,           0,   1]],
                      dtype=np.complex128)

# Note: NGATE and NGATE2 are also sqrt_SWAP gates (in the sense the they yield
# SWAP if executed twice)



def g1g2g3(U):
    """
    Given Numpy matrix U, calculate local invariants g1,g2,g3
    U must be in the canonical basis

    >>> "%.2f %.2f %.2f" % g1g2g3(CNOT)
    '0.00 0.00 1.00'
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

    >>> "%.2f %.2f %.2f" % c1c2c3(CNOT)
    '0.50 0.00 0.00'
    """
    U_tilde= SySy * U.transpose() * SySy
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

    >>> "%.2f %.2f %.2f" % g1g2g3_from_c1c2c3(*c1c2c3(CNOT))
    '0.00 0.00 1.00'
    """
    c1 *= np.pi
    c2 *= np.pi
    c3 *= np.pi
    g1 = np.cos(c1)**2 * np.cos(c2)**2 * np.cos(c3)**2 \
       - np.sin(c1)**2 * np.sin(c2)**2 * np.sin(c3)**2     + 0.0
    g2 = 0.25 * np.sin(2*c1) * np.sin(2*c2) * np.sin(2*c3) + 0.0
    g3 = 4*g1 - np.cos(2*c1) * np.cos(2*c2) * np.cos(2*c3) + 0.0
    return g1, g2, g3


def FPE(g1, g2, g3):
    """
    Evaluate the Perfect-Entangler Functional

    >>> FPE(*g1g2g3(CNOT))
    0.0
    >>> FPE(*g1g2g3(unity))
    2.0
    """
    return g3 * np.sqrt(g1**2 + g2**2) - g1 + 0.0


def point_in_weyl_chamber(c1, c2, c3):
    """
    Return True if the coordinates c1, c2, c3 are inside the Weyl chamber

    >>> point_in_weyl_chamber(*c1c2c3(BGATE))
    True
    >>> point_in_weyl_chamber(*c1c2c3(unity))
    True
    """
    return ( ((c1 < 0.5)  and (c2 <= c1)   and (c3 <= c2))
          or ((c1 >= 0.5) and (c2 <= 1-c1) and (c3 <= c2)) )


def point_in_PE(c1, c2, c3):
    """
    Return True if the coordinates c1, c2, c3 are inside the perfect-entangler
    polyhedron

    >>> point_in_PE(*c1c2c3(BGATE))
    True
    >>> point_in_PE(*c1c2c3(unity))
    False
    """
    if point_in_weyl_chamber(c1, c2, c3):
        return ( ((c1 + c2) >= 0.5) and (c1-c2 <= 0.5) and ((c2+c3) <= 0.5))
    else:
        return False


def get_F_avg_hilbert(O, U):
    """ Calculate the average gate fidelity of Hilbert space U with respect to
        (unitary) optimal gate O.

        U and O must be numpy Matrices, in canonical basis
    """
    n = O.shape[0]
    overlap = abs((O.H * U).trace()[0,0])**2 / n**2
    F_U_avg = (float(n*n) / float(n*(n+1))) * overlap
    F_TW_avg = 0
    for i in xrange(n):
        for j in xrange(n):
            O_dagger_U = 0.0
            for k in xrange(n):
                O_dagger_U += O[k,i].conj() * U[k,j]
            F_TW_avg += abs(O_dagger_U)**2
    F_TW_avg = F_TW_avg / float(n*(n+1))
    return F_TW_avg + F_U_avg


def concurrence(U):
    """
    Given U (canonical basis), calculate the maximum concurrence it
    generates.

    >>> round(concurrence(SWAP), 2)
    0.0
    >>> round(concurrence(CNOT), 2)
    1.0
    >>> round(concurrence(unity), 2)
    0.0
    """
    return concurrence_c1c2c3(*c1c2c3(U))


def concurrence_c1c2c3(c1, c2, c3):
    """
    Calculate the concurrence directly from the Weyl Chamber coordinates c1,
    c2, c3

    >>> round(concurrence_c1c2c3(*c1c2c3(SWAP)), 2)
    0.0
    >>> round(concurrence_c1c2c3(*c1c2c3(CNOT)), 2)
    1.0
    >>> round(concurrence_c1c2c3(*c1c2c3(unity)), 2)
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


def delta_uni(A):
    """ Assuming that A is a matrix obtained from projecting a unitary matrix
        to a smaller subspace, return the loss of population of subspace, as a
        distance measure of A from unitarity.

        Result is in [0,1], with 0 if A is unitary.
    """
    return 1.0 - (A.H * A).trace()[0,0].real / A.shape[0] + 0.0


def closest_unitary(A):
    """ Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.

        Return (U, d) where U is the closest unitary as a numpy matrix and d is
        the 2-norm operator distance between A and U
    """
    V, S, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    d = 0.0
    for s in S:
        if abs(s - 1.0) > d:
            d = abs(s-1.0)
    return U, d


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


def plot_weyl_chamber(points=None, labels=None, colors='red',
    weyl_alpha=0.1, PE_alpha=0.1, linecolor='black', bgcolor=None,
    pointsize=50, pointedge=True):
    """
    Show a plot of the Weyl chamber and the perfect-entangler polyhedron.
    Optionall, plot a number of points (given as c1, c2, c3 coordinates) with
    labels

    Parameters
    ----------

    points : array
        Array of coordinate tuples (c1, c2, c3)
    labels : None, array
        If not None, must be an array of the same size as points, containing a
        string label for each point (empty labels are allowed, of course)
        If None, the points will not be shown with labels
    colors : array
        If not None, color for each point, or single color do be used for each
        point
    weyl_alpha : float
        Alpha value (transparency) of the faces deliminating the Weyl Chamber
    PE_alpha : float
        Alpha value for the perfect-entangler polyhedron
    linecolor : str, tuple
        Color of lines making up the Weyl chamber and PE polyhedron. The color
        is given either as a string or as an (r, g, b, a) tuple
    bgcolor : None, tuple
        Color of background. If not None, must be (r, g, b, a) tuple. Color
        names are not accepted
    pointsize : int
        Size of a points
    pointedge : bool
        If False, do not draw a black edge around each point

    Reference
    ---------

    Zhang et al, PRA 67, 042313 (2003)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A1 = (1, 0, 0)
    A2 = (0.5, 0.5, 0)
    A3 = (0.5, 0.5, 0.5)
    O  = (0, 0, 0)
    L  = (0.5, 0, 0)
    M  = (0.75, 0.25, 0)
    N  = (0.75, 0.25, 0.25)
    P  = (0.25, 0.25, 0.25)
    Q  = (0.25, 0.25, 0)
    weyl_points = {'A1' : A1, 'A2' : A2, 'A3' : A3, 'O' : O, 'L' : L, 'M' : M,
                    'N' : N, 'P' : P, 'Q': Q}
    weyl_faces = []
    weyl_faces.append([[A1, A1, A3]])
    weyl_faces.append([[A1, A2, O]])
    weyl_faces.append([[A2, O, A3]])
    weyl_faces.append([[A1, O, A3]])
    for face in weyl_faces:
        pol = Poly3DCollection(face)
        pol.set_facecolor((0, 0, 1, weyl_alpha))
        pol.set_edgecolor(linecolor)
        ax.add_collection3d(pol)
    PE_faces = []
    PE_faces.append([[L, M, N]])
    PE_faces.append([[M, A2, N]])
    PE_faces.append([[L, M, A2, Q]])
    PE_faces.append([[A2, P, Q]])
    PE_faces.append([[N, P, A2]])
    PE_faces.append([[N, L, P]])
    PE_faces.append([[L, P, Q]])
    for face in PE_faces:
        pol = Poly3DCollection(face)
        pol.set_facecolor((0, 0, 0, PE_alpha))
        pol.set_edgecolor(linecolor)
        ax.add_collection3d(pol)
    ax.scatter(*zip(*weyl_points.values()))
    for label, coords in weyl_points.items():
        ax.text(coords[0], coords[1], coords[2], label, color=linecolor)
    ax.set_xlabel('$c_1/\\pi$')
    ax.set_ylabel('$c_2/\\pi$')
    ax.set_zlabel('$c_3/\\pi$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,0.5)
    ax.set_zlim(0,0.5)
    if bgcolor is not None:
        ax.w_xaxis.set_pane_color(bgcolor)
        ax.w_yaxis.set_pane_color(bgcolor)
        ax.w_zaxis.set_pane_color(bgcolor)
    if points is not None:
        if pointedge:
            ax.scatter(*zip(*points), c=colors, s=pointsize)
        else:
            ax.scatter(*zip(*points), c=colors, edgecolors='None', s=pointsize)
    if labels is not None:
        assert(len(labels) == len(points)), \
        "If labels are given, there must be a label for every point"
        for i, label in enumerate(labels):
            c1 = points[i][0]
            c2 = points[i][1]
            c3 = points[i][2]
            ax.text(c1, c2, c3, label, color=linecolor)
    plt.show()


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
    assert( np.max(np.abs(O_1*O_1.T - unity)) < 1.0e-12 ), "O_1 not orthogonal"
    assert( np.max(np.abs(O_2*O_2.T - unity)) < 1.0e-12 ), "O_2 not orthogonal"
    assert( np.max(np.abs((k1*A*k2 - Utilde))) < 1.0e-12 ), \
    "Cartan Decomposition Failed"

    return k1, A, k2
