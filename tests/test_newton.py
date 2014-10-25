from math import pi
from functools import partial
import QDYN
import numpy as np

WRITE_FILES = False # Set to True to write out populations and debug files

def write_v(v, outfile, write_size=False):
    if not WRITE_FILES:
        return
    with open(outfile, 'w') as out:
        if write_size:
            print >> out, "%d" % len(v)
        for val in v:
            print >> out, "%25.17E%25.17E" % (val.real, val.imag)


def plot_pops(rho, t, outfile):
    """
    Given rho(t), write the populations of each level, the trace, and the
    purity to the given outfile
    """
    if not WRITE_FILES:
        return
    N = rho.shape[0]
    if (t == 0.0):
        # Create a new file and write header
        with open(outfile, 'w') as out:
            header = "# %10s" % "time [ns]"
            for level in xrange(N):
                header += "%25s" % ("pop(%d)" % (level+1))
            header += "%25s" % ("trace")
            header += "%25s" % ("purity")
            print >> out, header
    # Append data to existing file
    with open(outfile, 'a') as out:
        datline = "%12.8f" % t
        for level in xrange(N):
            datline += "%25.17e" % np.abs(rho[level,level])
        datline += "%25.17e" % QDYN.linalg.trace(rho).real
        datline += "%25.17e" % QDYN.linalg.trace(np.dot(rho,rho)).real
        print >> out, datline


def generate_liouvillian(N, w_c, gamma, E0, tgrid):
    """
    Generator for apply_L Routine
    """

    t_start = tgrid[0]
    t_stop  = tgrid[-1]

    # Drift Hamiltonian
    H0 = np.matrix(np.zeros(shape=(N,N), dtype=np.complex128))
    for i in xrange(N):
        H0[i,i] = i * w_c #* GHz2au

    # Dissipator
    a = np.matrix(np.zeros(shape=(N,N), dtype=np.complex128))
    for i in xrange(1,N):
        a[i-1, i] = np.sqrt(i)

    def apply_L(rho, t):
        """ Calculate Lrho = L(t)[rho] """

        H1 = np.matrix(np.zeros(shape=(N,N), dtype=np.complex128))
        for i in xrange(1,N):
            H1[i-1,i] = np.sqrt(float(i)) \
                        * E0 * QDYN.pulse.blackman(t, t_start, t_stop)
            H1[i,i-1] = H1[i-1,i]
        H = H0 + H1

        vector_mode = False
        assert(len(rho.shape) == 2), \
        "rho must be NxN matrix or N**2 x 1 matrix"
        if (rho.shape[1] == 1):
            vector_mode = True
            rho = np.matrix(np.reshape(rho, (N,N), 'F'))

        # Evaluate RHS Liouville-von-Neumann equation
        Lrho = (H*rho - rho*H) # Commutator [H,rho]
        Lrho += 1.j * gamma \
                * (a * rho * a.H - 0.5 * (a.H * a * rho + rho * a.H * a))

        if vector_mode:
            return np.reshape(np.asarray(Lrho), (N**2,1), 'F') # N**2 x 1 matrix
        else:
            return Lrho

    return apply_L


def test_newton():

    print "*** Running newton_test"

    two_pi = 2.0 * pi

    # Parameters
    tgrid = np.linspace(0, 20, 100) # ns
    #tgrid = np.linspace(0, 0.1, 2) # ns
    N = 10
    w_c   = two_pi * 0.0       # GHz
    gamma = two_pi * 0.1       # GHZ
    E0 = 0.1                   # GHZ
    m = 3
    maxrestart = 500
    tol = 1.0e-13

    # convert to atomic units
    #w_c   *= 2.4188843274E-8
    #gamma *= 2.4188843274E-8
    #E0    *= 2.4188843274E-8
    #tgrid *=   41341373.0

    # initial state
    rho0 = np.matrix(np.zeros(shape=(N,N), dtype=np.complex128))
    #rho0[N-1,N-1] = 1.0
    rho0[:,:] = 1.0/float(N)
    #for diag in xrange(N):
        #rho0[diag,diag] = 1./N
    print "Trace of initial state: ", QDYN.linalg.trace(rho0).real

    # Liouvillian
    apply_L = generate_liouvillian(N=N, w_c=w_c, gamma=gamma,
                                   E0=E0, tgrid=tgrid)

    # Test simple application of Liouvillian first
    print ""
    print "**** Application of Liouvillian ****"
    print ""
    ti = len(tgrid) / 2
    print "ti = ", ti
    dt = tgrid[1] - tgrid[0]
    t = (float(ti)-0.5) * dt
    print "t = ", t
    pulse_value = E0 * QDYN.pulse.blackman(t, tgrid[0], tgrid[-1])
    print "pulse value = ", pulse_value
    rho = apply_L(rho0, t)
    print "Norm of output state:", QDYN.linalg.norm(rho)
    write_v(np.squeeze(np.reshape(np.asarray(rho), (1,N*N) )),
            "rho_%d.dat" % ti)
    L = QDYN.linalg.get_op_matrix(apply_L, rho0.shape, t)
    if WRITE_FILES:
        with open("liouvillian.dat", 'w') as out:
            for j in xrange(N**2):
                for i in xrange(N**2):
                    if (L[i,j] != 0.0):
                        print >> out, "%5d%5d%25.17E%25.17E" \
                        % (i+1, j+1, L[i,j].real, L[i,j].imag)

    #  propagate and write out population dynamics
    print ""
    print "**** Exact Propagation ****"
    print ""
    rho = rho0.copy()
    exact_storage = []

    rho = QDYN.prop.propagate(apply_L, rho, tgrid, 'exact',
          info_hook=partial(plot_pops, outfile="pops_%s.dat" % 'exact'),
          storage=exact_storage)
    print "Trace of propagated state: ", QDYN.linalg.trace(rho).real

    print ""
    print "**** Newton Propagation ****"
    print ""
    rho = rho0.copy()
    newton_storage = []
    rho = QDYN.prop.propagate(apply_L, rho, tgrid, 'newton',
          info_hook=partial(plot_pops, outfile="pops_%s.dat" % 'newton'),
          storage=newton_storage, m=m, maxrestart=maxrestart, tol=tol)
    print "Trace of propagated state: ", QDYN.linalg.trace(rho).real

    print ""
    print "Comparison of Newton and Exact Solution"
    print ""

    # compare propagated states
    for i, (rho_newton, rho_exact) \
    in enumerate(zip(newton_storage, exact_storage)):
        t = tgrid[i + 1]
        diff = QDYN.linalg.norm(rho_newton - rho_exact)
        print "t = %g, diff: %g" % (t, diff)
        if diff > 1.0e-12:
            raise AssertionError(
            "Newton propagation and exact propagation do not match")

    # cleanup
    if WRITE_FILES:
        files = ['liouvillian.dat', 'pops_exact.dat', 'pops_newton.dat',
                'pulse.out', 'rho_50.dat']
        for file in files:
            try:
                os.unlink(file)
            except:
                pass
