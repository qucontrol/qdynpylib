"""Test QDYN.dissipation module"""
from scipy import sparse
from QDYN.dissipation import lindblad_ops_to_dissipator


def test_tls_diss_sup():
    """Test dissipation operator for decay and dephasing in a TLS

    See http://nbviewer.jupyter.org/gist/goerz/865eb143cc615676ee76f257ae7459ee
    for analytical solution
    """
    import numpy as np
    from numpy import sqrt
    half = 0.5
    gamma1 = 1.5
    gamma2 = 1.2

    def LocalSigma(i, j):
        return sparse.coo_matrix(([1, ], ([i, ], [j, ])), shape=(2, 2))

    L1 = sqrt(gamma1) * LocalSigma(0, 1)
    L2 = sqrt(2*gamma2) * LocalSigma(1, 1)
    L = [L1, L2]

    DSup = lindblad_ops_to_dissipator(L)
    assert np.max(np.abs(DSup.todense() - np.array([
        [0,                       0,                       0,  gamma1],
        [0, -half * gamma1 - gamma2,                       0,       0],
        [0,                       0, -half * gamma1 - gamma2,       0],
        [0,                       0,                       0, -gamma1]
    ]))) < 1e-12
