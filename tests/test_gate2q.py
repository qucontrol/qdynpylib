"""Test the QDYN.gate2q module"""
import os

import QDYN
#
# buitin fixtures: request, tmpdir


def test_read(request):
    """Test read a two-qubit gate from file"""
    datadir = os.path.splitext(request.module.__file__)[0]

    infile = os.path.join(datadir, 'CNOT_matrix.dat')
    CNOT = QDYN.gate2q.Gate2Q.read(infile, name='CNOT', format='matrix')
    assert QDYN.linalg.norm(CNOT - QDYN.gate2q.CNOT) < 1e-15

    infile = os.path.join(datadir, 'CNOT_array.dat')
    CNOT = QDYN.gate2q.Gate2Q.read(infile, name='CNOT', format='array')
    assert QDYN.linalg.norm(CNOT - QDYN.gate2q.CNOT) < 1e-15
