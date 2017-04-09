"""Test the QDYN.prop_gate module"""
import os

import numpy as np

import QDYN
from QDYN.prop_gate import get_prop_gate_of_t
#
# buitin fixtures: request, tmpdir


def test_get_prop_gate_of_t(request):
    """Test reading for file with prop_gate_of_t routine"""
    datadir = os.path.splitext(request.module.__file__)[0]
    infile = os.path.join(datadir, 'U_of_t.dat')
    for gate in get_prop_gate_of_t(infile):
        assert isinstance(gate, QDYN.gate2q.Gate2Q)
    for gate, t in get_prop_gate_of_t(infile, with_t=True):
        assert isinstance(t, QDYN.units.UnitFloat)
        assert t.unit == 'microsec'
        assert isinstance(gate, QDYN.gate2q.Gate2Q)
    tgrid1 = np.array([float(t) for (gate, t)
                       in get_prop_gate_of_t(infile, with_t=True)])
    tgrid2 = np.genfromtxt(infile, usecols=(0, ))
    assert np.all(np.abs(tgrid1 - tgrid2) < 1e-15)
