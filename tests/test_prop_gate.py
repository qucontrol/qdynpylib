"""Test the QDYN.prop_gate module"""
import os

import numpy as np

import qdyn
from qdyn.prop_gate import get_prop_gate_of_t


# builtin fixtures: request, tmpdir


def test_get_prop_gate_of_t(request):
    """Test reading for file with prop_gate_of_t routine"""
    datadir = os.path.splitext(request.module.__file__)[0]
    infile = os.path.join(datadir, 'U_of_t.dat')
    for (_, t) in get_prop_gate_of_t(infile, with_t=True):
        assert isinstance(t, qdyn.units.UnitFloat)
        assert t.unit == 'microsec'
    tgrid1 = np.array(
        [float(t) for (gate, t) in get_prop_gate_of_t(infile, with_t=True)]
    )
    tgrid2 = np.genfromtxt(infile, usecols=(0,))
    assert np.all(np.abs(tgrid1 - tgrid2) < 1e-15)
