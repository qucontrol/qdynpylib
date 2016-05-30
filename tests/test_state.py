from __future__ import print_function, division, absolute_import

import QDYN
from cmath import sqrt
import pytest
import numpy as np

# built-in fixtures: tmpdir

def test_state_read_write(tmpdir):
    data_str = r'''# index             Re[Psi]             Im[Psi]
      1  1                   0
      3  0.7071067811865476  0.7071067811865476'''
    tmpdir.join('psi.in').write(data_str)
    state1 = QDYN.state.State.read(str(tmpdir.join('psi.in')), n=3)
    state1.write(str(tmpdir.join('psi.out')))
    state2 = QDYN.state.State.read(str(tmpdir.join('psi.out')), n=3)
    assert (QDYN.linalg.norm(state1.psi - state2.psi) <= 1e-15)
    state2 = QDYN.state.State(data=np.array([(1/sqrt(2)), 0, (0.5+0.5j)]))
    assert (QDYN.linalg.norm(state1.psi - state2.psi) <= 1e-15)
