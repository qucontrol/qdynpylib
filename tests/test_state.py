from __future__ import print_function, division, absolute_import

from QDYN.linalg import norm
from QDYN.state import read_psi_amplitudes, write_psi_amplitudes
from cmath import sqrt
import pytest
import numpy as np

# built-in fixtures: tmpdir

def test_state_read_write(tmpdir):
    data_str = r'''# index             Re[Psi]             Im[Psi]
      1  1                   0
      3  0.7071067811865476  0.7071067811865476'''
    tmpdir.join('psi.in').write(data_str)
    psi1 = read_psi_amplitudes(str(tmpdir.join('psi.in')), n=3)
    write_psi_amplitudes(psi1, str(tmpdir.join('psi.out')))
    psi2 = read_psi_amplitudes(str(tmpdir.join('psi.out')), n=3)
    assert norm(psi1 - psi2) <= 1e-15
    psi2 = np.array([(1/sqrt(2)), 0, (0.5+0.5j)])
    assert norm(psi1 - psi2) <= 1e-15
