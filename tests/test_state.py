from __future__ import print_function, division, absolute_import

from QDYN.linalg import norm
from QDYN.state import (
    read_psi_amplitudes, write_psi_amplitudes, iterate_psi_amplitudes)
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


def test_read_psi_blocks(tmpdir):
    data_str = r'''
# index             Re[Psi]             Im[Psi]
      1  1                   0


# index             Re[Psi]             Im[Psi]
      1  1                   0
      3  1.0000000000000000  0.0000000000000000


# index             Re[Psi]             Im[Psi]
      1  1                   0
      3  0.7071067811865476  0.7071067811865476
      '''.strip()
    tmpdir.join('psi_blocks.in').write(data_str)
    data_file_name = str(tmpdir.join('psi_blocks.in'))
    psi1, psi2, psi3 = iterate_psi_amplitudes(data_file_name, n=3)
    assert np.max(np.abs(psi1 - psi3)) > 0.5
    assert np.max(np.abs(psi1 - psi2)) > 0.5
    assert len(list(
        iterate_psi_amplitudes(data_file_name, n=3, start_from_block=2))) == 2
    phi2, phi3 = iterate_psi_amplitudes(
        data_file_name, n=3, start_from_block=2)
    assert np.max(np.abs(psi2 - phi2)) < 1e-14
    assert np.max(np.abs(psi3 - phi3)) < 1e-14
    for psi in iterate_psi_amplitudes(data_file_name, n=3, normalize=True):
        assert abs(1.0 - norm(psi)) < 1e-14
    for i, psi in enumerate(
            iterate_psi_amplitudes(data_file_name, n=3, normalize=False)):
        if i == 0:
            assert abs(1.0 - norm(psi)) < 1e-14
        else:
            assert abs(1.0 - norm(psi)) > 0.1
    phi3 = read_psi_amplitudes(data_file_name, n=3, block=3)
    assert np.max(np.abs(psi3 - phi3)) < 1e-14
    assert abs(1.0 - norm(phi3)) < 1e-14
    phi3 = read_psi_amplitudes(data_file_name, n=3, block=3, normalize=False)
    assert abs(np.sqrt(2.0) - norm(phi3)) < 1e-14
    with pytest.raises(ValueError):
        read_psi_amplitudes(data_file_name, n=3, block=4)
    with pytest.raises(ValueError):
        for psi in iterate_psi_amplitudes(
                data_file_name, n=3, start_from_block=4):
            pass
