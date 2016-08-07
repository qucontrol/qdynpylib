"""Tests for QDYN.model"""

import os
import filecmp
from functools import partial

import numpy as np

from QDYN.model import LevelModel
from QDYN.pulse import Pulse, blackman
from QDYN.io import read_indexed_matrix
from QDYN.linalg import norm
from QDYN.state import read_psi_amplitudes
# built-in fixtures: tmpdir, request

def test_level_model(tmpdir, request):
    """Test a simple two-qubit model"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    MHz = 2 * np.pi * 1e-3
    w1 = 0   * MHz
    w2 = 500 * MHz
    kappa1 = 0.01  * MHz
    kappa2 = 0.012 * MHz
    H0 = np.array(np.diag([0, w2, w1, w1+w2]), dtype=np.complex128)
    H1 = np.array(
         [[0, 1, 1, 0],
          [1, 0, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0]], dtype=np.complex128)
    pulse = partial(blackman, t_start=0, t_stop=50)
    L1 = np.sqrt(kappa1) * np.array(
         [[0, 0, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]], dtype=np.complex128)
    L2 = np.sqrt(kappa2) * np.array(
         [[0, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]], dtype=np.complex128)

    pop1 = np.array(
         [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]], dtype=np.complex128)
    pop2 = np.array(
         [[0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1]], dtype=np.complex128)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)

    model = LevelModel()
    model.add_ham(H0)
    model.add_ham(H1, pulse)
    model.add_lindblad_op(L1)
    model.add_lindblad_op(L2)
    model.set_propagation(psi, T=50, nt=1001, time_unit='ns', use_mcwf=True)
    model.add_observable(pop1, 'pops.dat', 'unitless', 'ns', '<P_1> (q1)')
    model.add_observable(pop2, 'pops.dat', 'unitless', 'ns', '<P_1> (q2)')
    model.write_to_runfolder(str(tmpdir.join('model_rf')))

    print(str(tmpdir.join('model_rf')))
    assert filecmp.cmp(os.path.join(test_dir, 'config'),
                       str(tmpdir.join('model_rf', 'config')), shallow=False)

    def file_matches_matrix(filename, matrix, limit=0.0):
        """Check that the indexed matrix in `filename` matches `matrix`, up to
        `limit`"""
        file_matrix = read_indexed_matrix(
                            str(tmpdir.join('model_rf', filename)),
                            expand_hermitian=False, shape=(4,4))
        return norm(file_matrix - matrix) <= limit

    def file_matches_psi(filename, psi, limit=0.0):
        """Check that the psi in `filename` matches `psi`, up to `limit`"""
        file_psi = read_psi_amplitudes(
                            str(tmpdir.join('model_rf', filename)), n=4)
        return norm(file_psi - psi) <= limit

    assert file_matches_matrix('H0.dat', H0)
    assert file_matches_matrix('H1.dat', H1)
    assert file_matches_matrix('L1.dat', L1)
    assert file_matches_matrix('L2.dat', L2)
    assert file_matches_matrix('O1.dat', pop1)
    assert file_matches_matrix('O2.dat', pop2)
    assert file_matches_psi('psi0.dat', psi, limit=1e-15)
    num_pulse = Pulse.read(str(tmpdir.join('model_rf', 'pulse1.dat')))
    assert num_pulse.time_unit == 'ns'
    assert num_pulse.T == '50_ns'
    expected_amplitude = blackman(num_pulse.tgrid, t_start=float(num_pulse.t0),
                                  t_stop=float(num_pulse.T))
    assert np.max(np.abs(num_pulse.amplitude - expected_amplitude)) < 1e-15

