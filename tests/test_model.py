"""Tests for QDYN.model"""

import os
import filecmp
from collections import OrderedDict
from functools import partial

import numpy as np
import pytest

from QDYN.model import LevelModel
from QDYN.pulse import Pulse, blackman
from QDYN.analytical_pulse import AnalyticalPulse
from QDYN.io import read_indexed_matrix
from QDYN.linalg import norm
from QDYN.state import read_psi_amplitudes
# built-in fixtures: tmpdir, request


@pytest.fixture
def H0():
    MHz = 2 * np.pi * 1e-3
    w1 = 0   * MHz
    w2 = 500 * MHz
    return np.array(np.diag([0, w2, w1, w1+w2]), dtype=np.complex128)


@pytest.fixture
def H1():
    return np.array(
         [[0, 1, 1, 0],
          [1, 0, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0]], dtype=np.complex128)


@pytest.fixture
def L1():
    MHz = 2 * np.pi * 1e-3
    kappa1 = 0.01 * MHz
    return np.sqrt(kappa1) * np.array(
         [[0, 0, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]], dtype=np.complex128)


@pytest.fixture
def L2():
    MHz = 2 * np.pi * 1e-3
    kappa2 = 0.012 * MHz
    return np.sqrt(kappa2) * np.array(
         [[0, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]], dtype=np.complex128)


@pytest.fixture
def pop1():
    return np.array(
         [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]], dtype=np.complex128)


@pytest.fixture
def pop2():
    return np.array(
         [[0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1]], dtype=np.complex128)


def two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi):
    """Construct model for two-qubit system"""
    # Not a fixture!
    model = LevelModel()
    model.add_ham(H0)
    model.add_ham(H1, pulse)
    model.add_lindblad_op(L1)
    model.add_lindblad_op(L2)
    model.set_propagation(psi, T=50, nt=1001, time_unit='ns', use_mcwf=True)
    model.add_observable(pop1, 'pops.dat', 'unitless', 'ns', '<P_1> (q1)')
    model.add_observable(pop2, 'pops.dat', 'unitless', 'ns', '<P_1> (q2)')
    return model


def test_level_model(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test a simple two-qubit model"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = partial(blackman, t_start=0, t_stop=50)
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    model.write_to_runfolder(str(tmpdir.join('model_rf')))

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
    assert model.ham() == [H0, H1]
    assert model.ham(with_attribs=True) == [
            (H0, OrderedDict([])),
            (H1, OrderedDict([('pulse_id', 1)]))
    ]
    assert model.observables() == [pop1, pop2]
    assert model.observables(with_attribs=True) == [
            (pop1, OrderedDict([('outfile', 'pops.dat'),
                                ('exp_unit', 'unitless'), ('is_real', True),
                                ('time_unit', 'ns'),
                                ('column_label', '<P_1> (q1)'),
                                ('op_unit', 'unitless')])),
            (pop2, OrderedDict([('outfile', 'pops.dat'),
                                ('exp_unit', 'unitless'), ('is_real', True),
                                ('time_unit', 'ns'),
                                ('column_label', '<P_1> (q2)'),
                                ('op_unit', 'unitless')]))
    ]
    assert model.lindblad_ops() == [L1, L2]
    assert model.lindblad_ops(with_attribs=True) == [
            (L1, OrderedDict([('add_to_H_jump', 'banded'),
                              ('conv_to_superop', False)])),
            (L2, OrderedDict([('add_to_H_jump', 'banded'),
                              ('conv_to_superop', False)]))
    ]
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


def test_target_psi(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test that we can add another 'target' state to the config file"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = partial(blackman, t_start=0, t_stop=50)
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    model.add_state(np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0),
                    label='target')
    model.write_to_runfolder(str(tmpdir.join('model_rf')),
                             config='target.config')

    assert filecmp.cmp(os.path.join(test_dir, 'target.config'),
                       str(tmpdir.join('model_rf', 'target.config')),
                       shallow=False)


def test_ensemble(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test ensemble of multiple Hamiltonians"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    pulse = partial(blackman, t_start=0, t_stop=50)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = partial(blackman, t_start=0, t_stop=50)
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    for i in range(5):
        r1 = (np.random.rand() - 0.5) * 0.01
        r2 = (np.random.rand() - 0.5) * 0.01
        ens = "ens%d" % (i+1)
        H0_fn = "H0_%s.dat" % ens
        H1_fn = "H1_%s.dat" % ens
        model.add_ham(H0+r1*H0, label=ens, filename=H0_fn)
        model.add_ham(H1+r2*H1, pulse, label=ens, filename=H1_fn)

    assert len(model.ham()) == 2
    assert len(model.ham(label='ens1')) == 2
    model.write_to_runfolder(str(tmpdir.join('model_rf')),
                             config='ensemble.config')
    assert filecmp.cmp(os.path.join(test_dir, 'ensemble.config'),
                       str(tmpdir.join('model_rf', 'ensemble.config')),
                       shallow=False)


def test_ensemble_shared_pulse(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test ensemble of multiple Hamiltonians that all share the same pulse"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    class MyAP(AnalyticalPulse):
        # We don't want register_formula to modify the underlying
        # AnalyticalPulse class, as this might leak into other tests
        _formulas = {}
        _allowed_args = {}
        _required_args = {}

    MyAP.register_formula('blackman', blackman)
    pulse = MyAP('blackman', T=50, nt=1000,
                 parameters={'t_start': 0, 't_stop': 50},
                 time_unit='ns', ampl_unit='unitless',
                 config_attribs={'label': ''})
    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    for i in range(5):
        r1 = (np.random.rand() - 0.5) * 0.01
        r2 = (np.random.rand() - 0.5) * 0.01
        ens = "ens%d" % (i+1)
        H0_fn = "H0_%s.dat" % ens
        H1_fn = "H1_%s.dat" % ens
        model.add_ham(H0+r1*H0, label=ens, filename=H0_fn)
        model.add_ham(H1+r2*H1, pulse, label=ens, filename=H1_fn)
    model.write_to_runfolder(str(tmpdir.join('model_rf')),
                             config='ensemble_shared.config')
    assert filecmp.cmp(os.path.join(test_dir, 'ensemble_shared.config'),
                       str(tmpdir.join('model_rf', 'ensemble_shared.config')),
                       shallow=False)
