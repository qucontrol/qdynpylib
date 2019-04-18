"""Tests for QDYN.model"""

import filecmp
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import pytest
from scipy import sparse
import qutip

from qdyn.analytical_pulse import AnalyticalPulse
from qdyn.io import read_indexed_matrix, read_psi_amplitudes
from qdyn.linalg import norm, tril, triu
from qdyn.model import LevelModel
from qdyn.pulse import Pulse, blackman
from qdyn.units import UnitFloat

# built-in fixtures: tmpdir, request


@pytest.fixture
def H0():
    MHz = 2 * np.pi * 1e-3
    w1 = 0 * MHz
    w2 = 500 * MHz
    return np.array(np.diag([0, w2, w1, w1 + w2]), dtype=np.complex128)


@pytest.fixture
def H1():
    return np.array(
        [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
        dtype=np.complex128,
    )


@pytest.fixture
def L1():
    MHz = 2 * np.pi * 1e-3
    kappa1 = 0.01 * MHz
    return np.sqrt(kappa1) * np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.complex128,
    )


@pytest.fixture
def L2():
    MHz = 2 * np.pi * 1e-3
    kappa2 = 0.012 * MHz
    return np.sqrt(kappa2) * np.array(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        dtype=np.complex128,
    )


@pytest.fixture
def pop1():
    return np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )


@pytest.fixture
def pop2():
    return np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi):
    """Construct model for two-qubit system"""
    # Not a fixture!
    model = LevelModel()
    model.add_ham(H0)
    model.add_ham(H1, pulse)
    model.add_lindblad_op(L1)
    model.add_lindblad_op(L2)
    model.set_propagation(
        initial_state=psi, T=50, nt=1001, time_unit='ns', use_mcwf=True
    )
    model.add_observable(pop1, 'pops.dat', 'unitless', 'ns', '<P_1> (q1)')
    model.add_observable(pop2, 'pops.dat', 'unitless', 'ns', '<P_1> (q2)')
    return model


def test_level_model(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test a simple two-qubit model"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = AnalyticalPulse.from_func(
        partial(blackman, t_start=0, t_stop=50),
        ampl_unit='unitless',
        time_unit='ns',
    )
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    model.user_data['my_str'] = 'This is a custom string value'
    model.user_data['i'] = 1
    model.user_data['j'] = 2
    model.user_data['propagate'] = True
    model.user_data['liouville_space'] = False
    model.user_data['precision'] = 1e-8
    model.write_to_runfolder(str(tmpdir.join('model_rf')))

    assert filecmp.cmp(
        os.path.join(test_dir, 'config'),
        str(tmpdir.join('model_rf', 'config')),
        shallow=False,
    )

    def file_matches_matrix(filename, matrix, limit=0.0):
        """Check that the indexed matrix in `filename` matches `matrix`, up to
        `limit`"""
        file_matrix = read_indexed_matrix(
            str(tmpdir.join('model_rf', filename)),
            expand_hermitian=False,
            shape=(4, 4),
        )
        return norm(file_matrix - matrix) <= limit

    def file_matches_psi(filename, psi, limit=0.0):
        """Check that the psi in `filename` matches `psi`, up to `limit`"""
        file_psi = read_psi_amplitudes(
            str(tmpdir.join('model_rf', filename)), n=4
        )
        return norm(file_psi - psi) <= limit

    assert file_matches_matrix('H0.dat', H0)
    assert file_matches_matrix('H1.dat', H1)
    assert model.ham() == [H0, H1]
    assert model.ham(with_attribs=True) == [
        (H0, OrderedDict([])),
        (H1, OrderedDict([('pulse_id', 1)])),
    ]
    assert model.observables() == [pop1, pop2]
    assert model.observables(with_attribs=True) == [
        (
            pop1,
            OrderedDict(
                [
                    ('outfile', 'pops.dat'),
                    ('exp_unit', 'unitless'),
                    ('is_real', True),
                    ('time_unit', 'ns'),
                    ('column_label', '<P_1> (q1)'),
                    ('op_unit', 'unitless'),
                ]
            ),
        ),
        (
            pop2,
            OrderedDict(
                [
                    ('outfile', 'pops.dat'),
                    ('exp_unit', 'unitless'),
                    ('is_real', True),
                    ('time_unit', 'ns'),
                    ('column_label', '<P_1> (q2)'),
                    ('op_unit', 'unitless'),
                ]
            ),
        ),
    ]
    assert model.lindblad_ops() == [L1, L2]
    assert model.lindblad_ops(with_attribs=True) == [
        (
            L1,
            OrderedDict(
                [('add_to_H_jump', 'banded'), ('conv_to_superop', False)]
            ),
        ),
        (
            L2,
            OrderedDict(
                [('add_to_H_jump', 'banded'), ('conv_to_superop', False)]
            ),
        ),
    ]
    assert file_matches_matrix('L1.dat', L1)
    assert file_matches_matrix('L2.dat', L2)
    assert file_matches_matrix('O1.dat', pop1)
    assert file_matches_matrix('O2.dat', pop2)
    assert file_matches_psi('psi.dat', psi, limit=1e-15)
    num_pulse = Pulse.read(str(tmpdir.join('model_rf', 'pulse1.dat')))
    assert num_pulse.time_unit == 'ns'
    assert num_pulse.T == '50_ns'
    expected_amplitude = blackman(
        num_pulse.tgrid, t_start=float(num_pulse.t0), t_stop=float(num_pulse.T)
    )
    assert np.max(np.abs(num_pulse.amplitude - expected_amplitude)) < 1e-15


def test_target_psi(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test that we can add another 'target' state to the config file"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = AnalyticalPulse.from_func(
        partial(blackman, t_start=0, t_stop=50),
        ampl_unit='unitless',
        time_unit='ns',
    )
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    model.add_state(
        np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0),
        label='target',
    )
    model.write_to_runfolder(
        str(tmpdir.join('model_rf')), config='target.config'
    )

    assert filecmp.cmp(
        os.path.join(test_dir, 'target.config'),
        str(tmpdir.join('model_rf', 'target.config')),
        shallow=False,
    )


def test_dissipation_superop(tmpdir, request, H0, L1, L2):
    """Test that we can add a dissipatio superoperator as an alternative to
    Lindblad operators"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    model = LevelModel()
    model.add_ham(H0)
    model.add_lindblad_op(L1)
    model.add_lindblad_op(L2)

    D = qutip.liouvillian(H=None, c_ops=[qutip.Qobj(L1), qutip.Qobj(L2)])
    with pytest.raises(ValueError):
        model.set_dissipator(D)

    model._lindblad_ops = []
    model.set_dissipator(D)

    with pytest.raises(ValueError):
        model.add_lindblad_op(L1)

    model.write_to_runfolder(
        str(tmpdir.join('model_rf')), config='dissipator.config'
    )

    assert filecmp.cmp(
        os.path.join(test_dir, 'dissipator.config'),
        str(tmpdir.join('model_rf', 'dissipator.config')),
        shallow=False,
    )


def test_oct(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test definition of OCT section"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = AnalyticalPulse.from_func(
        partial(blackman, t_start=0, t_stop=50),
        ampl_unit='unitless',
        time_unit='ns',
    )
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    with pytest.raises(KeyError) as exc_info:
        model.set_oct(method='krotovpk', J_T_conv=1e-3, max_ram_mb=10)
    assert "Key 'oct_lambda_a' is required" in str(exc_info)
    pulse.config_attribs.update(
        OrderedDict(
            [
                ('oct_lambda_a', 1e-3),
                ('oct_shape', 'flattop'),
                ('t_rise', UnitFloat(5, 'ns')),
                ('t_fall', UnitFloat(5, 'ns')),
            ]
        )
    )
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    with pytest.raises(TypeError):
        model.set_oct(
            method='krotovpk', J_T_conv=1e-3, bogus='val', max_ram_mb=10
        )
    model.set_oct(
        method='krotovpk', J_T_conv=1e-3, iter_stop=10, max_ram_mb=10
    )
    model.write_to_runfolder(str(tmpdir.join('model_rf')), config='oct.config')

    assert filecmp.cmp(
        os.path.join(test_dir, 'oct.config'),
        str(tmpdir.join('model_rf', 'oct.config')),
        shallow=False,
    )


def test_ensemble(tmpdir, request, H0, H1, L1, L2, pop1, pop2):
    """Test ensemble of multiple Hamiltonians"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    pulse = AnalyticalPulse.from_func(
        partial(blackman, t_start=0, t_stop=50),
        ampl_unit='unitless',
        time_unit='ns',
    )
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    for i in range(5):
        r1 = (np.random.rand() - 0.5) * 0.01
        r2 = (np.random.rand() - 0.5) * 0.01
        ens = "ens%d" % (i + 1)
        H0_fn = "H0_%s.dat" % ens
        H1_fn = "H1_%s.dat" % ens
        model.add_ham(H0 + r1 * H0, label=ens, filename=H0_fn)
        model.add_ham(H1 + r2 * H1, pulse, label=ens, filename=H1_fn)

    assert len(model.ham()) == 2
    assert len(model.ham(label='ens1')) == 2
    assert len(model.ham(label='*')) == 12
    assert len(model.pulses(label='ens1')) == 1
    assert len(model.pulses(label='*')) == 6
    model.write_to_runfolder(
        str(tmpdir.join('model_rf')), config='ensemble.config'
    )
    assert filecmp.cmp(
        os.path.join(test_dir, 'ensemble.config'),
        str(tmpdir.join('model_rf', 'ensemble.config')),
        shallow=False,
    )


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
    pulse = MyAP(
        'blackman',
        parameters={'t_start': 0, 't_stop': 50},
        time_unit='ns',
        ampl_unit='unitless',
        config_attribs={'label': ''},
    )
    psi = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
    model = two_level_model(H0, H1, L1, L2, pop1, pop2, pulse, psi)
    for i in range(5):
        r1 = (np.random.rand() - 0.5) * 0.01
        r2 = (np.random.rand() - 0.5) * 0.01
        ens = "ens%d" % (i + 1)
        H0_fn = "H0_%s.dat" % ens
        H1_fn = "H1_%s.dat" % ens
        model.add_ham(H0 + r1 * H0, label=ens, filename=H0_fn)
        model.add_ham(H1 + r2 * H1, pulse, label=ens, filename=H1_fn)
    model.write_to_runfolder(
        str(tmpdir.join('model_rf')), config='ensemble_shared.config'
    )
    assert filecmp.cmp(
        os.path.join(test_dir, 'ensemble_shared.config'),
        str(tmpdir.join('model_rf', 'ensemble_shared.config')),
        shallow=False,
    )


def complex_pulse_error_data():
    """Generate data for parametrization of test_complex_pulse_error"""
    # real pulse:
    pulse1 = AnalyticalPulse.from_func(partial(blackman, t_start=0, t_stop=50))
    # complex pulse:
    pulse2 = AnalyticalPulse.from_func(lambda t: complex(pulse1.as_func()(t)))
    H1 = np.array(
        [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
        dtype=np.complex128,
    )
    H2 = sparse.coo_matrix(H1)
    H3 = np.eye(4)
    H4 = triu(H1)
    H5 = tril(H2)
    return [
        (H1, pulse1, True),
        (H2, pulse1, True),
        (H3, pulse1, True),
        (H1, pulse2, False),
        (H2, pulse2, False),
        (H3, pulse2, False),
        (H4, pulse2, True),
        (H5, pulse2, True),
    ]


@pytest.mark.parametrize('H, pulse, ok', complex_pulse_error_data())
def test_complex_pulse_error(H, pulse, ok):
    """Test that we catch trying to connect a complex pulse to a matrix with
    data in both triangles"""
    model = LevelModel()
    if ok:
        model.add_ham(H, pulse)
        assert len(model._pulses) == 1
    else:
        with pytest.raises(ValueError) as exc_info:
            model.add_ham(H, pulse)
        assert "Cannot connect a complex pulse" in str(exc_info)
