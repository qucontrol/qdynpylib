import filecmp

import numpy as np
import pytest

from qdyn.analytical_pulse import AnalyticalPulse
from qdyn.pulse import CRAB_carrier, blackman, carrier, pulse_tgrid
from qdyn.units import UnitFloat


# built-in fixtures: tmpdir


def ampl_field_free(tgrid):
    return 0.0 * carrier(tgrid, 'ns', 0.0, 'GHz').real


def ampl_1freq(tgrid, E0, T, w_L):
    return E0 * blackman(tgrid, 0, T) * carrier(tgrid, 'ns', w_L, 'GHz').real


def ampl_1freq_rwa(tgrid, E0, T, w_L, w_d):
    # note: amplitude reduction by 1/2 is included in construction of ham
    return (
        E0
        * blackman(tgrid, 0, T)
        * carrier(tgrid, 'ns', (w_L - w_d), 'GHz', complex=True)
    )


def ampl_1freq_0(tgrid, E0, T, w_L=0.0):
    return E0 * blackman(tgrid, 0, T) * carrier(tgrid, 'ns', w_L, 'GHz').real


def ampl_2freq(tgrid, E0, T, freq_1, freq_2, a_1, a_2, phi):
    return (
        E0
        * blackman(tgrid, 0, T)
        * carrier(
            tgrid,
            'ns',
            freq=(freq_1, freq_2),
            freq_unit='GHz',
            weights=(a_1, a_2),
            phases=(0.0, phi),
        ).real
    )


def ampl_2freq_rwa(tgrid, E0, T, freq_1, freq_2, a_1, a_2, phi, w_d):
    # note: amplitude reduction by 1/2 is included in construction of ham
    return (
        E0
        * blackman(tgrid, 0, T)
        * carrier(
            tgrid,
            'ns',
            freq=(freq_1 - w_d, freq_2 - w_d),
            freq_unit='GHz',
            weights=(a_1, a_2),
            phases=(0.0, phi),
            complex=True,
        )
    )


def ampl_2freq_rwa_box(tgrid, E0, T, freq_1, freq_2, a_1, a_2, phi, w_d):
    # note: amplitude reduction by 1/2 is included in construction of ham
    return E0 * carrier(
        tgrid,
        'ns',
        freq=(freq_1 - w_d, freq_2 - w_d),
        freq_unit='GHz',
        weights=(a_1, a_2),
        phases=(0.0, phi),
        complex=True,
    )


def ampl_5freq(
    tgrid, E0, T, freq_low, a_low, b_low, freq_high, a_high, b_high
):
    norm_carrier = CRAB_carrier(
        tgrid, 'ns', freq_high, 'GHz', a_high, b_high, normalize=True
    )
    crab_shape = CRAB_carrier(
        tgrid, 'ns', freq_low, 'GHz', a_low, b_low, normalize=True
    )
    a = blackman(tgrid, 0, T) * crab_shape * norm_carrier
    return E0 * a / np.max(np.abs(a))


def ampl_5freq_rwa(
    tgrid, E0, T, freq_low, a_low, b_low, freq_high, a_high, b_high, w_d
):
    norm_carrier = CRAB_carrier(
        tgrid,
        'ns',
        freq_high - w_d,
        'GHz',
        a_high,
        b_high,
        normalize=True,
        complex=True,
    )
    crab_shape = CRAB_carrier(
        tgrid, 'ns', freq_low, 'GHz', a_low, b_low, normalize=True
    )
    # note: amplitude reduction by 1/2 is included in construction of ham
    a = blackman(tgrid, 0, T) * crab_shape * norm_carrier
    return E0 * a / np.max(np.abs(a))


def ampl_CRAB_rwa(tgrid, E0, T, r, a, b, w_d):
    # note that w_d is neccessary a pulse parameter, even though it does not
    # occur in the formula: the simplex adapts the config file based on the w_d
    # parameter in the pulse.
    #
    # frequencies are freq[k] = 2*pi*k*(1+r_k)/T, so the r vector must take
    # values in [-0.5, 0.5]
    n = len(a)
    freq = np.array([2 * np.pi * k * (1 + r[k]) / float(T) for k in range(n)])
    crab_shape = CRAB_carrier(tgrid, 'ns', freq, 'GHz', a, b, normalize=True)
    # note: amplitude reduction by 1/2 is included in construction of ham
    a = blackman(tgrid, 0, T) * crab_shape
    if np.max(np.abs(a)) > 1.0e-16:
        return E0 * a / np.max(np.abs(a))
    else:
        return np.zeros(len(a))


def test_analytical_pulse(tmpdir):

    AnalyticalPulse.register_formula('field_free', ampl_field_free)
    AnalyticalPulse.register_formula('1freq', ampl_1freq)
    AnalyticalPulse.register_formula('2freq', ampl_2freq)
    AnalyticalPulse.register_formula('5freq', ampl_5freq)
    AnalyticalPulse.register_formula('1freq_rwa', ampl_1freq_rwa)
    AnalyticalPulse.register_formula('2freq_rwa', ampl_2freq_rwa)
    AnalyticalPulse.register_formula('2freq_rwa_box', ampl_2freq_rwa_box)
    AnalyticalPulse.register_formula('5freq_rwa', ampl_5freq_rwa)
    AnalyticalPulse.register_formula('CRAB_rwa', ampl_CRAB_rwa)

    tmp = lambda filename: str(tmpdir.join(filename))

    with pytest.raises(TypeError) as exc_info:
        AnalyticalPulse.register_formula('1freq_0', 'bla')
    assert 'is not a Python function' in str(
        exc_info
    ) or 'unsupported callable' in str(  # <= 3.3
        exc_info
    )  # >= 3.4

    AnalyticalPulse.register_formula('1freq_0', ampl_1freq_0)
    with pytest.raises(ValueError) as exc_info:
        AnalyticalPulse(
            '1freq_0', parameters={'E0': 100}, time_unit='ns', ampl_unit='MHz'
        )
    assert 'Required parameter "T" for formula' in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        AnalyticalPulse(
            '1freq_0',
            parameters={'E0': 100, 'T': 200, 'extra': 0},
            time_unit='ns',
            ampl_unit='MHz',
        )
    assert 'Parameter "extra" does not exist in formula' in str(exc_info)

    p1 = AnalyticalPulse('field_free', time_unit='ns', ampl_unit='MHz')
    assert p1.header == '# Formula "field_free"'
    p1.write(tmp('p1.json'), pretty=True)
    p1.to_num_pulse(tgrid=pulse_tgrid(200, 500)).write(tmp('p1.dat'))
    p1_copy = AnalyticalPulse.read(tmp('p1.json'))
    p1_copy.write(tmp('p1_copy.json'), pretty=True)
    assert filecmp.cmp(tmp("p1.json"), tmp("p1_copy.json"))

    p2 = AnalyticalPulse(
        '1freq',
        parameters={'E0': 100, 'T': 200, 'w_L': 6.5},
        time_unit='ns',
        ampl_unit='MHz',
    )
    assert p2.header == '# Formula "1freq" with E0 = 100, T = 200, w_L = 6.5'
    p2.write(tmp('p2.json'), pretty=True)
    p2.to_num_pulse(tgrid=pulse_tgrid(200, 10000)).write(tmp('p2.dat'))
    p2_copy = AnalyticalPulse.read(tmp('p2.json'))
    p2_copy.write(tmp('p2_copy.json'), pretty=True)
    assert filecmp.cmp(tmp("p2.json"), tmp("p2_copy.json"))

    parameters = {
        'E0': 100,
        'T': 200,
        'freq_1': 6.0,
        'freq_2': 6.5,
        'a_1': 0.5,
        'a_2': 1.0,
        'phi': 0.0,
    }
    p3 = AnalyticalPulse(
        '2freq', parameters=parameters, time_unit='ns', ampl_unit='MHz'
    )
    assert (
        p3.header
        == '# Formula "2freq" with E0 = 100, T = 200, a_1 = 0.5, a_2 = 1.0, freq_1 = 6.0, freq_2 = 6.5, phi = 0.0'
    )
    p3.write(tmp('p3.json'), pretty=True)
    p3.to_num_pulse(tgrid=pulse_tgrid(200, 10000)).write(tmp('p3.dat'))
    p3_copy = AnalyticalPulse.read(tmp('p3.json'))
    p3_copy.write(tmp('p3_copy.json'), pretty=True)
    assert filecmp.cmp(tmp("p3.json"), tmp("p3_copy.json"))

    freq_low = np.array([0.01, 0.0243])
    freq_high = np.array([8.32, 10.1, 5.3])
    a_low = np.array([1.0, 0.21])
    a_high = np.array([0.58, 0.89, 0.1])
    b_low = np.array([1.0, 0.51])
    b_high = np.array([0.09, 0.12, 0.71])
    p4 = AnalyticalPulse(
        '5freq',
        parameters={
            'E0': 100,
            'T': 200,
            'freq_low': freq_low,
            'freq_high': freq_high,
            'a_low': a_low,
            'a_high': a_high,
            'b_low': b_low,
            'b_high': b_high,
        },
        time_unit='ns',
        ampl_unit='MHz',
    )
    assert (
        p4.header
        == '# Formula "5freq" with E0 = 100, T = 200, a_high = [0.58, 0.89, 0.1], a_low = [1.0, 0.21], b_high = [0.09, 0.12, 0.71], b_low = [1.0, 0.51], freq_high = [8.32, 10.1, 5.3], freq_low = [0.01, 0.0243]'
    )
    p4.write(tmp('p4.json'), pretty=True)
    p4.to_num_pulse(tgrid=pulse_tgrid(200, 10000)).write(tmp('p4.dat'))
    p4_copy = AnalyticalPulse.read(tmp('p4.json'))
    assert isinstance(
        p4_copy.parameters['a_low'], np.ndarray
    ), "Coefficients 'a_low' should be a numpy array"
    p4_copy.write(tmp('p4_copy.json'), pretty=True)
    assert filecmp.cmp(tmp("p4.json"), tmp("p4_copy.json"))


def test_analytical_pulse_with_tgrid():
    p0 = AnalyticalPulse.from_func(
        ampl_1freq,
        parameters={'E0': 100, 'T': 200, 'w_L': 6.5},
        time_unit='ns',
        ampl_unit='MHz',
    )
    assert p0.t0 == 0.0
    assert p0.T is None
    assert p0.nt is None
    assert p0.dt is None
    assert p0.tgrid is None
    assert p0.tgrid is None
    assert p0.w_max is None
    assert p0.dw is None
    assert p0.to_num_pulse() is None

    p1 = AnalyticalPulse.from_func(
        ampl_1freq,
        parameters={'E0': 100, 'T': 200, 'w_L': 6.5},
        time_unit='ns',
        ampl_unit='MHz',
        t0=0,
        T='T',
        nt=lambda _p: int(float(_p.T * 10)) + 1,
    )
    assert p0 != p1
    assert p1.time_unit == 'ns'
    assert p1.ampl_unit == 'MHz'
    assert p1.freq_unit == 'GHz'
    assert p1.copy() == p1
    assert p1.t0 == 0.0
    assert p1.T == UnitFloat(200, 'ns')
    assert p1.nt == 2001
    assert p1.dt == UnitFloat(0.1, 'ns')
    assert (
        np.max(np.abs(p1.tgrid - np.linspace(0.05, 200 - 0.05, 2000))) < 1e-14
    )
    assert p1.w_max == 5.0
    assert p1.dw == 0.005
    p1_num = p1.to_num_pulse()
    assert p1_num == p0.to_num_pulse(pulse_tgrid(200, 2001))
    assert np.max(np.abs(p1_num.tgrid - p1.tgrid)) < 1e-14
    assert np.max(np.abs(p1_num.states_tgrid - p1.states_tgrid)) < 1e-14
    assert p1_num.dt == p1.dt
    assert p1_num.w_max == p1.w_max
    assert p1_num.dw == p1.dw
