import os
import filecmp
import QDYN
import numpy as np
from QDYN.units import UnitFloat
from QDYN.pulse import tgrid_from_config, Pulse, pulse_tgrid, gaussian
import pytest

config_str = r'''
grid: system = initial, base = exp
* dim = 1, r_min = -10, r_max = 10, nr = 128, moveable = true

grid: system = target, base = exp
* dim = 1, r_min = 0.078740157480315, r_max = 20.078740157480315, nr = 128, &
moveable = true

psi:
* type = gauss, r_0 = 0.0, k_0 = 0.0, sigma = 1.0, system = initial
* type = gauss, r_0 = 10.0, k_0 = 0.0, sigma = 1.0, system = target

ham: type = op, op_type = pot, op_form = hamos, w_0 = 1.0, E_0 = 0.0, &
& mass = 1.0, specrad_method = molecular
* op_surf = 1, pulse_id = 1, offset = 0.
* op_surf = 1, pulse_id = 2, offset = 2.5
* op_surf = 1, pulse_id = 3, offset = 5
* op_surf = 1, pulse_id = 4, offset = 7.5
* op_surf = 1, pulse_id = 5, offset = 10

pulse : type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, &
& oct_shape = flattop, t_rise = 0.2, t_fall = 0.2, oct_lambda_a = 100, oct_increase_factor = 10
* id = 1, t_0 = 0, oct_outfile = pulse1.dat
* id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
* id = 3, t_0 = 5, oct_outfile = pulse3.dat
* id = 4, t_0 = 7.5, oct_outfile = pulse4.dat
* id = 5, t_0 = 10, oct_outfile = pulse5.dat

tgrid: t_start = 0.0, t_stop =  10.0, dt = 0.02, fixed = T

oct: method = krotovpk, iter_stop = 20, max_ram_mb = 1000, continue = false

prop: method = cheby

user_logicals: silent = true
'''


def test_tgrid_from_config():

    tgrid_dict = dict([('t_start', 0.0), ('t_stop', UnitFloat(10.0, 'ns')),
                       ('dt', UnitFloat(0.02, 'ns')), ('fixed', True)])
    tgrid = tgrid_from_config(tgrid_dict, time_unit='ns')
    assert ("%g" % tgrid[0]) == "0.01"
    assert ("%g" % tgrid[-1]) == "9.99"

    tgrid_dict = dict([('t_start', 0.0), ('t_stop', UnitFloat(10.0, 'ns')),
                       ('dt', UnitFloat(20, 'ps')), ('fixed', True)])
    tgrid = tgrid_from_config(tgrid_dict, time_unit='ns')
    assert ("%g" % tgrid[0]) == "0.01"
    assert ("%g" % tgrid[-1]) == "9.99"
    tgrid = tgrid_from_config(tgrid_dict, time_unit='ps')
    assert ("%g" % tgrid[0]) == "10"
    assert ("%g" % tgrid[-1]) == "9990"

    tgrid_dict = dict([('t_start', 0.0), ('t_stop', 10),
                    ('dt', 0.02), ('fixed', True)])
    with pytest.raises(ValueError) as exc_info:
        tgrid = tgrid_from_config(tgrid_dict, time_unit='ns')
    assert "Incompatible units in conversion: unitless, ns" in str(exc_info)

    tgrid = tgrid_from_config(tgrid_dict, time_unit=None)
    assert ("%g" % tgrid[0]) == "0.01"
    assert ("%g" % tgrid[-1]) == "9.99"

    tgrid = tgrid_from_config(tgrid_dict, time_unit='unitless')
    assert ("%g" % tgrid[0]) == "0.01"
    assert ("%g" % tgrid[-1]) == "9.99"

    tgrid_dict = dict([('t_start', 0.0), ('t_stop', UnitFloat(10.0, 'ns')),
                       ('dt', UnitFloat(0.02, 'ns')), ('fixed', True)])
    with pytest.raises(ValueError) as exc_info:
        tgrid = tgrid_from_config(tgrid_dict, time_unit=None)
    assert "Incompatible units in conversion: ns, unitless" in str(exc_info)


def test_pulse_derivative():
    tgrid = pulse_tgrid(10, 100)
    ampl = gaussian(tgrid, 5, 1)
    pulse = Pulse(tgrid=tgrid, amplitude=ampl, ampl_unit='MHz', time_unit='ns')
    deriv = pulse.derivative()
    assert deriv.dt == pulse.dt
    assert deriv.ampl_unit == 'unitless'
    assert deriv.time_unit == 'ns'
    assert "%.3f" % abs(np.max(deriv.amplitude)) == '0.613'
    assert "%.3f" % abs(np.min(deriv.amplitude)) == '0.613'


def test_write_oct_spectral_filter(tmpdir, request):
    tgrid = pulse_tgrid(10, 100)
    ampl = gaussian(tgrid, 5, 1)
    pulse = Pulse(tgrid=tgrid, amplitude=ampl, ampl_unit='MHz', time_unit='ns')
    filename = str(tmpdir.join("spectral_filter.dat"))
    pulse.write_oct_spectral_filter(
        filename, filter_func=lambda f: abs(f) < 2)
    datadir = os.path.splitext(request.module.__file__)[0]
    assert filecmp.cmp(filename, os.path.join(datadir, 'spectral_filter.dat'))
