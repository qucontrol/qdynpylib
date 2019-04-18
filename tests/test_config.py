from textwrap import dedent

import pytest

from qdyn.config import (
    _item_rxs,
    config_data_to_str,
    generate_make_config,
    get_config_user_value,
    get_config_value,
    read_config_str,
    set_config_user_value,
)


@pytest.fixture()
def config1():
    """Example config file"""
    config = r'''
    ! the following is a config file for 3 pulses and some custom
    ! data
    pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, &
    & oct_shape = flattop, ftrt = 0.2, oct_lambda_a = 100, &
    & oct_increase_factor = 10
    * id = 1, t_0 = 0, oct_outfile = pulse1.dat    ! pulse 1
    * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat  ! pulse 2
    * id = 3, t_0 = 5, oct_outfile = pulse3.dat    ! pulse 3

    user_strings: &
    A = "x**2", &
    B = 'B_{"avg"}' ! some "arbitrary" strings
    '''
    return read_config_str(config)


@pytest.fixture
def config_two_psi():
    """Config file with two psi sections"""
    config = r'''
    psi:
    * type = file, filename = psi_10.dat, label = 10

    psi:
    * type = file, filename = psi_01.dat, label = 01
    '''
    return read_config_str(config)


@pytest.fixture
def config_ham_ensemble():
    """Config with an ensemble of hamiltonians"""
    config = r'''
    tgrid: t_start = 0, t_stop = 100_ns, nt = 2000

    prop: method = newton, use_mcwf = F

    pulse:
    * is_complex = T, oct_spectral_filter = filter.dat, id = 1, &
    filename = pulse1.dat, oct_outfile = pulse.oct.dat, oct_lambda_a = 1.0, &
    oct_lambda_intens = 0.0, oct_increase_factor = 5.0, oct_shape = flattop, &
    shape_t_start = 0.0, t_rise = 2_ns, shape_t_stop = 100_ns, t_fall = 2_ns, &
    type = file, time_unit = ns, ampl_unit = MHz

    ham: type = matrix, real_op = F, n_surf = 150, sparsity_model = indexed
    * filename = H0.dat, op_unit = MHz, op_type = potential
    * filename = H1.dat, op_unit = dimensionless, op_type = dipole, &
    pulse_id = 1
    * filename = H2.dat, conjg_pulse = T, op_unit = dimensionless, &
    op_type = dipole, pulse_id = 1

    ham: type = matrix, real_op = F, n_surf = 150, label = gen1, &
    sparsity_model = indexed
    * filename = H3.dat, op_unit = MHz, op_type = potential
    * filename = H4.dat, op_unit = dimensionless, op_type = dipole, &
    pulse_id = 1
    * filename = H5.dat, conjg_pulse = T, op_unit = dimensionless, &
    op_type = dipole, pulse_id = 1

    ham: type = matrix, real_op = F, n_surf = 150, label = gen2, &
    sparsity_model = indexed
    * filename = H6.dat, op_unit = MHz, op_type = potential
    * filename = H7.dat, op_unit = dimensionless, op_type = dipole, &
    pulse_id = 1
    * filename = H8.dat, conjg_pulse = T, op_unit = dimensionless, &
    op_type = dipole, pulse_id = 1

    ham: type = matrix, real_op = F, n_surf = 150, label = gen3, &
    sparsity_model = indexed
    * filename = H9.dat, op_unit = MHz, op_type = potential
    * filename = H10.dat, op_unit = dimensionless, op_type = dipole, &
    pulse_id = 1
    * filename = H11.dat, conjg_pulse = T, op_unit = dimensionless, &
    op_type = dipole, pulse_id = 1

    ham: type = matrix, real_op = F, n_surf = 150, label = gen4, &
    sparsity_model = indexed
    * filename = H12.dat, op_unit = MHz, op_type = potential
    * filename = H13.dat, op_unit = dimensionless, op_type = dipole, &
    pulse_id = 1
    * filename = H14.dat, conjg_pulse = T, op_unit = dimensionless, &
    op_type = dipole, pulse_id = 1

    psi:
    * type = file, filename = psi_00.dat, label = 00

    psi:
    * type = file, filename = psi_01.dat, label = 01

    psi:
    * type = file, filename = psi_10.dat, label = 10

    psi:
    * type = file, filename = psi_11.dat, label = 11

    oct: method = krotovpk, J_T_conv = 1e-05, max_ram_mb = 8000, &
    iter_dat = oct_iters.dat, iter_stop = 1000, params_file = oct_params.dat

    user_strings: ensemble_gens = gen1\,gen2\,gen3\,gen4, time_unit = ns, &
    rwa_vector = rwa_vector.dat, write_gate = U_over_t.dat, &
    basis = 00\,01\,10\,11, gate = target_gate.dat, J_T = J_T_sm

    user_logicals: write_optimized_gate = T
    '''
    return read_config_str(config)


def test_item_rxs():
    """Test regular expressions for items"""
    rx_label = _item_rxs()[0][0]
    assert rx_label.match("label = 0")
    rx_int = _item_rxs()[2][0]
    assert rx_int.match("val=1")
    assert rx_int.match("val = 0")
    assert rx_int.match("val = 1")
    assert rx_int.match("val = 100")
    assert rx_int.match("val = -100")
    assert rx_int.match("val = +100")
    assert not rx_int.match("val = +100-100")
    assert not rx_int.match("val = 01")
    rx_float = _item_rxs()[3][0]
    assert rx_float.match('val=1.0')
    assert rx_float.match('val = 1.0')
    assert rx_float.match('val = -1.0')
    assert rx_float.match('val = -1.0e-5')
    assert rx_float.match('val = -1.0e5')
    assert rx_float.match('val = -1.0e+5')
    assert rx_float.match('val = 2e+5')


def test_get_config_value(config1):
    """Test that we can read value from a config_file"""
    assert get_config_value(config1, ('pulse', 1, 'id')) == 2
    assert get_config_value(config1, ('pulse', 1, 'w_L')) == 0.2
    with pytest.raises(ValueError) as exc_info:
        get_config_value(config1, ('pulse', '1', 'id'))
    assert 'list indices must be integers' in str(exc_info)
    assert get_config_value(config1, ('user_strings', 'B')) == 'B_{"avg"}'
    with pytest.raises(ValueError) as exc_info:
        get_config_value(config1, ('oct', 'method'))
    assert 'oct' in str(exc_info)
    print(str(exc_info))
    with pytest.raises(ValueError) as exc_info:
        get_config_value(config1, ('pulse', 5, 'xxx'))
    assert 'list index out of range' in str(exc_info)


def test_parse_two_psi(config_two_psi):
    """Test the a config file with two psi sections is read correctly"""
    assert len(config_two_psi['psi'])
    assert config_two_psi['psi'][0]['filename'] == 'psi_10.dat'
    assert config_two_psi['psi'][1]['filename'] == 'psi_01.dat'
    assert config_two_psi['psi'][1]['label'] == '01'


def test_get_set_user_value(config1):
    """Test that we can set and get user-defined values from a config_file"""
    config = config1.copy()
    A_val = get_config_value(config, ('user_strings', 'A'))
    assert get_config_user_value(config, "A") == A_val
    A_val = "This is a new value"
    set_config_user_value(config, "A", A_val)
    assert get_config_user_value(config, "A") == A_val
    with pytest.raises(KeyError):
        get_config_user_value(config, "my_float")
    val = -5.34e-2
    set_config_user_value(config, "my_float", val)
    assert abs(get_config_user_value(config, "my_float") - val) < 1e-14
    val = 1.0
    set_config_user_value(config, "my_float2", val)
    set_config_user_value(config, "my_float", val)
    assert abs(get_config_user_value(config, "my_float") - val) < 1e-14
    assert (
        abs(float(get_config_value(config, ('user_reals', "my_float"))) - val)
        < 1e-14
    )
    assert (
        abs(
            get_config_user_value(config, "my_float")
            - get_config_user_value(config, "my_float2")
        )
        < 1e-14
    )
    set_config_user_value(config, "my_logical", False)
    assert get_config_user_value(config, "my_logical") is False
    assert get_config_value(config, ('user_logicals', 'my_logical')) is False
    set_config_user_value(config, "my_int", 5)
    assert get_config_user_value(config, "my_int") == 5
    assert get_config_value(config, ('user_ints', 'my_int')) == 5
    set_config_user_value(config, "my_int2", 2)
    assert "\n" + config_data_to_str(config) == dedent(
        r'''
    pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, oct_shape = flattop, &
      ftrt = 0.2, oct_lambda_a = 100, oct_increase_factor = 10
    * id = 1, t_0 = 0, oct_outfile = pulse1.dat
    * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
    * id = 3, t_0 = 5, oct_outfile = pulse3.dat

    user_strings: A = "This is a new value", B = B_{"avg"}

    user_reals: my_float = 1.0, my_float2 = 1.0

    user_logicals: my_logical = F

    user_ints: my_int = 5, my_int2 = 2
    '''
    )


def test_generate_make_config(config1):
    make_config = generate_make_config(
        config1,
        variables={'E_0': ('pulse', 0, 'E_0'), 'A': ('user_strings', 'A')},
        dependencies={
            ('pulse', 1, 'E_0'): lambda kwargs: kwargs['E_0'],
            ('pulse', 2, 'E_0'): lambda kwargs: kwargs['E_0'],
        },
        checks={
            'E_0': lambda val: val > 0.0,
            'A': lambda val: isinstance(val, str),
        },
    )
    config = make_config(E_0=0.1)
    assert "\n" + config_data_to_str(config) == dedent(
        r'''
    pulse: type = gauss, t_FWHM = 1.8, E_0 = 0.1, w_L = 0.2, oct_shape = flattop, &
      ftrt = 0.2, oct_lambda_a = 100, oct_increase_factor = 10
    * id = 1, t_0 = 0, oct_outfile = pulse1.dat
    * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
    * id = 3, t_0 = 5, oct_outfile = pulse3.dat

    user_strings: A = x**2, B = B_{"avg"}
    '''
    )
    config = make_config(A="bla")
    assert "\n" + config_data_to_str(config) == dedent(
        r'''
    pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, oct_shape = flattop, &
      ftrt = 0.2, oct_lambda_a = 100, oct_increase_factor = 10
    * id = 1, t_0 = 0, oct_outfile = pulse1.dat
    * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
    * id = 3, t_0 = 5, oct_outfile = pulse3.dat

    user_strings: A = bla, B = B_{"avg"}
    '''
    )
    with pytest.raises(ValueError) as exc_info:
        config = make_config(E_0=-0.1)
    assert "does not pass check" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        config = make_config(A=1)
    assert "does not pass check" in str(exc_info)
    with pytest.raises(TypeError) as exc_info:
        config = make_config(B=1)
    assert "unexpected keyword" in str(exc_info)


def test_config_ham_ensemble(config_ham_ensemble):
    """Test config file with multiple hamiltonian blocks, each identified by a
    different label"""
    config = config_ham_ensemble
    assert len(config['ham']) == 15
    for line in config['ham'][0:3]:
        assert 'label' not in line
    for line in config['ham'][3:6]:
        assert line['label'] == 'gen1'
    for line in config['ham'][6:9]:
        line['label'] == 'gen2'
    for line in config['ham'][9:12]:
        line['label'] == 'gen3'
    for line in config['ham'][12:15]:
        line['label'] == 'gen4'
    assert len(config['psi']) == 4
