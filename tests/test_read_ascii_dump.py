"""Tests for reading ascii dumps"""

import os
from collections import OrderedDict

import numpy as np

from qdyn.io import read_ascii_dump


# built-in fixtures: request


def test_read_para_dump(request):
    """Test reading a dump of a config file structure"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    para = read_ascii_dump(os.path.join(test_dir, 'para.asciidump'))
    expected_keys = [
        'runfolder',
        'initialized',
        'grid',
        'tgrid',
        'ham',
        'observables',
        'dissipator',
        'eigensystem',
        'psi',
        'pulse',
        'oct',
        'prop',
        'bwr',
        'numerov',
        'user',
        'is_set',
    ]
    assert para.typename == 'para_t'
    assert list(para.keys()) == expected_keys
    assert para['runfolder'] == 'multi_config_run'
    assert para['initialized']
    assert isinstance(para['ham'], list)
    assert len(para['ham']) == 32
    assert para['ham'][1]['op_type'] == 'pot'
    assert para['ham'][1].typename == 'ham_pt'
    assert isinstance(para['user'], OrderedDict)
    assert isinstance(para['user']['reals_vals'], list)

    para = read_ascii_dump(
        os.path.join(test_dir, 'para.asciidump'), convert_boolean=False
    )
    assert para['initialized'] == 'T'


def test_read_state_dump(request):
    """Test reading a dump of a density matrix"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    state = read_ascii_dump(os.path.join(test_dir, 'state.asciidump'))
    expected_keys = [
        'psi',
        'rho',
        'coord_min',
        'mom_disp',
        'movgrid_nr',
        'surfs',
        'spindim',
    ]
    assert list(state.keys()) == expected_keys
    assert np.all(state['movgrid_nr'] == np.array([1]))
    assert state['rho'].shape == (1, 1, 3, 1, 1, 3)
    assert state['rho'][0, 0, 0, 0, 0, 2] == -1j

    state = read_ascii_dump(
        os.path.join(test_dir, 'state.asciidump'), flatten=True
    )
    assert state['rho'].shape == (9,)
