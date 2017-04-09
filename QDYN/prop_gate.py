"""Support routines for the qdyn_prop_gate utility"""
from __future__ import print_function, division, absolute_import

import re

import numpy as np

from .gate2q import Gate2Q
from .units import UnitFloat
from .math import isqrt


def get_prop_gate_of_t(gates_file, with_t=False):
    r'''Yield gates in `gates_file`, where `gates_file` is in the format
    written by the ``qdyn_prop_gate`` utility's ``--write-gate`` option. That
    is, each row in `gates_files` has $2 n^2 + 1$ columns. The first column is
    a time stamp, the remaining columns are the real and imaginary part for
    each entry in the $n \times n$ gate (vectorized in column-major format). If
    `with_t` is False (default), yield only the gates, otherwise yield both the
    gates and the time stamp for each gate

    Returns:
        * If ``with_t=False``, iterator over gates, where each gate is a
          complex $n \times n$ numpy matrix, or a Gate2Q instance for a $4
          \times 4$ gate
        * If ``with_t=True``, iterator of tuples ``(gate, t)``, where ``t`` is
          a float or an instance of UnitFloat if the time unit can be derived
          from the header of `gates_file`
    '''
    with open(gates_file) as in_fh:
        time_unit = None
        for line in in_fh:
            if line.startswith('#'):
                try:
                    time_unit = re.search(r't\s*\[(\w+)\]', line).group(1)
                except AttributeError:
                    pass
            else:
                vals = np.array([float(v) for v in line.split()])
                n = isqrt((len(vals)-1)//2)
                assert 2 * n * n + 1 == len(vals)
                shape = (n, n)
                gate = (np.reshape(vals[1::2], shape, order='F') +
                        1j * np.reshape(vals[2::2], shape, order='F'))
                if n == 4:
                    gate = Gate2Q(gate)
                if with_t:
                    if time_unit is not None:
                        yield gate, UnitFloat(vals[0], time_unit)
                    else:
                        yield gate, vals[0]
                else:
                    yield gate
