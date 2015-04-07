#!/usr/bin/env python
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
import scipy.linalg
from ..linalg import get_op_matrix, vectorize

def prop_step_exact(apply_op, state, t, dt):
    """
    Propagate state by one time step, i.e., calculate [exp(-i L(t) dt)].state
    by exact matrix exponetiation
    """

    # Construct explicit Liouvillian Matrix
    O = get_op_matrix(apply_op, state.shape, t)

    # Propagate
    U = scipy.linalg.expm(np.asarray((-1.j * O * dt))) # U(dt) = exp(-i*O*dt)
    state_out = U.dot(vectorize(state))
    return state_out.reshape(state.shape, order='F')

