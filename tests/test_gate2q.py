from __future__ import print_function, division, absolute_import, \
                       unicode_literals
from QDYN.gate2q import closest_SQ, closest_PE, random_Gate2Q, pop_loss, \
                        identity, CNOT
from QDYN.linalg import norm
import numpy as np
import logging

def test_closest_SQ():
    logger = logging.getLogger(__name__)
    assert norm(identity - identity.closest_SQ()) < 1.0e-14, \
    "closest_SQ should map identity onto itself"
    for method in ['Powell', 'Nelder-Mead', 'leastsq']:
        logger.info("Testing closest_SQ with method %s", method)
        gates = []
        for i in range(10):
            logger.info("[%d]", i+1)
            gates.append(closest_SQ(random_Gate2Q(), method=method,
                                    limit=1.0e-3))
        assert(np.all([(pop_loss(U) < 1.0e-14) for U in gates])), \
        "closest_SQ should yield unitary gates"
        assert(np.all([(U.concurrence() < 1.0e-14) for U in gates])), \
        "closest_SQ should yield non-entangling gates"

def test_closest_PE():
    logger = logging.getLogger(__name__)
    assert norm(CNOT - CNOT.closest_PE()) < 1.0e-14, \
    "closest_PE should map CNOT onto itself"
    for method in ['Powell', 'leastsq']:
        logger.info("Testing closest_PE with method %s", method)
        gates = []
        for i in range(10):
            logger.info("[%d]", i+1)
            gates.append(closest_PE(random_Gate2Q(), method=method,
                         limit=1.0e-3))
        assert(np.all([(pop_loss(U) < 1.0e-14) for U in gates])), \
        "closest_PE should yield unitary gates"
        assert(np.all([((1.0-U.concurrence()) < 1.0e-14) for U in gates])), \
        "closest_PE should yield perfect entanglers"
