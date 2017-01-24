"""Routines for calculating dissipators"""
from __future__ import print_function, division, absolute_import

from scipy import sparse


def lindblad_ops_to_dissipator(lindblad_ops):
    """Convert list of lindblad operators to a single superoperator sparse
    matrix

    Args:
        lindblad_ops (list of scipy.sparse.spmatrix):  List of Lindblad
            operators. Each Lindblad operator should be in a sparse matrix
            format.

    Returns:
        dissipator (sparse.coo_matrix): The dissipator superoperator in sparse
            format

    Note: Use `QDYN.io.write_indexed_matrix` to write the resulting dissipator
    to file. The routine corresponds to the Fortran QDYN
    `lindblad_ops_to_dissipator` routine. Due to better use of sparse matrix
    algebra, this Python version is usually substantially faster than the
    Fortran version.
    """
    super_data = []
    super_row_ind = []
    super_col_ind = []
    for L in lindblad_ops:
        n, m = L.shape
        assert n == m
        L_dag_L = L.conj().transpose() * L
        super_col = -1
        for j in range(n):
            for i in range(n):
                super_col += 1
                rho = sparse.coo_matrix(
                    ([1, ], ([i, ], [j, ])), shape=(n, m))
                diss = (L * rho * L.conj().transpose() -
                        0.5 * L_dag_L * rho -
                        0.5 * rho * L_dag_L).tocoo()
                for col, row, val in zip(diss.col, diss.row, diss.data):
                    super_row = n * col + row
                    super_row_ind.append(super_row)
                    super_col_ind.append(super_col)
                    super_data.append(val)
    return sparse.coo_matrix((super_data, (super_row_ind, super_col_ind)),
                             shape=(n*n, m*m))
