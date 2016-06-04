"""QuTiP compatiblity layers for propagation"""
from __future__ import print_function, division, absolute_import

from ..model import LevelModel
from ..analytical_pulse import AnalyticalPulse

def mcsolve(H, psi0, tlist, c_ops, e_ops, ntraj=None,
            args=None, options=None, progress_bar=False,
            map_func=None, map_kwargs=None):
    """Compatiblity wrapper to the :func:`qutip.mcsolve` routine. All
    parameters have the following options, with the following caveats:

    * The `options` may be an instance of `qutip.solver.Options` (or a
      compatible object), or a dictionary. Out of the options available in
      qutip, only `num_cpus` (defaults to the number of available cores) and
      `mc_avg` (default to True) are used. In addition to the options defined
      for qutip, `options` may also contain the following keys:
      - `pulse_ampl_unit`: physical unit of all pulse amplitude, defaults to
        'unitless'
      - `construct_mcwf_ham`: if False, assume that `H` contains the effective
        Hamiltonian for the MCWF method, instead of constructing it from the
        Lindblad operators. Defaults to True.
      - `cmd`: the command to perform the propagation. Defaults to
        ``['mpirun', '-n', '{num_cpuls}', 'qdyn_prop_traj', '{runfolder}']``
    * The `progress_bar` argument is ignored.
    * `map_func` and `map_kwargs` are ignored

    Returns a `Results` object compatible with `qutip.solver.Result`.
    """
    if args is None:
        args = {}
    model = LevelModel()


class Result():
    """Class for storing simulation results, see `qutip.solver.Result`"""
    def __init__(self):
        self.solver = None
        self.times = None
        self.states = []
        self.expect = []
        self.num_expect = 0
        self.num_collapse = 0
        self.ntraj = None
        self.seeds = None
        self.col_times = None
        self.col_which = None

    def __str__(self):
        s = "Result object "
        if self.solver:
            s += "with " + self.solver + " data.\n"
        else:
            s += "missing solver information.\n"
        s += "-" * (len(s) - 1) + "\n"
        if self.states is not None and len(self.states) > 0:
            s += "states = True\n"
        elif self.expect is not None and len(self.expect) > 0:
            s += "expect = True\nnum_expect = " + str(self.num_expect) + ", "
        else:
            s += "states = True, expect = True\n" + \
                "num_expect = " + str(self.num_expect) + ", "
        s += "num_collapse = " + str(self.num_collapse)
        if self.solver == 'mcsolve':
            s += ", ntraj = " + str(self.ntraj)
        return s

    def __repr__(self):
        return self.__str__()
