"""Abstraction for complete physical models that can be written to a runfolder
as a config file and dependent data"""

from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import os
import re

import numpy as np

from .io import write_indexed_matrix
from .analytical_pulse import AnalyticalPulse
from .pulse import Pulse, pulse_tgrid
from .config import write_config
from .state import write_psi_amplitudes
from .shutil import mkdir
from .units import UnitFloat
from .linalg import is_hermitian, choose_sparsity_model


class LevelModel(object):
    """Model for a system well-described in the energy basis. That is, all
    operators are (sparse) matrices, and all states are simple vectors

    Args:
        label (str): set the `label` attribute

    Attributes:
        label (str): label to be used for all config file items
        ham (matrix or list): the system Hamiltonian. Must be either a matrix
        or similar object (numpy matrix, numpy 2D array, scipy sparse matrix,
            `qutip.Qobj`) or a list. If a list, the total Hamiltonian is the
            sum of all list items. Each list item must be either a matrix
            (time-independent), or a list/tuple ``(matrix, pulse)`` for a
            time-dependent operator. The ``pulse`` must be an instance of
            `QDYN.pulse.Pulse`, `QDYN.pulse.AnalyticalPulse`, or a callable
            that takes a time value as a float and returns a pulse amplitude.
        t0 (float or QDYN.units.UnitFloat): Initial time.
        T (float or QDYN.units.UnitFloat): Final time.
        nt (int): Number of points in the time grid.
        prop_method (str): Propagation method
        use_mcwf: Propagate using the Monte-Carlo Wave Function (quantum
            jump) method
        construct_mcwf_ham (bool): When using the MCWF method, the propagation
            must use an "effective" Hamiltonian that includes a non-Hermitian
            decay term. This term is constructed from the Lindblad operators.
            If ``use_mcwf=True`` and ``construct_mcwf_ham=False``, it is the
            user's responsibility to ensure that `ham` is the proper effective
            Hamiltonian.

    Note:
        It is recommended to us `QDYN.pulse.AnalyticalPulse` to express
        time-dependency, as this allows to fully specify physical units.
        Instances of `QDYN.pulse.Pulse` must have a time grid that exactly
        matches the time grid specified via :meth:`set_propagation`. Using a
        callable means that the pulse amplitude is unitless, and the pulse as
        evaluated for time value with the unit specified in
        :meth:`set_propagation`.

        The attributes `t0`, `T`, `nt`, `prop_method`, `use-mcwf`, and
        `construct_mcwf_ham` are all set via :meth:`set_propagation`.
    """

    def __init__(self, label=''):
        if not re.match(r'^[\w\d]*$', label):
            raise ValueError("Invalid label: %s" % label)
        self.label = label
        self.ham = None
        self._observables = []
        self._obs_config_attribs = []
        self._lindblad_ops = []
        self._psi = None
        self.t0 = 0.0
        self.T = 0.0
        self.nt = 0
        self.prop_method = 'newton'
        self.use_mcwf = False
        self.construct_mcwf_ham = False
        self._config_data = OrderedDict([])
        self._pulse_id = 0 # last used pulse_id
        self._pulse_ids = {} # pulse -> pulse_id

    @property
    def observables(self):
        """list of all observables"""
        # protects _observables and _obs_config_attribs from going out of sync
        return list(self._observables)

    @property
    def lindblad_ops(self):
        """List of all Lindblad operators"""
        return list(self._lindblad_ops)

    def add_observable(self, O, outfile, exp_unit, time_unit, col_label,
            square=None):
        """Add an observable

        Args:
            O (tuple, matrix): Observable to add. Must be a matrix
                or a tuple ``(matrix, pulse)``, cf. the `ham` attribute.
            outfile (str): Name of output file to which to write expectation
                values of `O`
            exp_unit (str): The unit in which to write the expectation value in
                `outfile`
            time_unit (str):  The unit in which to write the time grid in
                `outfile`
            col_label (str): The label for the column in `outfile` containing
                the expectation value of `O`
            square (str or None): If not None, label for the column in
                `outfile` containing the expectation value for the square of
                `O`
        """
        self._observables.append(O)
        is_real = is_hermitian(O)
        self._obs_config_attribs.append(OrderedDict(
            [('outfile', outfile), ('exp_unit', exp_unit),
             ('is_real', is_real), ('time_unit', time_unit),
             ('column_label', col_label)]))
        if square is not None:
            self._obs_config_attribs[-1]['square'] = square

    def add_lindblad_op(self, L):
        """Add Lindblad operator.

        Args:
            L (tuple, matrix): Lindblad operator to  add. Must be a matrix
                or a tuple ``(matrix, pulse)``, cf. the `ham` attribute.
        """
        self._lindblad_ops.append(L)

    def set_propagation(self, initial_state, T, nt, time_unit, t0=0.0,
            prop_method='newton', use_mcwf=False, construct_mcwf_ham=True):
        """Set the time grid and other propagation-specific settings

        Args:
            initial_state (array): Initial wave function
            T (float): final time
            nt (int): number of points in the time grid
            time_unit (str): physical unit of `T`, `t0`
            t0 (float): initial time
            prop_method (str): method to be used for propagation
            use_mcwf (bool): If True, prepare for Monte-Carlo wave function
                propagation
            construct_mcwf_ham (bool): When using MCWF (`use_mcwf=True`), by
                default an additional inhomogeneous decay term is added to the
                Hamiltonian, based on the Lindblad operators. By passing
                `construct_mcwf_ham=False`, this does not happen. It is the
                user's responsibility then to ensure the Hamiltonian in the
                model is the correct "effective" Hamiltonian for a MCWF
                propagation.
        """
        self._psi = initial_state
        self.T = UnitFloat(T, time_unit)
        self.nt = nt
        self.t0 = UnitFloat(t0, time_unit)
        self.prop_method = prop_method
        if use_mcwf:
            self.use_mcwf = use_mcwf
        if construct_mcwf_ham is None:
            construct_mcwf_ham = use_mcwf
        if use_mcwf:
            self.construct_mcwf_ham = construct_mcwf_ham
        else:
            self.construct_mcwf_ham = False

    def write_to_runfolder(self, runfolder, config='config'):
        """Write the model to the given runfolder. This will create a config
        file (`config`) in the runfolder, as well as all dependent data file
        (operators, pulses)"""
        mkdir(runfolder)
        self._config_data = OrderedDict([])

        # time grid
        if float(self.T) >= 0.0 and self.nt > 0:
            self._config_data['tgrid'] = OrderedDict([
                ('t_start', self.t0), ('t_stop', self.T), ('nt', self.nt)])

        # propagation
        self._config_data['prop'] = OrderedDict([
            ('method', self.prop_method), ('use_mcwf', self.use_mcwf)])

        # pulses
        self._write_pulses(runfolder) # also sets self._pulse_ids

        # Hamiltonian
        if self.ham is not None:
            self._write_ham(runfolder)

        # Lindblad operators
        if len(self._lindblad_ops) > 0:
            self._write_lindblad_ops(runfolder)

        # initial state
        if self._psi is not None:
            self._write_psi(runfolder)

        # observables
        if len(self._observables) > 0:
            assert len(self._observables) == len(self._obs_config_attribs)
            self._write_observables(runfolder)

        write_config(self._config_data, os.path.join(runfolder, config))

    def _write_pulses(self, runfolder):
        """Write numerical pulse files to runfolder, add pulse data to
        self._config_data, and build the self._pulse_ids dict so that at some
        later point, operators may find the pulse ID for a pulse they are
        connected to"""
        nested_ops = [self.ham, ] + self._observables + self._lindblad_ops
        for nested_op in nested_ops:
            for element in nested_op:
                if isinstance(element, (list, tuple)):
                    try:
                        pulse = element[1]
                    except ValueError:
                        raise ValueError("nested_op must be either an "
                                "operator, or a list of operators or lists "
                                "`[Op, pulse]`")
                    if pulse not in self._pulse_ids:
                        self._write_pulse(pulse, runfolder)

    def _write_pulse(self, pulse, runfolder):
        tgrid = pulse_tgrid(self.T, self.nt, self.t0)
        self._pulse_id += 1
        self._pulse_ids[pulse] = self._pulse_id
        if 'pulse' not in self._config_data:
            self._config_data['pulse'] = []
        if self.label == '':
            filename = "pulse%d.dat" % self._pulse_id
        else:
            filename = "pulse_%s_%d.dat" \
                    % (self.label, self._pulse_id)
        if isinstance(pulse, AnalyticalPulse):
            p = pulse.to_num_pulse(tgrid)
            p.write(os.path.join(runfolder, filename))
            self._config_data['pulse'].append(
                    p.config_line(filename, self._pulse_id, self.label))
        elif isinstance(pulse, Pulse):
            if np.max(np.abs(tgrid - pulse.tgrid)) > 1e-12:
                raise ValueError("Mismatch of tgrid with pulse tgrid")
            pulse.write(os.path.join(runfolder, filename))
            self._config_data['pulse'].append(
                    pulse.config_line(filename, self._pulse_id, self.label))
        elif callable(pulse):
            ampl = np.array([pulse(t) for t in tgrid])
            p = Pulse(tgrid, amplitude=ampl,time_unit=self.T.unit,
                      ampl_unit='unitless', freq_unit='iu')
            p.write(os.path.join(runfolder, filename))
            self._config_data['pulse'].append(
                    p.config_line(filename, self._pulse_id, self.label))
        else:
            raise TypeError("Invalid pulse type")

    def _write_ham(self, runfolder):
        """Write operators in the Hamiltonian to data files inside runfolder,
        and add ham data to self._config_data"""
        ham = self.ham
        if not isinstance(ham, (list, tuple)):
            ham = [ham, ]
        op_counter = 0
        if 'ham' not in self._config_data:
            self._config_data['ham'] = []
        for element in ham:
            if isinstance(element, (list, tuple)):
                try:
                    H, pulse = element
                except ValueError:
                    raise ValueError("nested_op must be either an operator or "
                            "a list of operators or lists `[Op, pulse]`")
            else:
                H = element
                pulse = None
            if self.label == '':
                filename = "H%d.dat" % op_counter
            else:
                filename = "H_%s_%d.dat" % (self.label, op_counter)
            write_indexed_matrix(H,
                    filename=os.path.join(runfolder, filename),
                    hermitian=False)
            sparsity_model = choose_sparsity_model(H)
            self._config_data['ham'].append(
                    OrderedDict([('type', 'matrix'), ('n_surf', H.shape[0]),
                                 ('sparsity_model', sparsity_model),
                                 ('filename', filename),
                                 ('op_type', 'potential')]))
            if pulse is not None:
                self._config_data['ham'][-1]['pulse_id'] \
                = self._pulse_ids[pulse]
                self._config_data['ham'][-1]['op_type'] = 'dipole'
            if self.label != '':
                self._config_data['ham'][-1]['label'] = self.label
            op_counter +=1

    def _write_observables(self, runfolder):
        """Write operators describing all observables to the runfolder,
        and add observables data to self._config_data"""
        op_counter = 1
        for i, element in enumerate(self._observables):
            if isinstance(element, (list, tuple)):
                try:
                    O, pulse = element
                except ValueError:
                    raise ValueError("Each observable must either be an "
                            "operator or a two-element list consisting "
                            "of an operator and a pulse")
            else:
                O = element
                pulse = None
            if self.label == '':
                filename = "O%d.dat" % op_counter
            else:
                filename = "O_%s_%d.dat" % (self.label, op_counter)
            write_indexed_matrix(O,
                    filename=os.path.join(runfolder, filename),
                    hermitian=False)
            if 'observables' not in self._config_data:
                self._config_data['observables'] = []
            self._config_data['observables'].append(
                    OrderedDict(self._obs_config_attribs[i]))
            sparsity_model = choose_sparsity_model(O)
            self._config_data['observables'][-1].update(
                    OrderedDict([('type', 'matrix'), ('n_surf', O.shape[0]),
                                    ('sparsity_model', sparsity_model),
                                    ('filename', filename)]))
            if pulse is not None:
                self._config_data['observables'][-1]['pulse_id'] \
                = self._pulse_ids[pulse]
            if self.label != '':
                self._config_data['observables'][-1]['label'] = self.label
            op_counter +=1

    def _write_lindblad_ops(self, runfolder):
        """Write operators describing all Lindblad operators to the runfolder,
        and add dissipator data to self._config_data"""
        op_counter = 1
        for element in self._lindblad_ops:
            if isinstance(element, (list, tuple)):
                try:
                    L, pulse = element
                except ValueError:
                    raise ValueError("Each observable must either be an "
                            "operator or a two-element list consisting "
                            "of an operator and a pulse")
            else:
                L = element
                pulse = None
            if self.label == '':
                filename = "L%d.dat" % op_counter
            else:
                filename = "L_%s_%d.dat" % (self.label, op_counter)
            write_indexed_matrix(L,
                    filename=os.path.join(runfolder, filename),
                    hermitian=False)
            if 'dissipator' not in self._config_data:
                self._config_data['dissipator'] = []
            sparsity_model = choose_sparsity_model(L)
            self._config_data['dissipator'].append(
                    OrderedDict([('type', 'lindblad_ops'),
                                 ('sparsity_model', sparsity_model),
                                 ('conv_to_superop', False),
                                 ('filename', filename)]))
            if self.construct_mcwf_ham:
                self._config_data['dissipator'][-1]['add_to_H_jump'] \
                        = choose_sparsity_model(L)
            if pulse is not None:
                self._config_data['dissipator'][-1]['pulse_id'] \
                = self._pulse_ids[pulse]
            if self.label != '':
                self._config_data['dissipator'][-1]['label'] = self.label
            op_counter +=1

    def _write_psi(self, runfolder):
        """Write initial wave function to the runfolder, and add psi data to
        self._config_data"""
        if self.label == '':
            filename = "psi0.dat"
        else:
            filename = "psi_%s.dat" % self.label
        write_psi_amplitudes(self._psi, os.path.join(runfolder, filename))
        if 'psi' not in self._config_data:
            self._config_data['psi'] = []
        self._config_data['psi'].append(
                OrderedDict([('type', 'file'), ('filename', filename)]))
        if self.label != '':
            self._config_data['psi'][-1]['label'] = self.label


