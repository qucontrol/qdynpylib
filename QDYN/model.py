"""Abstraction for complete physical models that can be written to a runfolder
as a config file and dependent data"""

from __future__ import print_function, division, absolute_import
from collections import OrderedDict, defaultdict
import os

import numpy as np

from .io import write_indexed_matrix
from .analytical_pulse import AnalyticalPulse
from .pulse import Pulse, pulse_tgrid, pulse_config_line
from .config import write_config
from .state import write_psi_amplitudes
from .shutil import mkdir
from .units import UnitFloat
from .linalg import is_hermitian, choose_sparsity_model, iscomplexobj


class _SimpleNamespace:
    """Implementation of types.SimpleNamespace for Python 2"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LevelModel(object):
    """Model for a system well-described in the energy basis. That is, all
    operators are (sparse) matrices, and all states are simple vectors

    Attributes:
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

        After instantiation, the attributes `t0`, `T`, `nt`, `prop_method`,
        `use-mcwf`, and `construct_mcwf_ham` are all set via
        :meth:`set_propagation`.  Operators and pulses are added to the system
        through :meth:`add_ham`, :meth:`add_observable`, and
        :meth:`add_lindblad_op`. States are added through :meth:`add_state`.
        Both the general OCT settings (OCT section in the QDYN config file) and
        OCT-related settings for each control pulse are controlled through
        :meth:`set_oct`. After the model has been constructed, a config file
        and all dependent data input files for the operators, pulses, and
        states can be written via :meth:`write_to_runfolder`.
    """

    def __init__(self):
        self._ham = []  # list of (matrix, config_attribs)
        self._pulses = []  # list of (pulse, config_attribs)
        # the pulse config_attribs store the label, id, and filename, as well
        # as any attributes set by the `set_oct` method. Additional attributes
        # may exist in `pulse.config_attribs` of each pulse. All of these are
        # collected in the `pulses` method
        self._lindblad_ops = []  # list of (matrix, config_attribs)
        self._observables = []  # list of (matrix, config_attribs)
        self._psi = OrderedDict([])  # label => amplitude array
        self._oct = OrderedDict([])  # key => val for OCT section
        self.t0 = UnitFloat(0, 'iu')
        self.T = UnitFloat(0, 'iu')
        self.nt = 0
        self.prop_method = 'newton'
        self.use_mcwf = False
        self.mcwf_order = 2
        self.construct_mcwf_ham = False
        self._pulse_id = defaultdict(int)  # last used pulse_id, per label
        self._pulse_ids = {}  # (pulse, label) -> pulse_id

    @staticmethod
    def _obj_list(obj_list, label=None, with_attribs=False):
        """Common implementation of methods `observables`, 'lindbla_ops`,
        `ham`"""
        if label is None:
            label = ''
        result = []
        for (obj, attribs) in obj_list:
            obj_label = attribs.get("label", "")
            if (obj_label == label) or (label == '*'):
                if with_attribs:
                    result.append((obj, attribs))
                else:
                    result.append(obj)
        return result

    def observables(self, label=None, with_attribs=False):
        """Return list of all observables with the matching label (or all
        labels if `label` is '*'). If `with_attribs` is True, the result is a
        list of tuples ``(observable, attributes``) where ``attributes`` is a
        dictionary of config file attributes
        """
        return self._obj_list(self._observables, label, with_attribs)

    def lindblad_ops(self, label=None, with_attribs=False):
        """Return list of all Lindblad operators with the matching label (or
        all labels if `label` is '*'). If `with_attribs` is True, the result is
        a list of tuples ``(operator, attributes``) where ``attributes`` is a
        dictionary of config file attributes
        """
        return self._obj_list(self._lindblad_ops, label, with_attribs)

    def ham(self, label=None, with_attribs=False):
        """Return list of all Hamiltonian operators with the matching label (or
        all labels if `label` is '*'). If `with_attribs` is True, the result is
        a list of tuples ``(ham, attributes``) where ``attributes`` is a
        dictionary of config file attributes
        """
        return self._obj_list(self._ham, label, with_attribs)

    def pulses(self, label=None, with_attribs=False):
        """Return a list of all known pulses with the matching label (or all
        labels if `label` is '*'). The pulses originate from calls to e.g.
        `add_ham`. That is, they are introduced to the system via operators
        that couple to them. If `with_attribs` is True, the result is a list of
        tuples ``(pulse, attributes)`` where ``attributes`` is a (combined,
        read-only) dictionary of all config file attributes for the pulse.
        """
        if label is None:
            label = ''
        result = []
        for pulse, attribs in self._pulses:
            pulse_label = attribs['label']
            pulse_id = attribs['id']
            filename = attribs['filename']
            if (pulse_label == label) or (label == '*'):
                if with_attribs:
                    if callable(pulse):
                        p = _SimpleNamespace(
                                time_unit=self.T.unit, ampl_unit='unitless',
                                is_complex=iscomplexobj(pulse(0)),
                                config_attribs=[])
                    else:
                        p = pulse
                    config_attribs = pulse_config_line(p, filename, pulse_id,
                                                       pulse_label,
                                                       config_attribs=attribs)
                    result.append((pulse, config_attribs))
                else:
                    result.append(pulse)
        return result

    def _add_matrix(self, add_target, matrix, label, pulse=None,
                    check_matrix=True, kwargs=None):
        """Common implementation of `add_ham`, `add_observable`,
        `add_lindblad_op`"""
        if kwargs is None:
            # Note: we do not use **kwargs to preserve an OrderedDict
            kwargs = {}
        if check_matrix:
            if not hasattr(matrix, 'shape'):
                # we take the existence of the 'shape' attribute as a
                # least-effort indication that we have a proper matrix
                raise TypeError(str(matrix) + ' must be a matrix')
        if pulse is not None:
            if not (isinstance(pulse, (Pulse, AnalyticalPulse))
                    or callable(pulse)):
                raise TypeError("pulse must be an instance of "
                        "QDYN.analytical_pulse.AnalyticalPulse, "
                        "QDYN.pulse.Pulse, or a callable.")
        config_attribs = OrderedDict([])
        for key in kwargs:
            config_attribs[key] = kwargs[key]
        if label is not None:
            config_attribs['label'] = label
        if pulse is not None:
            pulse_id = self._add_pulse(pulse, system_label=label)
            config_attribs['pulse_id'] = pulse_id
        add_target.append(
            (matrix, config_attribs)
        )

    def _add_pulse(self, pulse, system_label):
        """Determine the `config_attribs` ``label``, ``id``, and ``filename``
        from the given `pulse`. The ``label`` is taken either from
        ``pulse.config_attribs['label']`` or from `system_label`. The
        ``filename`` is taken from either ``pulse.config_attribs['filename']``
        or it is chosen internally. If the combination ``(pulse, label)`` has
        not been encountered before, store ``(pulse, config_attribs)`` in
        `_pulses`. Increments ``self._pulse_id[label]`` and sets
        ``self._pulse_ids[(pulse, label)]``

        Return `pulse_id`
        """
        try:
            label = pulse.config_attribs['label']
        except (AttributeError, KeyError):
            label = system_label
        if label is None:
            label = ''
        key = (pulse, label)
        try:
            pulse_id = self._pulse_ids[key]
        except KeyError:
            self._pulse_id[label] += 1
            pulse_id = self._pulse_id[label]
        try:
            filename = pulse.config_attribs['filename']
        except (AttributeError, KeyError):
            if label == '':
                filename = "pulse%d.dat" % pulse_id
            else:
                filename = "pulse%d_%s.dat" % (pulse_id, label)
        if key not in self._pulse_ids:
            config_attribs = OrderedDict([
                ('label', label), ('id', pulse_id), ('filename', filename)
            ])
            self._pulses.append(
                (pulse, config_attribs)
            )
            self._pulse_ids[key] = pulse_id
        return pulse_id

    def add_ham(self, H, pulse=None, op_unit=None, sparsity_model=None,
            op_type=None, label=None, **kwargs):
        """Add a term to the system Hamiltonian. If called repeatedly, the
        total Hamiltonian is the sum of all added terms.

        Args:
            H: Hamiltonian matrix. Can be a numpy matrix or array,
                scipy sparse matrix, or `qutip.Qobj`
            pulse: if not None, `H` will couple to `pulse`. Can be an instance
                of `QDYN.analytical_pulse.AnalyticalPulse` (preferred, this is
                the only option to fully specify units), `QDYN.pulse.Pulse`, or
                a callable ``pulse(t)`` that returns a pulse value for a given
                float ``t`` (time)
            op_unit (None or str): Unit of the values in `H`.
            sparsity_model (None or str): sparsity model that QDYN should use
                to encode the data in `H`. If None, will be determined
                automatically
            op_type (None or str): the value for ``op_type`` in the config file
                that should be used for `H`. This determines how exactly `H`
                couples to `pulse`. Common values are 'dipole' (linear
                coupling) and 'dstark' (quadratic "Stark shift" coupling)
            label (str or None): Multiple Hamiltonians may be defined in the
                same config file if they are differentiated by label. The
                default label is the empty string
            kwargs: All other keyword arguments set options for `H` in the
                config file (e.g. `specrad_method`, `filename`)

        Note:
            It is recommended to us `QDYN.pulse.AnalyticalPulse` to express
            time-dependency, as this allows to fully specify physical units.
            Instances of `QDYN.pulse.Pulse` must have a time grid that exactly
            matches the time grid specified via :meth:`set_propagation`. Using
            a callable means that the pulse amplitude is unitless, and the
            pulse as evaluated for time value with the unit specified in
            :meth:`set_propagation`.
        """
        config_attribs = OrderedDict([])
        for key in sorted(kwargs):
            config_attribs[key] = kwargs[key]
        if op_unit is not None:
            config_attribs['op_unit'] = op_unit
        if sparsity_model is not None:
            config_attribs['sparsity_model'] = sparsity_model
        if op_type is not None:
            config_attribs['op_type'] = op_type
        if label is not None:
            config_attribs['label'] = label
        self._add_matrix(self._ham, H, label=label, pulse=pulse,
                         kwargs=config_attribs)

    def add_observable(self, O, outfile, exp_unit, time_unit, col_label,
            square=None, exp_surf=None, is_real=None, in_lab_frame=False,
            op_unit=None, label=None, pulse=None):
        """Add an observable

        Args:
            O (matrix, str): Observable to add. Must be a matrix, or
                one of "ham", "norm", "pop"
            outfile (str): Name of output file to which to write expectation
                values of `O`
            exp_unit (str): The unit in which to write the expectation value in
                `outfile`, as well as the default value for `op_unit`
            time_unit (str):  The unit in which to write the time grid in
                `outfile`
            col_label (str): The label for the column in `outfile` containing
                the expectation value of `O`
            square (str or None): If not None, label for the column in
                `outfile` containing the expectation value for the square of
                `O`
            exp_surf (int or None): The surface number the expectation value;
                only if `O` is a string
            is_real (bool or None): Whether or not the expectation value is
                real. If not given, this is determined  automatically
            in_lab_frame (bool): If True, indicates that the observable is
                defined in the lab frame. When expectation values are
                calculated, this should not be done with states in the rotating
                frame.
            op_unit (str or None): The unit in which the entries of `O` are
                given. By default, this is the same as `exp_unit`.
            label (str or None): Observables associated with different
                Hamiltonians may be defined in the same config file if they are
                differentiated by label. The default label is the empty string.
            pulse: If not None, a pulse for the observable to couple to (see
                `add_ham`)
        """
        config_attribs = OrderedDict([])
        if is_real is None:
            if isinstance(O, str):
                if O in ["norm", "pop"]:
                    is_real = True
                elif O == "ham":
                    is_real = False
                else:
                    raise ValueError("Invalid O")
            else:
                is_real = is_hermitian(O)
        config_attribs['outfile'] = outfile
        config_attribs['exp_unit'] = exp_unit
        config_attribs['is_real'] = is_real
        config_attribs['time_unit'] = time_unit
        config_attribs['column_label'] = col_label
        if square is not None:
            config_attribs['square'] = square
        if exp_surf is not None:
            config_attribs['exp_surf'] = exp_surf
        if label is not None:
            config_attribs['label'] = label
        if in_lab_frame:
            config_attribs['in_lab_frame'] = True
        if op_unit is None:
            config_attribs['op_unit'] = exp_unit
        else:
            config_attribs['op_unit'] = op_unit
        self._add_matrix(self._observables, O, label, pulse=pulse,
                         check_matrix=False, kwargs=config_attribs)

    def add_lindblad_op(self, L, op_unit=None, sparsity_model=None,
            add_to_H_jump=None, label=None, pulse=None, **kwargs):
        """Add Lindblad operator.

        Args:
            L (tuple, matrix): Lindblad operator to  add. Must be a matrix
                or a tuple ``(matrix, pulse)``, cf. the `ham` attribute.
            op_unit (None or str): Unit of the values in `L`.
            sparsity_model (None or str): sparsity model that QDYN should use
                to encode the data in `L`. If None, will be determined
                automatically
            add_to_H_jump (None, str): sparsity model to be set for
                `add_to_H_jump`, i.e. for the decay term in the effective MCWF
                Hamiltonian. If None, will be de determined automatically.
            label (str or None): Lindblad operators associated with different
                Hamiltonians may be defined in the same config file if they are
                differentiated by label. The default label is the empty string.
            pulse: If not None, a pulse for the Lindblad operator to couple to
                (see `add_ham`)

        All other keyword arguments set options for `L` in the config file.
        """
        config_attribs = OrderedDict([])
        for key in sorted(kwargs):
            config_attribs[key] = kwargs[key]
        if op_unit is not None:
            config_attribs['op_unit'] = op_unit
        if sparsity_model is not None:
            config_attribs['sparsity_model'] = sparsity_model
        if add_to_H_jump is not None:
            config_attribs['add_to_H_jump'] = add_to_H_jump
        if label is not None:
            config_attribs['label'] = label
        self._add_matrix(self._lindblad_ops, L, label, pulse=pulse,
                         kwargs=config_attribs)

    def set_propagation(self, T, nt, time_unit, t0=0.0,
            prop_method='newton', use_mcwf=False, mcwf_order=2,
            construct_mcwf_ham=True, label=None, initial_state=None):
        """Set the time grid and other propagation-specific settings

        Args:
            T (float): final time
            nt (int): number of points in the time grid
            time_unit (str): physical unit of `T`, `t0`
            t0 (float): initial time
            prop_method (str): method to be used for propagation
            use_mcwf (bool): If True, prepare for Monte-Carlo wave function
                propagation
            mcwf_order (int): Order for MCWF, must be 1 or 2
            construct_mcwf_ham (bool): When using MCWF (`use_mcwf=True`), by
                default an additional inhomogeneous decay term is added to the
                Hamiltonian, based on the Lindblad operators. By passing
                `construct_mcwf_ham=False`, this does not happen. It is the
                user's responsibility then to ensure the Hamiltonian in the
                model is the correct "effective" Hamiltonian for a MCWF
                propagation.
            label (str or None): The label for `initial_state`
            initial_state (array or None): Initial wave function

        Notes:
            When setting up an MCWF propagation, using the `mcwf_order=2` is
            usually the right thing to do. In some cases of strong dissipation,
            it may be numerically more efficient to use a first-order MCWF
            method, where in each time interval `dt` between two time grids
            there is at most one quantum jump, and the jumps takes place over
            the entire duration `dt`. The time steps must be very small
            accordingly. In contrast, for `mcwf_order=2`, all jumps are
            instantaneous, and there can be multiple jumps per time step, but
            the numerical overhead is larger.
        """
        if initial_state is not None:
            self.add_state(initial_state, label)
        self.T = UnitFloat(T, time_unit)
        self.nt = nt
        self.t0 = UnitFloat(t0, time_unit)
        self.prop_method = prop_method
        if use_mcwf:
            self.use_mcwf = use_mcwf
            self.mcwf_order = mcwf_order
        if construct_mcwf_ham is None:
            construct_mcwf_ham = use_mcwf
        if use_mcwf:
            self.construct_mcwf_ham = construct_mcwf_ham
        else:
            self.construct_mcwf_ham = False

    def set_oct(self, pulse_settings, method, J_T_conv, **kwargs):
        """Set config file data and pulse properties for optimal control

        Args:
            pulse_settings (dict): OCT settings for each known pulse. Must map
                each pulse (see :meth:`pulses`) to an `OrderedDict` of oct
                settings
            method (str): Optimization method. Allowed values are 'krotovpk',
                'krotov2', 'krotovbwr', and 'lbfgs'
            J_T_conv (foat): The value of the final time functional

        All other keyword arguments directly specify keys and values for the
        OCT config section. Allowed keys are `iter_start`, `iter_stop`,
        `max_ram_mb`, `max_disk_mb`, `lbfgs_memory`, `linesearch`,
        `grad_order`, `iter_dat`, `tau_dat`, `params_file`, `krotov2_conv_dat`,
        `ABC_dat`, `delta_J_conv`, `delta_J_T_conv`, `A`, `B`, `C`,
        `dynamic_sigma`, `dynamic_lambda_a`, `strict_convergence`,
        `limit_pulses`, `sigma_form`, `max_seconds`, `lambda_b`, `keep_pulses`,
        `re_init_prop`, `continue`, `storage_folder`, `bwr_nint`, `bwr_base`,
        `g_a`, see the QDYN Fortran library documentation for details.

        The settings for each pulse given by ``pulse_settings[pulse]``. It is
        recommended to be an `OrderedDict`, and may
        contain the keys `oct_parametrization`, `oct_pulse_max`,
        `oct_pulse_min`, `shape_t_start`, `shape_t_stop`, `ftbaseline`,
        `t_rise`, `t_fall`, `oct_lambda_intens`, `oct_lambda_a`,
        `oct_shape` (default to ``'const'``), `oct_outfile` (default to
        ``<file>.oct.dat``, where ``<file>`` is the pulse `filename` without
        extension), and `oct_increase_factor` (default 5.0).  See the QDYN
        Fortran library documentation for details. If Krotov's method is used,
        a value for `oct_lambda_a` must be given.

        Since the `set_oct` method sets attribute for existing pulses, it must
        be called *after* any methods that add the pulses (e.g.,
        :meth:`add_ham`)

        Raises:
            KeyError: If `pulse_settings` does not contain settings for all
                pulses, or if the settings for a pulse contain invalid or
                missing keys
        """
        allowed_keys = [
            'iter_start', 'iter_stop', 'max_ram_mb', 'max_disk_mb',
            'lbfgs_memory', 'linesearch', 'grad_order', 'iter_dat', 'tau_dat',
            'params_file', 'krotov2_conv_dat', 'ABC_dat', 'delta_J_conv',
            'delta_J_T_conv', 'A', 'B', 'C', 'dynamic_sigma',
            'dynamic_lambda_a', 'strict_convergence', 'limit_pulses',
            'sigma_form', 'max_seconds', 'lambda_b', 'keep_pulses',
            're_init_prop', 'continue', 'storage_folder', 'bwr_nint',
            'bwr_base', 'g_a'
        ]
        allowed_pulse_keys = [
            'oct_parametrization', 'oct_pulse_max', 'oct_pulse_min',
            'shape_t_start', 'shape_t_stop', 'ftbaseline', 't_rise', 't_fall',
            'oct_lambda_intens', 'oct_lambda_a', 'oct_shape', 'oct_outfile',
            'oct_increase_factor'
        ]
        krotov_required_pulse_keys = ['oct_lambda_a', ]

        def default_outfile(filename):
            return os.path.splitext(filename)[0] + ".oct.dat"

        self._oct = OrderedDict([('method', method), ('J_T_conv', J_T_conv)])
        for key in sorted(kwargs):
            if key in allowed_keys:
                self._oct[key] = kwargs[key]
            else:
                raise TypeError("got an unexpected keyword argument '%s'"
                                % key)
        for pulse, attribs in self._pulses:
            try:
                oct_attribs = pulse_settings[pulse]
            except KeyError:
                raise KeyError("pulse_settings must contain settings for all "
                               "pulses. Missing pulse: %s" % str(pulse))
            oct_attribs['oct_shape'] = oct_attribs.get('oct_shape', "const")
            octout = default_outfile(attribs['filename'])
            oct_attribs['oct_outfile'] = oct_attribs.get('oct_outfile', octout)
            oct_attribs['oct_increase_factor'] = \
                oct_attribs.get('oct_increase_factor', 5.0)
            if 'krotov' in method:
                for key in krotov_required_pulse_keys:
                    if key not in oct_attribs:
                        raise KeyError("Key '%s' is required for each pulse "
                                       "when using Krotov's method" % key)
            for key in oct_attribs:
                if key not in allowed_pulse_keys:
                    raise KeyError("Invalid key '%s'" % key)
            attribs.update(oct_attribs)

    def add_state(self, state, label):
        """Add a state (amplitude array) for the given label. Note that there
        can only be one state per label. Thus calling `add_state` with the same
        `label` of an earlier call will replace the `state`"""
        if label is None:
            label = ''
        self._psi[label] = state

    def write_to_runfolder(self, runfolder, config='config'):
        """Write the model to the given runfolder. This will create a config
        file (`config`) in the runfolder, as well as all dependent data file
        (operators, pulses)"""
        mkdir(runfolder)
        config_data = OrderedDict([])

        # time grid
        if self.nt > 0:
            config_data['tgrid'] = OrderedDict([
                ('t_start', self.t0), ('t_stop', self.T), ('nt', self.nt)])

        # propagation
        config_data['prop'] = OrderedDict([
            ('method', self.prop_method), ('use_mcwf', self.use_mcwf)])
        if self.use_mcwf:
            if self.mcwf_order not in [1, 2]:
                raise ValueError('mcwf_order must be in [1,2]')
            config_data['prop']['mcwf_order'] = self.mcwf_order

        # pulses
        self._write_pulses(runfolder, config_data)

        # Hamiltonian
        if len(self._ham) > 0:
            self._write_ham(runfolder, config_data)

        # Lindblad operators
        if len(self._lindblad_ops) > 0:
            self._write_lindblad_ops(runfolder, config_data)

        # observables
        if len(self._observables) > 0:
            self._write_observables(runfolder, config_data)

        # states
        if len(self._psi) > 0:
            self._write_psi(runfolder, config_data)

        # OCT
        if len(self._oct) > 0:
            config_data['oct'] = self._oct

        write_config(config_data, os.path.join(runfolder, config))

    def _write_pulses(self, runfolder, config_data):
        """Write numerical pulse files to runfolder, add pulse data to
        config_data,"""
        tgrid = pulse_tgrid(self.T, self.nt, self.t0)
        if 'pulse' not in config_data:
            config_data['pulse'] = []
        for pulse, attribs in self.pulses(label='*', with_attribs=True):
            filename = attribs['filename']
            if isinstance(pulse, AnalyticalPulse):
                p = pulse.to_num_pulse(tgrid)
                p.write(os.path.join(runfolder, filename))
                config_data['pulse'].append(attribs)
            elif isinstance(pulse, Pulse):
                if np.max(np.abs(tgrid - pulse.tgrid)) > 1e-12:
                    raise ValueError("Mismatch of tgrid with pulse tgrid")
                pulse.write(os.path.join(runfolder, filename))
                config_data['pulse'].append(attribs)
            elif callable(pulse):
                ampl = np.array([pulse(t) for t in tgrid])
                p = Pulse(tgrid, amplitude=ampl, time_unit=self.T.unit,
                          ampl_unit='unitless', freq_unit='iu')
                p.write(os.path.join(runfolder, filename))
                config_data['pulse'].append(attribs)
            else:
                raise TypeError("Invalid pulse type")

    @staticmethod
    def _write_matrices(runfolder, config_data, section, data, outprefix,
            type_attrib='matrix', set_n_surf=True, set_op_type=False,
            set_add_to_H_jump=False, counter0=0):
        """Common implementation of `_write_ham`, `_write_observables`, and
        `_write_lindblad_ops`"""
        if section not in config_data:
            config_data[section] = []
        for op_counter, (matrix, attribs) in enumerate(data):
            if 'filename' in attribs:
                filename = attribs['filename']
            else:
                filename = "%s%d.dat" % (outprefix, op_counter+counter0)
            write_indexed_matrix(matrix,
                    filename=os.path.join(runfolder, filename),
                    hermitian=False)
            config_attribs = OrderedDict([
                ('type', type_attrib), ('filename', filename),
                ('real_op', (not iscomplexobj(matrix)))
            ])
            if set_n_surf:
                config_attribs['n_surf'] = matrix.shape[0]
            for key, val in attribs.items():
                config_attribs[key] = val
            if 'sparsity_model' not in config_attribs:
                config_attribs['sparsity_model'] \
                        = choose_sparsity_model(matrix)
            if set_op_type:
                if 'op_type' not in config_attribs:
                    config_attribs['op_type'] = 'pot'
                    if 'pulse_id' in attribs:
                        config_attribs['op_type'] = 'dip'
            config_data[section].append(config_attribs)

    def _write_ham(self, runfolder, config_data):
        """Write operators in the Hamiltonian to data files inside `runfolder`,
        and add ham data to `config_data`"""
        self._write_matrices(runfolder, config_data, 'ham', self._ham,
                             outprefix='H', set_op_type=True, counter0=0)

    def _write_observables(self, runfolder, config_data):
        """Write operators describing all observables to the runfolder,
        and add observables data to `config_data`"""
        self._write_matrices(runfolder, config_data, 'observables',
                             self._observables, outprefix='O',
                             set_op_type=False, counter0=1)

    def _write_lindblad_ops(self, runfolder, config_data):
        """Write operators describing all Lindblad operators to the
        `runfolder`, and add dissipator data to `config_data`"""
        lindblad_ops = list(self._lindblad_ops)  # copy
        for L, attribs in lindblad_ops:
            if self.construct_mcwf_ham:
                if 'add_to_H_jump' not in attribs:
                    # TODO: choose sparsity model that is optimal for the
                    # entire decay term
                    attribs['add_to_H_jump'] = choose_sparsity_model(L)
                attribs['conv_to_superop'] = False
            else:
                if 'add_to_H_jump' in attribs:
                    # add_to_H_jump should never be defined outside of the MCWF
                    # method
                    del attribs['add_to_H_jump']
        self._write_matrices(runfolder, config_data, 'dissipator',
                             lindblad_ops, outprefix='L', set_n_surf=False,
                             type_attrib='lindblad_ops', counter0=1)

    def _write_psi(self, runfolder, config_data):
        """Write initial wave function to the runfolder, and add psi data to
        config_data"""
        if 'psi' not in config_data:
            config_data['psi'] = []
        for psi_counter, (label, psi) in enumerate(self._psi.items()):
            filename = "psi%d.dat" % psi_counter
            write_psi_amplitudes(psi, os.path.join(runfolder, filename))
            if 'psi' not in config_data:
                config_data['psi'] = []
            config_data['psi'].append(
                    OrderedDict([('type', 'file'), ('filename', filename)]))
            if label != '':
                config_data['psi'][-1]['label'] = label

