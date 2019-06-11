"""Describing pulses by an analytical formula"""
import json
import re
from collections import OrderedDict
from inspect import getfullargspec

import numpy as np

from .linalg import iscomplexobj
from .pulse import Pulse, _PulseConfigAttribs
from .units import UnitConvert, UnitFloat


class _NumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    to a special object. The result can be decoded using the
    NumpyAwareJSONDecoder to recover the numpy arrays."""

    def default(self, obj):  # pylint: disable=method-hidden, arguments-differ
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return {'type': 'np.' + obj.dtype.name, 'vals': obj.tolist()}
        elif isinstance(obj, _PulseConfigAttribs):
            d = OrderedDict([])
            for key, val in obj.items():
                d[key] = val
            return d
        return json.JSONEncoder.default(self, obj)


class _SimpleNumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    a list. Note that this does NOT allow to recover the original numpy array
    from the JSON data"""

    def default(self, obj):  # pylint: disable=method-hidden, arguments-differ
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, _PulseConfigAttribs):
            d = OrderedDict([])
            for key, val in obj.items():
                d[key] = val
            return d
        return json.JSONEncoder.default(self, obj)


class _NumpyAwareJSONDecoder(json.JSONDecoder):
    """Decode JSON data that hs been encoded with _NumpyAwareJSONEncoder"""

    def __init__(self, *args, **kargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.dict_to_object, *args, **kargs
        )

    def dict_to_object(self, d):
        """Convert (type, vals) to type(vals) for numpy-array-types"""
        inst = d
        if (len(d) == 2) and ('type' in d) and ('vals' in d):
            typ = d['type']
            vals = d['vals']
            if typ.startswith("np."):
                dtype = typ[3:]
            inst = np.array(vals, dtype=dtype)
        return inst


class AnalyticalPulse:
    """Representation of a pulse determined by an analytical formula

    Args:
        formula (str): Name of a previously registered formula
        parameters (dict): Dictionary of values for the pulse formula
        time_unit (str or None): The unit of the `tgrid` input parameter of the
            formula (None is equivalent to 'iu')
        ampl_unit (str or None): Unit in which the amplitude is defined. It is
            assumed that the formula gives values in the correct unit.
        t0 (float or str or callable): Start time of the pulse
        T (float or str or callable or None): End time of the pulse
        nt (int or str or callable or None): Number of grid points between `t0`
            and `T` (inclusive)
        freq_unit (str or None): Preferred unit for pulse spectra. If None,
            unit will be chose automatically.
        config_attribs (dict or None): Additional config data, for when
            generating a QDYN config file section describing the pulse (e.g.
            `{'oct_shape': 'flattop', 't_rise': '10_ns'}`)

    Attributes:
        parameters (dict): Dictionary of values for the pulse formula
        time_unit (str or None): Value of the `time_unit` arg
        ampl_unit (str): Value of the `ampl_unit` arg
        freq_unit (str): Value of the `freq_unit` arg
        config_attribs (collections.abc.MutableMapping): dictionary with the
            items from the `config_attribs` arg
    Notes:
        The `t0`, `T`, and `nt` may be given to specify a time grid that is
        used by default when converting to a numerical pulse
        (:meth:`to_num_pulse`). They may be a numerical value, which will
        be used directly. Alternatively, they may be a string, which is a
        key in the `parameters` dict, and the value of the corresponding
        parameter will be used. Lastly, they may be a callable the receives
        the entire :class:`AnalyticalPulse` object as its argument and
        returns and appropriate numerical value.
    """

    _formulas = (
        {}
    )  # formula name => formula callable, see `register_formula()`
    _allowed_args = {}  # formula name => allowed arguments
    _required_args = {}  # formula name => required arguments
    unit_convert = UnitConvert()

    @classmethod
    def register_formula(cls, name, formula):
        """Register a new analytical formula

        Parameters:
            name (str): Label for the formula
            formula (callable): callable that takes an tgrid numpy array and an
                arbitrary number of (keyword) arguments and returns a numpy
                array of amplitude values
        """
        argspec = getfullargspec(formula)
        if len(argspec.args) < 1:
            raise ValueError(
                "formula has zero arguments, must take at least "
                "a tgrid parameter"
            )
        cls._formulas[name] = formula
        cls._allowed_args[name] = argspec.args[1:]
        n_opt = 0
        if argspec.defaults is not None:
            n_opt = len(argspec.defaults)
        cls._required_args[name] = argspec.args[1:-n_opt]

    def __init__(
        self,
        formula,
        parameters=None,
        time_unit=None,
        ampl_unit=None,
        t0=0.0,
        T=None,
        nt=None,
        freq_unit=None,
        config_attribs=None,
    ):
        if formula not in self._formulas:
            raise ValueError("Unknown formula '%s'" % formula)
        self._formula = formula
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self._check_parameters()
        self.time_unit = time_unit or 'iu'  # cannot be None
        self.ampl_unit = ampl_unit or 'iu'  # cannot be None
        self.freq_unit = freq_unit
        if freq_unit is None:
            try:
                self.freq_unit = Pulse.freq_units[self.time_unit]
            except KeyError:
                raise TypeError("freq_unit must be specified")
        self.config_attribs = _PulseConfigAttribs(self)
        if config_attribs is not None:
            for key in config_attribs:
                self.config_attribs[key] = config_attribs[key]
        self._t0 = t0
        self._T = T
        self._nt = nt

    def _get(self, attr):
        val = getattr(self, attr)
        if isinstance(val, str):
            return self.parameters[val]
        elif callable(val):
            return val(self)
        else:
            return val

    @property
    def t0(self):
        """Time at which the pulse begins (dt/2 before the first point in the
        pulse), as instance of :class:`~qdyn.units.UnitFloat`.
        """
        t0 = self._get('_t0')
        if t0 is None:
            return None
        else:
            return UnitFloat(t0, self.time_unit)

    @property
    def T(self):
        """Time at which the pulse ends (dt/2 after the last point in the
        pulse), as an instance of :class:`~qdyn.units.UnitFloat`.

        None if T was given as None in initialization.
        """
        T = self._get('_T')
        if T is None:
            return None
        else:
            return UnitFloat(T, self.time_unit)

    @property
    def tgrid(self):
        """Time grid points for the numerical pulse values, as numpy array in
        units of :attr:`time_unit`.

        None if missing `T`, `nt` in initialization.

        The returned time grid has ``nt - 1`` values, and extends from ``t0 +
        dt/2`` to ``T - dt/2``, matching the requirements for the `tgrid`
        argument of :class:`~qdyn.pulse.Pulse`.

        See also:
            :attr:`states_tgrid` is the time grid of length ``nt`` from ``t0``
            to ``T``
        """
        try:
            return np.linspace(
                float(self.t0 + self.dt / 2),
                float(self.T - self.dt / 2),
                self.nt - 1,
                dtype=np.float64,
            )
        except TypeError:
            return None

    @property
    def states_tgrid(self):
        """Time grid values for the states propagated under the numerical pulse
        values, as numpy array in units of :attr:`time_unit`.

        None if missing `T`, `nt` in initialization.

        The returned time grid has ``nt`` values, and extends from ``t0``
        to ``T`` (inclusive).

        See also:
            attr:`tgrid` is the time grid for the numerical pulse values of
            length ``nt-1``, extending from ``t0 + dt/2`` to ``T - dt/2``.
        """
        try:
            return np.linspace(
                float(self.t0), float(self.T), self.nt, dtype=np.float64
            )
        except TypeError:
            return None

    @property
    def dt(self):
        """Time grid step, as instance of :class:`~qdyn.units.UnitFloat`

        None if time grid is not defined (missing T, nt in initialization).
        """
        t0 = self.t0
        nt = self.nt
        T = self.T
        if T is None or t0 is None or nt is None:
            return None
        else:
            return (T - t0) / (nt - 1)

    @property
    def nt(self):
        """Number of time steps in the time grid between :attr:`t0` and
        :attr:`T`, as an integer.

        None if `nt` missing in initialization.

        Note that this is the length of :attr:`states_tgrid`, not of
        :attr:`tgrid`.
        """
        nt = self._get('_nt')
        if nt is None:
            return None
        else:
            return int(nt)

    @property
    def w_max(self):
        """Maximum frequency that can be represented with the
        current sampling rate.

        None if time grid is not defined (missing T, nt in initialization).
        """
        try:
            return self.to_num_pulse().w_max
        except AttributeError:
            return None

    @property
    def dw(self):
        """Step width in the spectrum (i.e. the spectral resolution)
        based on the current pulse duration, as an instance of
        :class:`~qdyn.units.UnitFloat`.

        None if time grid is not defined (missing T, nt in initialization).
        """
        try:
            return self.to_num_pulse().dw
        except AttributeError:
            return None

    @classmethod
    def from_func(
        cls,
        func,
        parameters=None,
        time_unit=None,
        ampl_unit=None,
        t0=0.0,
        T=None,
        nt=None,
        freq_unit=None,
        config_attribs=None,
    ):
        """Instantiate directly from a callable `func`, without the need to
        register the formula first

        The callable `func` must fulfill the same requirements as `formula` in
        :meth:`register_formula`
        """
        name = repr(func)
        cls.register_formula(name, func)
        return cls(
            name,
            parameters=parameters,
            time_unit=time_unit,
            ampl_unit=ampl_unit,
            t0=t0,
            T=T,
            nt=nt,
            freq_unit=freq_unit,
            config_attribs=config_attribs,
        )

    @property
    def is_complex(self):
        """Is the pulse amplitude of complex type?"""
        return iscomplexobj(
            # pylint: disable=not-callable
            self.evaluate_formula(np.zeros(1), **self.parameters)
        )

    def __eq__(self, other):
        """Compare two pulses, within a precision of 1e-12"""
        if not isinstance(other, self.__class__):
            return False
        attribs = (
            '_formula',
            '_T',
            '_t0',
            '_nt',
            'time_unit',
            'ampl_unit',
            'freq_unit',
            'config_attribs',
        )
        for attr in attribs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        for key in self.parameters:
            try:
                if abs(self.parameters[key] - other.parameters[key]) > 1.0e-12:
                    return False
            except TypeError:
                if self.parameters[key] != other.parameters[key]:
                    return False
        return True

    def copy(self):
        """Return a copy of the analytical pulse"""
        return self.__class__(
            self._formula,
            parameters=self.parameters,
            time_unit=self.time_unit,
            ampl_unit=self.ampl_unit,
            freq_unit=self.freq_unit,
            config_attribs=self.config_attribs,
            t0=self._t0,
            T=self._T,
            nt=self._nt,
        )

    def array_to_parameters(self, array, keys=None):
        """Unpack the given array (numpy array or regular list) into the pulse
        parameters. This is especially useful for optimizing parameters with
        the :func:`scipy.optimize.minimize` routine.

        For each key, set the value of the `parameters[key]` attribute by
        popping values from the beginning of the array. If `parameters[key]` is
        an array, pop repeatedly to set every value.

        If `keys` is not given, all parameter keys are used, in sorted order.
        The array must contain exactly enough parameters, otherwise an
        `IndexError` is raised.
        """
        if keys is None:
            keys = sorted(self.parameters.keys())
        array = list(array)
        for key in keys:
            if np.isscalar(self.parameters[key]):
                self.parameters[key] = array.pop(0)
            else:
                for i in range(len(self.parameters[key])):
                    self.parameters[key][i] = array.pop(0)
        if len(array) > 0:
            raise IndexError("not all values in array placed in parameters")

    def parameters_to_array(self, keys=None):
        """Inverse method to `array_to_parameters`. Returns the "packed"
        parameter values for the given keys as a numpy array"""
        result = []
        if keys is None:
            keys = sorted(self.parameters.keys())
        for key in keys:
            if np.isscalar(self.parameters[key]):
                result.append(self.parameters[key])
            else:
                for i in range(len(self.parameters[key])):
                    result.append(self.parameters[key][i])
        return np.array(result)

    def _check_parameters(self):
        """Raise a ValueError if self.parameters is missing any required
        parameters for the current formula. Also raise ValueError is
        self.parameters contains any extra parameters"""
        formula = self._formula
        for arg in self._required_args[formula]:
            if arg not in self.parameters:
                raise ValueError(
                    (
                        'Required parameter "%s" for formula "%s" '
                        'not in parameters %s'
                    )
                    % (arg, formula, self.parameters)
                )
        for arg in self.parameters:
            if arg not in self._allowed_args[formula]:
                raise ValueError(
                    ('Parameter "%s" does not exist in formula ' '"%s"')
                    % (arg, formula)
                )

    @property
    def formula_name(self):
        """Name of the analytical formula that is used"""
        return self._formula

    @property
    def evaluate_formula(self):
        """The callable that numerically evaluates the used formula, for
        arbitrary parameters (keyword arguments)"""
        return self._formulas[self._formula]

    def as_func(self, allow_args=False):
        """Callable that evaluates the pulse for a given time value.

        If `allow_args` is True, the resulting function takes two parameters,
        `t` and `args` where `t` is a float for the time value at which to
        evaluate the pulse (in units of :attr:`time_unit`), and `args` is a
        dictionary that allows to override attr:`parameters`.

        If `allow_args` is False, the resulting function will only take a
        single parameter `t`, and evaluate the function for fixed
        :attr:`parameters`.
        """

        def func(t):
            tgrid = np.array([t])
            ampl = self._formulas[self._formula](tgrid, **self.parameters)
            return ampl[0]

        def func_with_args(t, args):
            tgrid = np.array([t])
            ampl = self._formulas[self._formula](
                tgrid,
                **{**self.parameters, **args}
                # args overrides self.parameters
            )
            return ampl[0]

        if allow_args:
            return func_with_args
        return func

    def to_json(self, pretty=True):
        """Return a json representation of the pulse"""
        self._check_parameters()
        json_opts = {
            'indent': None,
            'separators': (',', ':'),
            'sort_keys': True,
        }
        if pretty:
            json_opts = {
                'indent': 2,
                'separators': (',', ': '),
                'sort_keys': True,
            }
        attributes = self.__dict__.copy()
        for key in ['formula', 't0', 'T', 'nt']:
            attributes[key] = attributes.pop('_%s' % key)
        return json.dumps(attributes, cls=_NumpyAwareJSONEncoder, **json_opts)

    def __str__(self):
        """Return string representation (JSON)"""
        return self.to_json(pretty=True)

    def write(self, filename, pretty=True):
        """Write the analytical pulse to the given filename as a json data
        structure"""
        with open(filename, 'w') as out_fh:
            out_fh.write(self.to_json(pretty=pretty))

    @property
    def header(self):
        """Single line summarizing the pulse. Suitable as preamble for
        numerical pulse"""
        result = '# Formula "%s"' % self._formula
        if len(self.parameters) > 0:
            result += ' with '
            json_opts = {
                'indent': None,
                'separators': (', ', ': '),
                'sort_keys': True,
            }
            json_str = json.dumps(
                self.parameters, cls=_SimpleNumpyAwareJSONEncoder, **json_opts
            )
            result += re.sub(r'"(\w+)": ', r'\1 = ', json_str[1:-1])
        return result

    @classmethod
    def read(cls, filename):
        """Read in a json data structure and return a new `AnalyticalPulse`"""
        with open(filename, 'r') as in_fh:
            kwargs = json.load(in_fh, cls=_NumpyAwareJSONDecoder)
            pulse = cls(**kwargs)
        return pulse

    def to_num_pulse(
        self, tgrid=None, time_unit=None, ampl_unit=None, freq_unit=None
    ):
        """Return a :class:`.Pulse` instance that contains the
        corresponding numerical pulse.

        Args:
            tgrid (numpy.ndarray or None): The time grid on which to evaluate
                the pulse. Use :func:`~qdyn.pulse.pulse_tgrid` to generate this.
            time_unit (str or None): Unit of `tgrid`
            ampl_unit (str or None): Unit of pulse amplitude
            freq_unit (str or None): Unit of pulse frequencies

        For any missing argument (None value), the corresponding attribute is
        used.

        Returns None if `tgrid` is not given explicitly and no time grid was
        defined on initialization (arguments `T`, `nt`)
        """
        self._check_parameters()
        if tgrid is None:
            tgrid = self.tgrid
        if tgrid is None:
            return None
        if time_unit is None:
            time_unit = self.time_unit
        if ampl_unit is None:
            ampl_unit = self.ampl_unit
        if freq_unit is None:
            freq_unit = self.freq_unit
        amplitude = self._formulas[self._formula](tgrid, **self.parameters)
        if time_unit != self.time_unit:
            tgrid = self.unit_convert.convert(tgrid, self.time_unit, time_unit)
        if ampl_unit != self.ampl_unit:
            amplitude = self.unit_convert.convert(
                amplitude, self.ampl_unit, ampl_unit
            )
        if not (isinstance(amplitude, np.ndarray) and amplitude.ndim == 1):
            raise TypeError(
                (
                    'Formula "%s" returned type %s instead of the '
                    'required 1-D numpy array'
                )
                % (self._formula, type(amplitude))
            )

        pulse = Pulse(
            tgrid=tgrid,
            amplitude=amplitude,
            time_unit=time_unit,
            ampl_unit=ampl_unit,
            freq_unit=freq_unit,
            config_attribs=self.config_attribs,
        )
        pulse.preamble = [self.header]
        return pulse
