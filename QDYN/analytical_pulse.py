"""Describing pulses by an analytical formula"""
from __future__ import print_function, division, absolute_import
import re
import json
from functools import partial
try:

    from inspect import getfullargspec
    getfullargspec(partial(lambda x: None))

except (ImportError, TypeError):  # Python < 3.4

    import inspect

    def getfullargspec(func):
        """Improved version of inspect.getargspec with support for
        functools.partial"""
        if inspect.ismethod(func):
            func = func.__func__
        parts = 0, ()
        if type(func) is partial:
            keywords = func.keywords
            if keywords is None:
                keywords = {}
            parts = len(func.args), keywords.keys()
            func = func.func
        if not inspect.isfunction(func):
            raise TypeError('%r is not a Python function' % func)
        args, varargs, varkw = inspect.getargs(func.__code__)
        func_defaults = func.__defaults__
        if func_defaults is None:
            func_defaults = []
        else:
            func_defaults = list(func_defaults)
        if parts[0]:
            args = args[parts[0]:]
        if parts[1]:
            for arg in parts[1]:
                i = args.index(arg) - len(args)
                del args[i]
                try:
                    del func_defaults[i]
                except IndexError:
                    pass
        return inspect.ArgSpec(args, varargs, varkw, func_defaults)

from collections import OrderedDict

import numpy as np

from .pulse import Pulse, pulse_tgrid, _PulseConfigAttribs
from .linalg import iscomplexobj
from .units import UnitConvert


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    to a special object. The result can be decoded using the
    NumpyAwareJSONDecoder to recover the numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return {'type': 'np.'+obj.dtype.name, 'vals': obj.tolist()}
        elif isinstance(obj, _PulseConfigAttribs):
            d = OrderedDict([])
            for key, val in obj.items():
                d[key] = val
            return d
        return json.JSONEncoder.default(self, obj)


class SimpleNumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    a list. Note that this does NOT allow to recover the original numpy array
    from the JSON data"""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, _PulseConfigAttribs):
            d = OrderedDict([])
            for key, val in obj.items():
                d[key] = val
            return d
        return json.JSONEncoder.default(self, obj)


class NumpyAwareJSONDecoder(json.JSONDecoder):
    """Decode JSON data that hs been encoded with NumpyAwareJSONEncoder"""
    def __init__(self, *args, **kargs):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object,
                                  *args, **kargs)
    def dict_to_object(self, d):
        """Convert (type, vals) to type(vals) for numpy-array-types"""
        inst = d
        if (len(d) == 2) and ('type' in d) and ('vals' in d):
            type = d['type']
            vals = d['vals']
            if type.startswith("np."):
                dtype = type[3:]
            inst = np.array(vals, dtype=dtype)
        return inst


class AnalyticalPulse(object):
    """Representation of a pulse determined by an analytical formula

    The `formula` parameter must be the name of a previously registered
    formula. All other parameters set the corresponding attribute.

    Attributes:
        parameters (dict): Dictionary of values for the pulse formula
        time_unit (str): The unit of the `tgrid` input parameter of the formula
        ampl_unit (str): Unit in which the amplitude is defined. It is assumed
            that the formula gives values in the correct .
        freq_unit (str, None): Preferred unit for pulse spectra
        mode ("real", "complex", or None): If None, the mode will be selected
            depending on the whether the formula returns real or complex
            values. When set explicitly, the formula *must* give matching
            values
        config_attribs (dict): Additional config data, for the `config_line`
            method (e.g. `{'oct_shape': 'flattop', 't_rise': '10_ns'}`)
    """
    _formulas = {} # formula name => formula callable, see `register_formula()`
    _allowed_args = {}  # formula name => allowed arguments
    _required_args = {} # formula name => required arguments
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
            raise ValueError("formula has zero arguments, must take at least "
                             "a tgrid parameter")
        cls._formulas[name] = formula
        cls._allowed_args[name] = argspec.args[1:]
        n_opt = 0
        if argspec.defaults is not None:
            n_opt = len(argspec.defaults)
        cls._required_args[name] = argspec.args[1:-n_opt]

    def __init__(
            self, formula, parameters=None, time_unit=None, ampl_unit=None,
            t0=0.0, freq_unit=None, config_attribs=None):
        if formula not in self._formulas:
            raise ValueError("Unknown formula '%s'" % formula)
        self._formula = formula
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self._check_parameters()
        self.time_unit = time_unit
        self.ampl_unit = ampl_unit
        self.freq_unit = freq_unit
        self.config_attribs = _PulseConfigAttribs(self)
        if config_attribs is not None:
            for key in config_attribs:
                self.config_attribs[key] = config_attribs[key]

    @classmethod
    def from_func(
            cls, func, parameters=None, time_unit=None, ampl_unit=None,
            freq_unit=None, config_attribs=None):
        """Instantiate directly from a callable `func`, without the need to
        register the formula first

        The callable `func` must fulfill the same requirements as `formula` in
        :meth:`register_formula`
        """
        name = repr(func)
        cls.register_formula(name, func)
        return cls(
            name, parameters=parameters, time_unit=time_unit,
            ampl_unit=ampl_unit, freq_unit=freq_unit,
            config_attribs=config_attribs)

    @property
    def is_complex(self):
        """Is the pulse amplitude of complex type?"""
        return iscomplexobj(
            self.evaluate_formula(np.zeros(1), **self.parameters))

    def __eq__(self, other):
        """Compare two pulses, within a precision of 1e-12"""
        if not isinstance(other, self.__class__):
            return False
        for attr in ('_formula', 'mode', 'time_unit', 'ampl_unit', 'freq_unit',
                     'mode', 'postamble', 'config_attribs'):
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
            self._formula, parameters=self.parameters,
            time_unit=self.time_unit, ampl_unit=self.ampl_unit,
            freq_unit=self.freq_unit, config_attribs=self.config_attribs)

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
            if not arg in self.parameters:
                raise ValueError(('Required parameter "%s" for formula "%s" '
                                  'not in parameters %s')%(arg, formula,
                                  self.parameters))
        for arg in self.parameters:
            if not arg in self._allowed_args[formula]:
                raise ValueError(('Parameter "%s" does not exist in formula '
                                  '"%s"')%(arg, formula))

    @property
    def formula_name(self):
        """Name of the analytical formula that is used"""
        return self._formula

    @property
    def evaluate_formula(self):
        """The callable that numerically evaluates the used formula, for
        arbitrary parameters (keyword arguments)"""
        return self._formulas[self._formula]

    def as_func(self):
        """Return a callable that evaluates the pulse for a given (scalar) time
        value, for fixed parameters
        """

        def func(t):
            tgrid = np.array([t, ])
            ampl = self._formulas[self._formula](tgrid, **self.parameters)
            return ampl[0]

        return func

    def to_json(self, pretty=True):
        """Return a json representation of the pulse"""
        self._check_parameters()
        json_opts = {'indent': None, 'separators': (',', ':'),
                     'sort_keys': True}
        if pretty:
            json_opts = {'indent': 2, 'separators': (',', ': '),
                         'sort_keys': True}
        attributes = self.__dict__.copy()
        attributes['formula'] = attributes.pop('_formula')
        return json.dumps(attributes, cls=NumpyAwareJSONEncoder,
                          **json_opts)

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
            json_opts = {'indent': None, 'separators':(', ',': '),
                        'sort_keys': True}
            json_str = json.dumps(self.parameters,
                                  cls=SimpleNumpyAwareJSONEncoder,
                                  **json_opts)
            result += re.sub(r'"(\w+)": ', r'\1 = ', json_str[1:-1])
        return result

    @classmethod
    def read(cls, filename):
        """Read in a json data structure and return a new `AnalyticalPulse`"""
        with open(filename, 'r') as in_fh:
            kwargs = json.load(in_fh, cls=NumpyAwareJSONDecoder)
            pulse = cls(**kwargs)
        return pulse

    def to_num_pulse(
            self, tgrid, time_unit=None, ampl_unit=None, freq_unit=None,
            mode=None):
        """Return a :cls:`~QDYN.pulse.Pulse` instance that contains the
        corresponding numerical pulse"""
        self._check_parameters()
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
            amplitude = self.unit_convert.convert(amplitude, self.ampl_unit,
                                                  ampl_unit)
        if (not isinstance(amplitude, np.ndarray)
        and amplitude.ndim != 1):
            raise TypeError(('Formula "%s" returned type %s instead of the '
                             'required 1-D numpy array')%(
                             self._formula, type(amplitude)))

        pulse = Pulse(tgrid=tgrid, amplitude=amplitude, time_unit=time_unit,
                      ampl_unit=ampl_unit, freq_unit=freq_unit,
                      config_attribs=self.config_attribs)
        pulse.preamble = [self.header, ]
        return pulse
