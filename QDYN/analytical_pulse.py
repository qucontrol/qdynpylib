#!/usr/bin/env python
"""Describing pulses by an analytical formula"""
from __future__ import print_function, division, absolute_import
import re
import json
import inspect
from collections import OrderedDict

import numpy as np

from .pulse import Pulse, pulse_tgrid
from .units import UnitConvert


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    to a special object. The result can be decoded using the
    NumpyAwareJSONDecoder to recover the numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return {'type': 'np.'+obj.dtype.name, 'vals' :obj.tolist()}
        return json.JSONEncoder.default(self, obj)


class SimpleNumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    a list. Note that this does NOT allow to recover the original numpy array
    from the JSON data"""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
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
        t0 (float): Starting point of the pulse. When converting an analytical
            pulse to a numerical pulse, the first pulse value is at
            ``t0 + dt/2``)
        nt (integer): Number of time grid points. When converting an analytical
            pulse to a numerical pulse, the pulse will have nt-1 values
        T (float): End point of the pulse. When converting an analytical pulse
            to a numerical pulse, the last pulse value is at ``T - dt/2``
        parameters (dict): Dictionary of values for the pulse formula
        time_unit (str): Unit in which t0 and T are given
        ampl_unit (str): Unit in which the amplitude is defined. It is assumed
            that the formula gives values in the correct amplitude.
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
        argspec = inspect.getargspec(formula)
        # TODO: use https://github.com/aliles/funcsigs
        if len(argspec.args) < 1:
            raise ValueError("formula has zero arguments, must take at least "
                             "a tgrid parameter")
        cls._formulas[name] = formula
        cls._allowed_args[name] = argspec.args[1:]
        n_opt = 0
        if argspec.defaults is not None:
            n_opt = len(argspec.defaults)
        cls._required_args[name] = argspec.args[1:-n_opt]


    def __init__(self, formula, T, nt, parameters, time_unit,
        ampl_unit, t0=0.0, freq_unit=None, mode=None, config_attribs=None):
        if not formula in self._formulas:
            raise ValueError("Unknown formula '%s'" % formula)
        self._formula = formula
        self.parameters = parameters
        self._check_parameters()
        self.t0 = t0
        self.nt = nt
        self.T = T
        self.time_unit = time_unit
        self.ampl_unit = ampl_unit
        self.freq_unit = freq_unit
        self.mode = mode
        self.config_attribs = OrderedDict({})
        if config_attribs is not None:
            self.config_attribs = config_attribs

    def copy(self):
        """Return a copy of the analytical pulse"""
        return AnalyticalPulse(self._formula, self.T, self.nt, self.parameters,
                               self.t0, self.time_unit, self.ampl_unit,
                               self.freq_unit, self.mode)

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
        """The callable that numerically evaluates the used formula"""
        return self._formulas[self._formula]

    def to_json(self, pretty=True):
        """Return a json representation of the pulse"""
        self._check_parameters()
        json_opts = {'indent': None, 'separators':(',',':'), 'sort_keys': True}
        if pretty:
            json_opts = {'indent': 2, 'separators':(',',': '),
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

    def to_num_pulse(self, tgrid=None, time_unit=None, ampl_unit=None,
            freq_unit=None, mode=None):
        """Return a :cls:`QDYN.pulse.Pulse` instance that contains the
        corresponding numerical pulse"""
        self._check_parameters()
        if tgrid is None:
            tgrid = pulse_tgrid(self.T, self.nt, self.t0)
        if time_unit is None:
            time_unit = self.time_unit
        if ampl_unit is None:
            ampl_unit = self.ampl_unit
        if freq_unit is None:
            freq_unit = self.freq_unit
        if mode is None:
            mode = self.mode
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
        if mode is None:
            if np.isrealobj(amplitude):
                mode = 'real'
            else:
                mode = 'complex'
        else:
            if mode == 'real' and not np.isrealobj(amplitude):
                if np.max(np.abs(amplitude.imag)) > 0.0:
                    raise ValueError("mode is 'real', but amplitude has "
                                     "non-zero imaginary part")

        pulse = Pulse(tgrid=tgrid, amplitude=amplitude, time_unit=time_unit,
                      ampl_unit=ampl_unit, freq_unit=freq_unit, mode=mode)
        pulse.preamble = [self.header, ]
        pulse.config_attribs = OrderedDict(self.config_attribs)
        return pulse

    def config_line(self, filename, pulse_id, label=''):
        """Return an OrderedDict of attributes for a config file line
        describing the pulse"""
        result = OrderedDict(self.config_attribs)
        result.update(OrderedDict([
            ('type', 'file'), ('filename', filename), ('id', pulse_id),
            ('time_unit', self.time_unit), ('ampl_unit', self.ampl_unit)]))
        if label != '':
            result['label'] = label
        if self.mode == 'complex':
            result['is_complex'] = True

        return result

