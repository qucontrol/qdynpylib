"""
Module containing NumericConverter class
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import re

class NumericConverter:
    """ Manage the conversion of unit-numbers (e.g. 23121.4_cminv)  to pure
        numbers (in atomic units)
    """
    def __init__(self):
        """ Initialize conversion factors """
        self.au_convfactors = {
            'iu'      : 1.0,
            'au'      : 1.0,
            'fs'      : 0.02418884327440855991,  # atomictime to femtosecond
            'ps'      : 2.4188843274408560e-05,  # atomictime to picosecond
            'ns'      : 2.4188843274408560e-08,  # atomictime to nanosecond
            'microsec': 2.4188843274408560e-11,  # atomictime to microsecond
            'eV'      : 27.2113834492829731459,  # atomicenergy to electronvolt
            'cminv'   : 2.194746312856066506e5,  # atomicenergy to wavenumbers
            'K'       : 3.157746662555312504e5,  # atomicenergy to Kelvin
            'J'       : 4.3597438060897550e-18,  # atomicenergy to Joule
            'Hz'      : 6.579683918175572e15,    # atomicenergy to Hz * h
            'MHz'     : 6.579683918175572e9,     # atomicenergy to MHz * h
            'GHz'     : 6.579683918175572e6,     # atomicenergy to GHz * h
            'kHz'     : 6.579683918175572e12,    # atomicenergy to kHz * h
            'Vpm'     : 5.14220624463189208e11,  # electric field strength
            'm'       : 5.2917720827883533e-11,  # atomiclength to meter
            'nm'      : 5.2917720827883533e-2,   # atomiclength to nanometer
            'microm'  : 5.2917720827883535e-05,  # atomiclength to micrometer
            'pm'      : 52.917720827883533,      # atomiclength to picometer
            'angstrom': 0.5291772082788353,      # atomiclength to angstrom
            'kg'      : 9.1093818871545313e-31,  # atomicmass to kg
            'dalton'  : 5.48579911000000039e-4,  # atomicmass to dalton
        }
        self.patterns = {
            'int'        : re.compile(
                r'^([+-])?[0-9]+$'),
            'float'      : re.compile(
                r'^([+-])?[0-9]*\.[0-9]+([de][+-]?[0-9]+)?$'),
            'unit_float' : re.compile(
                r'^(?P<number>([+-])?[0-9]*\.[0-9]+([de][+-]?[0-9]+)?)'
                r'_(?P<unit>[a-zA-Z]+)$', re.X)
        }
    def parse(self, parse_string):
        """ 
        Take a string and try to return it as a number. If the string can
        be converted to an integer, return that integer. If the string can be
        converted to a float, return that float. If the string is a float with
        a unit (e.g. 23121.4_cminv), return the float converted to atomic
        units. If the unit is unrecognized, raise a KeyError. If the string
        cannot be parsed as either an int, a float or a unit-number, return the
        original string.
        """
        result = parse_string
        if self.patterns['int'].match(result):
            return int(result)
        float_match = self.patterns['float'].match(parse_string)
        if float_match:
            if 'd' in parse_string:
                result = result.replace("d", "e")
            return float(result)
        unit_float_match = self.patterns['unit_float'].match(parse_string)
        if unit_float_match:
            if 'd' in parse_string:
                result = result.replace("d", "e")
            number = unit_float_match.group('number')
            unit = unit_float_match.group('unit')
            return self.to_au(number, unit)
        return result
    def to_au(self, value, unit):
        """
        Convert value in given unit to atomic units. Raise KeyError if unit not
        recognized
        """
        return float(value) / self.au_convfactors[unit]
    def from_au(self, value, unit):
        """
        Convert value from atomic units to given unit. Raise KeyError if unit
        not recognized
        """
        return float(value) * self.au_convfactors[unit]
