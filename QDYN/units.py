"""Conversion between physical units"""
from __future__ import print_function, division
from functools import wraps
import uuid
import math
import re

DEFAULT_UNITS = r'''
# ------------------------------------------------------------------------------
#                               ATOMIC UNITS
# ------------------------------------------------------------------------------
#
# As the name suggest, these units are convenient when dealing with atoms or
# molecules, using quantities on the order of
# - energies ~ 10^{-18} J
# - times    ~ 10^{-17} s
# - masses   ~ 10^{-31} kg
# - lengths  ~ 10^{-11} m
#
# Atomic units are the default internal units of QDYN.

# This file lists the conversion factors from internal units (iu) to the
# given physical units

iu             1.00000000000000000000  # unity :-)
au             1.00000000000000000000  # internal units = atomic units
unitless       1.00000000000000000000  # unitless number


# ------------------------------------------------------------------------------
# Basic Units
# ------------------------------------------------------------------------------

# energy

Hartree        1.00000000000000000000  # Hartree
hartree        1.00000000000000000000  # Hartree
eV             27.2113834492829731459  # Hartree to electronvolt
cminv          2.194746312856066506e5  # Hartree to inverse cm (wavenumbers) (E = h c / lambda) with lambda in cm
K              3.157746662555312504e5  # Hartree to Kelvin (E = kb T)
J              4.3597438060897550e-18  # Hartree to Joule
Hz             6.579683918175572e15    # Hartree to Hz * h
GHz            6.579683918175572e6     # Hartree to 10^9 Hz * h
MHz            6.579683918175572e9     # Hartree to 10^6 Hz * h
kHz            6.579683918175572e12    # Hartree to 10^3 Hz * h

# square-root of energy (unit of dissipators)

sqrt_Hartree   1.0000000000000000e+00
sqrt_hartree   1.0000000000000000e+00
sqrt_eV        5.2164531483837724e+00
sqrt_cminv     4.6848119629885537e+02
sqrt_K         5.6193831178834148e+02
sqrt_J         2.0879999535655540e-09
sqrt_Hz        8.1115250835928336e+07
sqrt_GHz       2.5650894561741061e+03
sqrt_MHz       8.1115250835928338e+04
sqrt_kHz       2.5650894561741063e+06


# time

hartreeinv     1.00000000000000000000  # au = Hartree^{-1}
fs             0.02418884327440855991  # au to femtosecond
ps             2.4188843274408560e-05  # au to picosecond
ns             2.4188843274408560e-08  # au to nanosecond
microsec       2.4188843274408560e-11  # au to microsecond

# length

Bohr           1.00000000000000000000  # Bohr radius a_0
bohr           1.00000000000000000000  # a_0
m              5.2917720827883533e-11  # a_0 to meter
nm             5.2917720827883533e-2   # a_0 to nanometer
microm         5.2917720827883535e-05  # a_0 to micrometer
pm             52.917720827883533      # a_0 to picometer
Angstrom       0.5291772082788353      # a_0 to angstrom
angstrom       0.5291772082788353      # a_0 to angstrom

# mass

kg             9.1093818871545313e-31  # electron mass to kg
Dalton         5.48579911000000039e-4  # electron mass to Dalton (unified internal mass)
dalton         5.48579911000000039e-4  # electron mass to Dalton (unified internal mass)


# ------------------------------------------------------------------------------
# Units for couplings (product of coupling and control must yield energy)
# ------------------------------------------------------------------------------

# -----------------------------
# single-photon dipole coupling
# -----------------------------

# coupling in dipole strength (charge*length),

Debye          2.54174622886413859035  # e * a_0 to Debye with e = electron charge
debye          2.54174622886413859035  # e * a_0 to Debye
coulombmeter   8.4783528105438162e-30  # e * a_0 to Coulomb*meter
statCcm        2.5417462288641390e-18  # e * a_0 to statCoulomb*cm

# control in electric field strength (energy / (charge*length))

Vpm            5.14220624463189208e11  # Hartree/(a_0*e) to volt/meter
Vpcm           5.14220624463189208e09  # Hartree/(a_0*e) to volt/centimeter

# -----------
# Stark shift
# -----------

# Operator is given as polarizability
C2m2Jinv       1.6487772754e-41        # e^2 a_0^2 / Hartree to C^2 m^2 J^{-1}
cm3            1.4819210151E-25        # e^2 a_0^2 / Hartree to CGS cm^3

# See http://en.wikipedia.org/wiki/Polarizability for conversion au -> CGS

# QDYN knows how to square the pulse connected to a 'dstark' operator.
# Therefore, the field must sill be given in units of electric field strength


# ------------------------------------------------------------------------------
# Other units
# ------------------------------------------------------------------------------

# Laser field intensity
# The conversion factor results here from the standard relation between laser
# intensity I and electric field strength E,
#  I=1/2 * eps_zero * speed of light * E**2.
# (See, for example, Rep. Prog. Phys. 60, 389 (1997), Eq. 1.1)
# alpha in the lines below is the fine structure constant

Wpm2           3.50944521626224e20     # This corresponds to
#             1/8/Pi/alpha*atomicenergy/(atomictime*atomiclength**2) to Watt/m^2
Wpcm2          3.50944521626224e16     # This corresponds to
#            1/8/Pi/alpha*atomicenergy/(atomictime*atomiclength**2) to Watt/cm^2

# B-field strength (magnetic flux density)
gauss          2.35051746468947e09     # hbar/(a_0^2*e) to gauss
tesla          2.35051746468947e05     # hbar/(a_0^2*e) to tesla
'''


class UnitConvert(object):
    """Perform conversion between different physical units

    Parameters:
        units_file (str, optional): file from which to read unit definitions.
            The content in the file must be in the format specified in
            :meth:`read_units_definition`. The file  must contain two columns.
            The first columns contains the unit name, the second column the
            conversion factor between internal units and the unit. Units are
            separated into category blocks of compatible units by comment lines
            (starting with '#'). Only units in the same category (i.e., the
            same block in `units_file`) can be converted into each other.  If
            not given, the converter is initialized with a default set of
            units, using atomic units internally.

    Note:
        The special unit name 'iu' indicates "internal units", from and to
        which any unit can be converted. By default, this is atomic units (and
        'au' is an alias for 'iu'). Different internal units may be used by
        loading a `units_file`, although there is little reason to do so. It is
        required that hbar is 1 in internal units, making energy and time
        directly inverse quantities.

    Examples:

        >>> convert = UnitConvert()
        >>> print("%.2f" % convert.convert(1000, 'MHz', 'GHz'))
        1.00
        >>> v = UnitFloat(1000, 'MHz')
        >>> print("%s" % convert.convert(v, to_unit='GHz'))
        1_GHz
        >>> import numpy as np
        >>> v = np.linspace(0, 1000, 3)
        >>> convert.convert(v, 'MHz', 'GHz')
        array([ 0. ,  0.5,  1. ])
    """
    def __init__(self, units_file=None):
        self._convfactor = {} # unit => factor for internal unit to unit
        self._category = {} # unit => category ID or '*'
        if units_file is None:
            self._read_units_definition(DEFAULT_UNITS)
            self._convfactor['au'] = 1.0
            self._category['au'] = '*'
            # if the category is '*', conversion to/from any unit is possible
        else:
            with open(units_file) as in_fh:
                self._read_units_definition(in_fh.read())

    def convert(self, value, from_unit=None, to_unit=None):
        """Convert `value` between units. The result will be of the same type
        as the input `value`.

        Parameters:
            value (float, numpy array, UnitFloat): Value (or array of values)
                to convert
            from_unit (str, optional): Unit of `value`. Alternatively, if
                `value` is an instance of :class:`UnitFloat`, the `from_unit`
                can be taken from value. If given neither directly or obtained
                from `value`, convert from 'internal units'
            to_unit (str, optional): Unit to which to convert. If not given,
                convert to 'internal units'. Note that `from_unit` and
                `to_unit` must be compatible (i.e., both must be units of the
                same category, e.g. energy, length, or time)

        Raises:
            ValueError: if `from_unit` and `to_unit` are not compatible,
                `from_unit`  is incompatible with `value`, or either
                `from_unit` or `to_unit` are unknown units

        """
        if from_unit is None:
            if hasattr(value, 'unit'):
                from_unit = value.unit
        else:
            if hasattr(value, 'unit'):
                if from_unit != value.unit:
                    raise ValueError(("Value %s of from_unit clashes "
                                "with value %s") % (from_unit, str(value)))

        # handle a float of 0.0, which is 0.0 in any unit
        try:
            if (float(value) == 0.0):
                from_unit = to_unit
        except TypeError:
            pass # numpy arrays cannot be converted to float

        if from_unit is not None and to_unit is not None:
            try:
                from_cat = self._category[from_unit]
                to_cat   = self._category[to_unit]
                if from_cat == '*':
                    from_cat = to_cat
                if to_cat == '*':
                    to_cat = from_cat
            except KeyError:
                from_cat = '*'
                to_cat = '*'
            if from_cat != to_cat:
                raise ValueError("Incompatible units in conversion: %s, %s"
                                 % (from_unit, to_unit))
        if from_unit is None:
            convfactor = 1.0
        else:
            try:
                convfactor = 1.0 / self._convfactor[from_unit]
            except KeyError:
                raise ValueError("Unknown from_unit %s" % from_unit)
        if to_unit is not None:
            try:
                convfactor *= self._convfactor[to_unit]
            except KeyError:
                raise ValueError("Unknown to_unit %s" % to_unit)
        if isinstance(value, UnitFloat):
            return UnitFloat(val=float(value)*convfactor, unit=to_unit)
        else:
            return value * convfactor

    def add_unit(self, unit, convfactor, compatible_with=None):
        """Register a new unit

        Parameters:
            unit (str): name of the unit
            convfactor (float): conversion factor to internal units
            compatible_with (str): name of another unit that `unit` can be
                converted to. Compatibility is transitive, i.e. `unit` will
                also be convertible to anything compatible with
                `compatible_with`. If not given, the new `unit` can only be
                converted to internal units.


        >>> convert = UnitConvert()
        >>> convert.add_unit("THz", 6.579683918175572e3, compatible_with='J')
        >>> print(" ".join(sorted(convert.compatible_units('GHz'))))
        Hartree Hz J K MHz THz au cminv eV hartree iu kHz
        """
        self._convfactor[unit] = float(convfactor)
        if compatible_with is None or compatible_with == 'iu':
            new_category = str(uuid.uuid4())
            self._category[unit] = new_category
        else:
            self._category[unit] = self._category[compatible_with]

    def _read_units_definition(self, units_def_str):
        self._convfactor = {}
        self._category = {}
        rx_comment  = re.compile(r'^#\s*')
        rx_unit = re.compile(r'^(?P<unit>[A-Za-z\d]+)\s+'
                             r'(?P<factor>[0-9eE.+-]+)')
        category = ''
        for line in units_def_str.splitlines():
            if rx_comment.match(line):
                category = str(uuid.uuid4())
            else:
                m_unit = rx_unit.match(line)
                if m_unit:
                    unit = m_unit.group('unit')
                    convfactor = float(m_unit.group('factor'))
                    self._convfactor[unit] = convfactor
                    self._category[unit] = category
        self._convfactor['iu'] = 1.0
        self._category['iu'] = '*'
        self._convfactor['unitless'] = 1.0
        self._category['unitless'] = 'unitless'

    def compatible_units(self, unit):
        """Set of all units to which `unit` can be converted

        >>> convert = UnitConvert()
        >>> print(" ".join(sorted(convert.compatible_units('GHz'))))
        Hartree Hz J K MHz au cminv eV hartree iu kHz
        """
        result = set([])
        cat = self._category[unit]
        for unit2 in self.units:
            if unit2 != unit:
                cat2 = self._category[unit2]
                if cat2 == '*' or cat == '*':
                    cat2 = cat
                if cat == cat2:
                    result.add(unit2)
        return result

    @property
    def units(self):
        """Set of all defined unit names"""
        return set(self._convfactor.keys())


class UnitFloat(object):
    """Class for a float value with a physical unit. Behaves like a float in
    most contexts.

    Parameters:
        val (float, UnitFloat): The value. If `val` is an instance of
            `UnitFloat` and `unit` is given also, the value will be converted.
        unit (str, None): The unit. If None, the unit is take from `val` if
            `val` is an instance  of `UnitFloat`, or else the unit is set to
            ``unitless``. Any unit known to to the unternal unit convert may be
            used. Using internal units (``unit='iu'``) is valid, but should be
            avoided.

    Attributes:
        val (float): Value
        unit (str): Unit

    Class Attributes:
        unit_convert (UnitConvert): internal unit converter

    Examples:

        >>> v = UnitFloat(1.0, 'GHz')
        >>> print(v)
        1_GHz

        >>> v = UnitFloat(1.0)
        >>> print(v)
        1_unitless

        >>> v2 = UnitFloat(1.0, 'GHz')
        >>> v = UnitFloat(v2, 'MHz')
        >>> print(v)
        1000_MHz
    """

    _str_pattern = re.compile(r'^\s*(?P<val>[\d.Ee+-]+)(_(?P<unit>\w+))?\s*$')
    unit_convert = UnitConvert()

    def __init__(self, val=0.0, unit=None):
        if isinstance(val, UnitFloat):
            if unit is None:
                unit = val.unit
            self.val = float(val.convert(unit))
        else:
            self.val = float(val)
            if unit is None:
                unit = 'unitless'
        self.unit = unit

    @classmethod
    def from_str(cls, val_str):
        """Create instance from a string
        >>> v = UnitFloat.from_str('1.0_GHz')
        >>> v == UnitFloat(1.0, 'GHz')
        True
        >>> try:
        ...     v = UnitFloat.from_str('abcd')
        ... except ValueError as e:
        ...     print(e)
        String 'abcd' does not describe a UnitFloat
        """
        m = UnitFloat._str_pattern.match(val_str)
        if m:
            val = float(m.group('val'))
            unit = m.group('unit')
        else:
            raise ValueError("String '%s' does not describe a UnitFloat"
                                % val_str)
        return cls(val, unit)

    def to_str(self, fmt='%g'):
        """Convert to string while using the specified format for the value
        part.

        >>> print(UnitFloat(5.2, 'GHz').to_str('%.2f'))
        5.20_GHz
        """
        if self.val == 0:
            return fmt % 0
        else:
            if self.unit is None:
                return fmt % self.val
            else:
                return ((fmt+'_%s') % (self.val, self.unit))

    @classmethod
    def check_str(cls, val_str):
        """Check wether the given string describes a valid unit_float
        """
        m = UnitFloat._str_pattern.match(val_str)
        if m:
            return True
        else:
            return False

    def _with_unit_conversion(f):
        """Decoractor for a method taking two instances of UnitFloat (first,
        second) as parameter. Converts `second` to the same unit as `first`, if
        possible, and throws ValueError otherwise.
        """
        @wraps(f)
        def wrapped(self, other):
            if not isinstance(other, UnitFloat):
                if float(other) == 0.0:
                    other = UnitFloat(0.0, self.unit)
                else:
                    raise TypeError("All arguments must be instances of "
                                    "UnitFloat")
            return f(self, other.convert(self.unit))
        return wrapped

    def __str__(self):
        """String representation. Equivalent to :meth:`to_str`.

        >>> f = UnitFloat(1.0, 'GHz')
        >>> print(f)
        1_GHz
        >>> print(str(f))
        1_GHz
        >>> print("%s" % str(f))
        1_GHz
        >>> print(UnitFloat(5.2, 'GHz'))
        5.2_GHz
        >>> print(UnitFloat(1000.2, 'GHz'))
        1000.2_GHz
        >>> print(UnitFloat(100000000000.2, 'GHz'))
        1e+11_GHz
        >>> print(UnitFloat(1.2e12, 'GHz'))
        1.2e+12_GHz
        >>> print(UnitFloat(0))
        0
        >>> print(UnitFloat(1.2))
        1.2_unitless
        """
        if self.val == 0:
            return "0"
        return self.to_str()
    __repr__ = __str__

    def __eq__(self, other):
        """Test equality. Two instances of `UnitFloat` are the same if their
        string representation is the same, making the equality check robust
        against conversion rounding errors.

        >>> UnitFloat(1.0, 'GHz') == UnitFloat(1.0, 'GHz')
        True
        >>> UnitFloat(1.0, 'GHz') == UnitFloat(2.0, 'GHz')
        False
        >>> UnitFloat(1.0, 'GHz') != UnitFloat(2.0, 'GHz')
        True
        >>> UnitFloat(1.0, 'GHz') == UnitFloat(1000, 'MHz')
        True
        >>> UnitFloat(1.0, 'GHz') == '1_GHz'
        True
        >>> UnitFloat(1.0, 'GHz') == (1, 'GHz')
        True
        >>> UnitFloat(1.0) == 1
        True
        """
        if isinstance(other, UnitFloat):
            return repr(self) == repr(other.convert(self.unit))
        else: # we'll try other as string, tuple and float
            try:
                try:
                    return self == UnitFloat.from_str(other)
                except (TypeError, ValueError):
                    try:
                        return self == UnitFloat(other[0], other[1])
                    except (TypeError, IndexError):
                        return self == UnitFloat(other)
            except (TypeError, ValueError):
                raise TypeError("Cannot compare UnitFloat to %s" % other)

    @_with_unit_conversion
    def __gt__(self, other):
        """Compare two numbers

        >>> UnitFloat(1.0, 'GHz') > UnitFloat(1.0, 'GHz')
        False
        >>> UnitFloat(2.0, 'GHz') > UnitFloat(1.0, 'GHz')
        True
        >>> UnitFloat(1.0, 'GHz') >= UnitFloat(1.0, 'GHz')
        True
        >>> UnitFloat(1.0, 'GHz') < UnitFloat(2.0, 'GHz')
        True
        >>> UnitFloat(1.0, 'GHz') <= UnitFloat(1.0, 'GHz')
        True
        >>> UnitFloat(1.0, 'GHz') > UnitFloat(900, 'MHz')
        True
        >>> try:
        ...     UnitFloat(1.0, 'GHz') < UnitFloat(2.0, 'ns')
        ... except ValueError as e:
        ...     print(e)
        Incompatible units in conversion: GHz, ns
        """
        return self.val > other.val

    @_with_unit_conversion
    def __ge__(self, other):
        return self.val >= other.val

    def __neg__(self):
        """Negate number

        >>> print(-UnitFloat(1.1, 'GHz'))
        -1.1_GHz
        """
        return UnitFloat(val=(-self.val), unit=self.unit)

    def __pos__(self):
        """Unary positive"""
        return self

    def __abs__(self):
        """Absolute value of number

        >>> print(abs(UnitFloat(-1.1, 'GHz')))
        1.1_GHz
        >>> print(abs(UnitFloat(1.1, 'GHz')))
        1.1_GHz
        """
        return UnitFloat(val=abs(self.val), unit=self.unit)

    def __round__(self, n):
        """Round to n decimal places.
        This results in a :class:`UnitFloat` in Python 3 and a float in Python
        2.
        """
        return UnitFloat(val=round(self.val, n), unit=self.unit)

    def __floor__(self):
        """Round down to nearest integer.
        This results in a :class:`UnitFloat` in Python 3 and a float in Python
        2.
        """
        return UnitFloat(val=math.floor(self.val), unit=self.unit)

    def __ceil__(self):
        """Round up to nearest integer.
        This results in a :class:`UnitFloat` in Python 3 and a float in Python
        2.
        """
        return UnitFloat(val=math.ceil(self.val), unit=self.unit)

    @_with_unit_conversion
    def __add__(self, other):
        """Add two numbers, which must use exactly the same unit.

        >>> print(UnitFloat(1.1, 'GHz') + UnitFloat(2.1, 'GHz'))
        3.2_GHz
        >>> print(UnitFloat(1.1, 'GHz') - UnitFloat(2.0, 'GHz'))
        -0.9_GHz
        >>> print(UnitFloat(1.0, 'GHz') + UnitFloat(100, 'MHz'))
        1.1_GHz

        >>> try:
        ...     UnitFloat(1.0, 'GHz') + 1.0
        ... except TypeError as e:
        ...     print(e)
        All arguments must be instances of UnitFloat
        """
        return UnitFloat(val=(self.val+other.val), unit=self.unit)

    @_with_unit_conversion
    def __sub__(self, other):
        return UnitFloat(val=(self.val-other.val), unit=self.unit)

    def __mul__(self, factor):
        """Multiply with a number

        >>> print(2*UnitFloat(1.1, 'GHz'))
        2.2_GHz
        >>> print(UnitFloat(1.1, 'GHz')*2)
        2.2_GHz
        >>> print(UnitFloat(1.1, 'GHz')*"2.0")
        2.2_GHz
        >>> try:
        ...     print(UnitFloat(1.1, 'GHz')*UnitFloat(1.1, 'GHz'))
        ... except TypeError as e:
        ...     print(e)
        Factor cannot be an instance of UnitFloat
        >>> try:
        ...     print(UnitFloat(1.1, 'GHz')*2.0j)
        ... except TypeError as e:
        ...     print(e)
        can't convert complex to float
        """
        if isinstance(factor, UnitFloat):
            raise TypeError("Factor cannot be an instance of UnitFloat")
        return UnitFloat(val=(float(factor)*self.val), unit=self.unit)
    __rmul__ = __mul__

    def __iadd__(self, other):
        """Augmented assignment

        >>> v = UnitFloat(1.1, 'GHz')
        >>> v += 1
        >>> print(v)
        2.1_GHz
        >>> v -= 1
        >>> print(v)
        1.1_GHz
        >>> v2 = UnitFloat(1.0, 'GHz')
        >>> v += v2
        >>> print(v)
        2.1_GHz
        >>> v -= v2
        >>> print(v)
        1.1_GHz
        >>> v2 = UnitFloat(100.0, 'MHz')
        >>> v += v2
        >>> print(v)
        1.2_GHz
        >>> v2 = UnitFloat(1.0, 'ns')
        >>> try:
        ...     v += v2
        ... except ValueError as e:
        ...     print(e)
        Incompatible units in conversion: ns, GHz
        """
        if isinstance(other, UnitFloat):
            self.val += other.convert(self.unit).val
        else:
            self.val += float(other)
        return self

    def __isub__(self, other):
        if isinstance(other, UnitFloat):
            self.val -= other.convert(self.unit).val
        else:
            self.val -= float(other)
        return self

    def __imul__(self, factor):
        """Augmented multiplication

        >>> v = UnitFloat(1.1, 'GHz')
        >>> v *= 2
        >>> print(v)
        2.2_GHz
        """
        if isinstance(factor, UnitFloat):
            raise TypeError("Factor cannot be an instance of UnitFloat")
        self.val *= float(factor)
        return self

    def __truediv__(self, other):
        """Division

        >>> v = UnitFloat(1.1, 'GHz')
        >>> r = v / UnitFloat(0.1, 'GHz')
        >>> isinstance(r, float)
        True
        >>> print("%.1f" % r)
        11.0
        >>> print("%.1f" % (v / UnitFloat(100, 'MHz')))
        11.0
        >>> print(v / 10)
        0.11_GHz
        """
        if isinstance(other, UnitFloat):
            return (self.val)/(other.convert(self.unit).val)
        else:
            # `other` is assumed to be a float
            return UnitFloat(val=(self.val/float(other)), unit=self.unit)
    __div__ = __truediv__

    def __itruediv__(self, quotient):
        """Augmented division

        >>> v = UnitFloat(1.1, 'GHz')
        >>> v /= 10
        >>> print(v)
        0.11_GHz
        """
        if isinstance(quotient, UnitFloat):
            # allowing to divide by another UnitFloat would cause a type change
            raise TypeError("Quotient cannot be an instance of UnitFloat")
        self.val /= float(quotient)
        return self
    __idiv__ = __itruediv__

    def __float__(self):
        """Convert to float (discarding unit)

        >>> float(UnitFloat(1.1, 'GHz'))
        1.1
        >>> float(UnitFloat(1.1, 'MHz'))
        1.1
        >>> float(UnitFloat(1.1))
        1.1

        The conversion is also invoked for format strings:

        >>> print("%.1f" % UnitFloat(1.1, 'GHz'))
        1.1
        >>> print("%s" % UnitFloat(1.1, 'GHz'))
        1.1_GHz
        """
        return float(self.val)

    def convert(self, to_unit):
        """Convert to a different unit

        >>> v = UnitFloat(1.1, 'GHz')
        >>> str(v.convert('MHz'))
        '1100_MHz'
        """
        if to_unit == self.unit:
            return self
        else:
            return self.unit_convert.convert(self, to_unit=to_unit)

    def __hash__(self):
        return hash(repr(self))

