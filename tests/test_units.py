import QDYN
from QDYN.units import UnitFloat, UnitConvert
import six
import math
import pytest

def test_unit_float_round():
    if six.PY2:
        assert round(UnitFloat(1.11, 'GHz'), 1) == 1.1
    else:
        assert round(UnitFloat(1.11, 'GHz'), 1) == UnitFloat(1.1, 'GHz')
    if six.PY2:
        assert math.floor(UnitFloat(1.11, 'GHz')) == 1
    else:
        assert math.floor(UnitFloat(1.11, 'GHz')) == UnitFloat(1, 'GHz')
    if six.PY2:
        assert math.ceil(UnitFloat(1.11, 'GHz')) == 2
    else:
        assert math.ceil(UnitFloat(1.11, 'GHz')) == UnitFloat(2, 'GHz')

def test_unit_convert():
    # Note: we compare strings instead of instances of UnitFloat to deal with
    # rounding errors
    c = UnitConvert()

    # It's ok to convert to the the same unit we're already at
    v = UnitFloat(2.5, 'GHz')
    v2 = c.convert(v, to_unit='GHz')
    assert str(v) == str(v2)

    # we can leave out from_unit
    v2 = c.convert(v, to_unit='iu')
    assert v2.unit == 'iu'
    assert str(v) == str(c.convert(v2, from_unit='iu', to_unit='GHz'))

    # we can leave out both from_unit and to_unit, but then in the converted
    # result, we have no explicit 'iu' unit
    assert c.convert(v).unit is None
    assert float(c.convert(v)) - v2.val < 1.0e-12
    assert str(c.convert(v2, to_unit=v.unit)) == str(v)

    # using the default internal units, we can convert anything to au
    v2 = c.convert(v, to_unit='au')
    assert str(v) == str(c.convert(v2, from_unit='au',  to_unit='GHz'))

    # a UnitFloat with unit None is equivalent in the conversion to unit 'iu'
    v2 = c.convert(UnitFloat(float(c.convert(2.5, 'GHz'))), to_unit='MHz')
    assert str(v2) == '2500_MHz'

    # Invalid units produces ValueErrors
    with pytest.raises(ValueError) as excinfo:
        c.convert(UnitFloat(2.5, 'GHz'), 'MHz', 'J')
    assert "from_unit clashes" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        c.convert(2.5, 'MHz', 'bohr')
    assert "Incompatible units" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        c.convert(2.5, 'MHz', 'n/a')
    assert "Unknown to_unit" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        c.convert(2.5, 'n/a', 'MHz')
    assert "Unknown from_unit" in str(excinfo.value)

