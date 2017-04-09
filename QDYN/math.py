"""Math helper routine"""
from __future__ import print_function, division, absolute_import


def isqrt(n):
    """Integer square root of n > 0

    >>> isqrt(1024**2)
    1024
    >>> isqrt(10)
    3
    """
    assert n >= 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
