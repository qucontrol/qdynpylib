"""
Module containing utitlities for managin QDYN config files
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import json

def read_config(filename):
    """Parse the given config file and return a nested data structure to
    represent it.

    Returns a dict mapping

        section name => dict of key => value; or
        section name => list of dicts key => value
    """
    config_data = {}
    raise NotImplementedError()
    return config_data

def write_config(config_data, filename):
    """Write a config file to the given filename, based on the given
    config_data, where config_data is a dict as returned by `read_config`
    """

def read_old_config(filename):
    """Parse the given config file in the old format and return a nested data
    structure to represent it.

    Returns a dict mapping

        section name => dict of key => value; or
        section name => list of dicts key => value
    """
    config_data = {}
    raise NotImplementedError()
    return config_data


def convert_config(old_config_data, mappings):
    """Convert the old_config_data, as returned by `read_old_config` into
    a data structure compatible with the current config file. The conversion is
    performed according to the given list of mappings.

    Mappings is a data structure (to be determined) the describes how old
    values are to be mapped to new values. It should be read from a json file
    using the standard library json module

    * (old_section_name, old_key, new_section_name, new_key)
    """
    raise NotImplementedError()




