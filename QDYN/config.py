""" Module containing utilities for managing QDYN config files """
from __future__ import print_function, division, absolute_import
import json
import re
import base64
import random
import copy
from collections import OrderedDict

import six
import numpy as np

from .units import UnitFloat

def _protect_str_vals(str_line):
    r'''Parsing is greatly simplified if it can be assumed that key-value pairs
    in the config match the regular expression '\w+\s*=\s*[\w.+-]+'. That is,
    the values do not contain spaces, quotes, or escaped characters. This
    function replaces values in the given `str_line` by a base-64 string
    which is guaranteed to contain only letters and numbers. It returns the
    "protected" string and a list of replacement tuples. The protected string
    is guaranteed to not be shorter than the original string.

    >>> s_in = r'a = "text", '+"b = 'text'"
    >>> print(s_in)
    a = "text", b = 'text'
    >>> s, r = _protect_str_vals(s_in)
    >>> print(s)
    a = InRleHQi, b = J3RleHQn
    >>> print(r)
    [('InRleHQi', '"text"'), ('J3RleHQn', "'text'")]

    >>> s_in = r'a = this\ is\ an\ unquoted\ string, b = "text"'
    >>> print(s_in)
    a = this\ is\ an\ unquoted\ string, b = "text"
    >>> s, r = _protect_str_vals(s_in)
    >>> print(s)
    a = dGhpc1wgaXNcIGFuXCB1bnF1b3RlZFwgc3RyaW5n, b = InRleHQi
    >>> print(r)
    [('InRleHQi', '"text"'), ('dGhpc1wgaXNcIGFuXCB1bnF1b3RlZFwgc3RyaW5n', 'this\\ is\\ an\\ unquoted\\ string')]
    '''
    # handle quoted strings
    rx_dq = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')  # search for ...
    rx_sq = re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'")  # ... balanced quotes
    replacements = []
    n_replacements = -1
    while n_replacements < len(replacements):
        n_replacements = len(replacements)
        matches = rx_dq.findall(str_line)
        matches.extend(rx_sq.findall(str_line))
        for quoted_str  in matches:
            if six.PY2:
                b64 = base64.b64encode(quoted_str).replace("=", '')
            else:
                b64 = base64.b64encode(bytes(quoted_str, 'utf8'))\
                    .decode('ascii').replace("=", '')
            replacements.append((b64, quoted_str))
            str_line = str_line.replace(quoted_str, b64)

    # handle un-quoted, but escaped strings
    rx_escaped_word = re.compile(
            # any string of characters that does not include spaces or any
            # of the characters ,:!&=\, except when they are escaped (preceded
            # by a backslash)
            r'(([^\s,:!&=\\]|\\\s|\\\\|\\,|\\:|\\!|\\&|\\=|\\n|\\r|\\t|\\b)+)')
    rx_good_word = re.compile(r'^[\w.+-]+$')
    for match in rx_escaped_word.finditer(str_line):
        word = match.group(0)
        if not (rx_good_word.match(word) or (word == '*')):
            if six.PY2:
                b64 = base64.b64encode(word).replace("=", '')
            else:
                b64 = base64.b64encode(bytes(word, 'utf8'))\
                      .decode('ascii').replace("=", '')
            replacements.append((b64, word))
            str_line = str_line.replace(word, b64)

    return str_line, replacements


def _unprotect_str_vals(str_line, replacements):
    r'''Inverse to :func:`_protect_str_vals`

    >>> s, r = _protect_str_vals(r'a = "text", '+"b = 'text'")
    >>> print(_unprotect_str_vals(s, r))
    a = "text", b = 'text'
    '''
    for (b64, quoted_str) in replacements:
        str_line = str_line.replace(b64, quoted_str)
    result = str_line
    for (b64, quoted_str) in replacements:
        str_line = str_line.replace(b64, quoted_str)
    while result != str_line:
        result = str_line
        for (b64, quoted_str) in replacements:
            str_line = str_line.replace(b64, quoted_str)
    return result


def _escape_str(s):
    """Replace special characters in the given string with escape codes.
    Surround strings containing spaces with quotes"""
    replacements = [
        (",",  "\\,"),
        (":",  "\\:"),
        ("!",  "\\!"),
        ("&",  "\\&"),
        ("=",  "\\="),
        ("\n", "\\n"),
        ("\r", "\\r"),
        ("\t", "\\t"),
        ("\b", "\\b"),
    ]
    s = str(s)
    bs_protect = ''.join([chr(random.randint(33,127)) for i in range(15)])
    while bs_protect in s:
        bs_protect = ''.join([chr(random.randint(33,127)) for i in range(15)])
    s = s.replace("\\", bs_protect)
    for unescaped, escaped in replacements:
        s = s.replace(unescaped, escaped)
    if " " in s:
        if '"' in s:
            s = "'%s'" % s
        else:
            s = '"%s"' % s
    s = s.replace(bs_protect, "\\\\")
    return s


def _unescape_str(s):
    """Inverse to :func:`_escape_str`"""
    replacements = [
        ("\\,", ","),
        ("\\:", ":"),
        ("\\!", "!"),
        ("\\&", "&"),
        ("\\=", "="),
        ("\\n", "\n"),
        ("\\r", "\r"),
        ("\\t", "\t"),
        ("\\b", "\b"),
    ]
    bs_protect = ''.join([chr(random.randint(33,127)) for i in range(15)])
    while bs_protect in s:
        bs_protect = ''.join([chr(random.randint(33,127)) for i in range(15)])
    if s.startswith("'") or s.startswith('"'):
        s = s[1:]
    if s.endswith("'") or s.endswith('"'):
        s = s[:-1]
    s = s.replace("\\\\", bs_protect)
    for escaped, unescaped in replacements:
        s = s.replace(escaped, unescaped)
    s = s.replace(bs_protect, "\\")
    return s


def _val_to_str(val):
    """Convert `val` to a string that can be written directly to the config
    file"""
    logical_mapping = {True: 'T', False: 'F'}
    if isinstance(val, (bool, np.bool_)):
        return logical_mapping[val]
    elif isinstance(val, str):
        return _escape_str(val)
    else:
        return str(val)


def _process_raw_lines(raw_lines):
    r'''Return an iterator over the "processed" lines of a config file, based
    on the given raw lines. The processing of the raw lines consists of
    stripping out comments, and combining multiple continued raw lines into a
    single processed lines. The returned lines do not have a trailing newline.

    Arguments:
        raw_lines (iterable): Iterable of a raw list of lines, as they might
        appear in a config file. These may contain comments and continuation.
        Each line may or may not contain trailing newlines (trailing whitespace
        is ignored)

    Example:

        >>> config = r"""
        ... ! the following is a config file for 3 pulses and some custom
        ... ! data
        ... pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, &
        ... & oct_shape = flattop, ftrt = 0.2, oct_lambda_a = 100, &
        ... & oct_increase_factor = 10
        ... * id = 1, t_0 = 0, oct_outfile = pulse1.dat    ! pulse 1
        ... * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat  ! pulse 2
        ... * id = 3, t_0 = 5, oct_outfile = pulse3.dat    ! pulse 3
        ...
        ... user_string: &
        ...   A = "x**2", &
        ...   B = 'B_{"avg"}' ! some "arbitrary" strings
        ... """
        >>> for line in _process_raw_lines(config.splitlines()):
        ...     print(line)
        pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2,  oct_shape = flattop, ftrt = 0.2, oct_lambda_a = 100,  oct_increase_factor = 10
        * id = 1, t_0 = 0, oct_outfile = pulse1.dat
        * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
        * id = 3, t_0 = 5, oct_outfile = pulse3.dat
        user_string: A = "x**2", B = 'B_{"avg"}'
    '''
    line_nr = 0
    line = ''
    raw_lines = iter(raw_lines)
    while True: # stops when next(raw_lines) throws StopIteration
        raw_line = next(raw_lines)
        line_nr += 1
        line, replacements = _protect_str_vals(raw_line.strip())
        # strip out out comments
        line = re.sub('!.*$', '', line).strip()
        while line.endswith('&'):
            line = line[:-1] # strip out trailing '&' in continued line
            line2, replacements2 = _protect_str_vals(next(raw_lines).strip())
            line_nr += 1
            replacements.extend(replacements2)
            # strip out comments
            line2 = re.sub('!.*$', '', line2).strip()
            # strip out leading '&' in continuing line
            if line2.startswith('&') and len(line2) > 1:
                line2 = line2[1:]
            line += line2
        line = _unprotect_str_vals(line, replacements)
        if len(line) == 0:
            continue
        else:
            yield line


def _split_config_line(line, linewidth):
    """Split the given line into a multi-line string of continued lines, trying
    to keep the length of each line under `linewidth`. Trailing newlines in
    `line` are ignored."""
    full_line, replacements = _protect_str_vals(line.strip())
    if len(full_line) == 0:
        return "\n"
    rx_item = re.compile(r'(\w+\s*=\s*[\w.+-]+\s*,?\s*)')
    parts = [part for part in rx_item.split(full_line) if len(part) > 0]
    current_line = ''
    lines = []
    while len(parts) > 0:
        current_line += parts.pop(0)
        try:
            while len(current_line) + len(parts[0]) <= linewidth - 1:
                current_line += parts.pop(0)
        except IndexError:
            pass # We've exhausted all parts
        else:
            if len(parts) > 0:
                current_line += '&'
        current_line = _unprotect_str_vals(current_line, replacements)
        lines.append(current_line.rstrip())
        current_line = '  ' # indent for continuing line
    return "\n".join(lines)+"\n"


def _render_config_lines(section_name, section_data):
    r'''Render `section_data` into a list of lines.

    Example:
        >>> section_data = [
        ...     OrderedDict([
        ...         ('type', 'gauss'),
        ...         ('t_FWHM', UnitFloat.from_str('1.8_ns')),
        ...         ('E_0', UnitFloat.from_str('1_GHz')),
        ...         ('id', 1), ('oct_outfile', 'pulse1.dat')]),
        ...     OrderedDict([
        ...         ('type', 'gauss'),
        ...         ('t_FWHM', UnitFloat.from_str('1.8_ns')),
        ...         ('E_0', UnitFloat.from_str('1_GHz')),
        ...         ('id', 2), ('oct_outfile', 'pulse2.dat')])
        ... ]
        >>> for line in _render_config_lines('pulse', section_data):
        ...     print(line)
        pulse: type = gauss, t_FWHM = 1.8_ns, E_0 = 1_GHz
        * id = 1, oct_outfile = pulse1.dat
        * id = 2, oct_outfile = pulse2.dat
    '''
    # Passing `section_name` and `section_data` separately (instead of a full
    # `config_data` dict) allows this routine to be used for creating multiple
    # "blocks" of the same section, each with a different "label"
    lines = []
    if isinstance(section_data, list):
        # section name and common items
        if len(section_data) > 1:
            common_items = dict(set(section_data[0].items()).intersection(
                                *[set(d.items()) for d in section_data[1:]]))
        else:
            common_items = {}
        line = "%s:" % section_name
        for key in section_data[0]:
            # we *do not* iterate over keys in common_items, so that items
            # are ordered the same as in section_data[0], instead of randomly
            if key in common_items:
                line += " %s = %s," % (key, _val_to_str(common_items[key]))
        if line.endswith(","):
            lines.append(line[:-1].strip())
        else:
            lines.append(line.strip())
        # item lines
        for item_line in section_data:
            line = "*"
            for key in item_line:
                if key not in common_items:
                    line += " %s = %s," % (key, _val_to_str(item_line[key]))
            if line.endswith(","):
                lines.append(line[:-1].strip())
            else:
                lines.append(line.strip())
    else: # section does not contain item lines, but key-value pairs only
        line = "%s:" % section_name
        for key in section_data:
            line += " %s = %s," % (key, _val_to_str(section_data[key]))
        lines.append(line[:-1].strip())
    return lines


def _item_rxs():
    # the following regexes are encapsulated in this function to make the them
    # accessible to the config_conversion module
    logical_mapping = {
        'T': True, 'true': True, '.true.': True,
        'F': False, 'false': False, '.false.': False,
    }
    item_rxs = [
        (re.compile(r'(?P<key>\w+)\s*=\s*(?P<value>[\dEe.+-]+_\w+)$'),
        lambda v: UnitFloat.from_str(v)),
        (re.compile(r'(?P<key>\w+)\s*=\s*(?P<value>[\d+-]+)$'),
        lambda v: int(v)),
        (re.compile(r'(?P<key>\w+)\s*=\s*(?P<value>[\dEe.+-]+)$'),
        lambda v: float(v)),
        (re.compile(r'(?P<key>\w+)\s*=\s*(?P<value>(T|F|true|false|'
                    r'\.true\.|\.false\.))$'),
        lambda v: logical_mapping[v]),
        (re.compile(r'(?P<key>\w+)\s*=\s*(?P<value>.+)$'),
        lambda v: _unescape_str(v)),
    ]
    return item_rxs


def read_config_file(filename):
    """Equivalent to ``read_config_str(open(filename).read())``"""
    with open(filename) as in_fh:
        return _read_config_lines(_process_raw_lines(in_fh))


def read_config_str(config_str):
    """Parse the multi-line string containing the contents of a config file,
    and return a nested data structure of the config file data.

    Return an ordered dictionary containing the following mapping::

        section name => dict(key => value)

    or::

        section name => list of dicts(key => value)
    """
    return _read_config_lines(_process_raw_lines(config_str.splitlines()))


def _read_config_lines(lines):
    """Parse an iterable of lines and return the nested config data structure
    (see :func:`read_config_str`)

    Note:
        `lines` must not contain comments or continuations
    """
    rx_sec = re.compile(r'''
        (?P<section>[a-z_A-Z]+) \s*:\s*
        (?P<items>
            (\w+ \s*=\s* [\w.+-]+ [\s,]+)*
            (\w+ \s*=\s* [\w.+-]+ \s*)
        )?
        ''', re.X)
    rx_itemline = re.compile(r'''
        \* \s+
        (?P<items>
            (\w+ \s*=\s* [\w.+-]+ [\s,]+)*
            (\w+ \s*=\s* [\w.+-]+ \s*)
        )
        ''', re.X)
    rx_item = re.compile(r'(\w+\s*=\s*[\w.+-]+)')
    item_rxs = _item_rxs()

    # we need to make two passes over lines, so we may have to convert an
    # iterator to a list
    lines = list(lines)

    config_data = OrderedDict([])

    # first pass: identify sections, list of lines in each section
    current_section = ''
    for line in lines:
        line, replacements = _protect_str_vals(line)
        m_sec = rx_sec.match(line)
        m_itemline = rx_itemline.match(line)
        if m_sec:
            current_section = m_sec.group('section')
            config_data[current_section] = OrderedDict([])
        elif m_itemline:
            if isinstance(config_data[current_section], OrderedDict):
                config_data[current_section]  = []
            config_data[current_section].append(OrderedDict([]))

    # second pass: set items
    current_section = ''
    current_itemline = 0
    for line in lines:
        line, replacements = _protect_str_vals(line)
        m_sec = rx_sec.match(line)
        m_itemline = rx_itemline.match(line)
        line_items = OrderedDict([])
        for item in rx_item.findall(line):
            matched = False
            for rx, setter in item_rxs:
                m = rx.match(item)
                if m:
                    val = _unprotect_str_vals(m.group('value'), replacements)
                    line_items[m.group('key')] = setter(val)
                    matched = True
                    break
            if not matched:
                raise ValueError("Could not parse item '%s'" % str(item))
        if m_sec:
            current_itemline = 0
            current_section = m_sec.group('section')
            if isinstance(config_data[current_section], OrderedDict):
                config_data[current_section].update(line_items)
            else:
                for line_dict in config_data[current_section]:
                    line_dict.update(line_items)
        elif m_itemline:
            config_data[current_section][current_itemline].update(
                    line_items)
            current_itemline += 1
        else:
            raise ValueError("Could not parse line '%s'" % line)

    return config_data


def write_config(config_data, filename):
    """Write out a config file

    Arguments:
        config_data (dict): data structure as returned by :func:`read_config`.
        filename (str): name of file to which to write config
    """
    with open(filename, 'w') as out_fh:
        out_fh.write(config_data_to_str(config_data))


def config_data_to_str(config_data):
    """Inverse of :func:`read_config_str`"""
    lines = []
    for section in config_data:
        if isinstance(config_data[section], list):
            label_groups = OrderedDict([])
            for itemline in config_data[section]:
                label = itemline.get('label')
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(itemline)
            for group in label_groups.values():
                lines.extend(_render_config_lines(section, group))
                lines.append('')
        else:
            lines.extend(_render_config_lines(section, config_data[section]))
            lines.append('')
    result = ''
    for line in lines[:-1]: # we ignore the trailing blank line
        result += _split_config_line(line, 80)
    return result


def get_config_value(config_data, key_tuple):
    """Extract value from `config_data` by the given `key_tuple`

    Arguments:
        config_data (dict): data structure as returned by :func:`read_config`.
        key_tuple (tuple): tuple of keys. For example if `key_tuple` is
            ``('pulse', 0, 'id')``, then the returned value would be
            ``config_data['pulse'][0]['id']``

    Raises:
        ValueError: if any of the keys in `key_tuple` are invalid or cannot be
            found
    """
    try:
        key = key_tuple[0]
    except IndexError:
        raise ValueError("key_tuple must be a with at least one element")
    try:
        val = config_data[key]
        for key in key_tuple[1:]:
            val = val[key]
    except (TypeError, IndexError) as exc_info:
        raise ValueError("Invalid key '%s': %s" % (key, str(exc_info)))
    except KeyError:
        raise ValueError("Invalid key '%s'" % (key, ))
    return val


def set_config_value(config_data, key_tuple, value):
    """Set a value in `config_data`, cf. `get_config_value`"""
    if len(key_tuple) < 2:
        raise ValueError("key_tuple must have at least two elements")
    try:
        item = config_data[key_tuple[0]]
        for key in key_tuple[1:-1]:
            item = item[key]
        item[key_tuple[-1]] = value
    except (TypeError, IndexError) as exc_info:
        raise ValueError("Invalid key '%s': %s" % (key, str(exc_info)))
    except KeyError:
        raise ValueError("Invalid key '%s'" % (key, ))


def generate_make_config(config_template, variables, dependencies=None,
        checks=None):
    r'''Generate a routine, e.g. `make_config`, that may be used to generate
    config file data based on the given template.

    Arguments:
        config_template (dict): data structure as returned by
            :func:`read_config` that will serve as a template
        variables (dict): mapping of a keyword variable name to a key-tuple
            in the config (cf. :func:`set_config_value`)
        dependencies (dict): mapping of a key-tuple to a callable that
            calculates a value for that entry in the config file.
        checks (dict): mapping of a keyword variable name to a callable that
            checks whether a given value is acceptable.

    Example:

        >>> config_template = read_config_str(r"""
        ...     pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, &
        ...     & oct_shape = flattop, ftrt = 0.2, oct_lambda_a = 100, &
        ...     & oct_increase_factor = 10
        ...     * id = 1, t_0 = 0, oct_outfile = pulse1.dat    ! pulse 1
        ...     * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat  ! pulse 2
        ...     * id = 3, t_0 = 5, oct_outfile = pulse3.dat    ! pulse 3""")

        >>> make_config = generate_make_config(config_template,
        ...     variables={'E_0': ('pulse', 0, 'E_0'), },
        ...     dependencies={
        ...         ('pulse', 1, 'E_0'): lambda kwargs: kwargs['E_0'],
        ...         ('pulse', 2, 'E_0'): lambda kwargs: kwargs['E_0']},
        ...     checks={'E_0': lambda val: val >= 0.0})

        >>> print(make_config.__doc__)
        Generate config file data (``config_data``) based on the given keyword parameters.
        <BLANKLINE>
        Keyword Arguments:
            E_0: Set ``config_data['pulse'][0]['E_0']`` (default: 1.0)
        <BLANKLINE>
        Also, the following will be set to values that depend on the given keyword arguments:
        <BLANKLINE>
        * ``config_data['pulse'][0]['E_0']``
        * ``config_data['pulse'][0]['E_0']``
        <BLANKLINE>
        If called without arguments, data equivalent to the following config file is returned::
        <BLANKLINE>
            pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2, oct_shape = flattop, &
              ftrt = 0.2, oct_lambda_a = 100, oct_increase_factor = 10
            * id = 1, t_0 = 0, oct_outfile = pulse1.dat
            * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
            * id = 3, t_0 = 5, oct_outfile = pulse3.dat
        <BLANKLINE>
        Raises:
            TypeError: if an invalid keyword is passed.
            ValueError: if any value fails to pass checks.
        <BLANKLINE>

        >>> config = make_config(E_0=0.1)
        >>> print(config_data_to_str(config))
        pulse: type = gauss, t_FWHM = 1.8, E_0 = 0.1, w_L = 0.2, oct_shape = flattop, &
          ftrt = 0.2, oct_lambda_a = 100, oct_increase_factor = 10
        * id = 1, t_0 = 0, oct_outfile = pulse1.dat
        * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
        * id = 3, t_0 = 5, oct_outfile = pulse3.dat
        <BLANKLINE>
    '''
    if dependencies is None:
        dependencies = {}
    if checks is None:
        checks = {}
    defaults = {}
    for varname, key_tuple in variables.items():
        defaults[varname] = get_config_value(config_template, key_tuple)
    for key_tuple in dependencies.keys():
        # rely on get_config_value to throw Exception for invalid key_tuple
        get_config_value(config_template, key_tuple)
    def make_config(**kwargs):
        """Generate config file data"""
        config_data = copy.deepcopy(config_template)
        for varname in defaults:
            if varname not in kwargs:
                kwargs[varname] = defaults[varname]
        for varname, value in kwargs.items():
            if varname not in variables:
                raise TypeError(("Got unexpected keyword argument '%s'. "
                        "Valid keyword arguments are: %s")
                        % (varname, ", ".join(variables.keys())))
            if varname in checks:
                if not checks[varname](value):
                    raise ValueError("Value %s for '%s' does not pass check"
                                     % (value, varname))
            key_tuple = variables[varname]
            set_config_value(config_data, key_tuple, value)
        for key_tuple, eval_dependency in dependencies.items():
            value = eval_dependency(kwargs)
            set_config_value(config_data, key_tuple, value)
        return config_data
    # Set the docstring of `make_config`
    make_config.__doc__ = ("Generate config file data (``config_data``) "
                           "based on the given keyword parameters.\n\n")
    make_config.__doc__ += "Keyword Arguments:\n"
    for varname in variables:
        make_config.__doc__ += "    %s: Set ``%s`` (default: %s)\n" % (
                varname, 'config_data'+''.join(
                    ["[%s]" % repr(k) for k in variables[varname]]),
                repr(defaults[varname]))
    make_config.__doc__ += ("\nAlso, the following will be set to values "
                            "that depend on the given keyword arguments:\n\n")
    for key_tuple in dependencies:
        make_config.__doc__ += ("* ``config_data"
                                +''.join(["[%s]" % repr(k)
                                          for k in variables[varname]])+"``\n")
    make_config.__doc__ += ("\nIf called without arguments, data equivalent "
                            "to the following config file is returned::"
                            "\n\n    " + "\n    ".join(
                                config_data_to_str(make_config()).splitlines()
                            ))
    make_config.__doc__ += "\n\nRaises:\n"
    make_config.__doc__ += "    TypeError: if an invalid keyword is passed.\n"
    if len(checks) > 0:
        make_config.__doc__ += ("    ValueError: if any value fails to "
                                "pass checks.\n")
    return make_config


def get_config_structure(def_f90, outfile='new_config_structure.json'):
    """Get a dumped .json-file with all the allowed section names and
    correspondig items of the new config structure genereated by reading the
    'para_t' type in the def.f90 file
    """
    config_structure = {}

    # Local regular expression filters
    rx = {
        'get_pt_type' : re.compile(
            r'^type\s*\((?P<sec_name>[^\s]\w+)_pt\).*$'),
        'start_pt_sec' : re.compile(
            r'^type\s+(?P<sec_name>[^\s]\w+)_pt\s*$'),
        'get_pt_item' : re.compile(
            r'^.+::\s*(?P<item_name>.+)$')
    }

    # First pass: Get all *_pt sections in para_t
    config_sec_list = []
    in_para_t = False
    file = open(def_f90)
    for line in file:
        line = line.strip()
        m = re.compile(r'^type\s+para_t\s*$').match(line)
        if m:
            in_para_t = True
        m = re.compile(r'^end\s*type\s+para_t\s*$').match(line)
        if m:
            in_para_t = False
        if in_para_t:
            m = rx['get_pt_type'].match(line)
            if m:
                sec_name = m.group('sec_name').strip()
                if sec_name == 'user':
                    continue
                config_sec_list.append(sec_name)

    # Second pass: Get all allowed items from the *_pt type definitions
    in_pt_sec = False
    sec_name = ''
    file = open(def_f90)
    for line in file:
        line = line.strip()
        if in_pt_sec:
            m = re.compile(r'^end\s*type\s+(\w+)_pt$').match(line)
            if m:
                in_pt_sec = False
        m = rx['start_pt_sec'].match(line)
        if m:
            if in_pt_sec:
                raise AssertionError("Couldn't find end of last *_pt section")
            in_pt_sec = True
            sec_name = m.group('sec_name')
        if in_pt_sec and sec_name in config_sec_list:
            if sec_name not in config_structure:
                config_structure[sec_name] = []
            m = rx['get_pt_item'].match(line)
            if m:
                config_structure[sec_name].append(m.group('item_name').strip())

    # Write to file
    with open(outfile, 'w') as fh_out:
        json_opts = {'indent': 2, 'separators':(',',': '),
                     'sort_keys': True}
        fh_out.write(json.dumps(config_structure, **json_opts))


