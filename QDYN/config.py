""" Module containing utilities for managing QDYN config files """
from __future__ import print_function, division, absolute_import
import os
import json
import re
import base64
import random
from collections import OrderedDict

import six

from .units import UnitFloat

def protect_quotes(str_line):
    r'''Replace quoted expressions in the given `str_line` by a base-64 string
    which is guaranteed to contain only letters and numbers. Return the
    "protected" string and a list of tuples (base-64 replacement, quoted
    expression). Nested and escaped quotes are handled.

    >>> s_in = r'a = "text", '+"b = 'text'"
    >>> print(s_in)
    a = "text", b = 'text'
    >>> s, r = protect_quotes(s_in)
    >>> print(s)
    a = InRleHQi, b = J3RleHQn
    >>> print(r)
    [('InRleHQi', '"text"'), ('J3RleHQn', "'text'")]
    '''
    rx_dq = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')
    rx_sq = re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'")
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
    return str_line, replacements


def unprotect_quotes(str_line, replacements):
    r'''Inverse to :func:`protect_quotes`

    >>> s, r = protect_quotes(r'a = "text", '+"b = 'text'")
    >>> print(unprotect_quotes(s, r))
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


def escape_str(s):
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


def unescape_str(s):
    """Inverse to :func:`escape_str`"""
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


class ConfigReader(object):
    r"""Reader for a config file, allowing to obtain the lines with
    blank lines, comments and continuations stripped out.

    Arguments:
        filename (str or file handle): Name of a file (or open file handle)
            from which to read the raw config file.

    Example:

        >>> from six import StringIO
        >>> config_file = StringIO(r'''
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
        ... ''')
        >>> with ConfigReader(config_file) as config:
        ...     for line in config:
        ...         print(line)
        pulse: type = gauss, t_FWHM = 1.8, E_0 = 1.0, w_L = 0.2,  oct_shape = flattop, ftrt = 0.2, oct_lambda_a = 100,  oct_increase_factor = 10
        * id = 1, t_0 = 0, oct_outfile = pulse1.dat
        * id = 2, t_0 = 2.5, oct_outfile = pulse2.dat
        * id = 3, t_0 = 5, oct_outfile = pulse3.dat
        user_string: A = "x**2", B = 'B_{"avg"}'

    Note:
        If `filename` is an open filename, the filehandle is closed if
        `ConfigReader` is used as a context manager, it is not closed if the
        `ConfigReader` is used directly as an iterator,
        ``for line in ConfigReader(config_file)): print(line)`` in the above
        example. If `filename` is a string (i.e. the name of a file), the
        `ConfigReader` *must* be used as a context manager.

        The returned lines do *not* have a newline.
    """
    def __init__(self, filename):
        self.filename = filename
        self.fh = None
        self.line_nr = 0

    def __enter__(self):
        self.line_nr = 0
        try:
            self.fh = open(self.filename)
        except TypeError:
            self.fh = self.filename
        return self

    def next(self):
        line = ''
        if self.fh is None:
            raise ValueError("I/O operation on closed file")
        raw_line = self.fh.readline()
        self.line_nr += 1
        if len(raw_line) == 0:
            raise StopIteration()
        line, replacements = protect_quotes(raw_line.strip())
        # strip out out comments
        line = re.sub('!.*$', '', line).strip()
        while line.endswith('&'):
            line = line[:-1] # strip out trailing '&' in continued line
            line2, replacements2 = protect_quotes(self.fh.readline().strip())
            self.line_nr += 1
            replacements.extend(replacements2)
            # strip out comments
            line2 = re.sub('!.*$', '', line2).strip()
            # strip out leading '&' in continuing line
            if line2.startswith('&') and len(line2) > 1:
                line2 = line2[1:]
            line += line2
        line = unprotect_quotes(line, replacements)
        if len(line) == 0:
            return self.next()
        else:
            return line

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __exit__(self, exception_type, exception_value, traceback):
        self.fh.close()
        return False

    def readline(self):
        return self.next()


class ConfigWriter(object):
    r"""Writer for a config file, taking care of splitting lines at a
    reasonable line width

    Arguments:
        filename (str or file handle): Name of a file (or open file handle)
            to which to write the config file.
        linewidth (int): Approximate linewidth after which a line should be
        truncated

    Note: See :class:`ConfigReader` for caveats on using an open file handle
    for `filename`.
    """

    def __init__(self, filename, linewidth=80):
        self.filename = filename
        self.fh = None
        self.linewidth = linewidth
        self.line_nr = 0

    def __enter__(self):
        self.line_nr = 0
        try:
            self.fh = open(self.filename, 'w')
        except TypeError:
            self.fh = self.filename
        return self

    def write(self, line):
        """Write a single line. The line may be broken into several lines
        shorter than `linewidth`. Trailing newlines in `line` are ignored."""
        full_line, replacements = protect_quotes(line.strip())
        if len(full_line) == 0:
            self.fh.write("\n")
            self.line_nr += 1
            return
        rx_item = re.compile(r'(\w+\s*=\s*[\w.+-]+\s*,?\s*)')
        parts = [part for part in rx_item.split(full_line) if len(part) > 0]
        current_line = ''
        while len(parts) > 0:
            current_line += parts.pop(0)
            try:
                while len(current_line) + len(parts[0]) <= self.linewidth - 1:
                    current_line += parts.pop(0)
            except IndexError:
                pass # We've exhausted all parts
            else:
                if len(parts) > 0:
                    current_line += '&'
            current_line = unprotect_quotes(current_line, replacements)
            self.fh.write(current_line.strip() + "\n")
            self.line_nr += 1
            current_line = '  ' # indent for continuing line

    def writelines(self, lines):
        """Write a list of lines. The elements of `lines` should not have
        trailing newlines"""
        for line in lines:
            self.write(line)

    def __exit__(self, exception_type, exception_value, traceback):
        self.fh.close()
        return False


def render_config_lines(section_name, section_data):
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
        >>> for line in render_config_lines('pulse', section_data):
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
        line = "%s:" % section_name
        for key in section_data[0]:
            # we *do not* iterate over keys in common_items, so that items
            # are ordered the same as in section_data[0], instead of randomly
            if key in common_items:
                line += " %s = %s," % (key, escape_str(common_items[key]))
        if line.endswith(","):
            lines.append(line[:-1].strip())
        else:
            lines.append(line.strip())
        # item lines
        for item_line in section_data:
            line = "*"
            for key in item_line:
                if key not in common_items:
                    line += " %s = %s," % (key, escape_str(item_line[key]))
            if line.endswith(","):
                lines.append(line[:-1].strip())
            else:
                lines.append(line.strip())
    else: # section does not contain item lines, but key-value pairs only
        line = "%s:" % section_name
        for key in section_data:
            line += " %s = %s," % (key, escape_str(section_data[key]))
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
        lambda v: unescape_str(v)),
    ]
    return item_rxs


def read_config(filename):
    """Parse the given config file and return a nested data structure to
    represent it.

    Return a dict containing the following mapping::

        section name => dict(key => value)

    or::

        section name => list of dicts(key => value)
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

    # first pass: identify sections, list of lines in each section
    config_data = OrderedDict([])
    with ConfigReader(filename) as config:
        current_section = ''
        for line in config:
            line, replacements = protect_quotes(line)
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
    with ConfigReader(filename) as config:
        current_section = ''
        current_itemline = 0
        for line in config:
            line, replacements = protect_quotes(line)
            m_sec = rx_sec.match(line)
            m_itemline = rx_itemline.match(line)
            line_items = OrderedDict([])
            for item in rx_item.findall(line):
                matched = False
                for rx, setter in item_rxs:
                    m = rx.match(item)
                    if m:
                        val = unprotect_quotes(m.group('value'), replacements)
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
                    pass
            elif m_itemline:
                config_data[current_section][current_itemline].update(
                        line_items)
                current_itemline += 1
            else:
                raise ValueError("Could not parse line %d" % config.line_nr)
    return config_data


def write_config(config_data, filename):
    """Write out a config file

    Arguments:
        config_data (dict): data structure as returned by :func:`read_config`.
        filename (str): name of file to which to write config
    """
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
                lines.extend(render_config_lines(section, group))
                lines.append('')
        else:
            lines.extend(render_config_lines(section, config_data[section]))
            lines.append('')
    with ConfigWriter(filename) as config:
        config.writelines(lines)


def get_config_structure(def_f90, outfile='new_config_structure.json'):
    """Get a dumped .json-file with all the allowed section names and
    correspondig items of the new config structure genereated by reading the
    'para_t' type in the def.f90 file
    """
    config_structure = {}

    # Local regular expression filters
    RX = {
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
            m = RX['get_pt_type'].match(line)
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
        m = RX['start_pt_sec'].match(line)
        if m:
            if in_pt_sec:
                raise AssertionError("Couldn't find end of last *_pt section")
            in_pt_sec = True
            sec_name = m.group('sec_name')
        if in_pt_sec and sec_name in config_sec_list:
            if not sec_name in config_structure:
                config_structure[sec_name] = []
            m = RX['get_pt_item'].match(line)
            if m:
                config_structure[sec_name].append(m.group('item_name').strip())

    # Write to file
    with open(outfile, 'w') as fh_out:
        json_opts = {'indent': 2, 'separators':(',',': '),
                     'sort_keys': True}
        fh_out.write(json.dumps(config_structure, **json_opts))


