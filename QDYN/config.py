""" Module containing utitlities for managing QDYN config files """
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
        if `filename` is an open filename, the filehandle is closed if
        `ConfigReader` is used as a config manager, it is not closed if the
        `ConfigReader` is used directly as an iterator,
        ``for line in ConfigReader(config_file)): print(line)`` in the above
        example. If `filename` is a string (i.e. the name of a file), the
        `ConfigReader` *must* be used as a context manager.
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


def read_config(filename):
    """Parse the given config file and return a nested data structure to
    represent it.

    Return a dict containing the following mapping::

        section name => dict of key => value

    or::

        section name => list of dicts key => value
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


def write_config(config_data, filename='config_new'):
    """Write a config file to the given filename, based on the given
    config_data, where config_data is a dict as returned by `read_config`
    """
    lines = []
    line_break = 80

    # Part for non-user defined reals, integers or logicals section
    for sec_name in config_data:
        if sec_name.startswith('user'):
            continue
        line = sec_name + r':'
        item_line = '* '
        has_items = False
        for element in config_data[sec_name]:
            is_item_line = re.compile(r'^\d+$').match(str(element).strip())
            if is_item_line: # Corresponds to an item line
                if not has_items: # Only write section name once
                    lines.append(line)
                has_items = True
                item_line = '*'
                for item in config_data[sec_name][element]:
                    val = config_data[sec_name][element][item]
                    if len(item_line + r' ' + item + r' = ' + val + r',')\
                    > line_break:
                        lines.append(item_line + r' &')
                        item_line = ' '
                    item_line = item_line + r' ' + item + r' = ' + val + r','
                if item_line.endswith(','):
                    item_line = item_line[:-1]
                lines.append(item_line)
            else: # Corresponds to a non-allocatable section
                val = config_data[sec_name][element]
                if len(line + r' ' + element + r' = ' + val + r',')\
                > line_break:
                    lines.append(line + r' &')
                    line = ' ' * (len(sec_name)+1)
                line = line + r' ' + element + r' = ' + val + r','
        if line.endswith(','):
            line = line[:-1]
        if not is_item_line:
            lines.append(line)
        lines.append('')

    # Part for user defined reals, integers and logicals
    for sec_name in config_data:
        if not sec_name.startswith('user'):
            continue
        line = sec_name + r':'
        for element in config_data[sec_name]:
            element = str(element).strip()
            val = config_data[sec_name][element]
            if len(line + r' ' + element + r' = ' + val + r',')\
            > line_break:
                lines.append(line + r' &')
                line = ' ' * (len(sec_name)+1)
            line = line + r' ' + element + r' = ' + val + r','
        if line.endswith(','):
            line = line[:-1]
        lines.append(line)
        lines.append('')

    # If last line is empty, remove it
    while lines[-1] == '':
        del lines[-1]

    with open(filename, 'w') as out_fh:
        for line in lines:
            out_fh.write('%s\n' % line)


def get_new_config_structure(filename):
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
    file = open(filename)
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
    file = open(filename)
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
    with open('new_config_structure.json', 'w') as fh_out:
        json_opts = {'indent': 2, 'separators':(',',': '),
                     'sort_keys': True}
        fh_out.write(json.dumps(config_structure, **json_opts))


def get_old_config_structure(foldername):
    """Get a dumped .json-file with all the allowed section names and
    corresponding items of the old config struture generated by reading the
    *.fi files in qdyn/includes/parser
    """
    config_structure = {}

    for file in os.listdir(foldername):
        file = file.strip()
        if file == 'trap.fi':
            continue
        if file.endswith('.fi'):
            m = re.compile(r'^(?P<sec_name>[^\s].*)\.fi$').match(file)
            if m:
                sec_name = m.group('sec_name')
                if sec_name == 'user':
                    continue
            else:
                raise AssertionError("Unreadable file in passed folder")
            item_list = read_old_parser_fi(file)
            config_structure[sec_name] = item_list

    # Write to file
    with open('old_config_structure.json', 'w') as fh_out:
        json_opts = {'indent': 2, 'separators':(',',': '),
                     'sort_keys': True}
        fh_out.write(json.dumps(config_structure, **json_opts))


def read_old_parser_fi(filename):
    """Read an old *.fi file in the '../qdyn/includes/parser' folder and
    extract all allowed items for this section
    """
    item_list = []

    file = open(filename)
    for line in file:
        line = line.strip()
        m = re.compile(r'^case\s*\(\'(?P<item_name>[^\s].*).*\'\)$')\
            .match(line)
        if m:
            item_name = m.group('item_name')
            item_list.append(item_name)

    return item_list


def read_old_config(filename):
    """Parse the given config file in the old format and return a nested data
    structure to represent it.

    Returns a dict mapping

        section name => dict of key => value; or
        section name => list of dicts key => value
    """
    config_data = {}

    # Local regular expression filters
    RX = {
        'section': re.compile(
            r'^(?P<sec_name>[^\d]\w+)\s*:\s*(?P<sec_line>.+)$'),
        'item_line': re.compile(
            r'^(?P<item_num>\d+)\s*:\s*(?P<item_line>.+)$'),
        'item': re.compile(
            r'^(?P<item_name>.+)\s*=\s*(?P<item_val>.+)$'),
        'continue_line': re.compile(
            r'^(?P<line>.*)\&$')
    }

    # Collect all existing section types
    file = open(filename)
    for line in file:
        line = line.strip()
        m = RX['section'].match(line)
        if m:
            sec_name = m.group('sec_name')
            if not sec_name in config_data:
                config_data[sec_name]               = {}
                config_data[sec_name]['item_lines'] = {}
                config_data[sec_name]['num_lines']  = 0

    # Get lines and items for each section
    sec_name = ''
    has_line_break = False

    file = open(filename)
    for line in file:
        has_comment = False
        if '!' in line:
            has_comment = True
            line = line[0:line.index('!')]
        line = line.strip()
        if line.startswith('&'):
            line = line[1:]
        line = line.strip()
        if line == '':
            continue
        if not has_line_break:
            curr_line = ''

        m = RX['continue_line'].match(line)
        if m and not has_comment:
            has_line_break = True
            curr_line = curr_line + m.group('line')
            continue
        else:
            has_line_break = False
        curr_line = curr_line + line

        m = RX['section'].match(curr_line)
        if m:
            sec_name = m.group('sec_name').strip()
            sec_line_array = m.group('sec_line').strip().split(',')

        m = RX['item_line'].match(curr_line)
        if m: # Section item line
            config_data[sec_name]['num_lines'] += 1
            item_num = 1
            while item_num in config_data[sec_name]['item_lines']:
                item_num += 1
            config_data[sec_name]['item_lines'][item_num] = {}
            item_line_array = m.group('item_line').strip().split(',')
            # If 'item_num = 0' has been added in case of a section with item
            # lines, remove it from config_data, since we want the item lines
            # to start with 1 in that case
            if 0 in config_data[sec_name]['item_lines']:
                del config_data[sec_name]['item_lines'][0]
        else: # No section item line
            item_num = 0
            config_data[sec_name]['item_lines'][item_num] = {}
            item_line_array = []

        # If item line, remove entry 'n = {number}' from sec_line_array,
        # since this only indicates the number of following item lines.
        # This information is gathered in 'num_lines'
        for item in sec_line_array:
            m = RX['item'].match(item)
            if m:
                if ('n' == m.group('item_name').strip()):
                    sec_line_array.remove(item)
                    break
        # Join arrays with all necessary items
        item_array = sec_line_array + item_line_array

        for item in item_array:
            item = item.strip()
            # Check for missing ',' => otherwise additional '=' signs
            # will appear
            if item.count('=') != 1:
                print('Line |' + line)
                raise AssertionError("Number of '=' differs from one. "
                "Missing or too much '=' signs?")
            m_item = RX['item'].match(item)
            if m_item:
                item_name = m_item.group('item_name').strip()
                item_val  = m_item.group('item_val').strip()
                config_data[sec_name]['item_lines'][item_num]\
                    [item_name] = item_val
            else:
                print('Line |' + line)
                raise AssertionError("Item name and value have to be separated"
                " by a '=' sign")

    return config_data


def get_mappings(old_config_structure_file='old_config_structure.json',
                 new_config_structure_file='new_config_structure.json'):
    """Get the mappings between the old config structure sections/items and the
    new config strucuter sections/items
    """
    mappings = {}

    # Read from files
    with open(old_config_structure_file) as old_fh:
        old_config_structure = json.load(old_fh)
    with open(new_config_structure_file) as new_fh:
        new_config_structure = json.load(new_fh)

    for sec_name in old_config_structure:
        mappings[sec_name] = {}
        # Check if section exits in new config structure
        if sec_name in new_config_structure:
            for item in old_config_structure[sec_name]:
                # Check if item exists ..
                if item in new_config_structure[sec_name]:
                    mappings[sec_name][item] = (sec_name, item)
                # ... or otherwise replace the item name with the new name
                elif sec_name == 'pulse' and item == 'oct_alpha':
                    mappings[sec_name][item] = (sec_name, 'oct_lamba_a')
                elif sec_name == 'oct' and item == 'max_megs':
                    mappings[sec_name][item] = (sec_name, 'max_ram_mb')
                elif sec_name == 'oct' and item == 'type':
                    mappings[sec_name][item] = (sec_name, 'method')
                elif sec_name == 'grid' and item == 'gll':
                    mappings[sec_name][item] = (sec_name, 'coord_type')
                elif sec_name == 'grid' and item == 'rmin':
                    mappings[sec_name][item] = (sec_name, 'r_min')
                elif sec_name == 'grid' and item == 'rmax':
                    mappings[sec_name][item] = (sec_name, 'r_max')
                elif sec_name == 'grid' and item == 'Emax':
                    mappings[sec_name][item] = (sec_name, 'E_max')
                elif sec_name == 'grid' and item == 'map':
                    mappings[sec_name][item] = (sec_name, 'maptype')
                elif sec_name == 'grid' and item == 'n':
                    mappings[sec_name][item] = (sec_name, 'nr')
                elif sec_name == 'grid' and item == 'system':
                    mappings[sec_name][item] = (sec_name, 'label')
                elif sec_name == 'psi' and item == 'system':
                    mappings[sec_name][item] = (sec_name, 'label')
                elif sec_name == 'psi' and item == 'width':
                    mappings[sec_name][item] = (sec_name, 'sigma')
                else:
                    pass
                    # Silently ignore missing assigments
        # If sec_name has been replaced/renamed in the new config structure
        else:
            if sec_name in ['pot', 'dip', 'stark', 'so', 'cwlaser', 'spin']:
                # Move all operator like types to new ham_pt
                for item in old_config_structure[sec_name]:
                    target_item = item
                    if item == 'type':
                        target_item = 'op_form'
                    if item == 'system':
                        target_item = 'label'
                    if item == 'surf':
                        target_item = 'op_surf'
                    if item == 'surf1':
                        target_item = 'op_surf_1'
                    if item == 'surf2':
                        target_item = 'op_surf_2'
                    if item == 'photons':
                        target_item = 'n_photons'
                    if item == 'd':
                        target_item = 'offset'
                    if item == 'de':
                        target_item = 'depth'
                    if item == 'a':
                        target_item = 'width'
                    if item == 'r_0':
                        target_item = 'rat_r_0'
                    if item == 'mu_0' or item == 'val':
                        target_item = 'E_0'
                    if item == 'j':
                        target_item = 'rotbarr_j'
                    if item == 'imaginary':
                        target_item = 'imag_op'
                    if item == 'dip_unit':
                        target_item = 'op_unit'
                    if item == 'pot_unit':
                        target_item = 'op_unit'
                    if item == 'SO_0':
                        target_item = 'asym_SO'
                    if item == 'so_unit':
                        target_item = 'op_unit'
                    if item == 'step_ex':
                        target_item = 'ex_step'
                    if item == 'periodic':
                        target_item = 'is_periodic'
                    if item == 'stark_0':
                        target_item = 'asym_stark'
                    if item == 'w' or item =='negative':
                        pass # Ignore trap_pt types
                    # Check for existence in the new config structure
                    if not target_item in new_config_structure['ham']:
                        raise KeyError("Item '%s' doesn't exist in new config "
                                       "structure" % target_item)
                    mappings[sec_name][item] = ('ham', target_item)
            elif sec_name == 'misc':
                for item in old_config_structure[sec_name]:
                    target_item = item
                    # Part moved to new prop_pt
                    if item in ['prop', 'inhom_prop', 'ITO_guess_method',\
                                'ITO_do_iter', 'inhom_max_order', 'n_taylor',\
                                'inhom_expan_err', 'rk45_relerr',\
                                'rk45_abserr', 'newton_relerr',\
                                'newton_norm_min', 'newton_arnoldi_order',\
                                'newton_max_restarts']:
                        target_sec = 'prop'
                        if item == 'prop':
                            target_item = 'method'
                        if item == 'inhom_prop':
                            target_item = 'inhom_method'
                    # Part moved to new ham_pt
                    elif item in ['mass', 'map_to_j', 'specrad_method',\
                                  'base', 'memoize_pots']:
                        target_sec = 'ham'
                        if item == 'memoize_pots':
                            target_item = 'memoize_ops'
                        if item == 'base':
                            target_item = 'kin_base'
                    # Part moved to pulse_pt
                    elif item in ['rwa']:
                        target_sec = 'pulse'
                    # Part moved to grid_pt
                    elif item in ['spher_method']:
                        target_sec = 'grid'
                    # Part moved to oct_pt
                    elif item in ['bwr_nint', 'bwr_base']:
                        target_sec = 'oct'
                    else:
                        raise KeyError("Unknown item '%s' in misc" % item)
                    # Check for existence in the new config structure
                    if not target_item in new_config_structure[target_sec]:
                        raise KeyError("Item '%s' doesn't exist in new config "
                                       "structure" % target_item)
                    mappings[sec_name][item] = (target_sec, target_item)
            elif sec_name == 'trap':
                # Silently ignore trap sections
                continue
            else:
                raise KeyError("Unknown section name '%s' detected" % sec_name)

    # Add special cases
    mappings['oct']['max_seconds'] = ('oct', 'max_seconds')

    return mappings


def convert_config(filename, mappings):
    """Convert the old_config_data, as returned by `read_old_config` into
    a data structure compatible with the current config file. The conversion is
    performed using a list of mappings, where is encoded to which section and
    item of the new config structure an item from the old config structure is
    assigned
    """
    new_config_data = {}

    old_config_data = read_old_config(filename)

    obsolete_sec_items = [
        ('pulse', 'ftrt')
    ]

    ham_line = 0
    for sec_name in old_config_data:
        if sec_name.startswith('user') or sec_name == 'misc':
            continue

        for item_line in old_config_data[sec_name]['item_lines']:
            curr_item_line = item_line
            if sec_name in ['pot', 'dip', 'stark', 'so', 'cwlaser', 'spin']:
                ham_line += 1
                curr_item_line = ham_line

            for item in old_config_data[sec_name]['item_lines'][item_line]:
                # Check for obsolete items, will be ignored since they have to
                # be replaced by hand anyways
                curr_item = item
                if (sec_name, curr_item) in obsolete_sec_items\
                or sec_name == 'trap':
                    print("WARNING! Skipping item '%s' in section '%s'"\
                          % (curr_item, sec_name))
                    continue
                # Check for cases which require a special treatment
                if sec_name == 'oct' and curr_item == 'max_hours':
                    print("***** Converting 'max_hours' to 'max_seconds'")
                    item_val = str(int(old_config_data[sec_name]['item_lines']\
                               [item_line][curr_item])*3600)
                    curr_item = 'max_seconds'
                elif sec_name == 'grid' and curr_item == 'coord_type':
                    item_val = old_config_data[sec_name]['item_lines']\
                               [item_line][curr_item]
                    if item_val.startswith('cartesian'):
                        item_val = 'cartesian'
                else:
                    if not sec_name in mappings:
                        raise KeyError("Unknown section name: '%s'" % sec_name)
                    item_val = old_config_data[sec_name]['item_lines']\
                               [item_line][curr_item]
                new_sec_name, new_item_name = mappings[sec_name][curr_item]

                # Add section if necessary
                if not new_sec_name in new_config_data:
                    new_config_data[new_sec_name] = {}

                # Sections that doesn't allow for lines
                if new_sec_name in ['tgrid', 'oct', 'prop', 'numerov']:
                    curr_item_line = 0

                if curr_item_line > 0 and\
                not curr_item_line in new_config_data[new_sec_name]:
                    new_config_data[new_sec_name][curr_item_line] = {}

                # Add item to section or to section line
                if curr_item_line > 0:
                    new_config_data[new_sec_name][curr_item_line]\
                        [new_item_name] = item_val
                else:
                    new_config_data[new_sec_name][new_item_name] = item_val

                # Add `op_type` if necessary
                if sec_name in ['pot', 'dip', 'stark', 'so', 'cwlaser',\
                                'spin']:
                    new_config_data[new_sec_name][curr_item_line]['op_type']\
                        = sec_name
                    if not 'type' in\
                    new_config_data[new_sec_name][curr_item_line]:
                        new_config_data[new_sec_name][curr_item_line]['type']\
                            = 'op'

    # Add parameters from former 'misc' section to respective new section. If
    # they allow for lines, add the item to each line
    if 'misc' in old_config_data:
        for item in old_config_data['misc']['item_lines'][0]:
            new_sec_name, new_item_name = mappings['misc'][item]
            if not 'misc' in mappings:
                raise KeyError("Unknown section name: '%s'" % sec_name)
            item_val = old_config_data['misc']['item_lines'][0][item]
            if not new_sec_name in new_config_data:
                new_config_data[new_sec_name] = {}
            if new_sec_name in ['tgrid', 'oct', 'prop', 'numerov']:
                new_config_data[new_sec_name][new_item_name] = item_val
            else:
                # Check for empty dictionary
                if not new_config_data[new_sec_name]:
                    new_config_data[new_sec_name][1] = {}
                for item_line in new_config_data[new_sec_name]:
                    new_config_data[new_sec_name][item_line][new_item_name]\
                        = item_val

    # User defined parameters stay unaltered in the new config structure
    for sec_name in old_config_data:
        if sec_name.startswith('user'):
            for item in old_config_data[sec_name]['item_lines'][0]:
                new_sec_name  = sec_name
                new_item_name = item
                item_val = old_config_data[sec_name]['item_lines'][0][item]
                if not new_sec_name in new_config_data:
                    new_config_data[new_sec_name] = {}
                new_config_data[new_sec_name][new_item_name]\
                    = item_val

    # Set 'dim' parameter in for each grid line
    if 'grid' in new_config_data:
        for item_line in new_config_data['grid']:
            new_config_data['grid'][item_line]['dim'] = str(item_line)
            if not 'coord_type' in new_config_data['grid'][item_line]:
                new_config_data['grid'][item_line]['coord_type']\
                    = 'cartesian'

    return new_config_data
