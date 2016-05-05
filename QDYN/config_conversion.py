"""Converting the "old" config file format to the "new" format"""
import re
import os
import json
from collections import OrderedDict

from .config import _process_raw_lines, _item_rxs, _protect_quotes, _unprotect_quotes


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

    Returns a dict mapping::

        section name => dict of key => value; or
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
        [\d]+ \s*:\s*
        (?P<items>
            (\w+ \s*=\s* [\w.+-]+ [\s,]+)*
            (\w+ \s*=\s* [\w.+-]+ \s*)
        )
        ''', re.X)
    rx_item = re.compile(r'(\w+\s*=\s*[\w.+-]+)')
    item_rxs = _item_rxs()

    # first pass: identify sections, list of lines in each section
    config_data = OrderedDict([])
    with open(filename) as in_fh:
        current_section = ''
        for line in _process_raw_lines(in_fh):
            line, replacements = _protect_quotes(line)
            m_sec = rx_sec.match(line)
            m_itemline = rx_itemline.match(line)
            if m_sec:
                current_section = m_sec.group('section')
                config_data[current_section] = None
            elif m_itemline:
                if config_data[current_section] is None:
                    config_data[current_section]  = []
                config_data[current_section].append(OrderedDict([]))

    # second pass: set items
    with open(filename) as in_fh:
        current_section = ''
        current_itemline = 0
        for line in _process_raw_lines(in_fh):
            line, replacements = _protect_quotes(line)
            m_sec = rx_sec.match(line)
            m_itemline = rx_itemline.match(line)
            line_items = OrderedDict([])
            for item in rx_item.findall(line):
                matched = False
                for rx, setter in item_rxs:
                    m = rx.match(item)
                    if m:
                        val = _unprotect_quotes(m.group('value'), replacements)
                        line_items[m.group('key')] = setter(val)
                        matched = True
                        break
                if not matched:
                    raise ValueError("Could not parse item '%s'" % str(item))
            if m_sec:
                current_itemline = 0
                current_section = m_sec.group('section')
                if config_data[current_section] is None:
                    config_data[current_section] = [line_items, ]
                else:
                    try:
                        del line_items['n']
                    except KeyError:
                        pass
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


def convert_config(old_config_data, mappings):
    """Convert the old_config_data, as returned by `read_old_config` into
    a data structure compatible with the current config file. The conversion is
    performed using a list of mappings, where is encoded to which section and
    item of the new config structure an item from the old config structure is
    assigned
    """
    new_config_data = {}

    obsolete_sec_items = [
        ('pulse', 'ftrt')
    ]

    ham_index = 0
    for sec_name in old_config_data:
        if sec_name.startswith('user') or sec_name == 'misc':
            continue

        for i in range(len(old_config_data[sec_name])):
            item_index = i
            if sec_name in ['pot', 'dip', 'stark', 'so', 'cwlaser', 'spin']:
                ham_index += 1
                item_index = ham_index

            for item in old_config_data[sec_name][i]:
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
                    item_val = str(int(old_config_data[sec_name]\
                               [i][curr_item])*3600)
                    curr_item = 'max_seconds'
                elif sec_name == 'grid' and curr_item == 'coord_type':
                    item_val = old_config_data[sec_name][i][curr_item]
                    if item_val.startswith('cartesian'):
                        item_val = 'cartesian'
                else:
                    if not sec_name in mappings:
                        raise KeyError("Unknown section name: '%s'" % sec_name)
                    item_val = old_config_data[sec_name][i][curr_item]
                new_sec_name, new_item_name = mappings[sec_name][curr_item]

                # Add section if necessary
                if not new_sec_name in new_config_data:
                    new_config_data[new_sec_name] = {}

                # Sections that doesn't allow for lines
                if new_sec_name in ['tgrid', 'oct', 'prop', 'numerov']:
                    item_index = 0

                if item_index > 0 and\
                not item_index in new_config_data[new_sec_name]:
                    new_config_data[new_sec_name][item_index] = {}

                # Add item to section or to section line
                if item_index > 0:
                    new_config_data[new_sec_name][item_index]\
                        [new_item_name] = item_val
                else:
                    new_config_data[new_sec_name][new_item_name] = item_val

                # Add `op_type` if necessary
                if sec_name in ['pot', 'dip', 'stark', 'so', 'cwlaser',\
                                'spin']:
                    new_config_data[new_sec_name][item_index]['op_type']\
                        = sec_name
                    if not 'type' in\
                    new_config_data[new_sec_name][item_index]:
                        new_config_data[new_sec_name][item_index]['type']\
                            = 'op'

    # Add parameters from former 'misc' section to respective new section. If
    # they allow for lines, add the item to each line
    if 'misc' in old_config_data:
        for item in old_config_data['misc'][0]:
            new_sec_name, new_item_name = mappings['misc'][item]
            if not 'misc' in mappings:
                raise KeyError("Unknown section name: '%s'" % sec_name)
            item_val = old_config_data['misc'][0][item]
            if not new_sec_name in new_config_data:
                new_config_data[new_sec_name] = {}
            if new_sec_name in ['tgrid', 'oct', 'prop', 'numerov']:
                new_config_data[new_sec_name][new_item_name] = item_val
            else:
                # Check for empty dictionary
                if not new_config_data[new_sec_name]:
                    new_config_data[new_sec_name][1] = {}
                for item_line in new_config_data[new_sec_name]:
                    item_line[new_item_name] = item_val

    # User defined parameters stay unaltered in the new config structure
    for sec_name in old_config_data:
        if sec_name.startswith('user'):
            for item in old_config_data[sec_name][0]:
                new_sec_name  = sec_name
                new_item_name = item
                item_val = old_config_data[sec_name][0][item]
                if not new_sec_name in new_config_data:
                    new_config_data[new_sec_name] = {}
                new_config_data[new_sec_name][new_item_name]\
                    = item_val

    # Set 'dim' parameter in for each grid line
    if 'grid' in new_config_data:
        for i, item_line in enumerate(new_config_data['grid']):
            item_line['dim'] = str(i)
            if not 'coord_type' in new_config_data['grid'][i]:
                new_config_data['grid'][i]['coord_type'] = 'cartesian'

    return new_config_data
