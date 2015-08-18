from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import os
from .io import tempinput, open_file
import numpy as np

from collections import namedtuple

# "Filtered" representation of oct_iters data
ConvergenceData = namedtuple("Convergence",
['iter', 'J_T', 'g_a_int', 'g_b_int', 'J', 'Delta_J_T', 'Delta_J', 'sec'])

class OCTConvergences(object):
    """
    Collection of convergence data from multiple OCT runs.

    Attributes
    ----------

    data: dict
        key -> ConvergenceData namedtuple
    plot_label: dict
        key -> Label for legend in plot
    color: dict
        key -> color of line in plot
    splits: dict
        key -> tuple (iter, J_T, description)
    """

    def __init__(self, keys=None, files=None, colors=None, plot_labels=None):
        """
        Initialize Collection of convergence informations. All parameters are
        passed to the `load` method
        """
        self.data = {}
        self.splits = {}
        self.plot_label = {}
        self.color = {}
        self.J_T_max = -1.0e100
        self.J_T_min = 1.0e100
        if keys is not None:
            self.load(keys=keys, files=files, colors=colors,
                      plot_labels=plot_labels)

    def load(self, keys, files, colors=None, plot_labels=None):
        """
        Load data from a series of files. See `load_file` method for details.

        Parameters
        ----------

        keys: list of str
            For each file, a key under which to register the data (data will be
            stored in the attribute `data[key]`)

        files: list of str, list of file-like objects
            List of filenames, or filehandles from which to read data

        colors: list of str, str, None
            For each file, color in which the data should appear in a plot. If
            a single string, that color will be used for all files.

        plot_labels: list of str, None
            For each file, label for the legend of a plot. If None, the keys
            are used as labels.

        All parameters that are passed as lists must be of the same length.
        Otherwise, a ValueError is thrown.
        """
        if files is None:
            files = []
        if len(keys) != len(files):
            raise ValueError("keys and files must be of same length")
        default_color = 'black'
        if isinstance(colors, str):
            default_color = colors
            colors = None
        else:
            if len(colors) != len(keys):
                raise ValueError("keys and colors must be of same length")
        if plot_labels is not None:
            if len(plot_labels) != len(keys):
                raise ValueError("keys and plot_labels must be of same length")
        for i, key in enumerate(keys):
            plot_label = None
            if plot_labels is not None:
                plot_label = plot_labels[i]
            color = default_color
            if colors is not None:
                color = colors[i]
            self.load_file(key=key, file=files[i], color=color,
                           plot_label=plot_label)

    def load_file(self, key, file, color='black', plot_label=None):
        r'''
        Load convergence data from a single file, which must be in the format
        written by the QDYN fortran library when specifying the `iter_dat`
        attribute in the oct section of the config file.

        Paramters
        ---------

        key: str
            Key under which to register the data (i.e., data will be stored in
            the `data[key]` attribute)

        file: str of file-like object
            Name of file, or filehandle from which to read data

        plot_label: str, None
            Label of the data in the plot legend

        color: str
            Color

        Example
        -------

        >>> oct_iters = """# Fri Mar 27 23:44:56 +0100 2015
        ... # lambda_a =    5.000000E+08; lambda_intens =    0.000000E+00; lambda_b =    0.000000E+00
        ... # it.                 J_T             g_a_int             g_b_int                   J           Delta_J_T             Delta J sec/it
        ...     0  1.401102103266E-02  0.000000000000E+00  0.000000000000E+00  1.401102103266E-02  1.401102103266E-02  1.401102103266E-02      0
        ...     1  7.294159844106E-03  3.312556313731E-03  0.000000000000E+00  1.060671615784E-02 -6.716861188549E-03 -3.404304874818E-03     96
        ...     2  3.502016076673E-03  1.882015300126E-03  0.000000000000E+00  5.384031376798E-03 -3.792143767433E-03 -1.910128467308E-03     97
        ... """
        >>> c = OCTConvergences()
        >>> with tempinput(oct_iters) as oct_iters_dat:
        ...     c.load_file('iters', oct_iters_dat)
        >>> len(c.data)
        1
        >>> c.data['iters'].iter
        array([0, 1, 2])
        >>> c.data['iters'].J_T
        array([ 0.01401102,  0.00729416,  0.00350202])
        >>> c.data['iters'].J
        array([ 0.01401102,  0.01060672,  0.00538403])
        >>> c.data['iters'].Delta_J_T
        array([ 0.01401102, -0.00671686, -0.00379214])
        >>> c.data['iters'].Delta_J
        array([ 0.01401102, -0.0034043 , -0.00191013])
        >>> c.data['iters'].sec
        array([ 0, 96, 97])
        '''
        if plot_label is None:
            plot_label = key
        self.plot_label[key] = plot_label
        self.color[key] = color
        self.splits[key] = []
        prev_iter = -1
        with open_file(file) as in_fh:
            iters          = [] # vals[0]
            J_T_vals       = [] # vals[1]
            g_a_int_vals   = [] # vals[2]
            g_b_int_vals   = [] # vals[3]
            J_vals         = [] # vals[4]
            Delta_J_T_vals = [] # vals[5]
            Delta_J_vals   = [] # vals[6]
            sec_vals       = [] # vals[7]
            current_split = ""
            for line in in_fh:
                if line.startswith("#"):
                    if not line.startswith("# it."):
                        current_split += line[2:]
                else:
                    vals = line.split()
                    iter = int(vals[0])
                    if iter == prev_iter + 1:
                        J_T = float(vals[1])
                        if current_split != "":
                            self.splits[key].append(
                                (iter, J_T, current_split) )
                            current_split = ""
                        iters.append(iter)
                        J_T_vals.append(J_T)
                        if J_T < self.J_T_min:
                            self.J_T_min = J_T
                        if J_T > self.J_T_max:
                            self.J_T_max = J_T
                        g_a_int_vals.append(  float(vals[2]))
                        g_b_int_vals.append(  float(vals[3]))
                        J_vals.append(        float(vals[4]))
                        Delta_J_T_vals.append(float(vals[5]))
                        Delta_J_vals.append(  float(vals[6]))
                        sec_vals.append(        int(vals[7]))
                        prev_iter = iter
            self.data[key] = ConvergenceData(
                               iter      = np.array(iters, dtype=np.int),
                               J_T       = np.array(J_T_vals),
                               g_a_int   = np.array(g_a_int_vals),
                               g_b_int   = np.array(g_b_int_vals),
                               J         = np.array(J_vals),
                               Delta_J_T = np.array(Delta_J_T_vals),
                               Delta_J   = np.array(Delta_J_vals),
                               sec       = np.array(sec_vals, dtype=np.int))

    def show_bokeh(self, log_scale='y', outfile=None):
        """
        Show an interactive plot of all data, using the bokeh library

        In order to have the plot show up in an IPython Notebook, you must
        execute

        >>> import bokeh.plotting
        >>> bokeh.plotting.output_notebook()

        before calling the `show_bokeh` method. Alternatively, from a script,
        you may show the plot via a temporary html file, e.g.

        >>> bokeh.plotting.output_file('temp.html')

        Parameters
        ----------

        log_scale: str
            One of '', 'x', 'y', or 'xy', to indicate which axis (if any)
            should be plotted in a log scale

        outfile: str
            If given, write the plot to an html file instead of showing it.
        """
        from bokeh.plotting import figure, show, save, output_file
        from bokeh.models import ColumnDataSource, HoverTool
        fig_args = {'tools': "pan,box_zoom,reset,resize,hover",
                    'title': '', 'plot_width': 900, 'plot_height': 400,
                    'x_axis_label':"OCT iteration",
                    'y_axis_label':"Optimization Error"}
        if 'y' in log_scale:
            fig_args['y_axis_type'] = 'log'
            fig_args['y_range'] = [self.J_T_min, max(1.0, self.J_T_max)]
        if 'x' in log_scale:
            fig_args['x_axis_type'] = 'log'
            max_iter = 1
            for key in self.data.keys():
                if np.max(self.data[key].iter) > max_iter:
                    max_iter = np.max(self.data[key].iter)
            fig_args['x_range'] = [1, max_iter]
        p = figure(**fig_args)
        # collect the split points
        splits_J_T = []
        splits_iters = []
        splits_timestamps = []
        splits_lambda_vals = []
        splits_colors = []
        for key in self.data.keys():
            p.line(self.data[key].iter, self.data[key].J_T,
                  color=self.color[key], legend=self.plot_label[key])
            for (iter, J_T, description) in self.splits[key]:
                splits_J_T.append(J_T)
                splits_iters.append(iter)
                lines = [line for line in description.split("\n") if
                         line.strip() != "\n"]
                timestamp = lines[0]
                lambda_vals = " ".join(lines[1:])
                splits_timestamps.append(timestamp)
                splits_lambda_vals.append(lambda_vals)
                splits_colors.append(self.color[key])
        splits_data = ColumnDataSource(data=dict(
                    J_T=splits_J_T, iter=splits_iters,
                    timestamp=splits_timestamps,
                    lambda_vals=splits_lambda_vals,color=splits_colors))
        p.scatter(x='iter', y='J_T', source=splits_data, marker="circle")
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = [
            ("iter", "@iter"),
            ("J_T", "@J_T"),
            ("timestamp", "@timestamp"),
            ("lambdas", "@lambda_vals"),
            ]
        if outfile is None:
            show(p)
        else:
            output_file(outfile)
            save(p)
