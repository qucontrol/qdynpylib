import os
from .io import open_file
import numpy as np

from collections import namedtuple

# "Filtered" representation of oct_iters data
ConvergenceData = namedtuple("Convergence",
['iter', 'J_T', 'g_a_int', 'g_b_int', 'J', 'Delta_J_T', 'Delta_J', 'sec'])

class OCTConvergences(object):

    def __init__(self, oct_folders=None, iters_dat='oct_iters.dat'):
        self.data = {}
        self.splits = {}
        self.labels = {}
        self.colors = {}
        self.J_T_max = -1.0e100
        self.J_T_min = 1.0e100
        if oct_folders is not None:
            self.load(oct_folders, iters_dat)

    def load(self, oct_folders, iters_data='oct_iters.dat'):
        for folder in oct_folders:
            self.load_file(os.path.join(folder, iters_data), label=folder)

    def load_file(self, filename, key=None, label=None):
        r'''
        >>> from StringIO import StringIO
        >>> from textwrap import dedent
        >>> oct_iters = """# Fri Mar 27 23:44:56 +0100 2015
        ... # lambda_a =    5.000000E+08; lambda_intens =    0.000000E+00; lambda_b =    0.000000E+00
        ... # it.                 J_T             g_a_int             g_b_int                   J           Delta_J_T             Delta J sec/it
        ...     0  1.401102103266E-02  0.000000000000E+00  0.000000000000E+00  1.401102103266E-02  1.401102103266E-02  1.401102103266E-02      0
        ...     1  7.294159844106E-03  3.312556313731E-03  0.000000000000E+00  1.060671615784E-02 -6.716861188549E-03 -3.404304874818E-03     96
        ...     2  3.502016076673E-03  1.882015300126E-03  0.000000000000E+00  5.384031376798E-03 -3.792143767433E-03 -1.910128467308E-03     97
        ... """
        >>> c = OCTConvergences()
        >>> c.load_file(StringIO(oct_iters), key='oct_iters.dat')
        >>> c.data
        {'oct_iters.dat': Convergence(iter=array([0, 1, 2]), J_T=array([ 0.01401102,  0.00729416,  0.00350202]), g_a_int=array([ 0.        ,  0.00331256,  0.00188202]), g_b_int=array([ 0.,  0.,  0.]), J=array([ 0.01401102,  0.01060672,  0.00538403]), Delta_J_T=array([ 0.01401102, -0.00671686, -0.00379214]), Delta_J=array([ 0.01401102, -0.0034043 , -0.00191013]), sec=array([ 0, 96, 97]))}
        '''
        if key is None:
            key = str(filename)
        if label is None:
            if not key in self.labels:
                self.labels[key] = str(key)
        else:
            self.labels[key] = label
        if not key in self.colors:
            self.colors[key] = 'black' # TODO
        if not key in self.splits:
            self.splits[key] = []
        prev_iter = -1
        with open_file(filename) as in_fh:
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

    def bokeh(self, log_scale='y'):
        from bokeh.plotting import figure, show
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
                  color=self.colors[key], legend=self.labels[key])
            for (iter, J_T, description) in self.splits[key]:
                splits_J_T.append(J_T)
                splits_iters.append(iter)
                lines = [line for line in description.split("\n") if
                         line.strip() != "\n"]
                timestamp = lines[0]
                lambda_vals = " ".join(lines[1:])
                splits_timestamps.append(timestamp)
                splits_lambda_vals.append(lambda_vals)
                splits_colors.append(self.colors[key])
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
        show(p)
