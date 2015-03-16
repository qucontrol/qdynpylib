import os
from .io import open_file
import numpy as np

from collections import namedtuple

Convergence = namedtuple("Convergence", ['iter', 'J_T'])

class OCTConvergences(object):
    def __init__(self, oct_folders=None, iters_dat='oct_iters.dat'):
        self.convergence = {}
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

    def load_file(self, filename, label=None):
        key = str(filename)
        if label is None:
            if not key in self.labels:
                self.labels[key] = str(filename)
        else:
            self.labels[key] = label
        if not key in self.colors:
            self.colors[key] = 'black' # TODO
        if not key in self.splits:
            self.splits[key] = []
        prev_iter = -1
        with open_file(filename) as in_fh:
            iters = []
            J_T_vals = []
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
                        prev_iter = iter
            self.convergence[key] = Convergence(
                                        iter=np.array(iters),
                                        J_T=np.array(J_T_vals))

    def bokeh(self, log_scale=True):
        from bokeh.plotting import figure, show
        from bokeh.models import ColumnDataSource, HoverTool
        fig_args = {'tools': "pan,box_zoom,reset,resize,hover",
                    'title': '', 'plot_width': 900, 'plot_height': 400,
                    'x_axis_label':"OCT iteration",
                    'y_axis_label':"Optimization Error"}
        if log_scale:
            fig_args['y_axis_type'] = 'log'
            fig_args['y_range'] = [self.J_T_min, max(1.0, self.J_T_max)]
        p = figure(**fig_args)
        # collect the split points
        splits_J_T = []
        splits_iters = []
        splits_timestamps = []
        splits_lambda_vals = []
        splits_colors = []
        for key in self.convergence.keys():
            p.line(self.convergence[key].iter, self.convergence[key].J_T,
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
