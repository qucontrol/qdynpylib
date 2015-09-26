"""
Routines for calculating local invariants, concurrence, and related quantities
for two-qubit gates
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import numpy as np
from numpy import cos, sin, exp, pi
from numpy import less, greater
from numpy import logical_and as And
from numpy import logical_or as Or
from numpy import less_equal as less_eq
from numpy import greater_equal as greater_eq
from mpl_toolkits.mplot3d import Axes3D # required to activate 2D plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from six.moves import xrange
from .gate2q import Gate2Q
from .linalg import vectorize
from .memoize import memoize
from scipy.optimize import leastsq
from scipy.linalg import expm

# TODO: allow to obtain gates for names points in Weyl chamber

Qmagic = (1.0/np.sqrt(2.0)) * np.matrix(
                   [[ 1,  0,  0,  1j],
                    [ 0, 1j,  1,  0],
                    [ 0, 1j, -1,  0],
                    [ 1,  0,  0, -1j]], dtype=np.complex128)


SxSx   =  np.matrix( # sigma_x * sigma_x
                   [[ 0,  0,  0,  1],
                    [ 0,  0,  1,  0],
                    [ 0,  1,  0,  0],
                    [ 1,  0,  0,  0]], dtype=np.complex128)

SySy   =  np.matrix( # sigma_y * sigma_y
                   [[ 0,  0,  0, -1],
                    [ 0,  0,  1,  0],
                    [ 0,  1,  0,  0],
                    [-1,  0,  0,  0]], dtype=np.complex128)

SzSz   =  np.matrix( # sigma_x * sigma_x
                   [[ 1,  0,  0,  0],
                    [ 0, -1,  0,  0],
                    [ 0,  0, -1,  0],
                    [ 0,  0,  0,  1]], dtype=np.complex128)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class WeylChamber():
    """Class for plotting data in the Weyl Chamber

    Class Attributes
    ----------------

    weyl_points: dict
        Dictionary of Weyl chamber point names to (c1, c2, c3) coordinates (in
        units of pi). Each point name is also a class attribute itself
    normal: dict
        Dictionary of Weyl chamber region name to normal vectors for the
        surface that separates the region from the polyhedron of perfect
        entanglers (pointing outwards from the PE's). The three regions are
        'W0': region from the origin point (O) to the PE polyhedron
        'W0*': region from the A2 point to the PE polyhedron
        'W1': region from the A2 point (SWAP gate) to the PE polyhedron
    anchor: dict
        Dictionary of anchor points for the normal vectors (i.e., an arbitrary
        point on the surface that separates the region specified by the key
        from the perfect entanglers polyhedron

    Attributes
    ----------
    fig_width : float {8.5}
        Figure width, in cm
    fig_height: float {6.0}
        Figure height, in cm
    left_margin: float {0.0}
        Distance from left figure edge to axes, in cm
    bottom_margin: float {0.0}
        Distance from bottom figure edge to axes, in cm
    right_margin: float {0.3}
        Distance from right figure edge to axes, in cm
    top_margin: float {0.0}
        Distance from top figure edge to axes, in cm
    azim: float {-50}
        azimuthal view angle in degrees
    elev: float {-20}
        elevation view angle in degrees
    dpi: int {300}
        Resolution to use when saving to dpi, or showing interactively
    linecolor: str {'black'}
        Color to be used when drawing lines (e.g. the edges of the Weyl
        chamber)
    weyl_edges: list
        List of tuples (point1, point2, foreground) where point1 and point2 are
        keys in weyl_points, and foreground is a logical to indicate whether
        the edge is in the background or foreground (depending on the
        perspective of the plot). Describes the lines that make up the Weyl
        chamber.
    weyl_edge_fg_properties: dict
        Properties to be used when drawing foreground weyl_edges
    weyl_edge_bg_properties: dict
        Properties to be used when drawing background weyl_edges
    PE_edges: list
        List of tuples (point1, point2, foreground) where point1 and point2 are
        keys in weyl_points, and foreground is a logical to indicate whether
        the edge is in the background or foreground (depending on the
        perspective of the plot). Desribes the lines that make up the
        polyhedron of perfect entanglers
    PE_edge_fg_properties: dict
        Properties to be used when drawing foreground PE_edges
    PE_edge_bg_properties: dict
        Properties to be used when drawing background PE_edges
    labels: dict
        label names => array (c1, c2, c3) where label should be drawn
    tex_labels: logical {True}
        If True wrap label names in dollar signs to produce a latex string.
    label_properties: dict
        Properties to be used when drawing labels
    z_axis_left: logical {True}
        If True, draw z-axis on the left
    grid: logical {False}
        Show a grid on panes?
    panecolor: None or tuple  {(1.0, 1.0, 1.0, 0.0)}
        Color (r, g, b, alpha) with values in [0,1] for the c1, c2, and c3
        panes
    facecolor: str {'None'}
        Name of color for figure background
    ticklabelsize: float {7}
        font size for tick labels
    full_cube: logical {False}
        if True, draw all three axes in the range [0,1]. This may result in a
        less distorted view of the Weyl chamber
    """
    A1 = np.array((1, 0, 0))
    A2 = np.array((0.5, 0.5, 0))
    A3 = np.array((0.5, 0.5, 0.5))
    O  = np.array((0, 0, 0))
    L  = np.array((0.5, 0, 0))
    M  = np.array((0.75, 0.25, 0))
    N  = np.array((0.75, 0.25, 0.25))
    P  = np.array((0.25, 0.25, 0.25))
    Q  = np.array((0.25, 0.25, 0))
    weyl_points = {'A1' : A1, 'A2' : A2, 'A3' : A3, 'O' : O, 'L' : L,
                   'M' : M, 'N' : N, 'P' : P, 'Q': Q}
    normal = {'W0' : (np.sqrt(2.0)/2.0)*np.array((-1, -1, 0)),
              'W0*': (np.sqrt(2.0)/2.0)*np.array(( 1, -1, 0)),
              'W1' : (np.sqrt(2.0)/2.0)*np.array(( 0,  1, 1))}
    anchor = {'W0': L, 'W0*': L, 'W1': A2}

    def __init__(self):
        self._fig = None
        self._ax = None
        self._artists = None
        self.azim = -50
        self.elev = 20
        self.dpi = 300
        self.fig_width = 8.5
        self.fig_height = 6.0
        self.left_margin = 0.0
        self.bottom_margin =  0.0
        self.right_margin =  0.3
        self.top_margin =  0.0
        self.linecolor = 'black'
        self.weyl_edge_fg_properties = {
                'color':'black', 'linestyle':'-', 'lw':0.5}
        self.weyl_edge_bg_properties = {
                'color':'black', 'linestyle':'--', 'lw':0.5}
        self.PE_edge_fg_properties = {
                'color':'black', 'linestyle':'-', 'lw':0.5}
        self.PE_edge_bg_properties = {
                'color':'black', 'linestyle':'--', 'lw':0.5}
        self.weyl_edges = [
            [ 'O', 'A1', True],
            ['A1', 'A2', True],
            ['A2', 'A3', True],
            ['A3', 'A1', True],
            ['A3',  'O', True],
            [ 'O', 'A2', False]
        ]
        self.PE_edges = [
            ['L',  'N', True],
            ['L',  'P', True],
            ['N',  'P', True],
            ['N', 'A2', True],
            ['N',  'M', True],
            ['M',  'L', False],
            ['Q',  'L', False],
            ['P',  'Q', False],
            ['P', 'A2', False]
        ]
        self.labels = {
            'A_1' : self.A1 + np.array((-0.03, 0.04 , 0.00)),
            'A_2' : self.A2 + np.array((0.01, 0, -0.01)),
            'A_3' : self.A3 + np.array((-0.01, 0, 0)),
            'O'   : self.O  + np.array((-0.025,  0.0, 0.02)),
            'L'   : self.L  + np.array((-0.075, 0, 0.01)),
            'M'   : self.M  + np.array((0.05, -0.01, 0)),
            'N'   : self.N  + np.array((-0.075, 0, 0.009)),
            'P'   : self.P  + np.array((-0.05, 0, 0.008)),
            'Q'   : self.Q  + np.array((0, 0.01, 0.03)),
        }
        self.label_properties = {
            'color': 'black',  'fontsize': 'small'
        }
        self.tex_labels = True
        self.z_axis_left = True
        self.grid = False
        self.panecolor = (1.0, 1.0, 1.0, 0.0)
        self.facecolor = 'None'
        self.ticklabelsize = 7
        self.full_cube = False
        self._scatter = []

    @property
    def figsize(self):
        """Tuple (width, height) of figure size in inches"""
        cm2inch = 0.39370079 # conversion factor cm to inch
        return (self.fig_width*cm2inch, self.fig_height*cm2inch)

    @property
    def fig(self):
        """Return a reference to the figure on which the Weyl chamber has been
        rendered. Undefined unless the `render` method has been called."""
        return self._fig

    @property
    def ax(self):
        """Return a reference to the Axes instance on which the Weyl chamber
        has been rendered. Undefined unless the `render` method has been
        called."""
        return self._ax

    @property
    def artists(self):
        """Return a list of rendered artists. This includes only artists that
        were created as part of a plotting command, not things like the edges
        of the Weyl chmaber or the perfect entanglers polyhedron"""
        return self._artists

    def _repr_png_(self):
        from IPython.core.pylabtools import print_figure
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        self.render(fig, ax)
        fig_data = print_figure(fig, 'png')
        plt.close(fig)
        return fig_data

    def _repr_svg_(self):
        from IPython.core.pylabtools import print_figure
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        self.render(fig, ax)
        fig_data = print_figure(fig, 'svg').decode('utf-8')
        plt.close(fig)
        return fig_data

    def render(self, ax):
        """Render the Weyl chamber on the given Axes3D object
        """
        self._ax = ax
        self._fig = ax.figure
        self._artists = []
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.patch.set_facecolor(self.facecolor)
        if self.panecolor is not None:
            ax.w_xaxis.set_pane_color(self.panecolor)
            ax.w_yaxis.set_pane_color(self.panecolor)
            ax.w_zaxis.set_pane_color(self.panecolor)
        if self.z_axis_left:
            tmp_planes = ax.zaxis._PLANES
            ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                                tmp_planes[0], tmp_planes[1],
                                tmp_planes[4], tmp_planes[5])
            ax.zaxis.set_rotate_label(False)
            ax.zaxis.label.set_rotation(90)
        ax.grid(self.grid)
        # background lines
        for P1, P2, fg in self.weyl_edges:
            if not fg:
                self._draw_line(ax, P1, P2, zorder=-1,
                               **self.weyl_edge_bg_properties)
        for P1, P2, fg in self.PE_edges:
            if not fg:
                self._draw_line(ax, P1, P2, zorder=-1,
                               **self.PE_edge_bg_properties)
        # scatter plots
        for c1, c2, c3, kwargs in self._scatter:
            self._artists.append(ax.scatter3D(c1, c2, c3, **kwargs))
        pass # plot everything else
        # labels
        for label in self.labels:
            c1, c2, c3 = self.labels[label]
            if self.tex_labels:
                ax.text(c1, c2, c3, "$%s$" % label, **self.label_properties)
            else:
                ax.text(c1, c2, c3, label, **self.label_properties)
        # foreground lines
        for P1, P2, fg in self.weyl_edges:
            if fg:
                self._draw_line(ax, P1, P2, **self.weyl_edge_fg_properties)
        for P1, P2, fg in self.PE_edges:
            if fg:
                self._draw_line(ax, P1, P2, **self.PE_edge_fg_properties)
        ax.set_xlabel(r'$c_1/\pi$')
        ax.set_ylabel(r'$c_2/\pi$')
        ax.set_zlabel(r'$c_3/\pi$')
        if self.full_cube:
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_zlim(0,1)
        else:
            ax.set_xlim(0,1)
            ax.set_ylim(0,0.5)
            ax.set_zlim(0,0.5)
        ax.tick_params(axis='both', which='major',
                       labelsize=self.ticklabelsize)
        # try to fix positioning of tick labels
        ax.xaxis._axinfo['ticklabel']['space_factor'] = 0.5
        ax.yaxis._axinfo['ticklabel']['space_factor'] = 0.5
        ax.xaxis._axinfo['label']['space_factor'] = 1.8
        ax.yaxis._axinfo['label']['space_factor'] = 1.8
        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        [t.set_va('center') for t in ax.get_zticklabels()]
        [t.set_ha('center') for t in ax.get_zticklabels()]


    def plot(self, fig=None, outfile=None):
        """Generate a plot of the Weyl chamber on the given figure, or create a
        new figure if fig argument is not given. If `outfile` is given, write
        the reulting plot to the file.
        """
        pyplot_fig = False
        if fig is None:
            pyplot_fig = True
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        w = self.fig_width - (self.left_margin + self.right_margin)
        h = self.fig_height - (self.bottom_margin + self.top_margin)
        pos = [self.left_margin/self.fig_width,
               self.bottom_margin/self.fig_height,
               w/self.fig_width, h/self.fig_height]
        ax = fig.add_axes(pos, projection='3d')
        self.render(ax)
        if outfile is not None:
            fig.savefig(outfile)
            if pyplot_fig:
                plt.close(fig)

    def scatter(self, c1, c2, c3, **kwargs):
        """Add a scatter plot to the Weyl chamber. All keyword arguments will
        be passed to matplotlib scatter3D function"""
        self._scatter.append((c1, c2, c3, kwargs))

    def add_point(self, c1, c2, c3, scatter_index=0, **kwargs):
        """Add a point to a scatter plot with the given scatter_index. If there
        is no existing scatter plot with that index, a new one will be created.
        The arguments of the scatter plot are updated with the given kwargs."""
        try:
            c1s, c2s, c3s, kw = self._scatter[scatter_index]
            kw.update(kwargs)
            self._scatter[scatter_index] = (np.append(c1s, [c1, ]),
                                            np.append(c2s, [c2, ]),
                                            np.append(c3s, [c3, ]),
                                            kw)
        except IndexError:
            self._scatter.append((np.array([c1, ]), np.array([c2, ]),
                                  np.array([c3, ]), kwargs))

    def add_gate(self, U, scatter_index=0, **kwargs):
        """Call the add_point method for the Weyl chamber coordinates of the
        given gate."""
        self.add_point(*U.weyl_coordinates(), scatter_index=scatter_index,
                       **kwargs)

    def _draw_line(self, ax, origin, end, **kwargs):
        """Draw a line from origin to end onto the given axis. Both origin and
        end must either be 3-tuples, or names of weyl_points. All keyword
        arguments are passed to the `ax.plot` method
        """
        try:
            if origin in self.weyl_points:
                o1, o2, o3 = self.weyl_points[origin]
            else:
                o1, o2, o3 = origin
        except ValueError:
            raise ValueError("origin '%s' is not in weyl_points "
                             "or a list (c1, c2, c3)" % origin)
        try:
            if end in self.weyl_points:
                c1, c2, c3 = self.weyl_points[end]
            else:
                c1, c2, c3 = end
        except ValueError:
            raise ValueError("origin '%s' is not in weyl_points "
                             "or a list (c1, c2, c3)" % origin)
        ax.plot([o1, c1], [o2, c2] , [o3, c3], **kwargs)

    def _draw_arrow(self, ax, origin, end, **kwargs):
        try:
            if origin in self.weyl_points:
                o1, o2, o3 = self.weyl_points[origin]
            else:
                o1, o2, o3 = origin
        except ValueError:
            raise ValueError("origin '%s' is not in weyl_points "
                             "or a list (c1, c2, c3)" % origin)
        try:
            if end in self.weyl_points:
                c1, c2, c3 = self.weyl_points[end]
            else:
                c1, c2, c3 = end
        except ValueError:
            raise ValueError("origin '%s' is not in weyl_points "
                             "or a list (c1, c2, c3)" % origin)
        a = Arrow3D([o1, c1], [o2, c2] , [o3, c3], mutation_scale=10, lw=1.5,
                    arrowstyle="-|>", **kwargs)
        ax.add_artist(a)

    def show(self):
        """
        Show a plot of the Weyl chamber
        """
        self.plot()
        plt.show()



def g1g2g3(U, ndigits=8):
    """
    Given Numpy matrix U, calculate local invariants g1,g2,g3
    U must be in the canonical basis. For numerical stability, the resulting
    values are rounded to the given precision, cf. the `ndigits` parameter of
    the built-in `round` function.

    >>> from . gate2q import CNOT
    >>> print("%.2f %.2f %.2f" % g1g2g3(CNOT))
    0.00 0.00 1.00
    """
    # mathematically, the determinant of U and to_magic(U) is the same, but
    # we seem to get better numerical accuracy if we calculate detU with
    # the rotated U
    detU = np.linalg.det(to_magic(U))
    m = to_magic(U).T * to_magic(U)
    g1_2 = (np.trace(m))**2 / (16.0 * detU)
    g3   = (np.trace(m)**2 - np.trace(m*m)) / ( 4.0 * detU)
    g1 = round(g1_2.real, ndigits)
    g2 = round(g1_2.imag, ndigits)
    g3 = round(g3.real, ndigits)
    return (g1, g2, g3)


def c1c2c3(U, ndigits=8):
    """
    Given U (canonical basis), calculate the Weyl Chamber coordinates
    c1,c2,c3.

    In order to facility numerical stability, the resulting coordinates are
    rounded to the given precision (cf. `ndigits` parameter of the built-in
    `round` function). Otherwise, rounding errors would likely to result in
    points that are not in the Weyl chamber, e.g. (0.1, 0.0, 1.0e-13)

    Algorithm from Childs et al., PRA 68, 052311 (2003).

    >>> from . gate2q import CNOT
    >>> print("%.2f %.2f %.2f" % c1c2c3(CNOT))
    0.50 0.00 0.00
    """
    U_tilde = SySy * U.transpose() * SySy
    ev = np.linalg.eigvals((U * U_tilde)/np.sqrt(complex(np.linalg.det(U))))
    two_S = np.angle(ev) / np.pi
    for i in range(len(two_S)):
        if two_S[i] <= -0.5: two_S[i] += 2.0
    S = np.sort(two_S / 2.0)[::-1] # sort decreasing
    n = int(round(sum(S)))
    S -= np.r_[np.ones(n), np.zeros(4-n)]
    S = np.roll(S, -n)
    M = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    c1, c2, c3 = np.dot(M, S[:3])
    if c3 < 0:
        c1 = 1 - c1
        c3 = -c3
    return (round(c1, ndigits), round(c2, ndigits), round(c3, ndigits))


def g1g2g3_from_c1c2c3(c1, c2, c3):
    """
    Calculate the local invariants from the Weyl-chamber coordinates (c1, c2,
                pass
    c3, in units of pi)

    >>> from . gate2q import CNOT
    >>> print("%.2f %.2f %.2f" % g1g2g3_from_c1c2c3(*c1c2c3(CNOT)))
    0.00 0.00 1.00
    """
    c1 *= pi
    c2 *= pi
    c3 *= pi
    g1 = cos(c1)**2 * cos(c2)**2 * cos(c3)**2 \
       - sin(c1)**2 * sin(c2)**2 * sin(c3)**2     + 0.0
    g2 = 0.25 * sin(2*c1) * sin(2*c2) * sin(2*c3) + 0.0
    g3 = 4*g1 - cos(2*c1) * cos(2*c2) * cos(2*c3) + 0.0
    return g1, g2, g3


def point_in_weyl_chamber(c1, c2, c3, raise_exception=False):
    """
    Return True if the coordinates c1, c2, c3 are inside the Weyl chamber

    >>> from . gate2q import BGATE, identity
    >>> point_in_weyl_chamber(*c1c2c3(BGATE))
    True
    >>> point_in_weyl_chamber(*c1c2c3(identity))
    True

    The coordinates may also be array-like, in which case a boolean numpy array
    is returned.

    >>> point_in_weyl_chamber([0.0,0.5,0.8], [1.0,0.25,0.0], [1.0,0.25,0.0])
    array([False,  True,  True], dtype=bool)

    If raise_exception is True, raise an ValueError if any values are outside
    the Weyl chamber.

    >>> try:
    ...     point_in_weyl_chamber(1.0, 0.5, 0, raise_exception=True)
    ... except ValueError as e:
    ...     print(e)
    (1, 0.5, 0) is not in the Weyl chamber
    """
    result = Or(
        And(And(
          less(c1, 0.5), less_eq(c2, c1)), less_eq(c3, c2)),
        And(And(
          greater_eq(c1, 0.5), less_eq(c2, 1.0-np.array(c1))), less_eq(c3, c2))
    )
    if raise_exception:
        if not np.all(result):
            if np.isscalar(c1): # assume c2, c3 are scalar, too
                raise ValueError("(%g, %g, %g) is not in the Weyl chamber"
                                % (c1, c2, c3))
            else:
                raise ValueError("Not all values (c1, c2, c3) are in the "
                                    "Weyl chamber")
    return result


def point_in_PE(c1, c2, c3, check_weyl=False):
    """
    Return True if the coordinates c1, c2, c3 are inside the perfect-entangler
    polyhedron

    >>> from QDYN.gate2q import BGATE
    >>> point_in_PE(*c1c2c3(BGATE))
    True
    >>> from QDYN.gate2q import identity
    >>> point_in_PE(*c1c2c3(identity))
    False

    >>> point_in_PE([0.0, 0.5, 0.8], [1.0, 0.25, 0.0], [1.0, 0.25, 0.0])
    array([False,  True, False], dtype=bool)
    """
    in_weyl = point_in_weyl_chamber(c1, c2, c3, raise_exception=check_weyl)
    c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3)
    return And(in_weyl,
        And(And(
            greater_eq(c1+c2, 0.5), less_eq(c1-c2, 0.5)), less_eq(c2+c3, 0.5)
        )
    )


def point_in_region(region, c1, c2, c3, check_weyl=False):
    """Return True if the coordinates c1, c2, c3 are inside the given region
    of the Weyl chamber. The regions are 'W0' (between origin O and perfect
    entangerls polyhedron), 'W0*' (between point A1 and perfect entangler
    polyhedron), 'W1' (between A3 point and perfect entanglers polyhedron), and
    'PE' (inside perfect entanglers polyhedron)

    If the check_weyl parameter is given a True, raise a ValueError for any
    points outside of the Weyl chamber

    >>> point_in_region('W0', *WeylChamber.O)
    True
    >>> point_in_region('W0', 0.2, 0.05, 0.0)
    True
    >>> point_in_region('W0', *WeylChamber.L)
    False
    >>> point_in_region('W0', *WeylChamber.Q)
    False
    >>> point_in_region('PE', *WeylChamber.Q)
    True
    >>> point_in_region('W0*', *WeylChamber.A1)
    True
    >>> point_in_region('W0*', 0.8, 0.1, 0.1)
    True
    >>> point_in_region('W1', *WeylChamber.A3)
    True
    >>> point_in_region('W1', 0.5, 0.4, 0.25)
    True
    >>> point_in_region('W1', 0.5, 0.25, 0)
    False
    >>> point_in_region('PE', 0.5, 0.25, 0)
    True

    The function may be also applied against arrays:
    >>> point_in_region('W1', [0.5,0.5], [0.4,0.25], [0.25,0.0])
    array([ True, False], dtype=bool)
    """
    regions = ['W0', 'W0*', 'W1', 'PE']
    if region == 'PE':
        return point_in_PE(c1, c2, c3, check_weyl=check_weyl)
    else:
        in_weyl = point_in_weyl_chamber(c1, c2, c3, raise_exception=check_weyl)
        c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3)
        if region == 'W0':
            return And(in_weyl, less(c1+c2, 0.5))
        elif region == 'W0*':
            return And(in_weyl, greater(c1-c2, 0.5))
        elif region == 'W1':
            return And(in_weyl, greater(c2+c3, 0.5))
        else:
            raise ValueError("region %s is not in %s"%(region, regions))


def get_region(c1, c2, c3):
    """Return the region of the Weyl chamber ('W0', 'W0*', 'W1', 'PE') the the
    given point is in.

    >>> print(get_region(*WeylChamber.O))
    W0
    >>> print(get_region(*WeylChamber.A1))
    W0*
    >>> print(get_region(*WeylChamber.A3))
    W1
    >>> print(get_region(*WeylChamber.L))
    PE
    >>> print(get_region(0.2, 0.05, 0.0))
    W0
    >>> print(get_region(0.8, 0.1, 0.1))
    W0*
    >>> print(get_region(0.5, 0.25, 0))
    PE
    >>> print(get_region(0.5, 0.4, 0.25))
    W1
    >>> try:
    ...     get_region(1.0, 0.5, 0)
    ... except ValueError as e:
    ...     print(e)
    (1, 0.5, 0) is not in the Weyl chamber

    Only scalar values are accepted for c1, c2, c3
    """
    point_in_weyl_chamber(c1, c2, c3, raise_exception=True)
    if c1+c2 < 0.5:
        return 'W0'
    elif c1-c2 > 0.5:
        return 'W0*'
    elif c2+c3 > 0.5:
        return 'W1'
    else:
        return 'PE'


def project_to_PE(c1, c2, c3):
    """Return new tuple (c1', c2', c3') obtained by projecting the given input
    point (c1, c2, c3) onto the closest boundary of the perfect entanglers
    polyhedron. If the input point already is a perfect entangler, it will be
    returned unchanged

    >>> print("%s, %s, %s" % tuple(project_to_PE(*WeylChamber.A3)))
    0.5, 0.25, 0.25
    >>> print("%.3f, %.3f, %.3f" % tuple(project_to_PE(0.5, 0.5, 0.25)))
    0.500, 0.375, 0.125
    >>> print("%.3f, %.3f, %.3f" % tuple(project_to_PE(0.25, 0, 0)))
    0.375, 0.125, 0.000
    >>> print("%.3f, %.3f, %.3f" % tuple(project_to_PE(0.75, 0, 0)))
    0.625, 0.125, 0.000
    >>> print("%.3f, %.3f, %.3f" % tuple(project_to_PE(0.3125, 0.0625, 0.01)))
    0.375, 0.125, 0.010
    >>> print("%.1f, %.1f, %.1f" % tuple(project_to_PE(0.5, 0, 0)))
    0.5, 0.0, 0.0
    >>> print("%.1f, %.1f, %.1f" % tuple(project_to_PE(0.5, 0.2, 0.2)))
    0.5, 0.2, 0.2
    >>> try:
    ...     project_to_PE(1.0, 0.5, 0)
    ... except ValueError as e:
    ...     print(e)
    (1, 0.5, 0) is not in the Weyl chamber
    """
    if point_in_PE(c1, c2, c3):
        return c1, c2, c3
    else:
        region = get_region(c1, c2, c3)
        p = np.array((c1, c2, c3))
        n = WeylChamber.normal[region]
        a = WeylChamber.anchor[region]
        return p - np.inner((p-a), n) * n


def concurrence(c1, c2, c3):
    """
    Calculate the concurrence directly from the Weyl Chamber coordinates c1,
    c2, c3

    >>> from . gate2q import SWAP, CNOT, identity
    >>> round(concurrence(*c1c2c3(SWAP)), 2)
    0.0
    >>> round(concurrence(*c1c2c3(CNOT)), 2)
    1.0
    >>> round(concurrence(*c1c2c3(identity)), 2)
    0.0
    """
    if ((c1 + c2) >= 0.5) and (c1-c2 <= 0.5) and ((c2+c3) <= 0.5):
        # if we're inside the perfect-entangler polyhedron in the Weyl chamber
        # the concurrence is 1 by definition. the "regular" formula gives wrong
        # results in this case.
        return 1.0
    else:
        c1_c2_c3 = np.array([c1, c2, c3])
        c3_c1_c2 = np.roll(c1_c2_c3, 1)
        m = np.concatenate((c1_c2_c3 - c3_c1_c2, c1_c2_c3 + c3_c1_c2))
        return np.max(abs(sin(pi * m)))


def to_magic(A):
    """ Convert a matrix A that is represented in the canonical basis to a
        representation in the Bell basis
    """
    return Qmagic.conj().T * A * Qmagic


def from_magic(A):
    """ The opposite of to_magic """
    return Qmagic * A * Qmagic.conj().T


def J_T_LI(O, U, form='g'):
    """
    Given Numpy matrices O (optimal gate), U (obtained gate), calculate the
    value of the Local invariants-functional
    """
    if form == 'g':
        return np.sum(np.abs(np.array(g1g2g3(O)) - np.array(g1g2g3(U)))**2)
    elif form=='c':
        delta_c = np.array(c1c2c3(O)) - np.array(c1c2c3(U))
        return np.prod(cos(np.pi * (delta_c) / 2.0))
    else:
        raise ValueError("Illegal value for 'form'")


def F_PE(g1, g2, g3):
    """
    Evaluate the Perfect-Entangler Functional

    >>> from . gate2q import CNOT
    >>> F_PE(*g1g2g3(CNOT))
    0.0
    >>> from . gate2q import identity
    >>> F_PE(*g1g2g3(identity))
    2.0
    """
    return g3 * np.sqrt(g1**2 + g2**2) - g1 + 0.0


def canonical_gate(c1,c2,c3):
    """Return the canonical two-qubit gate characterized by the given Weyl
    chamber coordinates (in units of pi)
    >>> print(canonical_gate(0.5,0,0))
    [[0.707107+0.000000j, 0.000000+0.000000j, 0.000000+0.000000j, 0.000000+0.707107j],
     [0.000000+0.000000j, 0.707107+0.000000j, 0.000000+0.707107j, 0.000000+0.000000j],
     [0.000000+0.000000j, 0.000000+0.707107j, 0.707107+0.000000j, 0.000000+0.000000j],
     [0.000000+0.707107j, 0.000000+0.000000j, 0.000000+0.000000j, 0.707107+0.000000j]]
    >>> U = canonical_gate(0.5,0,0)
    >>> diff = np.array(U.weyl_coordinates()) - np.array([0.5, 0, 0])
    >>> print("%.1f" % np.max(np.abs(diff)))
    0.0
    """
    return Gate2Q(expm(pi*0.5j * (c1*SxSx +c2*SySy + c3*SzSz)))


def cartan_decomposition(U):
    """
    Calculate the Cartan Decomposition of the given U in U(4)

    U = k1 * A * k2

    up to a global phase

    Parameters
    ----------

    U : numpy matrix
        Two-qubit quantum gate. Must be unitary

    Returns
    -------

    k1 : numpy matrix
       left local operations in SU(2) x SU(2)
    A  : numpy matrix
        non-local operations, in SU(4)
    k2 : numpy matrix
       right local operations in SU(2) x SU(2)

    Notes
    -----

    If you are working with a logical subspace, you should unitarize U before
    calculating the Cartan decomposition

    References
    ----------

    * D. Reich. Optimising the nonlocal content of a two-qubit gate. Diploma
      Thesis. FU Berlin, 2010. Appendix E

    * Zhang et al. PRA 67, 042313 (2003)
    """
    U = np.matrix(U)                    # in U(4)
    Utilde = U / np.linalg.det(U)**0.25 # U in SU(4)

    found_branch = False
    # The fourth root has four branches; the correct solution could be in
    # any one of them
    for branch in xrange(4):

        UB = to_magic(Utilde) # in Bell basis
        m = UB.T * UB

        # The F-matrix can be calculated according to Eq (21) in PRA 67, 042313
        # It is a diagonal matrix containing the squares of the eigenvalues of
        # m
        c1, c2, c3 = c1c2c3(Utilde)
        F1 = exp( pi * 0.5j * ( c1 - c2 + c3) )
        F2 = exp( pi * 0.5j * ( c1 + c2 - c3) )
        F3 = exp( pi * 0.5j * (-c1 - c2 - c3) )
        F4 = exp( pi * 0.5j * (-c1 + c2 + c3) )
        Fd = np.array([F1, F2, F3, F4])
        F  = np.matrix(np.diag(Fd))

        # Furthermore, we have Eq (22), giving the eigen-decomposition of the
        # matrix m. This gives us the matrix O_2.T of the eigenvectors of m
        Fsq, O_2_transposed = np.linalg.eig(m)

        ord1 = np.argsort(np.angle(Fd**2)) # sort according to complex phase
        ord2 = np.argsort(np.angle(Fsq))   # ... (absolute value is 1)
        diff = np.sum( np.abs( (Fd**2)[ord1] - Fsq[ord2] ) )
        # Do Fd**2 and Fsq contain the same values (irrespective of order)?
        if  diff < 1.0e-12:
            found_branch = True
            break
        else:
            Utilde *= 1.0j

    # double check that we managed to find a branch (just to be 100% safe)
    assert(found_branch), \
    "Couldn't find correct branch of fourth root in mapping U(4) -> SU(4)"

    # Getting the entries of F from Eq (21) instead of by taking the square
    # root of Fsq has the benefit that we don't have to worry about whether we
    # need to take the positive or negative root.
    # However, we do need to make sure that the eigenstates are ordered to
    # correspond to F1, F2, F3, F4
    # After reordering, we need to transpose to get O_2 itself
    reordered = np.matrix(np.zeros((4,4)), dtype=np.complex128)
    order = []
    for i in xrange(4):
        for j in xrange(4):
            if (abs(Fd[i]**2 - Fsq[j]) < 1.0e-12):
                if not j in order:
                    order.append(j)
    assert len(order) == 4, "Couldn't order O_2" # should not happen
    # Reorder using the order we just figured out, and transpose
    for i in xrange(4):
        reordered[:,i] = O_2_transposed[:,order[i]]
    O_2 = reordered.T

    # Now that we have O_2 and F, completing the Cartan decomposition is
    # straightforward, following along Appendix E of Daniel's thesis
    k2 = from_magic(O_2)
    O_1 = UB * O_2.T * F.H
    k1 = from_magic(O_1)
    A = canonical_gate(c1, c2, c3)

    # Check our results
    from . gate2q import identity
    assert( np.max(np.abs(O_1*O_1.T - identity)) < 1.0e-12 ), \
    "O_1 not orthogonal"
    assert( np.max(np.abs(O_2*O_2.T - identity)) < 1.0e-12 ), \
    "O_2 not orthogonal"
    assert( np.max(np.abs((k1*A*k2 - Utilde))) < 1.0e-12 ), \
    "Cartan Decomposition Failed"

    return k1, A, k2


def _U2(phi, theta, phi1, phi2):
    r"""Return a unitary gate using the parametrization

        U = e^{i \phi} \begin{bmatrix}
             \cos\theta e^{ i\phi_1}  & \sin\theta e^{ i\phi_2}\\
            -\sin\theta e^{-i\phi_2}  & \cos\theta e^{-i\phi_1}\\
            \end{bmatrix}

    >>> from .gate2q import pop_loss
    >>> gates = [_U2(*(2*np.pi*np.random.random(4))) for i in range(100)]
    >>> np.any([(pop_loss(U) > 1.0e-14) for U in gates])
    False
    """
    return exp(1j*phi) * np.array([
        [  cos(theta) * exp( 1j*phi1), sin(theta) * exp( 1j*phi2) ],
        [ -sin(theta) * exp(-1j*phi2), cos(theta) * exp(-1j*phi1) ]])


def _SQ_unitary(phi_left, theta_left, phi1_left, phi2_left,
    phi_right, theta_right, phi1_right, phi2_right):
    """
    Return a non-entangling two-qubit gate (a two-qubit gate locally equivalent
    to the identity)

    >>> from .gate2q import pop_loss
    >>> gates = [_SQ_unitary(*(2*np.pi*np.random.random(8))) for i in range(100)]
    >>> np.any([(pop_loss(U) > 1.0e-14) for U in gates])
    False
    >>> np.any([(U.concurrence() > 1.0e-14) for U in gates])
    False
    """
    I = np.identity(2)
    return Gate2Q(
           np.kron(_U2(phi_left, theta_left, phi1_left, phi2_left), I).dot(
           np.kron(I, _U2(phi_right, theta_right, phi1_right, phi2_right))))


def closest_LI(U, c1, c2, c3, method='Powell', limit=1.0e-6):
    """Find the closest gate that has the given Weyl chamber coordinates (in
    units of pi)"""
    from .gate2q import _closest_gate
    A = canonical_gate(c1, c2, c3)
    def f_U(p):
        return _SQ_unitary(*p[:8]).dot(A).dot(_SQ_unitary(*p[8:]))
    return _closest_gate(U, f_U, n=16, method=method, limit=limit)


def random_weyl_point(region=None):
    """Return a random point (c1, c2, c3) in the Weyl chamber, in units of pi.
    If region is given in ['W0', 'W0*', 'W1', 'PE'], the point will be in the
    specified region of the Weyl chamber

    >>> c1, c2, c3 = random_weyl_point()
    >>> point_in_weyl_chamber(c1, c2, c3)
    True
    >>> c1, c2, c3 = random_weyl_point(region='PE')
    >>> point_in_region('PE', c1, c2, c3)
    True
    >>> c1, c2, c3 = random_weyl_point(region='W0')
    >>> point_in_region('W0', c1, c2, c3)
    True
    >>> c1, c2, c3 = random_weyl_point(region='W0*')
    >>> point_in_region('W0*', c1, c2, c3)
    True
    >>> c1, c2, c3 = random_weyl_point(region='W1')
    >>> point_in_region('W1', c1, c2, c3)
    True
    """
    while True:
        c1 = np.random.rand()
        c2 = 0.5*np.random.rand()
        c3 = 0.5*np.random.rand()
        if point_in_weyl_chamber(c1, c2, c3):
            if region is None:
                return c1, c2, c3
            else:
                if point_in_region(region, c1, c2, c3):
                    return c1, c2, c3

