from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import QDYN
import numpy as np
from numpy import sqrt
from matplotlib.figure import Figure as figure


def compare_images(expected, actual, tol=1.0e-3):
    """Raise an AssertionError if the images are not the same"""
    from matplotlib.testing.compare import compare_images as mpl_compare_images
    try:
        result = mpl_compare_images(expected, actual, tol=tol)
    except Exception as e:
        raise AssertionError(e)
    assert result is None, result


def test_bloch_plot():
    bloch = QDYN.bloch.Bloch()
    bloch.wireframe=False
    pnt = QDYN.bloch.bloch_coordinates(np.array((3,1+1j))/sqrt(11.0))
    bloch.add_precession(pnt, [1,0,0])
    bloch.add_precession(pnt, [0,-1,0])
    bloch.add_precession(pnt, [0,0,1])
    bloch.add_points(pnt)
    bloch.add_vectors(pnt)
    bloch.sphere_alpha=0.3
    bloch.set_label_convention("xyz01")
    bloch.precession_color = ['orange', 'blue','red']
    bloch.vector_color = ['black']
    bloch.precession_width = [2.0]
    bloch.xyz_axes = 2
    bloch.view = [-68, 16]

    fig = figure()
    bloch.plot(fig)

    assert len(fig.axes) == 1, "Expected 1 axes instance in figure"

    ax = fig.axes[0]
    assert len(ax.get_children()) == 34, "Plot should have 34 children"

    for val in ax.get_w_lims():
        diff = abs(abs(val) - 0.7)
        if diff > 1.0e-12:
            raise AssertionError("3D world limits should all be 0.7, diff: %s"
                                 % diff)
    bloch.plot(outfile='tests/result_images/bloch_test.png')
    compare_images(expected='tests/baseline_images/bloch_test.png',
                   actual='tests/result_images/bloch_test.png', tol=1.0)

