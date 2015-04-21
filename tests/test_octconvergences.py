from __future__ import print_function, division, absolute_import, \
                       unicode_literals
from os.path import join
import QDYN

def test_oct_convergences_show_bokeh():
    keys = ['octJhol', 'octJsm', 'postJhol', 'postJsm']
    root = join('tests', 'oct_iters')
    c = QDYN.octutils.OCTConvergences(keys=keys,
                    files=[join(root, '%s.dat' % k) for k in keys],
                    colors=['black', '#377eb8', '#ff7f00', 'red'])
    c.show_bokeh(outfile='test_octconvergences.html')
