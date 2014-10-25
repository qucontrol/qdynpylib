# The QDYN Python Package

The QDYN Python Package provides a collection of utilties to augment the Fortran
QDYN library for quantum dynamics and control, developed in-house in the 
[Koch group at the University of Kassel][AGKOCH]

The package will read and write some of the files generated
by the Fortran QDYN routines, and provides additional tools (such as signal
processing for pulses) that are easier to implement in Python and that will
greatly benefit from (i)Python's interactiveness.


## Prerequisites ##

The QDYN package depends on the [Python scientific stack](Scipy)
(numpy/scipy/matplotlib) in a recent version. It is recommended to use one of
the standalone scientific Python distributions like [Enthought Canopy][EPD] 
or [Anaconda][Anaconda]. You might want to use a [virtual environment][VE} as
well.

The QDYN library is written for Python 2.7. It will not run on Python 3.x


## Installation ##

Assuming that you have either a scientific Python distribution installed in your
home directory (recommended), or created and activated a virtual environment,
you can install the latest official release of the QDYN package with

    pip install QDYN

Alternatively, from a checkout of the source code, you can run

    python setup.py install

to install the most current revision. The same command is also executed by

    make install

In order to run the automated tests, run

    make test

## Usage ##

Load the package with `import QDYN` in your python script.

There are a lof parts that can be used interactively in
[IPython](IPython) (either the shell or the notebook interface).
For example:

    >>> import QDYN
    >>> from QDYN.pulse import Pulse
    >>> p = Pulse('pulse.dat')
    >>> p.amplitude *= QDYN.pulse.flattop(p.tgrid, p.t0(), p.T(), t_rise=5)
    >>> p.show()
    >>> p.resample(upsample=2)
    >>> p.write('pulse_doublesampled.dat')


[AGKOCH]: http://www.uni-kassel.de/fb10/en/institutes/physics/research-groups/quantum-dynamics-and-control/homepage.html
[VE]: http://bitbucket.org/ianb/virtualenv/raw/tip/virtualenv.py
[EPD]: https://www.enthought.com/products/canopy/
[Scipy]: http://www.scipy.org
[IPython]: http://ipython.org
[Anaconda]: https://store.continuum.io/cshop/anaconda/
