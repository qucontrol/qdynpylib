# The QDYN Python Package

[![Build Status](https://travis-ci.org/goerz/qdynpylib.svg?branch=master)](https://travis-ci.org/goerz/qdynpylib)
[![Documentation Status](https://readthedocs.org/projects/qdynpylib/badge/?version=latest)](https://qdynpylib.readthedocs.org/en/latest/?badge=latest)

The QDYN Python Package provides a collection of utilities to augment the
[QDYN Fortran library][QDYN] for quantum dynamics and control, developed in the
[Koch group at the University of Kassel][AGKOCH]

## Installation ##

Assuming that you have either a scientific Python distribution installed in your
home directory (recommended), or created and activated a virtual environment,
you can install the latest official release of the QDYN package with

    pip install QDYN

Lastly, to install the latest master from the github repository:

    pip install -I git+https://github.com/goerz/qdynpylib.git#egg=QDYN

To uninstall, run

    pip uninstall QDYN

Note that a "manual" installation via `python setup.py install` from a checkout
of the source code is *not* recommended, as it provides no possibility for an
automatic uninstall. When installing from a local checkout, you may use
`make install` or `make develop`.

It is strongly recommended to use the [Anaconda][] Python distribution.

## Development ##

Development of the QDYN package follows the [git-flow][] branching model with
the default settings. After cloning the repository, you must run

    git checkout master
    git checkout develop
    git flow init -d

The first two commands are essential for ensuring that both the `master` and
`develop` branch exist and are tracking their respective branch in `origin`.
Otherwise, `git flow init` will fail, or produce incorrectly set up branches.
All pull requests must be against the `develop` branch.

## Tests ##

From a checkout of the library source code, execute

    make test

to run the automated tests.


[git-flow]: https://github.com/nvie/gitflow#git-flow
[AGKOCH]: http://www.uni-kassel.de/fb10/en/institutes/physics/research-groups/quantum-dynamics-and-control/homepage.html
[QDYN]: https://www.qdyn-library.net/
[Anaconda]: https://store.continuum.io/cshop/anaconda/
