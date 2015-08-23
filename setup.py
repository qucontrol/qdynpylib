#!/usr/bin/env python
from setuptools import setup


def get_version(filename):
    with open(filename) as in_fh:
        for line in in_fh:
            if line.startswith('__version__'):
                return line.split('=')[1].strip()[1:-1]
    raise ValueError("Cannot extract version from %s" % filename)


setup(name='QDYN',
      version=get_version("QDYN/__init__.py"),
      description='Package providing some Python modules for working with ' \
                  'the QDYN Fortran library',
      author='Michael Goerz',
      author_email='goerz@physik.uni-kassel.de',
      url='https://github.com/goerz/qdynpylib',
      license='GPL',
      install_requires=[
          'numpy>=1.9',
          'matplotlib>=1.3',
          'scipy>=0.15',
          'sympy>=0.7',
          'bokeh>=0.8',
          'click>=5.0',
      ],
      packages=['QDYN', 'QDYN.prop'],
      scripts=[],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ]
     )
