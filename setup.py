#!/usr/bin/env python
from distutils.core import setup
from QDYN import __version__
from setuptools.command.install import install
from setuptools.command.sdist import sdist


def write_git_info(command_subclass):
    """
    A decorator for classes subclassing one of the setuptools commands.

    It modifies the run() method so that it generates a file QDYN/__git__.py
    containing the git revision number
    """
    orig_run = command_subclass.run

    def modified_run(self):
        import subprocess
        try:
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
            with open("QDYN/__git__.py", 'w') as f:
                print >> f, "__revision__ = '%s'" % sha
        except subprocess.CalledProcessError:
            pass
        orig_run(self)

    command_subclass.run = modified_run
    return command_subclass


@write_git_info
class custom_install(install):
    pass


@write_git_info
class custom_sdist(sdist):
    pass


setup(name='QDYN',
      version=__version__,
      description='Package providing some Python modules for working with QDYN',
      author='Michael Goerz',
      author_email='goerz@physik.uni-kassel.de',
      license='GPL',
      packages=['QDYN', 'QDYN.prop'],
      scripts=[],
      cmdclass={'install': custom_install, 'sdist': custom_sdist},
     )
