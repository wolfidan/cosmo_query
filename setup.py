 #! /usr/bin/env python

# System imports
from distutils.core import setup, Command, Extension
from distutils.command.build_py import build_py as _build_py
from distutils      import sysconfig
from pip import __file__ as pip_loc
import os
import subprocess

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

install_to = os.path.join(os.path.split(os.path.split(pip_loc)[0])[0],
                   'pycosmo', 'templates')

class build_py(_build_py):
    description = "Custom command that concatenates COSMO height files"
    def run(self):
        folder = './cosmo_query/cosmo_info/'
        # Concatenate all partial COSMO files
        for res in ['1','2','7']:
            cmd = 'cat ' + folder + 'cosmo_' + res + '_heights_split_a* > ' \
                + folder + 'cosmo_' + res + '_heights.npz'
            print(cmd)
            subprocess.call(cmd, shell=True)
            _build_py.run(self)
            
# interp1 extension module
_interp1_c = Extension("_interp1_c",
                   ["./cosmo_query/c/interp1_c.i","./cosmo_query/c/interp1_c.c"],
                   include_dirs = [numpy_include],
                   )

_radar_interp_c = Extension("_radar_interp_c",
                  ["./cosmo_query/c/radar_interp_c.i","./cosmo_query/c/radar_interp_c.c"],
                   include_dirs = [numpy_include],
                  )
# ezrange setup
setup(  name        = "cosmo_query",
        cmdclass={'build_py': build_py},
        description = "Tools for retrieving COSMO data from CSCS servers",
        version     = "1.0",
        url='http://gitlab.epfl.ch/wolfensb/cosmo_query/',
        author='Daniel Wolfensberger - LTE EPFL',
        author_email='daniel.wolfensberger@epfl.ch',
        license='GPL-3.0',
        packages=['cosmo_query','cosmo_query/c'],
        package_data   = {'cosmo_query/c' : ['*.o','*.i','*.c'],
			 'cosmo_query':['./cosmo_info/list_vars/*','./cosmo_info/file_logs/*',
                   './cosmo_info/*.npz']},
        data_files = [(install_to, ["LICENSE"])],
        include_package_data=True,
        install_requires=[
          'pyproj',
          'numpy',
          'scipy',
          'netCDF4',
          'paramiko',
          'h5py'
        ],
        zip_safe=False,
        ext_modules = [_interp1_c,_radar_interp_c ]
        )


