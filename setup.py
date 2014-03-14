from distutils.core import setup
import latbin

packages = ['latbin']
requires = ['numpy','scipy']
ext_modules = []

setup(
    name='latbin',
    author="Tim Anderton, Dylan Gregersen",
    author_email='<quidditymaster@gmail.com>;<dylan.gregersen@utah.edu>',
    url="https://github.com/astrodsg/latbin",
    license="3-clause BSD style license",
    description="Python lattice binning package for large data",
    long_description=open("README.rst").read(),
    classifiers=["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: BSD License",
                 "Natural Language :: English",
                 "Programming Language :: Python",
                 "Topic :: Scientific/Engineering :: Mathematics",
                 "Topic :: Scientific/Engineering :: Physics"],
    platforms='any',
    version=latbin.__version__,
    packages=packages,
    ext_modules=ext_modules,
    requires=requires,
    )
