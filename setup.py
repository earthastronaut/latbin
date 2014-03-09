from distutils.core import setup
import latbin

packages = ['latbin']
package_data = {'latbin':[]}
requires = ['numpy','scipy']
ext_modules = []

setup(
    name='latbin',
    author="Tim Anderton, Dylan Gregersen",
    # author_email='',
    url="https://github.com/astrodsg/latbin",
    license="3-clause BSD style license",
    # url='',
    description="Python lattice binning package for large data",
    long_description=open("README.md").read(),
    classifiers=["Intended Audience :: Science Researchers",
                 "Topic :: Scientific/Engineering"],
    platforms='any',
    version=latbin.__version__,
    packages=packages,
    package_data=package_data,
    ext_modules=ext_modules,
    requires=requires,
    )