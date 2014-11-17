#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For handling the Python 2 and Python 3 support
AUTHOR: Dylan Gregersen
DATE: Sun Nov 16 12:23:36 2014
"""
# ########################################################################### #

# import modules 
import sys 
PY3 = sys.version[0] == "3"

# ########################################################################### #
if PY3: # ----------------------- python 3 imports
    import pickle

    import builtins
    range = builtins.range 
    
else: # ----------------------- python 2 imports    
    import cPickle as pickle

    # modify builtins to generator versions
    range = xrange 
    import itertools
    zip = itertools.izip
    map = itertools.imap 

# ########################################################################### #

def iteritems (dict_obj):
    if PY3:
        return dict_obj.items()
    else:
        return dict_obj.iteritems()

def isstr (obj):
    if PY3:
        return isinstance(obj,str)
    else:
        return isinstance(obj,basestring)

