# Licensed under a 3-clause BSD style license - see LICENSE

# Standard Library
from __future__ import print_function, division

# 3rd Party
import numpy as np

# Internal

__all__ = ["PointSet","Lattice"]

# ########################################################################### #

class PointSet (object):
    
    def __init__ (self,lattice_points):
        """
        Paramters
        ---------
        lattice_points : ndarray, size=(U,V) 
            where U is the number of points in V dimensional space
        
        """
        self.lattice_points = np.asarray(lattice_points)
    
    @property
    def ndim (self):
        return self.lattice_points.ndim
        
    def quantize (self,points):
        """
        Takes points and returns point set representation
        
        Parameters
        ----------
        points : ndarray, size=(M,N)
            array of points to quantize M long by N dimensions
            
        Returns
        -------
        reps : list, length M
            a list of representations for each point in pts
            
        """
        pts = np.asarray(points)    
    
    def cannonize (self,reps):
        """
        Takes representations and return unique representation for each point
        
        """
        pass

class Lattice (PointSet):
    
    def __init__ (self,lattice_points):
        PointSet.__init__(self,lattice_points)

    def quantize (self,points):
        # TODO: following Algorithim 3 on p445
        pts = np.asarray(points)
        s = np.sum(pts,axis=1)
        pts -= s/(self.ndim+1)
        deficiency = np.sum(pts,axis=0)
        
    quantize.__doc__ = PointSet.quantize.__doc__
    