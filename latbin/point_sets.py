# Licensed under a 3-clause BSD style license - see LICENSE

# Standard Library
from __future__ import print_function, division

# 3rd Party
import numpy as np
#vector quantization
import scipy.cluster.vq as vq

# Internal

__all__ = ["PointSet","Lattice"]

# ########################################################################### #

class PointSet (object):
    
    def __init__ (self, point_coordinates, k=None):
        """
        Paramters
        ---------
        point_coordinates : ndarray, size=(n_points , n_dims) 
        
        """
        self.points = np.asarray(point_coordinates)
    
    @property
    def ndim (self):
        return self.points.shape[1]
    
    def coordinates(self, point_index):
        return self.points[point_index]
    
    def quantize (self,points, return_distortion=False):
        """
        Takes points and returns point set representation
        
        Parameters
        ----------
        points : ndarray, size=(n_points , n_dims)
            array of points to quantize
            
        Returns
        -------
        reps : list, length M
            a list of representations for each point in pts
            
        """
        vqres = vq.vq(np.asarray(points), self.points)
        if return_distortion:
            return vqres
        else:
            return vqres[0]
    
    def cannonize (self,reps):
        """
        Takes representations and return unique representation for each point
        
        """
        return np.asarray(reps, dtype=int)

class Lattice (PointSet):
    
    def __init__ (self, packing_radius, offset=None, scale=None, family="A"):
        pass

    def quantize (self,points):
        # TODO: following Algorithim 3 on p445
        pts = np.asarray(points)
        s = np.sum(pts,axis=1)
        pts -= s/(self.ndim+1)
        deficiency = np.sum(pts,axis=0)
        
    quantize.__doc__ = PointSet.quantize.__doc__
    
