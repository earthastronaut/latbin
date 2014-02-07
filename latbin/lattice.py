# Licensed under a 3-clause BSD style license - see LICENSE

# Standard Library
from __future__ import print_function, division
from collections import Iterable

# 3rd Party
import numpy as np
#vector quantization

# Internal
from .point_sets import PointSet, PointInformation

__all__ = ["Lattice","ZLattice","generate_lattice"]

# ########################################################################### #

class Lattice (PointSet):
    
    def __init__(self, packing_radius, ndim, origin, scale, rotation):
        self.packing_radius = float(packing_radius)
        
        if ndim <= 0:
            raise ValueError("ndim must be > 0")
        self.ndim = ndim
        
        if origin is None:
            origin = np.zeros(ndim,dtype=float)
        self.origin = np.asarray(origin,dtype=float)
        if len(self.origin) != ndim or self.origin.ndim != 1:
            raise ValueError("origin must be a float vector of length={}".format(ndim))
        
        if scale is None:
            scale = np.ones(ndim,dtype=float)
        if isinstance(scale,Iterable):
            self.scale = np.asarray(scale,dtype=float)
        else:
            self.scale = np.ones(ndim,dtype=float)*scale
        if len(self.scale) != ndim or self.scale.ndim != 1:
            raise ValueError("scale must be a float or a float vector of length={}".format(ndim))

        # TODO: finish rotation
        if rotation is None:
            rotation = np.eye(ndim) 
        self.rotation = np.asarray(rotation)
     
    def to_data_coords(self, lattice_coords):
        """
        transforms from the internal lattice coordinates to the original 
        data coordinates.
        
        lattice_coords: ndarray shape = (n_points, n_dims)  or (n_dims,)
        """
        lattice_coords = np.asarray(lattice_coords)
        
        if lattice_coords.shape[-1] != self.ndim:
            raise ValueError("lattice_coords must be ndim={}".format(self.ndim))
        
        if lattice_coords.ndim == 1:
            lc = lattice_coords.reshape((1,)+lattice_coords.shape)
            data_coords = self.scale*lc+self.origin
        else:
            data_coords = self.scale*lattice_coords+self.origin
        
        return data_coords        
     
    def to_lattice_coords (self,data_coords):
        """
        transforms from the internal lattice coordinates to the original 
        data coordinates.
        
        data_coords: ndarray shape = (n_points, n_dims)  or (n_dims,)
        
        """
        data_coords = np.asarray(data_coords)
        
        if data_coords.shape[-1] != self.ndim:
            raise ValueError("lattice_coords must be ndim={}".format(self.ndim))
        
        if data_coords.ndim == 1:
            dc = data_coords.reshape((1,)+data_coords.shape)
            lattice_coords = (dc-self.origin)/self.scale
        else:
            lattice_coords = (data_coords-self.origin)/self.scale
        
        return lattice_coords        
        
class ZLattice (Lattice):
    
    def __init__ (self, packing_radius, ndim, origin=None, scale=None, rotation=None):
        Lattice.__init__(self, packing_radius, ndim, origin, scale, rotation)

    def coordinates(self,point_index):
        return self.to_data_coords(point_index)
    
    def quantize (self,points,return_distortion=False):
        """
        Parameters
        ----------
        data_points : ndarray
        return_distortion : boolean
        
        Return
        ------
        
        """
        return Lattice.quantize(self, points, return_distortion)


def histogram(points, bins):
    """histogram a set of points onto a pointset
    points: ndarray
    bins: a PointSet or Lattice instance
    """
    keys = bins.quantize(points)
    pi = PointInformation()
    for i, key in enumerate(keys):
        pi[key] = points[i]
    return pi

  
def generate_lattice (packing_radius, ndim, origin=None, scale=None, family="z"):
    family = family.lower()
    if family == 'z':
        return ZLattice(packing_radius, ndim, origin, scale)
    # TODO: add for all families
    pass