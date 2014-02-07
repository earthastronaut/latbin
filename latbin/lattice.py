# Licensed under a 3-clause BSD style license - see LICENSE

# Standard Library
from __future__ import print_function, division
from collections import Iterable

# 3rd Party
import numpy as np
#vector quantization

# Internal
from .point_sets import PointSet, PointInformation

__all__ = ["Lattice","ZLattice",#"DLattice","ALattice","ELattice",
           "generate_lattice"]

# ########################################################################### #

class Lattice (PointSet):
    """
    The basic Lattice class. To have a specific lattice please use the subclass
    
    """
    def __init__(self, ndim, origin, scale, rotation, packing_radius):
        
        
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

        if packing_radius != 1.0:
            self.scale /= packing_radius 

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
    
    def __init__ (self, ndim, origin=None, scale=None, rotation=None, packing_radius=1.0):
        Lattice.__init__(self, ndim, origin, scale, rotation, packing_radius)

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

class DLattice (Lattice):

    def __init__ (self, ndim, origin=None, scale=None, rotation=None, packing_radius=1.0):
        Lattice.__init__(self, ndim, origin, scale, rotation, packing_radius)
       
class ALattice (Lattice):

    def __init__ (self, ndim, origin=None, scale=None, rotation=None, packing_radius=1.0):
        Lattice.__init__(self, ndim, origin, scale, rotation, packing_radius)

class ELattice (Lattice):

    def __init__ (self, ndim, origin=None, scale=None, rotation=None, packing_radius=1.0):
        Lattice.__init__(self, ndim, origin, scale, rotation, packing_radius)
       
families = {'z':ZLattice,
            }
# TODO: add class to families when known

# ########################################################################### #

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

 
# ########################################################################### #
 
def generate_lattice (ndim, origin=None, scale=None, family="z", packing_radius=1.0):
    rotation = None
    family = family.lower()
    if not families.has_key(family):
        raise ValueError("family must be in ({}), see NOTE1 in doc string".format(", ".join(families.keys())))
    latclass = families[family] 
    return latclass(ndim, origin, scale, rotation, packing_radius) #TODO: this only works if all the arguments for every lattice class is the same. is it?

generate_lattice.__doc__ =     """
    Function for getting a lattice object based on input parameters
    
    Parameters
    ----------
    ndim : integer
        Number of dimensions
    origin : array-like of floats, shape=(ndim,)
        1D array-like object which gives the origin of lattice in ndim
    scale : float or array-like of floats, shape=(ndim,)
        If a float then cast to an 1D array of length ndim. The 1D array is used to scale the data space
    family : string in ({0})
        Gives the family of lattices to generate. See NOTES1.
    packing_radius : float (optional)
        This is used to modify the scale. scale \= packing_radius
        
    Returns
    -------
    Lattice : {1}
        Depending on family, this returns a lattice object
    
    
    Notes
    -----
    __1)__ Families of lattices are defined TODO: finish
        * ALattice : ndim=2 is a hexbin lattice
    
    """.format(", ".join(families.keys()),
               ", ".join([val.__name__ for val in families.values()]))

    
    