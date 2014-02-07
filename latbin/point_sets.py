# Licensed under a 3-clause BSD style license - see LICENSE

# Standard Library
from __future__ import print_function, division
from copy import deepcopy

# 3rd Party
import numpy as np
#vector quantization
import scipy.cluster.vq as vq

# Internal

__all__ = ["PointInformation","PointSet"]

# ########################################################################### #

class PointInformation (dict):
    """
    TODO: finish doc string
    """
    
    def __init__ (self,*args,**kwargs):
        super(PointInformation,self).__init__(*args,**kwargs)
        
    def _type_check_other (self,other):
        if not isinstance(other,dict):
            raise TypeError("other must be a subclass of dict")
    
    def __add__ (self,other):
        return self.add(other,fill=0)
            
#     def __iadd__ (self,other):
#         self._type_check_other(other)
#         for key in other:
#             if self.has_key(key):
#                 continue
#             self[key] = other[key]
#         return self    
      
    def __setitem__ (self,key,value):
        dict.__setitem__(self,key,value)
      
    def _operation (self,other,operator,fill):
        pass
     
    def add (self,other,fill=0):
        self._type_check_other(other)
        key_set = set(self.keys())
        for key in other.keys():
            key_set.add(key)
        pi = PointInformation()
        for key in key_set:
            first = self.get(key, fill)
            second = other.get(key, fill)
            pi[key] = first + second
        return pi
    
    def sub (self,other,fill=0):
        self._type_check_other(other)
        key_set = set(self.keys())
        for key in other.keys():
            key_set.add(key)
        pi = PointInformation()
        for key in key_set:
            first = self.get(key, fill)
            second = other.get(key, fill)
            pi[key] = first - second
        return pi
        pass
    
    def mul (self,other,fill=1):
        self._type_check_other(other)
        key_set = set(self.keys())
        for key in other.keys():
            key_set.add(key)
        pi = PointInformation()
        for key in key_set:
            first = self.get(key, fill)
            second = other.get(key, fill)
            pi[key] = first * second
        return pi
     
    def div (self,other,fill=np.nan):
        self._type_check_other(other)
        key_set = set(self.keys())
        for key in other.keys():
            key_set.add(key)
        pi = PointInformation()
        for key in key_set:
            first = self.get(key, fill)
            second = other.get(key, fill)
            pi[key] = first / second
        return pi
    
    def copy (self):
        return deepcopy(self)
    
class PointSet (object):
    
    def __init__ (self, point_coordinates):
        """
        Paramters
        ---------
        point_coordinates : ndarray, size=(n_points , n_dims) 
        
        """
        self.points = np.asarray(point_coordinates)
    
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
            return vqres[0],vqres[1]
        else:
            return vqres[0],None
        
        
        
    