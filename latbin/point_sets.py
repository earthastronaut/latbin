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
           
    def __setitem__ (self,key,value):
        dict.__setitem__(self,key,value)
 
    def copy (self):
        return deepcopy(self)
     
    pass 
    # --------------------------------------------------------------------------- #
      
    def operation (self,other,operator,fill=np.nan):
        """
        This performs an operation on the information
        
        Parameters
        ----------
        other : dict, PointInformation, any
        operator : callable, takes two values
        fill : any
        
        Returns
        -------
        point_info : PointInformation
        
        
        """
        point_info = PointInformation()
        
        if not isinstance(other,dict):
            for key in self.keys():
                point_info[key] = operator(self[key],other)
            return point_info
         
        key_set = set(self.keys())
        for key in other.keys():
            key_set.add(key)
        for key in key_set:
            first = self.get(key, fill)
            second = other.get(key, fill)
            point_info[key] = operator(first,second)
            
        return point_info
 
    def inplace_operation (self,other,operator,fill=np.nan):
        """
        This performs an operation on the information
        
        Parameters
        ----------
        other : dict, PointInformation, any
        operator : callable, takes two values
        fill : any
        
        Returns
        -------
        None - changes the current data in self
        
        
        """
        if not isinstance(other,dict):
            for key in self.keys():
                self[key] = operator(self[key],other)
        
        for key,second in other.iteritems():
            first = self.get(key,fill)
            self[key] = operator(first,second)            
                 
    def __add__ (self,other):
        # Note: having this as a separate function call is slightly slower
        # computationally for low numbers (~10), but at high numbers (>100) is 
        # a small effect. 
        return self.operation(other, operator=lambda x,y:x+y, fill=0)

    def __iadd__ (self,other):
        self.inplace_operation(other, operator=lambda x,y:x+y, fill=0)
        
    def __sub__ (self,other):
        return self.operation(other, operator=lambda x,y:x-y, fill=0)    

    def __isub__ (self,other):
        return self.inplace_operation(other, operator=lambda x,y:x-y, fill=0)    

    def __mul__ (self,other):
        return self.operation(other, operator=lambda x,y:x*y, fill=1)    

    def __imul__ (self,other):
        return self.inplace_operation(other, operator=lambda x,y:x*y, fill=1)    

    def __div__ (self,other):
        return self.operation(other, operator=lambda x,y:x/y, fill=np.nan)        
     
    def __idiv__ (self,other):
        return self.inplace_operation(other, operator=lambda x,y:x*y, fill=1)    
    
    def __rshift__ (self,other):
        return self.inplace_operation(other, operator=lambda x,y:x<<y, fill=0)    

    def __irshift__ (self,other):
        return self.inplace_operation(other, operator=lambda x,y:x<<y, fill=0)        

    def __neg__ (self):
        return self.__mul__(-1)

    def add (self,other,fill=0):
        return self.operation(other, operator=lambda x,y:x+y, fill=fill)
        
    def sub (self,other,fill=0):
        return self.operation(other, operator=lambda x,y:x-y, fill=fill)    
    
    def mul (self,other,fill=1):
        return self.operation(other, operator=lambda x,y:x*y, fill=fill)    
     
    def div (self,other,fill=np.nan):
        return self.operation(other, operator=lambda x,y:x/y, fill=fill)            
        
class PointSet (object):
    
    def __init__ (self, point_coordinates):
        """
        Parameters
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
        
        
        
    