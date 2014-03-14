# Licensed under a 3-clause BSD style license - see LICENSE

# Standard Library
from __future__ import print_function, division
from copy import deepcopy

# 3rd Party
import numpy as np

# Internal
import latbin

# ########################################################################### #

__all__ = ["PointInformation"]

# ########################################################################### #

# TODO: Make this a Pandas.dataframe or Pandas.dataseries

class InformationDict (dict):
    """
    Stores dict information which can be combined and operated upon
    
    """
    def __init__ (self,*args,**kwargs):
        super(InformationDict,self).__init__(*args,**kwargs)
           
    def __setitem__ (self,key,value):
        dict.__setitem__(self,key,value)
 
    def copy (self):
        return deepcopy(self)
     
    pass 
    # --------------------------------------------------------------------------- #
      
    def operation (self,other,operator,fill=np.nan):
        """This performs an operation on the information and returns a new
        object
        
        Parameters
        ----------
        other : dict, PointInformation, any
        operator : callable, takes two values
        fill : any
        
        Returns
        -------
        point_info : PointInformation
        
        """
        point_info = InformationDict()

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
        """This performs an operation on the information inplace
        
        Parameters
        ----------
        other : dict, PointInformation, any
        operator : callable, takes two values
        fill : any
        
        Returns
        -------
        self : type(self)
            changes the current data in self
        
        """
        if not isinstance(other,dict):
            for key in self.keys():
                self[key] = operator(self[key],other)
            return self
            
        for key,second in other.iteritems():
            first = self.get(key,fill)
            self[key] = operator(first,second)            
        return self
         
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
        return self.inplace_operation(other, operator=lambda x,y:x/y, fill=1)    
    
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

class PointInformation (InformationDict):
    """
    Stores information for each point
        
    """
    def __init__ (self,lattice,*args,**kwargs):
        """
        Parameters
        ----------
        lattice : `latbin.Lattice`
            The associated lattice object of the point information
        *args, **kwargs : defaults for dict(*args,**kwargs)
        
        """
        if not isinstance(lattice, latbin.Lattice):
            raise TypeError("lattice must be instance of `latbin.Lattice`")
        self.lattice = lattice 
        super(PointInformation, self).__init__(*args,**kwargs)
       
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
        point_info = PointInformation(self.lattice)

        if not isinstance(other,dict):
            for key in self.keys():
                point_info[key] = operator(self[key],other)
            return point_info
        
        if not (self.lattice == other.lattice):
            raise ValueError(("can only compare identical lattices: "
                             "'{}' != '{}'").format(repr(self.lattice),
                                                    repr(other.lattice)))                 
        key_set = set(self.keys())
        for key in other.keys():
            key_set.add(key)
        for key in key_set:
            first = self.get(key, fill)
            second = other.get(key, fill)
            point_info[key] = operator(first,second)
            
        return point_info

    operation.__doc__ = InformationDict.operation.__doc__
    
    def inplace_operation (self,other,operator,fill=np.nan):       
        if not isinstance(other,dict):
            for key in self.keys():
                self[key] = operator(self[key],other)
            return self
        
        if not (self.lattice == other.lattice):
            raise ValueError(("can only compare identical lattices: "
                             "'{}' != '{}'").format(repr(self.lattice),
                                                    repr(other.lattice)))        
        for key,second in other.iteritems():
            first = self.get(key,fill)
            self[key] = operator(first,second)            
        return self
    
    inplace_operation.__doc__ = InformationDict.inplace_operation.__doc__
    
    def centers (self):
        """ Get the centers of the lattice points with values
        
        Returns
        -------
        centers : ndarray, shape=(N,ndim)
        
        """
        return self.lattice.lattice_to_data_space(self.keys())
        