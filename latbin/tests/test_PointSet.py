# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard Library
import unittest

# 3rd Party
import numpy as np

# Internal
import latbin
from latbin import PointInformation
from latbin import ZLattice, Lattice, generate_lattice


# ########################################################################### #

class TestPointInformation (unittest.TestCase):
    
    def setUp(self):
        
    
        self.pi1 = PointInformation(a=1,b=3,c='hello',e=[1,2,3,4])
        self.pi2 = PointInformation(    b=2,c=' world',e=[5,6])
        
        self.pi3 = PointInformation(a=3.2,b=2.3,c=1.2)
        self.pi4 = PointInformation(      b=5,  c=0, d=-2.3)
        
        unittest.TestCase.setUp(self)   

    def test_add (self):
        
        PI = PointInformation
        
        # these pairs are the (test,answer)
        rpairs = [((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ((self.pi1, self.pi2), PI(a=1,b=5,c='hello world',e=[1,2,3,4,5,6])),
                  #((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  #((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ]
        
        for i,(operands, true_result) in enumerate(rpairs):
            op1, op2 = operands
            add_result = op1.add(op2,fill=0)
            
            msg = " .add failed for example {}".format(i)
            self.assertEqual(add_result, true_result , msg)
        
            dunder_add_result = op1 + op2
            msg = " __add__  failed for example {}".format(i)
            self.assertEqual(dunder_add_result, true_result , msg)
        
        # TODO: finish the tests
        
    def test_sub(self):
        pass

    def test_mul(self):
        pass

    def test_div(self):
        pass
    
class TestCore (unittest.TestCase):
    
    
    def setUp (self):
        self.random_2d_pts = 20.0*(np.random.random((100, 1))-0.5)
        self.random_2d_pts = 20.0*(np.random.random((100, 2))-0.5)
        self.random_3d_pts = 20.0*(np.random.random((100, 3))-0.5)
        self.random_4d_pts = 20.0*(np.random.random((100, 4))-0.5)
    
        #parameters = []
        #self.point_set = latbin.PointSet()
    
        self.result2 = None
        
        self.pad = None
        
        unittest.TestCase.setUp(self)
        
        self.z1 = ZLattice(1, ndim=1, origin=(-1,), scale=2.1)
        self.z2 = ZLattice(1, ndim=2, origin=(-1, 0), scale=(2.1,2))
        # TODO: write test for ndim=5
    
    def test_z1_coords(self):
        #lattice coordinate, data coordinate pairs
        rpairs = [([[0],[1]],  [[-1], [1.1]])]
        
        for i, (lc, dc) in enumerate(rpairs):
            output_dc = self.z1.to_data_coords(lc)
            self.assertTrue(np.all(dc==output_dc), msg="bad {}".format(i))
            
            output_lc = self.z1.to_lattice_coords(dc)
            self.assertTrue(np.all(lc==output_lc), msg="bad {}".format(i))

    def test_z2_coords(self):
        #lattice coordinate, data coordinate pairs
        rpairs = [([[0, 0],[1, 3.5]],  [[-1, 0], [1.1, 7.0]])]
        
        for i, (lc, dc) in enumerate(rpairs):
            output_dc = self.z2.to_data_coords(lc)
            self.assertTrue(np.all(dc==output_dc), msg="bad {}".format(i))
            
            output_lc = self.z2.to_lattice_coords(dc)
            self.assertTrue(np.all(lc==output_lc), msg="bad {}".format(i))
       
    def test_z1_quantize (self):
        
        pass
        
        
            
    def test_2d (self):
        
        # set up the point set latbinner 
        
        # bin two data sets
            
        # subtract the two data sets
        
        # compare to correct result

        pass
    
    def test_quantize(self):
        pass
        


# ########################################################################### #

if __name__ == "__main__":
    unittest.main()
